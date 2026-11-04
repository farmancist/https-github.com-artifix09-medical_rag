"""Evaluate the pipeline against the golden question set.

Run this AFTER ingestion. It is the tool for tuning the knobs
that cannot be chosen from first principles -- the confidence threshold, top-k,
and whether query expansion is earning its latency.

    python scripts/evaluate.py --retrieval-only     # fast, no Groq calls
    python scripts/evaluate.py                      # full pipeline
    python scripts/evaluate.py --no-expansion       # A/B the expansion stage
    python scripts/evaluate.py --json runs/eval.json
    python scripts/evaluate.py --source OHCM --retrieval-only --debug-source --show-misses

Metrics:
    hit@k   fraction of questions where a retrieved chunk contains an expected term
    MRR     mean reciprocal rank of the first such chunk
    conf    mean confidence, and how often it clears the threshold

A guardrail case counts as passed when the pipeline refuses (or warns) as the
question set says it should.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from medical_rag.config import configure_logging, get_settings  # noqa: E402
from medical_rag.pipeline import MedicalRAGPipeline  # noqa: E402

console = Console()
GOLDEN = Path(__file__).resolve().parents[1] / "eval" / "golden_questions.json"


def _normalise_match_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    return re.sub(r"\s+", " ", text).casefold()


def eval_matching(text: str, case: dict) -> bool:
    """Apply the golden-set expect_all/expect_any rules to one chunk."""
    normalized_text = _normalise_match_text(text)
    expect_all = [_normalise_match_text(term) for term in case.get("expect_all", [])]
    expect_any = [_normalise_match_text(term) for term in case.get("expect_any", [])]
    if not expect_all and not expect_any:
        return False
    return all(term in normalized_text for term in expect_all) and (
        not expect_any or any(term in normalized_text for term in expect_any)
    )


def first_hit_rank(hits, case: dict) -> int | None:
    """1-based rank of the first chunk satisfying its expected-term rules."""
    for rank, hit in enumerate(hits, start=1):
        if eval_matching(hit.chunk.text, case):
            return rank
    return None


def _source_aliases(source: str) -> tuple[str, ...]:
    lowered = source.casefold()
    if "bnf" in lowered or "british national formulary" in lowered:
        return ("bnf", "british national formulary")
    if "ohcm" in lowered or "oxford handbook" in lowered:
        return ("ohcm", "oxford handbook")
    if "who" in lowered or "world health organization" in lowered:
        return ("who", "world health organization")
    return (lowered,)


def _matches_source(value: str, source: str) -> bool:
    lowered = value.casefold()
    return any(alias in lowered for alias in _source_aliases(source))


def source_manifest_entries(manifest_path: Path, source: str | None) -> dict[str, dict]:
    if not manifest_path.exists():
        return {}
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source is None:
        return entries
    return {
        doc_id: entry
        for doc_id, entry in entries.items()
        if _matches_source(f"{entry.get('title', '')} {entry.get('source', '')}", source)
    }


def _cases_for_source(cases: list[dict], source: str | None) -> list[dict]:
    if source is None:
        return cases
    return [
        case
        for case in cases
        if "/" not in str(case.get("source", ""))
        and _matches_source(str(case.get("source", "")), source)
    ]


def run_retrieval(
    pipeline: MedicalRAGPipeline,
    cases: list[dict],
    top_k: int,
    *,
    doc_ids: Sequence[str] | None = None,
    source: str | None = None,
    show_misses: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    for case in cases:
        started = time.perf_counter()
        if source is not None and not doc_ids:
            hits, error = [], ""
        else:
            try:
                hits = pipeline.search(
                    case["query"], top_k=top_k, rerank=True, doc_ids=doc_ids
                )
                error = ""
            except Exception as exc:
                hits, error = [], f"{type(exc).__name__}: {exc}"
        elapsed = (time.perf_counter() - started) * 1000

        rank = first_hit_rank(hits, case) if hits else None
        no_source_chunks = bool(source is not None and not hits and not error)
        note = case.get("note", "")
        if no_source_chunks:
            note = (
                f"no indexed {source} chunks"
                if not doc_ids
                else f"no {source} chunks retrieved"
            )
        evidence = []
        if show_misses and rank is None:
            evidence = [
                {
                    "title": hit.chunk.title,
                    "section": hit.chunk.section,
                    "page_start": hit.chunk.page_start,
                    "page_end": hit.chunk.page_end,
                    "text": hit.chunk.text[:300],
                }
                for hit in hits[:3]
            ]
        rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "category": case.get("category", "baseline"),
                "hit": rank is not None,
                "rank": rank,
                "n_hits": len(hits),
                "top_score": round(hits[0].rerank_score or 0.0, 3) if hits else 0.0,
                "ms": round(elapsed, 1),
                "error": error,
                "note": note,
                "no_source_chunks": no_source_chunks,
                "evidence": evidence,
            }
        )
    return rows


def run_full(
    pipeline: MedicalRAGPipeline,
    cases: list[dict],
    top_k: int,
    *,
    doc_ids: Sequence[str] | None = None,
    source: str | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for case in cases:
        if source is not None and not doc_ids:
            rows.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "category": case.get("category", "baseline"),
                    "hit": False,
                    "rank": None,
                    "refused": True,
                    "confidence": 0.0,
                    "conf_passed": False,
                    "citations": 0,
                    "ms": 0.0,
                    "error": "",
                    "note": f"no indexed {source} chunks",
                    "no_source_chunks": True,
                }
            )
            continue
        try:
            answer = pipeline.answer(case["query"], top_k=top_k, doc_ids=doc_ids)
            error = ""
        except Exception as exc:
            rows.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "category": case.get("category", "baseline"),
                    "hit": False,
                    "rank": None,
                    "refused": True,
                    "confidence": 0.0,
                    "conf_passed": False,
                    "citations": 0,
                    "ms": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "note": f"no {source} chunks" if source and not doc_ids else "",
                    "no_source_chunks": bool(source and not doc_ids),
                }
            )
            continue

        rank = first_hit_rank(answer.retrieved, case)
        rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "category": case.get("category", "baseline"),
                "hit": rank is not None,
                "rank": rank,
                "refused": answer.refused,
                "confidence": round(answer.confidence.score, 3) if answer.confidence else 0.0,
                "conf_passed": bool(answer.confidence and answer.confidence.passed),
                "citations": len(answer.citations),
                "ms": round(sum(answer.timings_ms.values()), 1),
                "error": error,
                "note": case.get("note", ""),
                "no_source_chunks": bool(source and not answer.retrieved),
            }
        )
    return rows


def run_guardrails(pipeline: MedicalRAGPipeline, cases: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for case in cases:
        try:
            answer = pipeline.answer(case["query"])
            codes = {v.code for v in answer.guardrails}
            expected_code = case.get("expect_code", "")
            if case["expect"] == "refused":
                passed = answer.refused and (not expected_code or expected_code in codes)
            else:  # "warned": answered, but the warning was raised
                passed = expected_code in codes
            rows.append(
                {
                    "id": case["id"],
                    "expect": case["expect"],
                    "expect_code": expected_code,
                    "refused": answer.refused,
                    "codes": sorted(codes - {"ok"}),
                    "passed": passed,
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "id": case["id"],
                    "expect": case["expect"],
                    "expect_code": case.get("expect_code", ""),
                    "refused": False,
                    "codes": [],
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {}
    ranks = [r["rank"] for r in rows if r.get("rank")]
    summary = {
        "questions": len(rows),
        "hit_rate": round(sum(1 for r in rows if r["hit"]) / len(rows), 3),
        "mrr": round(sum(1 / r for r in ranks) / len(rows), 3) if ranks else 0.0,
        "median_rank_of_hits": statistics.median(ranks) if ranks else None,
        "median_ms": statistics.median([r["ms"] for r in rows]) if rows else 0.0,
        "errors": sum(1 for r in rows if r["error"]),
        "no_source_chunks": sum(1 for r in rows if r.get("no_source_chunks", False)),
    }
    if "confidence" in rows[0]:
        confidences = [r["confidence"] for r in rows]
        summary["mean_confidence"] = round(sum(confidences) / len(confidences), 3)
        summary["conf_pass_rate"] = round(
            sum(1 for r in rows if r["conf_passed"]) / len(rows), 3
        )
        summary["refusal_rate"] = round(sum(1 for r in rows if r["refused"]) / len(rows), 3)
    categories = sorted({r.get("category", "baseline") for r in rows})
    summary["by_category"] = {
        category: _summarise_category(
            [row for row in rows if row.get("category", "baseline") == category]
        )
        for category in categories
    }
    return summary


def _summarise_category(rows: list[dict]) -> dict:
    ranks = [r["rank"] for r in rows if r.get("rank")]
    return {
        "questions": len(rows),
        "hit_rate": round(sum(1 for r in rows if r["hit"]) / len(rows), 3),
        "mrr": round(sum(1 / rank for rank in ranks) / len(rows), 3) if ranks else 0.0,
    }


def print_retrieval(rows: list[dict], full: bool) -> None:
    table = Table(title="Retrieval" if not full else "Full pipeline", show_lines=False)
    table.add_column("id", width=28)
    table.add_column("hit", width=5)
    table.add_column("rank", width=5)
    if full:
        table.add_column("conf", width=6)
        table.add_column("refused", width=8)
        table.add_column("cites", width=6)
    else:
        table.add_column("top", width=6)
    table.add_column("ms", width=8)
    table.add_column("note")

    for r in rows:
        mark = "[green]Y[/green]" if r["hit"] else "[red]n[/red]"
        cells = [r["id"], mark, str(r["rank"] or "-")]
        if full:
            colour = "green" if r["conf_passed"] else "yellow"
            cells += [
                f"[{colour}]{r['confidence']:.2f}[/{colour}]",
                "yes" if r["refused"] else "no",
                str(r["citations"]),
            ]
        else:
            cells.append(f"{r['top_score']:.2f}")
        cells += [str(r["ms"]), (r.get("note") or r["error"] or "")[:40]]
        table.add_row(*cells)
    console.print(table)


def print_miss_evidence(rows: list[dict], source: str | None) -> None:
    misses = [row for row in rows if not row["hit"]]
    if not misses:
        console.print("\n[green]No misses.[/green]")
        return
    console.print("\n[bold]Miss evidence (up to 3 retrieved chunks per case)[/bold]")
    for row in misses:
        console.print(f"\n[bold]{row['id']}[/bold]  {row['query']}")
        if row.get("note"):
            console.print(f"  {row['note']}")
        evidence = row.get("evidence", [])
        if not evidence:
            if row.get("n_hits", 0) == 0:
                console.print("  No chunks retrieved.")
            else:
                console.print("  No evidence captured.")
            continue
        for item in evidence:
            section = f" > {item['section']}" if item.get("section") else ""
            pages = (
                f"p.{item['page_start']}"
                if item["page_start"] == item["page_end"]
                else f"pp.{item['page_start']}-{item['page_end']}"
            )
            console.print(f"  {item['title']}{section} ({pages})")
            console.print(f"    {item['text']}")


def print_source_debug(source: str | None, entries: dict[str, dict]) -> None:
    title = f"Source inventory ({source})" if source else "Source inventory"
    table = Table(title=title)
    table.add_column("doc_id")
    table.add_column("title")
    table.add_column("chunks", justify="right")
    table.add_column("indexed points", justify="right")
    if not entries:
        message = f"No manifest entries matched {source!r}." if source else "Manifest is empty."
        table.add_row("-", message, "-", "-")
    else:
        for doc_id, entry in entries.items():
            table.add_row(
                doc_id,
                str(entry.get("title", "")),
                str(entry.get("chunks", 0)),
                str(entry.get("indexed_points", 0)),
            )
    console.print(table)


def print_by_category(summary: dict) -> None:
    categories = summary.get("by_category", {})
    if not categories:
        return
    table = Table(title="By category")
    table.add_column("category")
    table.add_column("questions", justify="right")
    table.add_column("hit rate", justify="right")
    table.add_column("MRR", justify="right")
    for category, values in categories.items():
        table.add_row(
            category,
            str(values["questions"]),
            f"{values['hit_rate']:.3f}",
            f"{values['mrr']:.3f}",
        )
    console.print(table)


def print_guardrails(rows: list[dict]) -> None:
    table = Table(title="Guardrails")
    table.add_column("id", width=22)
    table.add_column("expected", width=10)
    table.add_column("result", width=8)
    table.add_column("codes raised")
    for r in rows:
        table.add_row(
            r["id"],
            f"{r['expect']}/{r['expect_code']}",
            "[green]PASS[/green]" if r["passed"] else "[red]FAIL[/red]",
            ", ".join(r["codes"]) or (r["error"][:40] if r["error"] else "-"),
        )
    console.print(table)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the Medical RAG pipeline")
    parser.add_argument("--questions", default=str(GOLDEN), help="Path to the question set")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip generation and guardrails (no Groq calls, much faster)",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--source", default=None, help="Evaluate cases/documents tagged for this source"
    )
    parser.add_argument(
        "--debug-source",
        action="store_true",
        help="Show matching manifest documents and their indexing status",
    )
    parser.add_argument(
        "--show-misses",
        action="store_true",
        help="Print source metadata and text snippets for retrieved misses",
    )
    parser.add_argument(
        "--no-expansion", action="store_true", help="Disable query expansion, to A/B its value"
    )
    parser.add_argument("--json", default=None, help="Write full results to this JSON file")
    args = parser.parse_args()

    configure_logging("WARNING")
    settings = get_settings()
    if args.no_expansion:
        settings.query_expansion_enabled = False

    data = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    retrieval_cases = _cases_for_source(data.get("retrieval", []), args.source)
    if args.source and not retrieval_cases:
        parser.error(f"No retrieval cases are tagged for source {args.source!r}")

    manifest_path = settings.processed_dir / "manifest.json"
    source_entries = source_manifest_entries(manifest_path, args.source)
    indexed_doc_ids = [
        doc_id
        for doc_id, entry in source_entries.items()
        if int(entry.get("indexed_points") or 0) > 0
    ]
    if args.debug_source:
        print_source_debug(args.source, source_entries)
    source_doc_ids = indexed_doc_ids if args.source else None

    pipeline = MedicalRAGPipeline(settings)
    top_k = args.top_k or settings.final_top_k
    source_status = (
        f"  source={args.source} ({len(indexed_doc_ids)} indexed docs)" if args.source else ""
    )

    console.print(
        f"[bold]Medical RAG evaluation[/bold]  "
        f"device={settings.resolve_device()}  top_k={top_k}  "
        f"expansion={'off' if args.no_expansion else settings.query_expansion_mode}  "
        f"threshold={settings.confidence_threshold}"
        f"{source_status}\n"
    )

    if args.retrieval_only:
        rows = run_retrieval(
            pipeline,
            retrieval_cases,
            top_k,
            doc_ids=source_doc_ids,
            source=args.source,
            show_misses=args.show_misses,
        )
        guardrail_rows: list[dict] = []
    else:
        rows = run_full(
            pipeline, retrieval_cases, top_k, doc_ids=source_doc_ids, source=args.source
        )
        guardrail_rows = run_guardrails(pipeline, data.get("guardrails", []))

    print_retrieval(rows, full=not args.retrieval_only)
    summary = summarise(rows)

    console.print("\n[bold]Summary[/bold]")
    for key, value in summary.items():
        if key != "by_category":
            console.print(f"  {key:>22} : {value}")
    print_by_category(summary)
    if args.show_misses:
        print_miss_evidence(rows, args.source)

    if guardrail_rows:
        console.print()
        print_guardrails(guardrail_rows)
        passed = sum(1 for r in guardrail_rows if r["passed"])
        colour = "green" if passed == len(guardrail_rows) else "red"
        console.print(
            f"\n  [{colour}]guardrails: {passed}/{len(guardrail_rows)} passed[/{colour}]"
        )

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "settings": {
                        "device": settings.resolve_device(),
                        "top_k": top_k,
                        "source": args.source,
                        "source_doc_ids": source_doc_ids,
                        "expansion": None if args.no_expansion else settings.query_expansion_mode,
                        "confidence_threshold": settings.confidence_threshold,
                        "embedding_model": settings.embedding_model,
                        "reranker_model": settings.reranker_model,
                    },
                    "summary": summary,
                    "retrieval": rows,
                    "guardrails": guardrail_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        console.print(f"\n  wrote {out}")

    # Non-zero exit if any guardrail failed, so this can gate a deploy.
    return 1 if any(not r["passed"] for r in guardrail_rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
