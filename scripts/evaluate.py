"""Evaluate the pipeline against the golden question set.

Run this AFTER ingestion. It is the tool for tuning the knobs
that cannot be chosen from first principles -- the confidence threshold, top-k,
and whether query expansion is earning its latency.

    python scripts/evaluate.py --retrieval-only     # fast, no Groq calls
    python scripts/evaluate.py                      # full pipeline
    python scripts/evaluate.py --no-expansion       # A/B the expansion stage
    python scripts/evaluate.py --json runs/eval.json

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
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from medical_rag.config import configure_logging, get_settings  # noqa: E402
from medical_rag.pipeline import MedicalRAGPipeline  # noqa: E402

console = Console()
GOLDEN = Path(__file__).resolve().parents[1] / "eval" / "golden_questions.json"


def first_hit_rank(hits, expect_any: list[str]) -> int | None:
    """1-based rank of the first chunk containing any expected term."""
    wanted = [t.lower() for t in expect_any]
    for rank, hit in enumerate(hits, start=1):
        text = hit.chunk.text.lower()
        if any(term in text for term in wanted):
            return rank
    return None


def run_retrieval(pipeline: MedicalRAGPipeline, cases: list[dict], top_k: int) -> list[dict]:
    rows: list[dict] = []
    for case in cases:
        started = time.perf_counter()
        try:
            hits = pipeline.search(case["query"], top_k=top_k, rerank=True)
            error = ""
        except Exception as exc:
            hits, error = [], f"{type(exc).__name__}: {exc}"
        elapsed = (time.perf_counter() - started) * 1000

        rank = first_hit_rank(hits, case.get("expect_any", [])) if hits else None
        rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "hit": rank is not None,
                "rank": rank,
                "n_hits": len(hits),
                "top_score": round(hits[0].rerank_score or 0.0, 3) if hits else 0.0,
                "ms": round(elapsed, 1),
                "error": error,
            }
        )
    return rows


def run_full(pipeline: MedicalRAGPipeline, cases: list[dict], top_k: int) -> list[dict]:
    rows: list[dict] = []
    for case in cases:
        try:
            answer = pipeline.answer(case["query"], top_k=top_k)
            error = ""
        except Exception as exc:
            rows.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "hit": False,
                    "rank": None,
                    "refused": True,
                    "confidence": 0.0,
                    "conf_passed": False,
                    "citations": 0,
                    "ms": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        rank = first_hit_rank(answer.retrieved, case.get("expect_any", []))
        rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "hit": rank is not None,
                "rank": rank,
                "refused": answer.refused,
                "confidence": round(answer.confidence.score, 3) if answer.confidence else 0.0,
                "conf_passed": bool(answer.confidence and answer.confidence.passed),
                "citations": len(answer.citations),
                "ms": round(sum(answer.timings_ms.values()), 1),
                "error": error,
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
    }
    if "confidence" in rows[0]:
        confidences = [r["confidence"] for r in rows]
        summary["mean_confidence"] = round(sum(confidences) / len(confidences), 3)
        summary["conf_pass_rate"] = round(
            sum(1 for r in rows if r["conf_passed"]) / len(rows), 3
        )
        summary["refusal_rate"] = round(sum(1 for r in rows if r["refused"]) / len(rows), 3)
    return summary


def print_retrieval(rows: list[dict], full: bool) -> None:
    table = Table(title="Retrieval" if not full else "Full pipeline", show_lines=False)
    table.add_column("id", width=22)
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
        cells += [str(r["ms"]), (r["error"] or "")[:40]]
        table.add_row(*cells)
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
        "--no-expansion", action="store_true", help="Disable query expansion, to A/B its value"
    )
    parser.add_argument("--json", default=None, help="Write full results to this JSON file")
    args = parser.parse_args()

    configure_logging("WARNING")
    settings = get_settings()
    if args.no_expansion:
        settings.query_expansion_enabled = False

    data = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    pipeline = MedicalRAGPipeline(settings)
    top_k = args.top_k or settings.final_top_k

    console.print(
        f"[bold]Medical RAG evaluation[/bold]  "
        f"device={settings.resolve_device()}  top_k={top_k}  "
        f"expansion={'off' if args.no_expansion else settings.query_expansion_mode}  "
        f"threshold={settings.confidence_threshold}\n"
    )

    retrieval_cases = data.get("retrieval", [])
    if args.retrieval_only:
        rows = run_retrieval(pipeline, retrieval_cases, top_k)
        guardrail_rows: list[dict] = []
    else:
        rows = run_full(pipeline, retrieval_cases, top_k)
        guardrail_rows = run_guardrails(pipeline, data.get("guardrails", []))

    print_retrieval(rows, full=not args.retrieval_only)
    summary = summarise(rows)

    console.print("\n[bold]Summary[/bold]")
    for key, value in summary.items():
        console.print(f"  {key:>22} : {value}")

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
