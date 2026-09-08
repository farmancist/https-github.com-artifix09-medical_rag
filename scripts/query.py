"""Interactive CLI for querying the Medical RAG pipeline.

    python scripts/query.py "what are the contraindications for metformin?"
    python scripts/query.py            # REPL
    python scripts/query.py --search "heart attack"   # retrieval only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

from medical_rag.config import configure_logging  # noqa: E402
from medical_rag.models import RAGAnswer  # noqa: E402
from medical_rag.pipeline import MedicalRAGPipeline  # noqa: E402

console = Console()


def show_answer(result: RAGAnswer, verbose: bool) -> None:
    style = "red" if result.refused else "green"
    console.print(Panel(Markdown(result.answer), title="Answer", border_style=style))

    if result.expanded_query and result.expanded_query.variants:
        console.print(
            f"[dim]expansion ({result.expanded_query.method}): "
            f"{result.expanded_query.variants}[/dim]"
        )

    if result.confidence:
        c = result.confidence
        verdict = "[green]PASS[/green]" if c.passed else "[red]FAIL[/red]"
        console.print(
            f"[bold]confidence[/bold] {c.score:.3f} / {c.threshold:.2f} {verdict}  "
            f"[dim]{c.components}[/dim]"
        )

    if result.citations:
        table = Table(title="Citations", show_lines=False)
        table.add_column("#", width=4)
        table.add_column("Source")
        table.add_column("Pages", width=10)
        for citation in result.citations:
            pages = (
                str(citation.page_start)
                if citation.page_start == citation.page_end
                else f"{citation.page_start}-{citation.page_end}"
            )
            table.add_row(citation.marker, citation.title[:60], pages)
        console.print(table)

    for verdict in result.guardrails:
        if verdict.code == "ok":
            continue
        colour = {"block": "red", "warn": "yellow", "info": "blue"}[verdict.severity]
        console.print(f"[{colour}]guardrail {verdict.code}[/{colour}]: {verdict.reason[:200]}")

    if verbose and result.retrieved:
        table = Table(title="Retrieved")
        table.add_column("rerank", width=8)
        table.add_column("rrf", width=8)
        table.add_column("source")
        table.add_column("snippet")
        for hit in result.retrieved:
            table.add_row(
                f"{hit.rerank_score:.3f}" if hit.rerank_score is not None else "-",
                f"{hit.rrf_score:.4f}" if hit.rrf_score is not None else "-",
                f"{hit.chunk.title[:30]} p.{hit.chunk.page_start}",
                hit.chunk.text[:120].replace("\n", " "),
            )
        console.print(table)

    console.print(f"[dim]timings(ms): {result.timings_ms}[/dim]\n")


def show_search(hits) -> None:
    table = Table(title="Search results")
    table.add_column("#", width=3)
    table.add_column("rerank", width=8)
    table.add_column("dense", width=8)
    table.add_column("sparse", width=8)
    table.add_column("source", width=34)
    table.add_column("snippet")
    for i, hit in enumerate(hits, start=1):
        table.add_row(
            str(i),
            f"{hit.rerank_score:.3f}" if hit.rerank_score is not None else "-",
            f"{hit.dense_score:.3f}" if hit.dense_score is not None else "-",
            f"{hit.sparse_score:.2f}" if hit.sparse_score is not None else "-",
            f"{hit.chunk.title[:28]} p.{hit.chunk.page_start}",
            hit.chunk.text[:100].replace("\n", " "),
        )
    console.print(table)


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the Medical RAG pipeline")
    parser.add_argument("query", nargs="*", help="Question (omit for REPL)")
    parser.add_argument("--search", action="store_true", help="Retrieval only, no generation")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true", help="Show retrieved passages")
    args = parser.parse_args()

    configure_logging()
    pipeline = MedicalRAGPipeline()

    def handle(text: str) -> None:
        if args.search:
            show_search(pipeline.search(text, top_k=args.top_k))
        else:
            show_answer(pipeline.answer(text, top_k=args.top_k), args.verbose)

    if args.query:
        handle(" ".join(args.query))
        return 0

    console.print("[bold]Medical RAG[/bold] — type a question, or 'exit'.\n")
    while True:
        try:
            text = console.input("[bold cyan]> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0
        if text.lower() in {"exit", "quit", "q"}:
            return 0
        if text:
            handle(text)


if __name__ == "__main__":
    raise SystemExit(main())
