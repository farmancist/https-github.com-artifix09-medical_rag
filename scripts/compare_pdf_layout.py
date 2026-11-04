"""Compare PDF extraction with PyMuPDF layout analysis enabled and disabled.

By default, compares a deterministic, evenly distributed sample of 100 pages.
Use --all-pages to compare every page in the document.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pymupdf  # noqa: E402

from medical_rag.ingest.pdf_parser import PDFParser  # noqa: E402


def sample_page_indices(total_pages: int, sample_pages: int) -> list[int]:
    if sample_pages >= total_pages:
        return list(range(total_pages))
    if sample_pages <= 0:
        raise ValueError("sample_pages must be positive")
    if sample_pages == 1:
        return [0]
    return sorted({round(i * (total_pages - 1) / (sample_pages - 1)) for i in range(sample_pages)})


def page_text(blocks, page: int, block_type: str | None = None) -> str:
    return "\n".join(
        block.text
        for block in blocks
        if block.page == page and (block_type is None or block.block_type == block_type)
    )


def summarize(parsed, selected_pages: list[int], elapsed_seconds: float) -> dict:
    page_metrics = {}
    for page_index in selected_pages:
        page_no = page_index + 1
        page_blocks = [block for block in parsed.blocks if block.page == page_no]
        page_metrics[str(page_no)] = {
            "blocks": len(page_blocks),
            "table_blocks": sum(block.block_type == "table" for block in page_blocks),
            "text_chars": sum(
                len(block.text) for block in page_blocks if block.block_type == "text"
            ),
            "table_chars": sum(
                len(block.text) for block in page_blocks if block.block_type == "table"
            ),
        }
    return {
        "elapsed_seconds": round(elapsed_seconds, 3),
        "pages_parsed": len(selected_pages),
        "blocks": len(parsed.blocks),
        "table_blocks": sum(block.block_type == "table" for block in parsed.blocks),
        "text_chars": sum(
            len(block.text) for block in parsed.blocks if block.block_type == "text"
        ),
        "table_chars": sum(
            len(block.text) for block in parsed.blocks if block.block_type == "table"
        ),
        "pages": page_metrics,
    }


def compare(path: Path, selected_pages: list[int]) -> dict:
    started = time.perf_counter()
    without_layout = PDFParser(use_layout_analysis=False).parse(
        path, page_indices=selected_pages
    )
    without_layout_seconds = time.perf_counter() - started

    layout_started = time.perf_counter()
    with_layout = PDFParser(use_layout_analysis=True).parse(path, page_indices=selected_pages)
    with_layout_seconds = time.perf_counter() - layout_started

    page_comparisons = {}
    changed_pages = []
    for page_index in selected_pages:
        page_no = page_index + 1
        text_without = page_text(without_layout.blocks, page_no)
        text_with = page_text(with_layout.blocks, page_no)
        tables_without = page_text(without_layout.blocks, page_no, "table")
        tables_with = page_text(with_layout.blocks, page_no, "table")
        without_table_count = sum(
            block.page == page_no and block.block_type == "table"
            for block in without_layout.blocks
        )
        with_table_count = sum(
            block.page == page_no and block.block_type == "table"
            for block in with_layout.blocks
        )
        text_similarity = SequenceMatcher(
            None, text_without, text_with, autojunk=False
        ).ratio()
        text_identical = text_without == text_with
        table_text_identical = tables_without == tables_with
        comparison = {
            "text_similarity": round(text_similarity, 4),
            "text_identical": text_identical,
            "table_blocks_without_layout": without_table_count,
            "table_blocks_with_layout": with_table_count,
            "table_text_identical": table_text_identical,
        }
        page_comparisons[str(page_no)] = comparison
        if not text_identical or not table_text_identical or without_table_count != with_table_count:
            changed_pages.append(
                {
                    "page": page_no,
                    "without_layout_excerpt": text_without[:800],
                    "with_layout_excerpt": text_with[:800],
                    "without_layout_table_excerpt": tables_without[:800],
                    "with_layout_table_excerpt": tables_with[:800],
                }
            )

    return {
        "pdf": str(path.resolve()),
        "total_document_pages": without_layout.n_pages,
        "sampled_page_numbers": [index + 1 for index in selected_pages],
        "without_layout": summarize(without_layout, selected_pages, without_layout_seconds),
        "with_layout": summarize(with_layout, selected_pages, with_layout_seconds),
        "comparison": page_comparisons,
        "changed_pages": changed_pages,
        "interpretation": (
            "Text similarity and table counts describe extraction differences, not correctness. "
            "Review differing pages against the source PDF before selecting a mode."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="PDF to compare")
    parser.add_argument(
        "--sample-pages",
        type=int,
        default=100,
        help="Evenly distributed page count (default: 100; ignored with --all-pages)",
    )
    parser.add_argument(
        "--all-pages",
        action="store_true",
        help="Compare every page; can be substantially slower on long PDFs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/pdf-layout-comparison.json"),
        help="JSON report destination (default: runs/pdf-layout-comparison.json)",
    )
    args = parser.parse_args()

    if not args.pdf.is_file():
        parser.error(f"PDF not found: {args.pdf}")
    if args.sample_pages < 1:
        parser.error("--sample-pages must be positive")

    try:
        document = pymupdf.open(args.pdf)
        try:
            total_pages = document.page_count
        finally:
            document.close()
        selected_pages = (
            list(range(total_pages))
            if args.all_pages
            else sample_page_indices(total_pages, args.sample_pages)
        )
        report = compare(args.pdf, selected_pages)
    except Exception as exc:
        print(f"PDF layout comparison failed: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Compared {len(selected_pages)} of {total_pages} pages: {args.pdf}")
    for mode in ("without_layout", "with_layout"):
        result = report[mode]
        print(
            f"{mode:>16}: {result['table_blocks']} table blocks, "
            f"{result['text_chars']:,} text chars, {result['elapsed_seconds']:.1f}s"
        )
    print(f"Report: {args.output}")
    print(report["interpretation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
