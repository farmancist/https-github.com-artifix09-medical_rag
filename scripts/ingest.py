"""Ingest medical PDFs from RAW_DATA/ into Qdrant.

This is the GPU-side job. Typical use on the desktop:

    python scripts/ingest.py --recreate          # full rebuild
    python scripts/ingest.py                     # resume / add new PDFs
    python scripts/ingest.py --status            # what is done so far

Both stages are resumable, so a crash or Ctrl-C costs one document, not the run:

    python scripts/ingest.py --stage chunk       # parse + semantic split only
    python scripts/ingest.py --stage index       # embed + upload only

To sanity-check the parser and chunker on the laptop before pushing, use
--dry-run, which touches neither the GPU nor Qdrant beyond chunking:

    python scripts/ingest.py --dry-run --source "RAW_DATA/B09474-eng.pdf"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_rag.config import configure_logging, get_settings  # noqa: E402
from medical_rag.ingest.pipeline import IngestionPipeline  # noqa: E402


def print_stats(title: str, stats: dict) -> None:
    print(f"\n=== {title} ===")
    for key, value in stats.items():
        print(f"{key:>20} : {value}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest medical PDFs into Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", default=None, help="PDF file or directory (default: RAW_DATA/)")
    parser.add_argument(
        "--stage",
        choices=("all", "chunk", "index"),
        default="all",
        help="Which stage to run (default: all)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop the collection, clear the manifest, and rebuild everything from scratch",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redo documents already marked done in the manifest, keeping the collection",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk only; print statistics, upload nothing",
    )
    parser.add_argument("--status", action="store_true", help="Show ingestion state and exit")
    args = parser.parse_args()

    configure_logging()
    settings = get_settings()
    pipeline = IngestionPipeline(settings)

    if args.status:
        print(json.dumps(pipeline.status(), indent=2, default=str))
        return 0

    source = Path(args.source) if args.source else settings.raw_dir
    if not source.exists():
        print(f"ERROR: source not found: {source}", file=sys.stderr)
        print("Put your PDFs in RAW_DATA/ or pass --source.", file=sys.stderr)
        return 1

    if args.dry_run:
        pdfs = pipeline.discover(source)
        if not pdfs:
            print("No PDFs found.", file=sys.stderr)
            return 1
        doc = pipeline.parser.parse(pdfs[0])
        chunks = pipeline.chunker.chunk_document(doc)
        print(f"\ndocument : {doc.title[:70]}")
        print(f"pages    : {doc.n_pages}")
        print(f"blocks   : {len(doc.blocks)}")
        print(f"chunks   : {len(chunks)}")
        if chunks:
            sizes = [c.n_tokens for c in chunks]
            print(f"tokens   : min={min(sizes)} mean={sum(sizes) // len(sizes)} max={max(sizes)}")
            print("\n--- first chunk ---")
            print(chunks[0].text[:700])
            print("\n--- a middle chunk ---")
            print(chunks[len(chunks) // 2].text[:700])
        return 0

    if args.stage == "chunk":
        print_stats("chunking", pipeline.stage_chunk(source, force=args.force or args.recreate))
    elif args.stage == "index":
        print_stats("indexing", pipeline.stage_index(recreate=args.recreate, force=args.force))
    else:
        print_stats(
            "ingestion complete",
            pipeline.run(source, recreate=args.recreate, force=args.force),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
