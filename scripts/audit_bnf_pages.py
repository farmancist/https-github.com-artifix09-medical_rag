"""Run a sampled BNF parser -> chunker -> isolated Qdrant indexing audit.

The 24 page screenshots supplied for the BNF audit include two close-up
reference images (dose-box color and typography) that are not separate pages.
This script uses the actual PDF page sequence and leaves the active collection
untouched by indexing into an in-memory Qdrant client.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from qdrant_client import QdrantClient

from medical_rag.config import get_settings
from medical_rag.embeddings.encoder import get_encoder
from medical_rag.embeddings.sparse import BM25Encoder
from medical_rag.ingest.pipeline import IngestionPipeline
from medical_rag.ingest.pdf_parser import PDFParser
from medical_rag.ingest.semantic_chunker import SemanticChunker
from medical_rag.models import Chunk
from medical_rag.vectorstore.qdrant_store import QdrantStore


# One-based PDF page numbers mapped from the printed folios in the screenshots.
# Front matter pages are identified by their displayed contents (PDF pages 6,
# 14 and 15); Arabic BNF folios in this edition are offset by 19 PDF pages.
PDF_PAGES = [
    6, 14, 15, 72, 396, 398, 706, 713, 822, 831, 911, 1146,
    1282, 1283, 1287, 1289, 1291, 1292, 1297,
    1530, 1535, 1536, 1541, 1713,
]
PRINTED_PAGES = {
    6: "iv", 14: "xii", 15: "xiii", 72: "53", 396: "377", 398: "379",
    706: "687", 713: "694", 822: "803", 831: "812", 911: "892", 1146: "1127",
    1282: "1263", 1283: "1264", 1287: "1268", 1289: "1270", 1291: "1272",
    1292: "1273", 1297: "1278", 1530: "1511", 1535: "1516", 1536: "1517",
    1541: "1522", 1713: "1694",
}
QUERIES = [
    "TABLE 2 drugs that cause nephrotoxicity",
    "Naproxen acute gout initially 750 mg",
    "Tiaprofenic acid adult 300 mg twice daily",
    "opioid toxicity naloxone child dose intravenous injection",
    "Acalabrutinib drug interactions abemaciclib",
    "Osilodrostat opioids interaction predicted exposure",
]


def _hit(chunk: Chunk, score: float) -> dict:
    return {
        "score": round(score, 5),
        "pdf_pages": [chunk.page_start, chunk.page_end],
        "printed_pages": chunk.metadata.get("printed_pages", []),
        "entity": chunk.metadata.get("entity"),
        "field": chunk.metadata.get("field"),
        "indication": chunk.metadata.get("indication"),
        "table_title": chunk.metadata.get("table_title"),
        "has_table": chunk.metadata.get("has_table", False),
        "has_dose_box": chunk.metadata.get("has_dose_box", False),
        "excerpt": chunk.text[:260],
    }


def run(source: Path, report_path: Path) -> dict:
    source = source.resolve()
    parser = PDFParser()
    pages = [number - 1 for number in PDF_PAGES]
    document = parser.parse(source, page_indices=pages)
    encoder = get_encoder()
    chunks = SemanticChunker(encoder=encoder).chunk_document(document)

    with tempfile.TemporaryDirectory(prefix="bnf-page-audit-") as temp:
        temporary_root = Path(temp)
        settings = get_settings().model_copy(update={
            "processed_dir": temporary_root / "processed",
            "qdrant_path": temporary_root / "qdrant-unused",
            "qdrant_url": "",
            "qdrant_collection": "bnf_page_audit",
        })
        store = QdrantStore(settings, client=QdrantClient(":memory:"))
        pipeline = IngestionPipeline(settings, encoder=encoder, store=store)
        pipeline._write_chunks(document.doc_id, chunks)
        pipeline.manifest.mark_chunked(document, len(chunks))
        indexed = pipeline.stage_index(source=source, force=True)
        bm25 = BM25Encoder.load(settings.processed_dir / "bm25_stats.json")

        retrieval = []
        for query in QUERIES:
            dense_vector = encoder.encode([query])[0]
            sparse_vector = bm25.encode_query(query)
            dense_hits = store.search_dense(dense_vector, limit=5)
            sparse_hits = store.search_sparse(sparse_vector, limit=5)
            retrieval.append({
                "query": query,
                "dense": [_hit(chunk, score) for chunk, score in dense_hits],
                "sparse": [_hit(chunk, score) for chunk, score in sparse_hits],
            })

        indexed_chunks = list(store.iter_chunks())
        round_trip_tables = [
            chunk for chunk in indexed_chunks
            if chunk.metadata.get("has_table") and chunk.metadata.get("table_title")
        ]
        page_summaries = []
        for pdf_page in PDF_PAGES:
            page_blocks = [block for block in document.blocks if block.page == pdf_page]
            page_chunks = [
                chunk for chunk in chunks
                if chunk.metadata.get("pdf_page_start", chunk.page_start) <= pdf_page
                <= chunk.metadata.get("pdf_page_end", chunk.page_end)
            ]
            page_summaries.append({
                "pdf_page": pdf_page,
                "expected_printed_page": PRINTED_PAGES[pdf_page],
                "observed_printed_pages": sorted({
                    str(block.metadata["printed_page"])
                    for block in page_blocks if block.metadata.get("printed_page")
                }),
                "blocks": len(page_blocks),
                "chunks": len(page_chunks),
                "table_blocks": sum(block.block_type == "table" for block in page_blocks),
                "dose_box_blocks": sum(bool(block.metadata.get("dose_box")) for block in page_blocks),
                "fields": sorted({
                    str(block.metadata["field"])
                    for block in page_blocks if block.metadata.get("field")
                }),
            })

        result = {
            "source": str(source),
            "sample_pages": len(PDF_PAGES),
            "column_order": "right column top-to-bottom, then left column top-to-bottom per page",
            "parsed_profile": document.metadata.get("parser_profile"),
            "pages_parsed": document.metadata.get("pages_parsed"),
            "parsed_blocks": len(document.blocks),
            "chunks_created": len(chunks),
            "table_chunks": sum(bool(chunk.metadata.get("has_table")) for chunk in chunks),
            "dose_chunks": sum(bool(chunk.metadata.get("has_dose_box")) for chunk in chunks),
            "interaction_chunks": sum(chunk.metadata.get("field") == "INTERACTIONS" for chunk in chunks),
            "dose_chunks_misclassified_as_table": sum(
                bool(chunk.metadata.get("has_dose_box") and chunk.metadata.get("has_table"))
                for chunk in chunks
            ),
            "index": {
                "isolated_in_memory_collection": store.collection,
                "index_stats": indexed,
                "point_count": store.count(),
                "round_trip_titled_table_chunks": len(round_trip_tables),
            },
            "pages": page_summaries,
            "retrieval_smoke_queries": retrieval,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to the BNF PDF")
    parser.add_argument(
        "--report", type=Path, default=Path("runs/bnf-26-page-e2e.json"),
        help="Where to write the audit report (default: runs/bnf-26-page-e2e.json)",
    )
    args = parser.parse_args()
    result = run(args.source, args.report)
    print(json.dumps({
        key: value for key, value in result.items()
        if key not in {"pages", "retrieval_smoke_queries"}
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
