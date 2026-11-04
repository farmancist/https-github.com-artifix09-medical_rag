from __future__ import annotations

from pathlib import Path

import numpy as np

from medical_rag.config import Settings
from medical_rag.ingest.pdf_parser import PDFParser
from medical_rag.ingest.pipeline import IngestionPipeline
from medical_rag.models import Chunk, PageBlock, ParsedDocument


class _Encoder:
    dim = 2
    device = "cpu"
    model = object()

    def encode(self, texts, **kwargs):
        return np.zeros((len(texts), self.dim), dtype=np.float32)


class _Store:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.upserted: list[str] = []

    def create_collection(self, **kwargs) -> None:
        pass

    def delete_document(self, doc_id: str) -> None:
        self.deleted.append(doc_id)

    def upsert(self, chunks, dense, sparse):
        self.upserted.extend(chunk.doc_id for chunk in chunks)
        return len(chunks)


def _chunk(doc_id: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}-chunk",
        doc_id=doc_id,
        text=f"{doc_id} source text",
        page_start=1,
        page_end=1,
        title=doc_id,
        source_path=f"{doc_id}.pdf",
    )


def test_force_index_replaces_only_the_selected_document(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    bnf_pdf = raw_dir / "bnf.pdf"
    other_pdf = raw_dir / "other.pdf"
    bnf_pdf.touch()
    other_pdf.touch()

    store = _Store()
    pipeline = IngestionPipeline(
        Settings(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            cache_dir=tmp_path / "cache",
            qdrant_url="http://localhost:6333",
            pdf_layout_analysis=True,
        ),
        encoder=_Encoder(),
        store=store,
    )
    assert pipeline.parser.use_layout_analysis is True
    bnf_id = PDFParser.doc_id_for(bnf_pdf)
    other_id = PDFParser.doc_id_for(other_pdf)
    pipeline._write_chunks(bnf_id, [_chunk(bnf_id)])
    pipeline._write_chunks(other_id, [_chunk(other_id)])
    pipeline.manifest.entries = {
        bnf_id: {"indexed_at": "old", "indexed_points": 1},
        other_id: {"indexed_at": "old", "indexed_points": 1},
    }
    pipeline.manifest.save()

    stats = pipeline.stage_index(source=bnf_pdf, force=True)

    assert stats == {"documents": 1, "indexed": 1, "skipped": 0}
    assert store.deleted == [bnf_id]
    assert store.upserted == [bnf_id]
    assert pipeline.manifest.is_indexed(bnf_id)
    assert pipeline.manifest.is_indexed(other_id)



def test_chunk_stage_roundtrips_bnf_provenance(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    bnf_pdf = raw_dir / "British National Formulary.pdf"
    bnf_pdf.touch()
    doc_id = PDFParser.doc_id_for(bnf_pdf)
    parsed = ParsedDocument(
        doc_id=doc_id,
        source_path=str(bnf_pdf),
        title="British National Formulary",
        n_pages=1906,
        blocks=[PageBlock(
            text="Adult: 1 g once daily.", page=1282, section="Naproxen",
            metadata={
                "entity": "Naproxen", "field": "INDICATIONS AND DOSE",
                "indication": "Rheumatic disease", "route": "BY MOUTH",
                "dose_box": True, "printed_page": "1263", "pdf_page": 1282,
            },
        )],
        metadata={"parser_profile": "bnf", "parser_version": "bnf-layout-1.0.0"},
    )
    pipeline = IngestionPipeline(
        Settings(
            raw_dir=raw_dir, processed_dir=processed_dir,
            cache_dir=tmp_path / "cache", qdrant_url="http://localhost:6333",
        ),
        encoder=_Encoder(), store=_Store(),
    )
    pipeline.parser.parse = lambda _path: parsed

    stats = pipeline.stage_chunk(bnf_pdf, force=True)

    assert stats == {"documents": 1, "skipped": 0, "chunks": 1}
    [chunk] = pipeline.load_chunks(pipeline.chunk_path(doc_id))
    assert chunk.metadata["entity"] == "Naproxen"
    assert chunk.metadata["field"] == "INDICATIONS AND DOSE"
    assert chunk.metadata["printed_page"] == "1263"
    assert chunk.metadata["pdf_page_start"] == 1282
    assert chunk.metadata["source"] == str(bnf_pdf)
    assert chunk.metadata["has_dose_box"] is True
    assert chunk.metadata["parser_profile"] == "bnf"
    assert chunk.metadata["parser_version"] == "bnf-layout-1.0.0"
    assert chunk.metadata["chunker_version"] == "bnf-structural-1.1.0"
    assert chunk.metadata["ingestion_schema_version"] == "1.0"
