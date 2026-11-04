"""End-to-end ingestion: PDFs -> parsed -> chunked -> embedded -> Qdrant.

This is the compute-heavy half of the system, and the only part a GPU meaningfully
speeds up (nothing here is trained; the models run inference only, so CPU works
and simply takes longer). The corpus is a handful of very large
references (the BNF alone is ~1800 pages), so the run is split into two
independently resumable stages, both tracked in a manifest:

    stage 1  CHUNK   parse + sentence-embed + semantic split, per document,
                     written to data/processed/chunks/<doc_id>.jsonl
    stage 2  INDEX   exact BM25 statistics over all chunks, then embed and
                     upsert per document

Splitting them matters for two reasons. A failure at 90% costs one document
rather than the whole corpus. And BM25's `avgdl` can be computed exactly over
the finished chunk set instead of drifting as documents stream in — which it
would if we indexed while chunking.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from ..config import Settings, get_settings
from ..embeddings.encoder import DenseEncoder, get_encoder
from ..embeddings.sparse import BM25Encoder, tokenize
from ..models import Chunk, ParsedDocument
from ..vectorstore.qdrant_store import QdrantStore
from .bnf_chunking import BNF_CHUNKER_VERSION
from .pdf_parser import PDFParser
from .semantic_chunker import SEMANTIC_CHUNKER_VERSION, SemanticChunker
from .katzung_chunking import KATZUNG_CHUNKER_VERSION

logger = logging.getLogger(__name__)

BM25_STATS_FILE = "bm25_stats.json"
MANIFEST_FILE = "manifest.json"
CHUNKS_SUBDIR = "chunks"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Manifest:
    """Records what has been chunked and what has been indexed, per document."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[str, dict] = {}
        if path.exists():
            try:
                self.entries = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Manifest at %s is corrupt; starting fresh", path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=2), encoding="utf-8")

    def is_chunked(self, doc_id: str) -> bool:
        return bool(self.entries.get(doc_id, {}).get("chunked_at"))

    def is_indexed(self, doc_id: str) -> bool:
        return bool(self.entries.get(doc_id, {}).get("indexed_at"))

    def mark_chunked(self, doc: ParsedDocument, n_chunks: int) -> None:
        entry = self.entries.setdefault(doc.doc_id, {})
        # New chunks invalidate any prior index state for this document.
        entry.pop("indexed_at", None)
        entry.pop("indexed_points", None)
        entry.update(
            {
                "title": doc.title,
                "source": doc.source_path,
                "pages": doc.n_pages,
                "chunks": n_chunks,
                "parser_profile": doc.metadata.get("parser_profile"),
                "parser_version": doc.metadata.get("parser_version"),
                "chunker_version": (
                    BNF_CHUNKER_VERSION
                    if doc.metadata.get("parser_profile") == "bnf"
                    else KATZUNG_CHUNKER_VERSION
                    if doc.metadata.get("parser_profile") == "katzung"
                    else SEMANTIC_CHUNKER_VERSION
                ),
                "chunked_at": _now(),
            }
        )
        self.save()

    def mark_indexed(self, doc_id: str, n_points: int) -> None:
        self.entries.setdefault(doc_id, {}).update(
            {"indexed_points": n_points, "indexed_at": _now()}
        )
        self.save()

    def reset(self) -> None:
        self.entries = {}
        self.save()


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        encoder: DenseEncoder | None = None,
        store: QdrantStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.chunks_dir = self.settings.processed_dir / CHUNKS_SUBDIR
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

        self.parser = PDFParser(use_layout_analysis=self.settings.pdf_layout_analysis)
        self.encoder = encoder or get_encoder()
        self.chunker = SemanticChunker(self.settings, self.encoder)
        self.store = store or QdrantStore(self.settings)
        self.manifest = Manifest(self.settings.processed_dir / MANIFEST_FILE)

    # ------------------------------------------------------------------
    # stage 1 — chunk
    # ------------------------------------------------------------------

    def discover(self, source: str | Path | None = None) -> list[Path]:
        source = Path(source) if source else self.settings.raw_dir
        if source.is_file():
            return [source]
        pdfs = sorted(source.rglob("*.pdf"))
        if not pdfs:
            logger.warning("No PDFs found under %s", source)
        return pdfs

    def chunk_path(self, doc_id: str) -> Path:
        return self.chunks_dir / f"{doc_id}.jsonl"

    def stage_chunk(
        self, source: str | Path | None = None, *, force: bool = False
    ) -> dict[str, int]:
        pdfs = self.discover(source)
        stats = {"documents": 0, "skipped": 0, "chunks": 0}

        for path in pdfs:
            doc_id = PDFParser.doc_id_for(path)
            if not force and self.manifest.is_chunked(doc_id) and self.chunk_path(doc_id).exists():
                logger.info("Already chunked, skipping: %s", path.name)
                stats["skipped"] += 1
                continue

            started = time.perf_counter()
            logger.info("Parsing %s", path.name)
            try:
                doc = self.parser.parse(path)
            except Exception:
                logger.exception("Failed to parse %s — skipping", path.name)
                continue

            logger.info("Chunking %s (%d pages) on %s", doc.title[:60], doc.n_pages, self.encoder.device)
            chunks = self.chunker.chunk_document(doc)
            self._write_chunks(doc_id, chunks)
            self.manifest.mark_chunked(doc, len(chunks))

            stats["documents"] += 1
            stats["chunks"] += len(chunks)
            logger.info(
                "%s -> %d chunks in %.1fs", path.name, len(chunks), time.perf_counter() - started
            )
        return stats

    def _write_chunks(self, doc_id: str, chunks: Sequence[Chunk]) -> Path:
        path = self.chunk_path(doc_id)
        with path.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")
        return path

    @staticmethod
    def load_chunks(path: str | Path) -> list[Chunk]:
        path = Path(path)
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as fh:
            return [Chunk(**json.loads(line)) for line in fh if line.strip()]

    def all_chunk_files(self) -> list[Path]:
        return sorted(self.chunks_dir.glob("*.jsonl"))

    def iter_all_chunks(self) -> Iterable[Chunk]:
        for path in self.all_chunk_files():
            yield from self.load_chunks(path)

    # ------------------------------------------------------------------
    # stage 2 — index
    # ------------------------------------------------------------------

    def compute_bm25_stats(self) -> BM25Encoder:
        """Exact corpus statistics over every chunk written in stage 1."""
        total_tokens = 0
        n_docs = 0
        for chunk in self.iter_all_chunks():
            total_tokens += len(tokenize(chunk.text))
            n_docs += 1
        encoder = BM25Encoder(
            avgdl=(total_tokens / n_docs) if n_docs else 0.0, n_docs=n_docs
        )
        encoder.save(self.settings.processed_dir / BM25_STATS_FILE)
        logger.info("BM25 stats over %d chunks: avgdl=%.1f", n_docs, encoder.avgdl)
        return encoder

    def stage_index(
        self,
        *,
        recreate: bool = False,
        force: bool = False,
        source: str | Path | None = None,
    ) -> dict[str, int]:
        files = self.all_chunk_files()
        if not files:
            logger.warning("No chunk files in %s — run the chunk stage first", self.chunks_dir)
            return {"documents": 0, "indexed": 0, "skipped": 0}
        if recreate and source is not None:
            raise ValueError("Cannot recreate the collection while indexing only one source.")
        if source is not None:
            source_doc_ids = {PDFParser.doc_id_for(path) for path in self.discover(source)}
            files = [path for path in files if path.stem in source_doc_ids]
            if not files:
                logger.warning("No staged chunks found for source %s", source)
                return {"documents": 0, "indexed": 0, "skipped": 0}

        bm25 = self.compute_bm25_stats()

        # Force the model to load so the collection is created with the model's
        # real dimension rather than the configured guess.
        _ = self.encoder.model

        if recreate:
            # Collection is dropped, so every document must be re-uploaded.
            self.store.create_collection(dim=self.encoder.dim, recreate=True)
            for entry in self.manifest.entries.values():
                entry.pop("indexed_at", None)
            self.manifest.save()
        else:
            self.store.create_collection(dim=self.encoder.dim, recreate=False)

        stats = {"documents": 0, "indexed": 0, "skipped": 0}
        for path in files:
            doc_id = path.stem
            if not force and not recreate and self.manifest.is_indexed(doc_id):
                logger.info("Already indexed, skipping: %s", doc_id)
                stats["skipped"] += 1
                continue

            chunks = self.load_chunks(path)
            if not chunks:
                continue

            title = self.manifest.entries.get(doc_id, {}).get("title", doc_id)
            logger.info("Embedding %d chunks for %s", len(chunks), str(title)[:60])

            started = time.perf_counter()
            texts = [c.text for c in chunks]
            dense = self.encoder.encode(texts, show_progress=True)
            sparse = bm25.encode_documents(texts)
            if force and not recreate:
                entry = self.manifest.entries.get(doc_id)
                if entry:
                    entry.pop("indexed_at", None)
                    entry.pop("indexed_points", None)
                    self.manifest.save()
                self.store.delete_document(doc_id)
            written = self.store.upsert(chunks, dense, sparse)

            self.manifest.mark_indexed(doc_id, written)
            stats["documents"] += 1
            stats["indexed"] += written
            logger.info(
                "Indexed %d points for %s in %.1fs",
                written,
                str(title)[:60],
                time.perf_counter() - started,
            )
        return stats

    # ------------------------------------------------------------------
    # orchestration
    # ------------------------------------------------------------------

    def run(
        self,
        source: str | Path | None = None,
        *,
        recreate: bool = False,
        force: bool = False,
    ) -> dict:
        started = time.perf_counter()
        if recreate:
            logger.warning("--recreate: clearing manifest and re-chunking everything")
            self.manifest.reset()

        chunked = self.stage_chunk(source, force=force or recreate)
        indexed = self.stage_index(recreate=recreate, force=force)

        return {
            "documents_chunked": chunked["documents"],
            "documents_skipped": chunked["skipped"],
            "chunks_created": chunked["chunks"],
            "documents_indexed": indexed["documents"],
            "points_indexed": indexed["indexed"],
            "device": self.encoder.device,
            "collection": self.settings.qdrant_collection,
            "total_points": self.store.count(),
            "elapsed_s": round(time.perf_counter() - started, 2),
        }

    def status(self) -> dict:
        return {
            "raw_pdfs": len(self.discover()),
            "chunk_files": len(self.all_chunk_files()),
            "documents": {
                doc_id: {
                    "title": str(entry.get("title", ""))[:60],
                    "pages": entry.get("pages"),
                    "chunks": entry.get("chunks"),
                    "chunked": bool(entry.get("chunked_at")),
                    "indexed": bool(entry.get("indexed_at")),
                }
                for doc_id, entry in self.manifest.entries.items()
            },
            "qdrant": self.store.info(),
        }
