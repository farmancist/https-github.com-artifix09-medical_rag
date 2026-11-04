"""Qdrant collection management and search.

Collection layout:
  named dense vector  "dense"   -> PubMedBERT, cosine, HNSW m=32 / ef_construct=200
  named sparse vector "bm25"    -> BM25 weights, Modifier.IDF (IDF applied server-side)

The dense and sparse channels are queried separately and fused with RRF in
`retrieval/hybrid.py` rather than using Qdrant's server-side fusion, because
query expansion produces several query variants and we need to fuse across
*variants x channels*, not just channels.
"""

from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from qdrant_client import QdrantClient, models

from ..config import Settings, get_settings
from ..embeddings.sparse import SparseVector
from ..models import Chunk

logger = logging.getLogger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"


def _point_uuid(chunk_id: str) -> str:
    """Qdrant point ids must be uint64 or UUID; our chunk ids are sha1 hex.

    Deterministic UUID5 keeps re-ingest idempotent: the same chunk overwrites
    itself instead of creating a duplicate.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore:
    def __init__(self, settings: Settings | None = None, client: QdrantClient | None = None):
        self.settings = settings or get_settings()
        self.collection = self.settings.qdrant_collection
        self._client = client
        # Hybrid retrieval fans out across threads. Without this lock they race
        # to construct the client; in embedded mode that is fatal, because the
        # on-disk store takes an exclusive file lock and the losers crash with
        # "already accessed by another instance of Qdrant client".
        self._client_lock = threading.Lock()

    @property
    def client(self) -> QdrantClient:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:  # built while we waited for the lock
                return self._client
            if self.settings.qdrant_is_embedded:
                # Embedded on-disk mode: no server, no container, no network.
                # Good for development and smoke tests; a remote cluster is the
                # right home for the full corpus and for sharing one index
                # between machines.
                path = Path(self.settings.qdrant_path)
                path.mkdir(parents=True, exist_ok=True)
                logger.info("Qdrant: embedded on-disk store at %s", path)
                self._client = QdrantClient(path=str(path))
            else:
                logger.info("Qdrant: remote at %s", self.settings.qdrant_url)
                self._client = QdrantClient(
                    url=self.settings.qdrant_url,
                    api_key=self.settings.qdrant_api_key or None,
                    timeout=120,
                    prefer_grpc=False,
                )
            return self._client

    # -- schema ---------------------------------------------------------

    def collection_exists(self) -> bool:
        try:
            return self.client.collection_exists(self.collection)
        except Exception:
            return False

    def create_collection(self, *, dim: int | None = None, recreate: bool = False) -> None:
        dim = dim or self.settings.embedding_dim
        if self.collection_exists():
            if not recreate:
                logger.info("Collection %s already exists", self.collection)
                return
            logger.warning("Deleting existing collection %s", self.collection)
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(
                        m=self.settings.qdrant_hnsw_m,
                        ef_construct=self.settings.qdrant_hnsw_ef_construct,
                    ),
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,  # Qdrant supplies the IDF factor
                )
            },
            optimizers_config=models.OptimizersConfigDiff(default_segment_number=2),
        )
        logger.info(
            "Created %s (dim=%d, m=%d, ef_construct=%d)",
            self.collection,
            dim,
            self.settings.qdrant_hnsw_m,
            self.settings.qdrant_hnsw_ef_construct,
        )
        self._create_payload_indexes()

    def _create_payload_indexes(self) -> None:
        """Indexes for the filters the API exposes (source/section/doc)."""
        if self.settings.qdrant_is_embedded:
            # Embedded mode scans exhaustively, so payload indexes are a no-op
            # there and only emit warnings.
            return
        fields = {
            "doc_id": models.PayloadSchemaType.KEYWORD,
            "title": models.PayloadSchemaType.KEYWORD,
            "section": models.PayloadSchemaType.KEYWORD,
            "page_start": models.PayloadSchemaType.INTEGER,
            "metadata.entity": models.PayloadSchemaType.KEYWORD,
            "metadata.field": models.PayloadSchemaType.KEYWORD,
            "metadata.table_title": models.PayloadSchemaType.KEYWORD,
        }
        for field, schema in fields.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection, field_name=field, field_schema=schema
                )
            except Exception as exc:  # index may already exist
                logger.debug("payload index %s: %s", field, exc)

    def count(self) -> int:
        try:
            return int(self.client.count(self.collection, exact=True).count)
        except Exception:
            return 0

    def info(self) -> dict[str, Any]:
        if not self.collection_exists():
            return {"exists": False, "collection": self.collection}
        raw = self.client.get_collection(self.collection)
        return {
            "exists": True,
            "collection": self.collection,
            "points": self.count(),
            "status": str(raw.status),
            "optimizer_status": str(raw.optimizer_status),
        }

    # -- write ----------------------------------------------------------

    @staticmethod
    def _payload(chunk: Chunk) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "text": chunk.text,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "section": chunk.section,
            "title": chunk.title,
            "source_path": chunk.source_path,
            "chunk_index": chunk.chunk_index,
            "n_tokens": chunk.n_tokens,
            "metadata": chunk.metadata,
        }

    def upsert(
        self,
        chunks: Sequence[Chunk],
        dense: np.ndarray,
        sparse: Sequence[SparseVector],
        *,
        batch_size: int = 128,
    ) -> int:
        if not (len(chunks) == len(dense) == len(sparse)):
            raise ValueError(
                f"length mismatch: chunks={len(chunks)} dense={len(dense)} sparse={len(sparse)}"
            )

        written = 0
        for start in range(0, len(chunks), batch_size):
            stop = start + batch_size
            points = [
                models.PointStruct(
                    id=_point_uuid(chunk.chunk_id),
                    vector={
                        DENSE_VECTOR: dense[i].tolist(),
                        SPARSE_VECTOR: models.SparseVector(
                            indices=sparse[i].indices, values=sparse[i].values
                        ),
                    },
                    payload=self._payload(chunk),
                )
                for i, chunk in enumerate(chunks[start:stop], start=start)
            ]
            self.client.upsert(collection_name=self.collection, points=points, wait=True)
            written += len(points)
            logger.info("Upserted %d/%d", written, len(chunks))
        return written

    def delete_document(self, doc_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
                )
            ),
        )

    # -- read -----------------------------------------------------------

    @staticmethod
    def _to_chunk(payload: dict[str, Any]) -> Chunk:
        return Chunk(
            chunk_id=payload.get("chunk_id", ""),
            doc_id=payload.get("doc_id", ""),
            text=payload.get("text", ""),
            page_start=payload.get("page_start", 0),
            page_end=payload.get("page_end", 0),
            section=payload.get("section"),
            title=payload.get("title", ""),
            source_path=payload.get("source_path", ""),
            chunk_index=payload.get("chunk_index", 0),
            n_tokens=payload.get("n_tokens", 0),
            metadata=payload.get("metadata") or {},
        )

    def search_dense(
        self,
        vector: np.ndarray,
        *,
        limit: int,
        query_filter: models.Filter | None = None,
    ) -> list[tuple[Chunk, float]]:
        # Embedded mode is exact brute-force search, so HNSW ef is meaningless
        # there and passing it only produces a warning.
        search_params = (
            None
            if self.settings.qdrant_is_embedded
            else models.SearchParams(hnsw_ef=self.settings.qdrant_hnsw_ef_search)
        )
        result = self.client.query_points(
            collection_name=self.collection,
            query=vector.tolist() if isinstance(vector, np.ndarray) else list(vector),
            using=DENSE_VECTOR,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
            search_params=search_params,
        )
        return [(self._to_chunk(p.payload or {}), float(p.score)) for p in result.points]

    def search_sparse(
        self,
        vector: SparseVector,
        *,
        limit: int,
        query_filter: models.Filter | None = None,
    ) -> list[tuple[Chunk, float]]:
        if not len(vector):
            return []
        result = self.client.query_points(
            collection_name=self.collection,
            query=models.SparseVector(indices=vector.indices, values=vector.values),
            using=SPARSE_VECTOR,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        )
        return [(self._to_chunk(p.payload or {}), float(p.score)) for p in result.points]

    def iter_chunks(self, batch_size: int = 256) -> Iterable[Chunk]:
        """Full scan — used by evaluation and by BM25 stat rebuilds."""
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                yield self._to_chunk(p.payload or {})
            if offset is None:
                break


def build_filter(
    *,
    doc_ids: Sequence[str] | None = None,
    sections: Sequence[str] | None = None,
    entity: str | None = None,
    field: str | None = None,
) -> models.Filter | None:
    conditions: list[models.Condition] = []
    if doc_ids:
        conditions.append(
            models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids)))
        )
    if sections:
        conditions.append(
            models.FieldCondition(key="section", match=models.MatchAny(any=list(sections)))
        )
    if entity:
        conditions.append(models.FieldCondition(
            key="metadata.entity", match=models.MatchValue(value=entity)
        ))
    if field:
        conditions.append(models.FieldCondition(
            key="metadata.field", match=models.MatchValue(value=field)
        ))
    return models.Filter(must=conditions) if conditions else None
