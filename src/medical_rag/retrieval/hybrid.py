"""Hybrid retrieval: BM25 + dense, fused with Reciprocal Rank Fusion.

RRF is done client-side rather than with Qdrant's server-side fusion because
query expansion hands us N query variants. We therefore have up to 2N ranked
lists (dense and sparse per variant) and fuse all of them at once:

    RRF(chunk) = sum over lists L of  1 / (k + rank_L(chunk))

with k = 60 (the standard constant; it damps the influence of any single list's
top hit so one over-confident channel cannot dominate the fusion).

Variant lists are weighted below the original query's list — an expanded query
is a hypothesis about what the user meant, not the user's actual words.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence

from qdrant_client import models as qmodels

from ..config import Settings, get_settings
from ..embeddings.encoder import DenseEncoder, get_encoder
from ..embeddings.sparse import BM25Encoder
from ..models import Chunk, ExpandedQuery, RetrievedChunk
from ..vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

ORIGINAL_QUERY_WEIGHT = 1.0
VARIANT_QUERY_WEIGHT = 0.7


@dataclass
class _RankedList:
    query: str
    channel: str  # "dense" | "sparse"
    weight: float
    results: list[tuple[Chunk, float]]


class HybridRetriever:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store: QdrantStore | None = None,
        encoder: DenseEncoder | None = None,
        bm25: BM25Encoder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or QdrantStore(self.settings)
        self.encoder = encoder or get_encoder()
        self.bm25 = bm25 or BM25Encoder.load(self.settings.processed_dir / "bm25_stats.json")

    # -- public ---------------------------------------------------------

    def retrieve(
        self,
        expanded: ExpandedQuery | str,
        *,
        limit: int | None = None,
        query_filter: qmodels.Filter | None = None,
    ) -> list[RetrievedChunk]:
        if isinstance(expanded, str):
            expanded = ExpandedQuery(original=expanded, method="none")

        queries = expanded.all_queries
        if not queries:
            return []

        ranked_lists = self._search_all(queries, expanded.original, query_filter)
        fused = self._rrf(ranked_lists)
        limit = limit or self.settings.rerank_candidates
        return fused[:limit]

    # -- search ---------------------------------------------------------

    def _search_all(
        self,
        queries: Sequence[str],
        original: str,
        query_filter: qmodels.Filter | None,
    ) -> list[_RankedList]:
        """Run every (variant x channel) search concurrently.

        These are network-bound Qdrant calls, so threads are the right tool;
        the dense vectors are all encoded up front in one batch instead.
        """
        vectors = self.encoder.encode(list(queries))

        jobs: list[tuple[str, str, float]] = []
        for query in queries:
            weight = ORIGINAL_QUERY_WEIGHT if query == original else VARIANT_QUERY_WEIGHT
            jobs.append((query, "dense", weight))
            jobs.append((query, "sparse", weight))

        def run(job: tuple[str, str, float]) -> _RankedList:
            query, channel, weight = job
            try:
                if channel == "dense":
                    results = self.store.search_dense(
                        vectors[queries.index(query)],
                        limit=self.settings.dense_top_k,
                        query_filter=query_filter,
                    )
                else:
                    results = self.store.search_sparse(
                        self.bm25.encode_query(query),
                        limit=self.settings.sparse_top_k,
                        query_filter=query_filter,
                    )
            except Exception:
                logger.exception("%s search failed for %r", channel, query)
                results = []
            return _RankedList(query=query, channel=channel, weight=weight, results=results)

        # Threads only pay off against a server, where these calls are
        # network-bound. The embedded store is local numpy under an exclusive
        # file lock, so concurrency buys nothing and risks contention.
        if self.settings.qdrant_is_embedded:
            _ = self.store.client  # open once, before any fan-out
            return [run(job) for job in jobs]

        with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
            return list(pool.map(run, jobs))

    # -- fusion ---------------------------------------------------------

    def _rrf(self, ranked_lists: Sequence[_RankedList]) -> list[RetrievedChunk]:
        k = self.settings.rrf_k
        accumulated: dict[str, RetrievedChunk] = {}

        for ranked in ranked_lists:
            for rank, (chunk, score) in enumerate(ranked.results, start=1):
                key = chunk.chunk_id or f"{chunk.doc_id}:{chunk.chunk_index}"
                entry = accumulated.get(key)
                if entry is None:
                    entry = RetrievedChunk(chunk=chunk, rrf_score=0.0)
                    accumulated[key] = entry

                entry.rrf_score = (entry.rrf_score or 0.0) + ranked.weight / (k + rank)

                # Keep the best raw score seen per channel, for diagnostics.
                if ranked.channel == "dense":
                    entry.dense_score = max(entry.dense_score or 0.0, score)
                else:
                    entry.sparse_score = max(entry.sparse_score or 0.0, score)

                if ranked.query not in entry.matched_queries:
                    entry.matched_queries.append(ranked.query)

        fused = sorted(accumulated.values(), key=lambda r: r.rrf_score or 0.0, reverse=True)
        logger.debug(
            "RRF fused %d lists -> %d unique chunks", len(ranked_lists), len(fused)
        )
        return fused
