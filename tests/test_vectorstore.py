"""Vector store tests against a real (embedded) Qdrant.

These are the only tests that exercise Qdrant itself: the collection schema,
the upsert path, dense search, BM25 sparse search with server-side IDF, and RRF
fusion over real results. Embedded mode needs no server and no network, so they
run in CI alongside everything else.

Dense vectors are deterministic fakes so no model is needed; everything else is
the production code path.
"""

from __future__ import annotations

import numpy as np
import pytest

from medical_rag.config import Settings
from medical_rag.embeddings.sparse import BM25Encoder
from medical_rag.models import Chunk
from medical_rag.retrieval.hybrid import HybridRetriever
from medical_rag.vectorstore.qdrant_store import QdrantStore, build_filter

TEXTS = [
    "Metformin is contraindicated in severe renal impairment due to lactic acidosis risk.",
    "The usual adult dose of amoxicillin is 500 mg three times a day.",
    "Vitamin D deficiency is treated with colecalciferol supplementation.",
    "Anaphylaxis is managed with intramuscular adrenaline 500 micrograms.",
    "Warfarin interacts with many drugs; monitor the INR closely.",
]
DIM = 8


def make_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id=Chunk.make_id("doc", i, text),
            doc_id="doc-a" if i < 3 else "doc-b",
            text=text,
            page_start=i + 1,
            page_end=i + 1,
            title="Test Formulary",
            section=f"Section {i}",
        )
        for i, text in enumerate(TEXTS)
    ]


@pytest.fixture(scope="module")
def populated(tmp_path_factory):
    """An embedded Qdrant collection loaded with the sample corpus."""
    path = tmp_path_factory.mktemp("qdrant")
    settings = Settings(
        qdrant_url="",
        qdrant_path=path,
        qdrant_collection="test_collection",
        embedding_dim=DIM,
    )
    chunks = make_chunks()

    rng = np.random.default_rng(0)
    dense = rng.normal(size=(len(TEXTS), DIM)).astype(np.float32)
    dense /= np.linalg.norm(dense, axis=1, keepdims=True)

    bm25 = BM25Encoder().fit(TEXTS)
    sparse = bm25.encode_documents(TEXTS)

    store = QdrantStore(settings)
    store.create_collection(dim=DIM, recreate=True)
    store.upsert(chunks, dense, sparse)

    yield {
        "settings": settings,
        "store": store,
        "chunks": chunks,
        "dense": dense,
        "bm25": bm25,
    }
    store.client.close()


class TestCollection:
    def test_created_and_populated(self, populated) -> None:
        assert populated["store"].collection_exists()
        assert populated["store"].count() == len(TEXTS)

    def test_info_reports_points(self, populated) -> None:
        info = populated["store"].info()
        assert info["exists"] is True
        assert info["points"] == len(TEXTS)

    def test_scroll_returns_every_chunk(self, populated) -> None:
        assert len(list(populated["store"].iter_chunks())) == len(TEXTS)

    def test_payload_round_trips(self, populated) -> None:
        by_id = {c.chunk_id: c for c in populated["store"].iter_chunks()}
        original = populated["chunks"][0]
        restored = by_id[original.chunk_id]
        assert restored.text == original.text
        assert restored.page_start == original.page_start
        assert restored.title == original.title
        assert restored.section == original.section

    def test_upsert_is_idempotent(self, populated) -> None:
        """Re-ingesting the same PDF must overwrite, not duplicate."""
        store = populated["store"]
        before = store.count()
        store.upsert(
            populated["chunks"],
            populated["dense"],
            populated["bm25"].encode_documents(TEXTS),
        )
        assert store.count() == before

    def test_length_mismatch_is_rejected(self, populated) -> None:
        with pytest.raises(ValueError):
            populated["store"].upsert(
                populated["chunks"][:2], populated["dense"], [None] * 5  # type: ignore[list-item]
            )


class TestDenseSearch:
    def test_returns_the_nearest_chunk(self, populated) -> None:
        hits = populated["store"].search_dense(populated["dense"][2], limit=3)
        assert hits
        assert hits[0][0].chunk_id == populated["chunks"][2].chunk_id

    def test_respects_the_limit(self, populated) -> None:
        assert len(populated["store"].search_dense(populated["dense"][0], limit=2)) == 2

    def test_filter_restricts_to_a_document(self, populated) -> None:
        hits = populated["store"].search_dense(
            populated["dense"][0], limit=5, query_filter=build_filter(doc_ids=["doc-b"])
        )
        assert hits
        assert all(chunk.doc_id == "doc-b" for chunk, _ in hits)


class TestSparseSearch:
    """BM25 with IDF applied server-side by Qdrant."""

    @pytest.mark.parametrize(
        "query,expected_index",
        [
            ("vitamin D deficiency", 2),
            ("amoxicillin dose", 1),
            ("adrenaline anaphylaxis", 3),
            ("renal impairment metformin", 0),
            ("warfarin INR monitoring", 4),
        ],
    )
    def test_lexical_queries_hit_the_right_chunk(
        self, populated, query: str, expected_index: int
    ) -> None:
        hits = populated["store"].search_sparse(
            populated["bm25"].encode_query(query), limit=3
        )
        assert hits, f"no sparse hits for {query!r}"
        assert hits[0][0].chunk_id == populated["chunks"][expected_index].chunk_id

    def test_single_letter_term_is_searchable(self, populated) -> None:
        """Regression: dropping single characters would break 'vitamin D'."""
        hits = populated["store"].search_sparse(
            populated["bm25"].encode_query("vitamin D"), limit=1
        )
        assert hits and "Vitamin D" in hits[0][0].text

    def test_empty_query_returns_nothing(self, populated) -> None:
        assert populated["store"].search_sparse(
            populated["bm25"].encode_query("   "), limit=5
        ) == []


class _FixedEncoder:
    """Returns one fixed vector, so dense ranking is known in advance."""

    device = "cpu"

    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector

    def encode(self, texts, **kwargs):
        return np.tile(self.vector, (len(list(texts)), 1))


class TestHybridAgainstRealStore:
    def test_fusion_ranks_the_dual_channel_match_first(self, populated) -> None:
        """A chunk found by BOTH dense and BM25 must outrank one found by either."""
        retriever = HybridRetriever(
            populated["settings"],
            store=populated["store"],
            encoder=_FixedEncoder(populated["dense"][2]),
            bm25=populated["bm25"],
        )
        results = retriever.retrieve("vitamin D deficiency treatment")
        assert results
        assert results[0].chunk.chunk_id == populated["chunks"][2].chunk_id
        assert results[0].dense_score is not None
        assert results[0].sparse_score is not None

    def test_scores_are_descending(self, populated) -> None:
        retriever = HybridRetriever(
            populated["settings"],
            store=populated["store"],
            encoder=_FixedEncoder(populated["dense"][0]),
            bm25=populated["bm25"],
        )
        scores = [r.rrf_score for r in retriever.retrieve("metformin renal")]
        assert scores == sorted(scores, reverse=True)



def test_bnf_entity_and_field_filters_are_exact() -> None:
    query_filter = build_filter(entity="Naproxen", field="INDICATIONS AND DOSE")
    assert query_filter is not None
    conditions = {condition.key: condition.match.value for condition in query_filter.must}
    assert conditions == {
        "metadata.entity": "Naproxen",
        "metadata.field": "INDICATIONS AND DOSE",
    }
