"""Retrieval-side tests: BM25 sparse encoding, RRF fusion, query expansion.

All offline — Qdrant and the encoder are stubbed.
"""

from __future__ import annotations

import numpy as np

from medical_rag.config import Settings
from medical_rag.embeddings.sparse import BM25Encoder, tokenize, token_id
from medical_rag.models import Chunk, ExpandedQuery
from medical_rag.retrieval.hybrid import HybridRetriever
from medical_rag.retrieval.query_expansion import QueryExpander, load_lexicon


# --------------------------------------------------------------------- BM25


class TestTokenizer:
    def test_stopwords_dropped_medical_terms_kept(self) -> None:
        tokens = tokenize("What is the treatment for the acute myocardial infarction")
        assert "the" not in tokens and "is" not in tokens
        assert "myocardial" in tokens and "infarction" in tokens

    def test_plural_folding(self) -> None:
        assert tokenize("contraindications")[0] == tokenize("contraindication")[0]

    def test_hyphenated_terms_indexed_both_ways(self) -> None:
        tokens = tokenize("co-trimoxazole")
        assert "co-trimoxazole" in tokens
        assert "trimoxazole" in tokens

    def test_dosage_tokens_survive(self) -> None:
        assert "500mg" in tokenize("give 500mg orally")

    def test_single_letter_medical_terms_survive(self) -> None:
        """Dropping single characters would turn 'vitamin D' into 'vitamin'."""
        assert tokenize("vitamin D deficiency") == ["vitamin", "d", "deficiency"]
        assert "b" in tokenize("hepatitis B surface antigen")
        assert "t" in tokenize("T cell lymphoma")

    def test_common_function_words_still_dropped(self) -> None:
        assert tokenize("the treatment of the patient") == ["treatment", "patient"]

    def test_token_id_is_stable_and_in_uint32_range(self) -> None:
        assert token_id("metformin") == token_id("metformin")
        assert 0 <= token_id("metformin") < 2**31


class TestBM25Encoder:
    def test_fit_computes_average_length(self) -> None:
        encoder = BM25Encoder().fit(["alpha beta gamma", "alpha beta gamma delta epsilon"])
        assert encoder.n_docs == 2
        assert encoder.avgdl == 4.0

    def test_update_folds_into_running_average(self) -> None:
        encoder = BM25Encoder().fit(["a b c d"])  # avgdl 4 over 1 doc
        encoder.update(["a b c d e f g h"])  # 8 tokens
        assert encoder.n_docs == 2
        assert encoder.avgdl == 6.0

    def test_document_vector_saturates_term_frequency(self) -> None:
        """BM25 tf-saturation: 10 occurrences must not weigh 10x one."""
        encoder = BM25Encoder(avgdl=100.0, n_docs=10)
        once = encoder.encode_document("sepsis " + "filler " * 99)
        many = encoder.encode_document("sepsis " * 10 + "filler " * 90)
        idx = once.indices.index(token_id("sepsis"))
        w_once = once.values[idx]
        w_many = many.values[many.indices.index(token_id("sepsis"))]
        assert w_many > w_once
        assert w_many < 10 * w_once

    def test_query_vector_uses_presence_weights(self) -> None:
        vector = BM25Encoder().encode_query("metformin dosing")
        assert len(vector) == 2
        assert all(v >= 1.0 for v in vector.values)

    def test_empty_input_yields_empty_vector(self) -> None:
        assert len(BM25Encoder().encode_query("   ")) == 0

    def test_roundtrip_persistence(self, tmp_path) -> None:
        path = tmp_path / "bm25.json"
        BM25Encoder(avgdl=42.5, n_docs=7).save(path)
        loaded = BM25Encoder.load(path)
        assert loaded.avgdl == 42.5 and loaded.n_docs == 7

    def test_missing_file_loads_defaults(self, tmp_path) -> None:
        assert BM25Encoder.load(tmp_path / "nope.json").n_docs == 0


# ---------------------------------------------------------------------- RRF


class _StubEncoder:
    device = "cpu"

    def encode(self, texts, **kwargs):
        return np.zeros((len(texts), 4), dtype=np.float32)

    def encode_query(self, text):
        return np.zeros(4, dtype=np.float32)


class _StubStore:
    """Returns a fixed ranking per (channel, query)."""

    def __init__(self, dense: dict, sparse: dict) -> None:
        self.dense = dense
        self.sparse = sparse

    @staticmethod
    def _chunks(ids):
        return [
            (
                Chunk(
                    chunk_id=cid,
                    doc_id="d",
                    text=f"text {cid}",
                    page_start=1,
                    page_end=1,
                ),
                1.0 - i * 0.01,
            )
            for i, cid in enumerate(ids)
        ]

    def search_dense(self, vector, *, limit, query_filter=None):
        key = tuple(sorted(self.dense))[0] if len(self.dense) == 1 else None
        return self._chunks(self.dense.get(key, next(iter(self.dense.values()))))[:limit]

    def search_sparse(self, vector, *, limit, query_filter=None):
        return self._chunks(next(iter(self.sparse.values())))[:limit]


def make_retriever(dense_ids, sparse_ids) -> HybridRetriever:
    return HybridRetriever(
        Settings(),
        store=_StubStore({"q": dense_ids}, {"q": sparse_ids}),
        encoder=_StubEncoder(),
        bm25=BM25Encoder(avgdl=10.0, n_docs=5),
    )


class TestRRF:
    def test_chunk_in_both_channels_outranks_either_alone(self) -> None:
        # "b" is #2 dense and #1 sparse; "a" is #1 dense only.
        retriever = make_retriever(["a", "b", "c"], ["b", "d", "e"])
        results = retriever.retrieve(ExpandedQuery(original="q"))
        assert results[0].chunk.chunk_id == "b"

    def test_scores_are_descending(self) -> None:
        retriever = make_retriever(["a", "b", "c"], ["c", "b", "a"])
        results = retriever.retrieve(ExpandedQuery(original="q"))
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_union_of_both_channels_is_returned(self) -> None:
        retriever = make_retriever(["a", "b"], ["c", "d"])
        ids = {r.chunk.chunk_id for r in retriever.retrieve(ExpandedQuery(original="q"))}
        assert ids == {"a", "b", "c", "d"}

    def test_per_channel_scores_recorded(self) -> None:
        retriever = make_retriever(["a"], ["a"])
        result = retriever.retrieve(ExpandedQuery(original="q"))[0]
        assert result.dense_score is not None and result.sparse_score is not None

    def test_plain_string_query_is_accepted(self) -> None:
        retriever = make_retriever(["a"], ["a"])
        assert retriever.retrieve("plain question") != []


# --------------------------------------------------------- query expansion


class TestQueryExpansion:
    def expander(self, **kwargs) -> QueryExpander:
        settings = Settings(query_expansion_mode="lexicon", **kwargs)
        return QueryExpander(settings)

    def test_lexicon_loads(self) -> None:
        lexicon = load_lexicon()
        assert "heart attack" in lexicon
        assert "myocardial infarction" in lexicon["heart attack"]

    def test_layman_term_is_rewritten(self) -> None:
        expanded = self.expander().expand("what causes a heart attack")
        assert expanded.variants
        assert any("myocardial infarction" in v for v in expanded.variants)
        assert expanded.method == "lexicon"

    def test_longest_match_wins(self) -> None:
        """'high blood pressure' must not be shredded by a 'pressure' entry."""
        expanded = self.expander().expand("treatment for high blood pressure")
        assert any("hypertension" in v for v in expanded.variants)

    def test_clinical_query_is_left_alone(self) -> None:
        expanded = self.expander().expand("management of acute pancreatitis in adults")
        assert expanded.variants == []

    def test_original_query_always_comes_first(self) -> None:
        expanded = self.expander().expand("what causes a heart attack")
        assert expanded.all_queries[0] == "what causes a heart attack"

    def test_variant_count_is_capped(self) -> None:
        expanded = self.expander(query_expansion_max_variants=2).expand(
            "heart attack and stroke and high blood pressure"
        )
        assert len(expanded.variants) <= 2

    def test_disabled_expansion_is_a_no_op(self) -> None:
        expanded = self.expander(query_expansion_enabled=False).expand("heart attack")
        assert expanded.variants == []
        assert expanded.method == "disabled"

    def test_duplicates_removed(self) -> None:
        expanded = self.expander().expand("heart attack")
        assert len(set(expanded.all_queries)) == len(expanded.all_queries)
