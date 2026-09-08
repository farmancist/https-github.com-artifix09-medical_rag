"""Semantic chunker tests with a deterministic stub encoder (no model download)."""

from __future__ import annotations

import numpy as np

from medical_rag.config import Settings
from medical_rag.ingest.semantic_chunker import SemanticChunker, count_tokens, split_sentences
from medical_rag.models import PageBlock, ParsedDocument


class _TopicEncoder:
    """Embeds by keyword: sentences sharing a topic word get identical vectors,
    so distances are 0 within a topic and 1 across topics — which makes the
    percentile breakpoint behaviour testable without a real model."""

    device = "cpu"
    topics = ("cardiac", "renal", "hepatic")

    def encode(self, texts, **kwargs):
        vectors = np.zeros((len(texts), len(self.topics) + 1), dtype=np.float32)
        for i, text in enumerate(texts):
            lowered = text.lower()
            hit = next((j for j, t in enumerate(self.topics) if t in lowered), len(self.topics))
            vectors[i, hit] = 1.0
        return vectors


def make_doc(sentences: list[str], *, pages: list[int] | None = None) -> ParsedDocument:
    pages = pages or [1] * len(sentences)
    return ParsedDocument(
        doc_id="doc",
        source_path="/tmp/doc.pdf",
        title="Test Document",
        n_pages=max(pages, default=1),
        blocks=[PageBlock(text=s, page=p) for s, p in zip(sentences, pages)],
    )


def chunker(**kwargs) -> SemanticChunker:
    kwargs.setdefault("chunk_min_tokens", 1)
    kwargs.setdefault("chunk_max_tokens", 512)
    return SemanticChunker(Settings(**kwargs), _TopicEncoder())


class TestSentenceSplitting:
    def test_basic_split(self) -> None:
        assert len(split_sentences("First sentence. Second sentence. Third one.")) == 3

    def test_decimal_dose_not_split(self) -> None:
        sentences = split_sentences("Give 0.5 mg daily. Then reassess.")
        assert len(sentences) == 2
        assert "0.5 mg" in sentences[0]

    def test_abbreviations_not_split(self) -> None:
        sentences = split_sentences("Dr. Smith prescribed 500 mg. The patient improved.")
        assert len(sentences) == 2

    def test_eg_ie_not_split(self) -> None:
        sentences = split_sentences("Use an NSAID, e.g. ibuprofen. Avoid in renal failure.")
        assert len(sentences) == 2

    def test_abbreviation_before_capital_not_split(self) -> None:
        sentences = split_sentences("Reviewed by Dr. Ahmed. The dose was reduced.")
        assert len(sentences) == 2
        assert sentences[0].startswith("Reviewed by Dr. Ahmed")

    def test_initials_not_split(self) -> None:
        assert len(split_sentences("Reported by J. Smith. Findings were normal.")) == 2

    def test_unit_before_capital_does_split(self) -> None:
        """'500 mg. The patient' is a real boundary; units are not abbreviations."""
        sentences = split_sentences("Give metformin 500 mg. The patient improved.")
        assert len(sentences) == 2
        assert sentences[1] == "The patient improved."

    def test_unit_before_lowercase_does_not_split(self) -> None:
        assert len(split_sentences("Give 500 mg. of the suspension orally.")) == 1

    def test_question_and_exclamation_split(self) -> None:
        assert len(split_sentences("Is it safe? Yes it is. Good!")) == 3

    def test_token_count(self) -> None:
        assert count_tokens("500 mg twice daily") == 4


class TestSemanticChunking:
    def test_topic_shift_creates_a_boundary(self) -> None:
        doc = make_doc(
            [
                "Cardiac output is reduced in shock.",
                "Cardiac monitoring is required.",
                "Renal perfusion falls accordingly.",
                "Renal replacement may be needed.",
            ]
        )
        chunks = chunker().chunk_document(doc)
        assert len(chunks) >= 2

    def test_uniform_text_stays_together(self) -> None:
        doc = make_doc(["Cardiac point one." , "Cardiac point two.", "Cardiac point three."])
        chunks = chunker(chunk_overlap_ratio=0.0).chunk_document(doc)
        assert len(chunks) == 1

    def test_page_range_is_tracked(self) -> None:
        doc = make_doc(
            ["Cardiac sentence one.", "Cardiac sentence two."],
            pages=[3, 4],
        )
        chunks = chunker(chunk_overlap_ratio=0.0).chunk_document(doc)
        assert chunks[0].page_start == 3
        assert chunks[0].page_end == 4

    def test_max_tokens_is_enforced(self) -> None:
        sentences = [f"Cardiac finding number {i} was recorded." for i in range(60)]
        chunks = chunker(chunk_max_tokens=40, chunk_overlap_ratio=0.0).chunk_document(
            make_doc(sentences)
        )
        assert len(chunks) > 1
        # Allow a small overshoot: a chunk never splits mid-sentence.
        assert all(c.n_tokens <= 40 + 12 for c in chunks)

    def test_overlap_never_pushes_a_chunk_over_budget(self) -> None:
        """Overlap once inflated real chunks to 897 tokens against a 384 limit,
        past the encoder's window, so the added context was silently dropped."""
        sentences = [
            f"Cardiac observation number {i} was recorded in the trial cohort during review."
            for i in range(40)
        ]
        chunks = chunker(chunk_max_tokens=60, chunk_overlap_ratio=0.5).chunk_document(
            make_doc(sentences)
        )
        assert chunks
        assert all(c.n_tokens <= 60 + 20 for c in chunks), [c.n_tokens for c in chunks]

    def test_overlap_repeats_previous_content(self) -> None:
        doc = make_doc(
            [
                "Cardiac output is reduced in shock.",
                "Cardiac monitoring is required continuously.",
                "Renal perfusion falls accordingly in this state.",
                "Renal replacement therapy may become necessary here.",
            ]
        )
        with_overlap = chunker(chunk_overlap_ratio=0.5).chunk_document(doc)
        without = chunker(chunk_overlap_ratio=0.0).chunk_document(doc)
        assert sum(len(c.text) for c in with_overlap) > sum(len(c.text) for c in without)

    def test_oversized_table_is_split_by_rows_with_header_repeated(self) -> None:
        """Measured on the real corpus, un-split tables produced chunks where
        94% of the text fell outside the model's context window."""
        rows = "\n".join(
            f"| drug{i} | {i * 100} mg injection in vial for treatment |" for i in range(60)
        )
        table = f"| Medicine | Dose |\n|---|---|\n{rows}"
        doc = ParsedDocument(
            doc_id="doc",
            source_path="/tmp/d.pdf",
            title="T",
            n_pages=1,
            blocks=[PageBlock(text=table, page=1, block_type="table")],
        )
        chunks = chunker(chunk_max_tokens=100, chunk_overlap_ratio=0.0).chunk_document(doc)
        assert len(chunks) > 1, "oversized table was not split"
        for c in chunks:
            assert c.n_tokens <= 100 + 30, f"chunk still oversized: {c.n_tokens}"
            # Each part must stay self-describing.
            assert "| Medicine | Dose |" in c.text
        # No row may be lost.
        joined = " ".join(c.text for c in chunks)
        for i in range(60):
            assert f"drug{i} " in joined or f"drug{i} |" in joined

    def test_no_chunk_exceeds_the_budget(self) -> None:
        """A single un-splittable blob must not become an oversized chunk."""
        blob = " ".join(f"word{i}" for i in range(2000))  # no sentence punctuation
        chunks = chunker(chunk_max_tokens=120, chunk_overlap_ratio=0.0).chunk_document(
            make_doc([blob])
        )
        assert chunks
        assert all(c.n_tokens <= 120 + 30 for c in chunks), [c.n_tokens for c in chunks]

    def test_tables_are_kept_atomic(self) -> None:
        doc = ParsedDocument(
            doc_id="doc",
            source_path="/tmp/d.pdf",
            title="T",
            n_pages=1,
            blocks=[
                PageBlock(text="Cardiac introduction sentence.", page=1),
                PageBlock(text="| Drug | Dose |\n|---|---|\n| Aspirin | 300 mg |", page=1,
                          block_type="table"),
                PageBlock(text="Renal follow-up sentence.", page=1),
            ],
        )
        chunks = chunker(chunk_overlap_ratio=0.0).chunk_document(doc)
        table_chunks = [c for c in chunks if "| Aspirin |" in c.text]
        assert len(table_chunks) == 1
        assert table_chunks[0].metadata.get("has_table") is True

    def test_section_change_forces_a_break(self) -> None:
        doc = ParsedDocument(
            doc_id="doc",
            source_path="/tmp/d.pdf",
            title="T",
            n_pages=1,
            blocks=[
                PageBlock(text="Cardiac indication text here.", page=1, section="Indications"),
                PageBlock(text="Cardiac contraindication text.", page=1, section="Contraindications"),
            ],
        )
        chunks = chunker(chunk_overlap_ratio=0.0).chunk_document(doc)
        assert len(chunks) == 2

    def test_empty_document_yields_nothing(self) -> None:
        assert chunker().chunk_document(make_doc([])) == []

    def test_chunk_ids_are_deterministic(self) -> None:
        doc = make_doc(["Cardiac one.", "Renal two."])
        first = chunker(chunk_overlap_ratio=0.0).chunk_document(doc)
        second = chunker(chunk_overlap_ratio=0.0).chunk_document(doc)
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    def test_citation_label_formats_page_ranges(self) -> None:
        doc = make_doc(["Cardiac one.", "Cardiac two."], pages=[7, 9])
        chunk = chunker(chunk_overlap_ratio=0.0).chunk_document(doc)[0]
        assert chunk.citation_label() == "Test Document, pp.7-9"
