"""PDF parser helper tests.

These cover the pure text-processing functions, which is where the parser's
real behaviour lives. They run without opening a PDF, so they work everywhere.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from medical_rag.ingest.pdf_parser import (
    PDFParser,
    _boilerplate_key,
    _clean,
    _looks_like_heading,
    _title_from_filename,
    build_vocabulary,
)


class TestClean:
    def test_line_wraps_become_spaces(self) -> None:
        """PDFs break lines mid-sentence; those newlines must not survive."""
        text = "Review intravenous antimicrobials\nwithin 48 hours of starting."
        assert _clean(text) == "Review intravenous antimicrobials within 48 hours of starting."

    def test_paragraph_breaks_are_preserved(self) -> None:
        assert _clean("First para.\n\nSecond para.") == "First para.\n\nSecond para."

    def test_three_or_more_newlines_collapse_to_one_break(self) -> None:
        assert _clean("A.\n\n\n\nB.") == "A.\n\nB."

    def test_hyphenated_line_break_is_rejoined_by_default(self) -> None:
        assert _clean("inflamma-\ntion of the joint") == "inflammation of the joint"

    def test_vocabulary_keeps_a_real_compound(self) -> None:
        """Without this, the BNF's 'side-\\neffects' becomes 'sideeffects'."""
        vocab = build_vocabulary("side-effects are common. side-effects were noted.")
        assert _clean("common side-\neffects include", vocab) == "common side-effects include"

    def test_vocabulary_joins_a_syllable_break(self) -> None:
        vocab = build_vocabulary("the product is safe. another product exists.")
        assert _clean("the prod-\nuct is safe", vocab) == "the product is safe"

    def test_ligatures_are_expanded(self) -> None:
        # A ligature left in place breaks BM25: "inﬂammation" != "inflammation".
        assert _clean("in\ufb02ammation and e\ufb00usion") == "inflammation and effusion"

    def test_non_breaking_space_is_normalised(self) -> None:
        assert _clean("500\u00a0mg daily") == "500 mg daily"

    def test_carriage_returns_handled(self) -> None:
        assert _clean("line one\r\nline two") == "line one line two"

    def test_repeated_spaces_collapse(self) -> None:
        assert _clean("dose    of     500 mg") == "dose of 500 mg"


class TestBoilerplateKey:
    def test_page_numbers_are_folded(self) -> None:
        """Running footers differ only by page number, so exact matching misses them."""
        assert _boilerplate_key("WHO Model List page 28") == _boilerplate_key(
            "WHO Model List page 29"
        )

    def test_case_is_folded(self) -> None:
        assert _boilerplate_key("BNF 85") == _boilerplate_key("bnf 85")

    def test_distinct_text_stays_distinct(self) -> None:
        assert _boilerplate_key("Contraindications") != _boilerplate_key("Interactions")


class TestVocabulary:
    """A line-break hyphen is ambiguous; the document's own usage decides it.

    Measured on the real corpus, the Oxford Handbook is ~89% syllable breaks
    while the BNF is ~80% real compounds, so no fixed rule works for both.
    """

    def test_hyphenated_form_used_elsewhere_wins(self) -> None:
        vocab = build_vocabulary("cross-reference here and cross-reference there")
        assert vocab.resolve("cross", "reference") == "cross-reference"

    def test_joined_form_used_elsewhere_wins(self) -> None:
        vocab = build_vocabulary("gastroenterologist on call")
        assert vocab.resolve("gas", "troenterologist") == "gastroenterologist"

    def test_two_standalone_words_are_treated_as_a_compound(self) -> None:
        vocab = build_vocabulary("risk risk risk benefit benefit benefit")
        assert vocab.resolve("risk", "benefit") == "risk-benefit"

    def test_unknown_fragments_are_joined(self) -> None:
        assert build_vocabulary("nothing relevant here").resolve("inflamma", "tion") == (
            "inflammation"
        )

    def test_truncated_halves_do_not_pollute_the_vocabulary(self) -> None:
        """'prod-' and 'uct' must not be counted as standalone words, or they
        would vote to keep the hyphen that created them."""
        vocab = build_vocabulary("the prod-\nuct is here")
        assert "prod" not in vocab.words
        assert "uct" not in vocab.words

    def test_case_is_folded(self) -> None:
        vocab = build_vocabulary("Side-Effects were noted")
        assert vocab.resolve("side", "effects") == "side-effects"

    def test_original_casing_is_preserved_in_output(self) -> None:
        vocab = build_vocabulary("side-effects and side-effects")
        assert vocab.resolve("Side", "Effects") == "Side-Effects"

    def test_empty_document_is_safe(self) -> None:
        assert build_vocabulary("").resolve("a", "b") == "ab"


class TestTitleFromFilename:
    def test_libgen_suffix_stripped(self) -> None:
        title = _title_from_filename(Path("Some Medical Book - libgen.li.pdf"))
        assert title == "Some Medical Book"

    def test_publisher_parenthetical_stripped(self) -> None:
        title = _title_from_filename(Path("British National Formulary (2023, The Press).pdf"))
        assert title == "British National Formulary"

    def test_leading_series_bracket_stripped(self) -> None:
        title = _title_from_filename(Path("[Oxford Medical Handbooks] Clinical Medicine.pdf"))
        assert title == "Clinical Medicine"

    def test_underscores_become_spaces(self) -> None:
        assert _title_from_filename(Path("BNF85_March 2023.pdf")) == "BNF85 March 2023"

    def test_real_bnf_filename_is_identifiable(self) -> None:
        """The BNF's embedded metadata title is empty, so this path is load-bearing."""
        title = _title_from_filename(
            Path(
                "Joint Formulary Committee - British National Formulary (BNF85)_ "
                "March 2023-September 2023 (2023, The Pharmaceutical Press) - libgen.li.pdf"
            )
        )
        assert "British National Formulary" in title
        assert "libgen" not in title
        assert "Pharmaceutical Press" not in title

    def test_plain_name_survives(self) -> None:
        assert _title_from_filename(Path("B09474-eng.pdf")) == "B09474-eng"


class TestHeadingDetection:
    def test_numbered_heading(self) -> None:
        assert _looks_like_heading("3.2 Management of Sepsis")

    def test_plain_heading(self) -> None:
        assert _looks_like_heading("Contraindications")

    def test_multiline_is_not_a_heading(self) -> None:
        assert not _looks_like_heading("Contraindications\nand cautions")

    def test_long_text_is_not_a_heading(self) -> None:
        assert not _looks_like_heading("This is a full sentence of body text " * 4)

    def test_lowercase_start_is_not_a_heading(self) -> None:
        assert not _looks_like_heading("contraindications for use")


class TestPageBlockOrder:
    def test_heading_precedes_table_in_reading_order(self) -> None:
        class Table:
            bbox = (10, 120, 400, 300)

            @staticmethod
            def extract():
                return [["Indication", "Dose"], ["Infection", "150 mg"]]

        class Page:
            @staticmethod
            def find_tables():
                return SimpleNamespace(tables=[Table()])

            @staticmethod
            def get_text(kind):
                assert kind == "blocks"
                return [(10, 80, 150, 105, "Clindamycin", 0, 0)]

        page = Page()
        blocks = PDFParser(use_layout_analysis=True)._page_blocks(
            SimpleNamespace(load_page=lambda _index: page), 0
        )

        assert blocks[0] == ("Clindamycin", "text")
        assert blocks[1][1] == "table"
        assert "150 mg" in blocks[1][0]

    def test_line_based_table_detection_fallback_when_layout_finds_none(self) -> None:
        class Table:
            bbox = (10, 120, 400, 300)

            @staticmethod
            def extract():
                return [["Indication", "Dose"], ["Infection", "150 mg"]]

        calls = []

        class Page:
            layout_information = []

            @staticmethod
            def find_tables(**kwargs):
                calls.append(kwargs)
                return SimpleNamespace(tables=[Table()] if kwargs else [])

            @staticmethod
            def get_text(kind):
                assert kind == "blocks"
                return []

        blocks = PDFParser(use_layout_analysis=True)._page_blocks(
            SimpleNamespace(load_page=lambda _index: Page()), 0
        )

        assert calls == [{}, {"use_layout": False}]
        assert len(blocks) == 1
        assert blocks[0][1] == "table"
        assert "150 mg" in blocks[0][0]

    def test_layout_disabled_uses_line_based_detection_directly(self) -> None:
        calls = []

        class Page:
            @staticmethod
            def find_tables(**kwargs):
                calls.append(kwargs)
                return SimpleNamespace(tables=[])

            @staticmethod
            def get_text(kind):
                assert kind == "blocks"
                return []

        PDFParser(use_layout_analysis=False)._page_blocks(
            SimpleNamespace(load_page=lambda _index: Page()), 0
        )

        assert calls == [{"use_layout": False}]


    def test_e0e7f5_panel_with_dose_label_is_a_dose_box(self) -> None:
        from medical_rag.ingest import pdf_parser

        doc = pdf_parser.pymupdf.open()
        page = doc.new_page(width=595, height=842)
        dose_fill = (224 / 255, 231 / 255, 245 / 255)
        page.draw_rect((50, 80, 285, 220), color=None, fill=dose_fill)
        page.insert_text((62, 105), "INDICATIONS AND DOSE", fontsize=12)
        page.insert_text((62, 132), "Pain and inflammation", fontsize=11)
        page.insert_text((72, 157), "Adult: 1 g once daily", fontsize=11)

        # The same fill without the BNF dose label is a callout, not a dose box.
        page.draw_rect((310, 80, 550, 220), color=None, fill=dose_fill)
        page.insert_text((322, 105), "IMPORTANT SAFETY INFORMATION", fontsize=10)
        page.insert_text((322, 130), "Use with care and monitor the patient.", fontsize=10)
        page.insert_text((50, 260), "Nearby text remains outside the dose box.", fontsize=10)

        blocks = PDFParser(extract_tables=False)._page_blocks(doc, 0)
        doc.close()

        dose = [text for text, kind in blocks if kind == "dose_box"]
        assert len(dose) == 1
        assert "Adult: 1 g once daily" in dose[0]
        assert "IMPORTANT SAFETY INFORMATION" not in dose[0]
        assert any(
            kind == "text" and "Nearby text remains outside" in text
            for text, kind in blocks
        )


class TestLayoutActivation:
    def test_parse_activates_layout_when_extracting_tables(self, monkeypatch) -> None:
        import pymupdf.layout

        from medical_rag.ingest import pdf_parser

        activated = []
        monkeypatch.setattr(pymupdf.layout, "activate", lambda: activated.append(True))
        doc = SimpleNamespace(
            page_count=0,
            metadata={},
            close=lambda: None,
        )
        monkeypatch.setattr(pdf_parser.pymupdf, "open", lambda _path: doc)

        PDFParser(use_layout_analysis=True).parse("test.pdf")

        assert activated == [True]

    def test_parse_does_not_activate_layout_without_table_extraction(self, monkeypatch) -> None:
        import pymupdf.layout

        from medical_rag.ingest import pdf_parser

        activated = []
        monkeypatch.setattr(pymupdf.layout, "activate", lambda: activated.append(True))
        doc = SimpleNamespace(
            page_count=0,
            metadata={},
            close=lambda: None,
        )
        monkeypatch.setattr(pdf_parser.pymupdf, "open", lambda _path: doc)

        PDFParser(extract_tables=False).parse("test.pdf")

        assert activated == []

    def test_layout_analysis_is_disabled_by_default(self) -> None:
        assert PDFParser().use_layout_analysis is False

    def test_parse_preserves_original_page_numbers_for_selected_pages(self, monkeypatch) -> None:
        from medical_rag.ingest import pdf_parser

        pages = [
            SimpleNamespace(get_text=lambda: "first page vocabulary"),
            SimpleNamespace(get_text=lambda: "second page vocabulary"),
            SimpleNamespace(get_text=lambda: "third page vocabulary"),
        ]
        doc = SimpleNamespace(
            page_count=3,
            metadata={},
            load_page=lambda index: pages[index],
            close=lambda: None,
        )
        monkeypatch.setattr(pdf_parser.pymupdf, "open", lambda _path: doc)
        parser = PDFParser(extract_tables=False)
        monkeypatch.setattr(
            parser,
            "_page_blocks",
            lambda _doc, index, _vocab: [(f"Page {index + 1} text long enough", "text")],
        )

        parsed = parser.parse("test.pdf", page_indices=[0, 2])

        assert [block.page for block in parsed.blocks] == [1, 3]
        assert parsed.metadata["pages_parsed"] == 2

    def test_bnf_selected_pages_use_bnf_profile(self, monkeypatch) -> None:
        from medical_rag.ingest import pdf_parser
        from medical_rag.models import ParsedDocument

        doc = SimpleNamespace(page_count=3, metadata={}, close=lambda: None)
        monkeypatch.setattr(pdf_parser.pymupdf, "open", lambda _path: doc)
        parser = PDFParser()
        calls = []

        def parse_bnf(_doc, path, pages):
            calls.append(list(pages))
            return ParsedDocument(
                doc_id="bnf", source_path=str(path), title="BNF", n_pages=3,
                metadata={"parser_profile": "bnf", "pages_parsed": len(pages)},
            )

        monkeypatch.setattr(parser, "_parse_bnf", parse_bnf)
        parsed = parser.parse("BNF-sample.pdf", page_indices=[0, 2])
        assert calls == [[0, 2]]
        assert parsed.metadata["parser_profile"] == "bnf"

    def test_parse_rejects_conflicting_or_invalid_page_selection(self, monkeypatch) -> None:
        from medical_rag.ingest import pdf_parser

        doc = SimpleNamespace(page_count=2, close=lambda: None)
        monkeypatch.setattr(pdf_parser.pymupdf, "open", lambda _path: doc)

        with pytest.raises(ValueError, match="cannot be used together"):
            PDFParser().parse("test.pdf", max_pages=1, page_indices=[0])
        with pytest.raises(ValueError, match="between 0 and 1"):
            PDFParser().parse("test.pdf", page_indices=[2])


class TestDocId:
    def test_is_stable(self) -> None:
        assert PDFParser.doc_id_for("a/b/book.pdf") == PDFParser.doc_id_for("a/b/book.pdf")

    def test_ignores_directory_so_both_machines_agree(self) -> None:
        """Two machines store the PDFs at different paths; a path-derived id
        would duplicate every document in Qdrant."""
        assert PDFParser.doc_id_for(r"D:\medical rag\RAW_DATA\bnf.pdf") == PDFParser.doc_id_for(
            r"C:\Users\Khan-PC\Downloads\medical_rag\RAW_DATA\bnf.pdf"
        )

    def test_different_files_differ(self) -> None:
        assert PDFParser.doc_id_for("bnf.pdf") != PDFParser.doc_id_for("oxford.pdf")


def test_continuation_table_row_is_not_promoted_to_header() -> None:
    class ContinuationTable:
        @staticmethod
        def extract():
            return [["Maintenance dose", "1 g daily"], ["Adult", "2 g daily"]]

    markdown = PDFParser._table_to_markdown(ContinuationTable())
    assert markdown.splitlines()[0] == "| Content |  |"
    assert "| Maintenance dose | 1 g daily |" in markdown
    assert "| Adult | 2 g daily |" in markdown
