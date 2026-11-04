from __future__ import annotations

from collections import Counter

import pymupdf

from medical_rag.ingest.katzung_layout import (
    KATZUNG_PARSER_VERSION,
    KatzungCtx,
    extract_katzung_blocks,
    parse_katzung_pages,
)

RAW = (
    "RAW_DATA/Katzung &amp_ Trevor&_039_s Pharmacology Examination and Board Review"
    "{Bertram G. Katzung, Marieke Kruidering-Hall, Rupa Lalchandani Tuan, "
    "Todd W. Vanderah, Anthony J.}...(2021, McGraw Hill Medical){105357206} lib.pdf"
)


def _doc():
    return pymupdf.open(RAW)


def test_parser_version_is_declared():
    assert KATZUNG_PARSER_VERSION == "katzung-layout-1.0.0"


def test_parser_captures_chapters_and_parts():
    doc = _doc()
    try:
        tags = parse_katzung_pages(doc, list(range(doc.page_count)))
    finally:
        doc.close()

    assert len(tags) > 50_000
    chapters = {t["ctx"].chapter_title for t in tags if t["ctx"].chapter_title}
    parts = {t["ctx"].part_title for t in tags if t["ctx"].part_title}
    assert len(chapters) >= 20
    assert len(parts) >= 5
    assert "Introduction" in chapters
    assert "Autonomic Drugs" in parts


def test_two_column_reading_order_is_left_then_right():
    """Katzung runs left column top->bottom, then right column top->bottom."""
    doc = _doc()
    try:
        page = doc.load_page(18)  # chapter 1 body page, both columns populated
        blocks = extract_katzung_blocks(page)
        body = [b for b in blocks if b.zone == "body"]
        left = [b for b in body if b.x0 < 300]
        right = [b for b in body if b.x0 >= 300]
        assert left and right
        # The left column's first block must precede the right column's first
        # block in reading order.
        from medical_rag.ingest.katzung_layout import order_katzung_blocks

        ordered = order_katzung_blocks(body, page.rect.width)
        left_in_ordered = [b for b in ordered if b in left]
        right_in_ordered = [b for b in ordered if b in right]
        assert left_in_ordered and right_in_ordered
        assert ordered.index(left_in_ordered[0]) < ordered.index(right_in_ordered[0])
    finally:
        doc.close()


def test_footer_watermark_is_dropped():
    doc = _doc()
    try:
        page = doc.load_page(18)
        blocks = extract_katzung_blocks(page)
        assert all("ALGRAWANY" not in b.text for b in blocks)
    finally:
        doc.close()


def test_table_headings_are_detected_as_tables():
    doc = _doc()
    try:
        tags = parse_katzung_pages(doc, list(range(doc.page_count)))
    finally:
        doc.close()

    kinds = Counter(t["kind"] for t in tags)
    assert kinds["table"] > 50
    assert kinds["high_yield"] > 0
    # Every table tag carries its own heading as the ctx label.
    assert all(t["ctx"].table_title for t in tags if t["kind"] == "table")
    assert any("TABLE 1-1" in t["text"] for t in tags if t["kind"] == "table")


def test_table_title_does_not_leak_into_following_paragraph():
    """A table heading only qualifies the table it heads."""
    doc = _doc()
    try:
        tags = parse_katzung_pages(doc, list(range(doc.page_count)))
    finally:
        doc.close()

    for i, t in enumerate(tags):
        if t["kind"] == "table":
            # The tag immediately after a table heading is not itself a table.
            nxt = tags[i + 1]
            assert nxt["kind"] != "table"
            assert nxt["ctx"].table_title is None


def test_katzung_ctx_label_includes_table_title():
    ctx = KatzungCtx(
        chapter_title="Introduction",
        subsection="A. Size and Molecular Weight",
        table_title="TABLE 1-1 Common routes of drug administration.",
    )
    assert ctx.label() == (
        "Introduction > A. Size and Molecular Weight > "
        "TABLE 1-1 Common routes of drug administration."
    )