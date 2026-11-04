"""Katzung-aware layout analysis on top of PyMuPDF.

Katzung & Trevor's Pharmacology Examination and Board Review has a distinctive layout:
  - Header band (y0 < 45):
      * Even pages: "N PART X Book Title" or page number + Part title
      * Odd pages: "CHAPTER N Title <printed_page>"
      * Appendices & Index have corresponding running headers.
  - Body band (45 <= y0 < 745): Two columns (left x0 ~40-290, right x0 ~305-554).
      * Major section headings: Roman numerals / uppercase ("I. THE NATURE OF DRUGS", size ~13-15).
      * Subsection headings: Capital letters ("A. Size and Molecular Weight", "B. Drug-Receptor Bonds", size ~9-10 bold).
      * Special sections: "High-Yield Terms to Learn" (two-column definition list), "Summary Table", "Skill Keeper".
  - Footer band (y0 >= 745): Page watermarks ("ALGRAWANY ...") dropped.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, replace

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

KATZUNG_PARSER_VERSION = "katzung-layout-1.0.0"

_PAGE_TOKEN_RE = re.compile(r"^\d{1,4}$")
_CHAPTER_HEADER_RE = re.compile(r"CHAPTER\s+(\d+)\s+(.+?)\s+(\d{1,4})$", re.I)
_PART_HEADER_RE = re.compile(r"(?:(\d+)\s+)?PART\s+([IVXLCDM]+)\s+(.+)", re.I)
_NUM_HEADING_RE = re.compile(r"^(?:[IVXLCDM]+\.)\s+([A-Z][A-Za-z0-9 ,\-/()']{2,90})$")
_SUB_HEADING_RE = re.compile(r"^[A-Z]\.\s+([A-Z][A-Za-z0-9 ,\-/()']{2,80})$")

# Katzung's special blocks. PyMuPDF find_tables() returns no tables on these
# pages -- the tables are plain text spans -- so the heading is the only signal
# we get, and it is the thing that makes the table retrievable at all.
_SUMMARY_TABLE_RE = re.compile(r"^(?:DRUG\s+)?SUMMARY\s+TABLE(?:\s*:\s*(.+))?$", re.I)
_SKILL_KEEPER_RE = re.compile(r"^SKILL\s+KEEPER(?:\s*:\s*(.+))?$", re.I)
_HIGH_YIELD_RE = re.compile(r"^HIGH-YIELD\s+TERMS\s+TO\s+LEARN$", re.I)
_TABLE_HEADING_RE = re.compile(r"^TABLE\s+(\d+)-(\d+)\s*(.*)$", re.I)


@dataclass
class KatzungCtx:
    part_no: str | None = None
    part_title: str | None = None
    chapter_no: str | None = None
    chapter_title: str | None = None
    section: str | None = None
    subsection: str | None = None
    table_title: str | None = None
    printed_page: str | None = None

    def label(self) -> str:
        parts = [self.chapter_title, self.section, self.subsection, self.table_title]
        return " > ".join(p for p in parts if p)


@dataclass
class KatzungTagged:
    text: str
    kind: str  # text | table | high_yield | summary_table
    page: int
    ctx: KatzungCtx


@dataclass
class KLBlock:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    size: float = 0.0
    bold: bool = False
    zone: str = "body"
    kind: str = "text"


def extract_katzung_blocks(page) -> list[KLBlock]:
    width, height = page.rect.width, page.rect.height
    out: list[KLBlock] = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        for line in b["lines"]:
            spans = line["spans"]
            text = "".join(s["text"] for s in spans)
            if not text.strip():
                continue
            sizes, bolds = Counter(), Counter()
            for span in spans:
                count = len(span["text"].strip())
                if count:
                    sizes[round(span["size"], 1)] += count
                    bolds[bool(span["flags"] & 16)] += count
            x0, y0, x1, y1 = line.get("bbox", b["bbox"])
            if y0 < 45:
                zone = "header"
            elif y0 > 745 or "ALGRAWANY" in text:
                zone = "footer"
            else:
                zone = "body"
            out.append(
                KLBlock(
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    text=text,
                    size=sizes.most_common(1)[0][0] if sizes else 0.0,
                    bold=bolds.most_common(1)[0][0] if bolds else False,
                    zone=zone,
                )
            )
    return out


def order_katzung_blocks(blocks: list[KLBlock], page_w: float) -> list[KLBlock]:
    """Two-column reading order (right column then left column, or left then right depending on layout)."""
    # Katzung uses a two-column layout separated by a gutter around x ~295-310.
    gutter = 300.0
    key = lambda b: (b.y0, b.x0)
    left = sorted([b for b in blocks if b.x1 <= gutter or (b.x0 < gutter and b.x1 < page_w * 0.55)], key=key)
    right = sorted([b for b in blocks if b.x0 >= gutter or b.x0 >= page_w * 0.55], key=key)
    # Check if page is single column (e.g. Q&A or TOC)
    if not left or not right or max(b.x0 for b in blocks) - min(b.x0 for b in blocks) < page_w * 0.4:
        return sorted(blocks, key=key)
    # In Katzung, running order can be left then right, or right then left for some sections.
    # Standard reading order for Katzung: left column first, then right column.
    return left + right


def parse_katzung_pages(doc, page_indices) -> list[dict]:
    ctx = KatzungCtx()
    results: list[dict] = []
    for idx in page_indices:
        page = doc.load_page(idx)
        page_no = idx + 1
        blocks = extract_katzung_blocks(page)
        header_blocks = [b for b in blocks if b.zone == "header"]
        body_blocks = [b for b in blocks if b.zone == "body"]

        # Parse header
        header_text = " ".join(b.text.replace("\n", " ") for b in header_blocks)
        m_chap = _CHAPTER_HEADER_RE.search(header_text)
        if m_chap:
            ctx.chapter_no = m_chap.group(1)
            ctx.chapter_title = m_chap.group(2).strip()
            ctx.printed_page = m_chap.group(3)
        else:
            m_part = _PART_HEADER_RE.search(header_text)
            if m_part:
                ctx.part_no = m_part.group(2)
                ctx.part_title = m_part.group(3).strip()

        ordered = order_katzung_blocks(body_blocks, page.rect.width)
        for b in ordered:
            text = b.text.strip()
            if not text:
                continue
            # Check for major section heading
            m_sec = _NUM_HEADING_RE.match(text)
            if m_sec and b.size >= 12:
                ctx.section = m_sec.group(0)
                ctx.subsection = None
                continue
            m_sub = _SUB_HEADING_RE.match(text)
            if m_sub and b.size >= 9:
                ctx.subsection = text
                continue

            # Special blocks: Katzung's tables are plain text spans, so
            # PyMuPDF finds none -- the heading is the only boundary signal.
            kind = "text"
            m_tbl = _TABLE_HEADING_RE.match(text)
            if m_tbl:
                ctx.table_title = text
                kind = "table"
            elif (m_sum := _SUMMARY_TABLE_RE.match(text)):
                ctx.table_title = m_sum.group(1) or text
                kind = "table"
            elif (m_skill := _SKILL_KEEPER_RE.match(text)):
                ctx.table_title = m_skill.group(1) or text
                kind = "table"
            elif _HIGH_YIELD_RE.match(text):
                ctx.table_title = text
                kind = "high_yield"
            else:
                # A table heading only qualifies the table it heads; the
                # paragraph that follows is not part of it.
                ctx.table_title = None

            results.append(
                {
                    "text": text,
                    "page": page_no,
                    "kind": kind,
                    "ctx": replace(ctx),
                }
            )
    return results
