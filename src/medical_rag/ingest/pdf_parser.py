"""PDF -> ParsedDocument.

Medical PDFs are heavy on multi-column layout, headers/footers and tables.
We use PyMuPDF because it is pure-wheel (no system deps — important since we are
not running Docker on the laptop) and gives us block-level geometry, which is
what we need to (a) drop running headers/footers and (b) read columns in order.

Page numbers are preserved on every block because citations are mandatory
downstream: an answer that cannot name a page cannot pass the guardrails.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

from ..models import PageBlock, ParsedDocument

logger = logging.getLogger(__name__)

# A heading looks like "3.2 Management of Sepsis" or "MANAGEMENT OF SEPSIS".
_HEADING_RE = re.compile(
    r"^\s*(?:(?:\d+\.)*\d+\s+)?(?P<title>[A-Z][A-Za-z0-9 ,\-/()']{3,80})\s*$"
)
_WS_RE = re.compile(r"[ \t ]+")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _clean(text: str) -> str:
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)  # de-hyphenate across line breaks
    text = text.replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def _looks_like_heading(text: str) -> bool:
    if len(text) > 90 or "\n" in text.strip():
        return False
    return bool(_HEADING_RE.match(text.strip()))


class PDFParser:
    """Extracts ordered, de-noised text blocks (plus tables) from a PDF."""

    def __init__(
        self,
        *,
        extract_tables: bool = True,
        drop_repeating_headers: bool = True,
        min_block_chars: int = 40,
    ) -> None:
        self.extract_tables = extract_tables
        self.drop_repeating_headers = drop_repeating_headers
        self.min_block_chars = min_block_chars

    # -- public ---------------------------------------------------------

    def parse(self, path: str | Path) -> ParsedDocument:
        path = Path(path)
        doc = fitz.open(path)
        try:
            raw_pages = [self._page_blocks(doc, i) for i in range(doc.page_count)]
            boilerplate = (
                self._find_boilerplate(raw_pages) if self.drop_repeating_headers else set()
            )

            blocks: list[PageBlock] = []
            current_section: str | None = None
            for page_no, page_blocks in enumerate(raw_pages, start=1):
                for text, kind in page_blocks:
                    if kind == "text":
                        norm = text.strip().lower()
                        if norm in boilerplate:
                            continue
                        if _looks_like_heading(text):
                            current_section = text.strip()
                            # Headings are kept: they carry real clinical meaning
                            # ("Contraindications", "Adverse effects").
                        if len(text) < self.min_block_chars and not _looks_like_heading(text):
                            continue
                    blocks.append(
                        PageBlock(
                            text=text,
                            page=page_no,
                            block_type=kind,  # type: ignore[arg-type]
                            section=current_section,
                        )
                    )

            title = self._title(doc, path, blocks)
            return ParsedDocument(
                doc_id=self._doc_id(path),
                source_path=str(path.resolve()),
                title=title,
                n_pages=doc.page_count,
                blocks=blocks,
                metadata={
                    "pdf_title": (doc.metadata or {}).get("title") or "",
                    "author": (doc.metadata or {}).get("author") or "",
                    "n_blocks": len(blocks),
                },
            )
        finally:
            doc.close()

    def parse_directory(self, directory: str | Path) -> list[ParsedDocument]:
        directory = Path(directory)
        pdfs = sorted(directory.rglob("*.pdf"))
        if not pdfs:
            logger.warning("No PDFs found under %s", directory)
        docs: list[ParsedDocument] = []
        for p in pdfs:
            try:
                docs.append(self.parse(p))
                logger.info("Parsed %s", p.name)
            except Exception:
                logger.exception("Failed to parse %s — skipping", p)
        return docs

    # -- internals ------------------------------------------------------

    def _page_blocks(self, doc: "fitz.Document", index: int) -> list[tuple[str, str]]:
        page = doc.load_page(index)
        out: list[tuple[str, str]] = []

        table_boxes: list[fitz.Rect] = []
        if self.extract_tables:
            try:
                for table in page.find_tables():
                    md = self._table_to_markdown(table)
                    if md:
                        out.append((md, "table"))
                        table_boxes.append(fitz.Rect(table.bbox))
            except Exception:
                logger.debug("Table extraction failed on page %d", index + 1)

        raw = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        # Sort into reading order: coarse vertical bands, then left-to-right.
        # The 30pt band tolerates baseline jitter within a line of columns.
        raw = [b for b in raw if len(b) >= 5 and isinstance(b[4], str) and b[4].strip()]
        raw.sort(key=lambda b: (round(b[1] / 30.0), b[0]))

        for x0, y0, x1, y1, text, *_ in raw:
            if any(fitz.Rect(x0, y0, x1, y1).intersects(tb) for tb in table_boxes):
                continue  # already captured as a table
            cleaned = _clean(text)
            if cleaned:
                out.append((cleaned, "text"))
        return out

    @staticmethod
    def _table_to_markdown(table) -> str:
        try:
            rows = table.extract()
        except Exception:
            return ""
        rows = [[(c or "").strip().replace("\n", " ") for c in row] for row in rows if any(row)]
        if len(rows) < 2:
            return ""
        header, *body = rows
        width = max(len(r) for r in rows)
        header = header + [""] * (width - len(header))
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * width]
        for r in body:
            r = r + [""] * (width - len(r))
            lines.append("| " + " | ".join(r) + " |")
        return "\n".join(lines)

    @staticmethod
    def _find_boilerplate(pages: list[list[tuple[str, str]]], min_ratio: float = 0.5) -> set[str]:
        """Short lines repeated on >=50% of pages are headers/footers, not content."""
        if len(pages) < 4:
            return set()
        counts: Counter[str] = Counter()
        for blocks in pages:
            seen_on_page = {
                t.strip().lower() for t, kind in blocks if kind == "text" and len(t.strip()) < 120
            }
            counts.update(seen_on_page)
        cutoff = max(3, int(len(pages) * min_ratio))
        return {text for text, n in counts.items() if n >= cutoff}

    @staticmethod
    def _title(doc: "fitz.Document", path: Path, blocks: list[PageBlock]) -> str:
        meta_title = ((doc.metadata or {}).get("title") or "").strip()
        if len(meta_title) > 3:
            return meta_title
        for b in blocks:
            if b.page == 1 and 10 < len(b.text) < 150:
                return b.text.strip().split("\n")[0]
        return path.stem

    @staticmethod
    def doc_id_for(path: str | Path) -> str:
        """Stable id for a source file, so re-runs address the same document."""
        return hashlib.sha1(str(Path(path).resolve()).encode("utf-8")).hexdigest()[:16]

    _doc_id = doc_id_for
