"""PDF -> ParsedDocument.

Medical PDFs are heavy on multi-column layout, headers/footers and tables.
We use PyMuPDF because it is pure-wheel (no system deps, no build toolchain, so
a fresh clone installs anywhere) and gives us block-level geometry -- what we
need to (a) drop running headers/footers and (b) read columns in order.

Page numbers are preserved on every block because citations are mandatory
downstream: an answer that cannot name a page cannot pass the guardrails.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from collections import Counter
from pathlib import Path

try:  # PyMuPDF >= 1.24 exposes `pymupdf`; the `fitz` alias is deprecated in 1.28.
    import pymupdf
except ImportError:  # pragma: no cover - older PyMuPDF
    import fitz as pymupdf

from ..models import PageBlock, ParsedDocument

logger = logging.getLogger(__name__)

# A heading looks like "3.2 Management of Sepsis" or "MANAGEMENT OF SEPSIS".
_HEADING_RE = re.compile(r"^\s*(?:(?:\d+\.)*\d+\s+)?(?P<title>[A-Z][A-Za-z0-9 ,\-/()']{3,80})\s*$")
_BNF_ABOVE_MARKER_RE = re.compile(r"^e+i+\s+F above\s+", re.IGNORECASE)
# Spaces, tabs and U+00A0 (non-breaking space, common in typeset medical PDFs).
_WS_RE = re.compile("[ \t ]+")
_HYPHEN_BREAK_RE = re.compile(r"([A-Za-z]+)-\n([A-Za-z]+)")
_MULTI_NEWLINE_RE = re.compile(r"\n{2,}")
# A newline with no blank line either side is a typesetting line wrap, not a
# paragraph break -- PDFs break lines mid-sentence.
_LINE_WRAP_RE = re.compile(r"(?<!\n)\n(?!\n)")
# Running headers carry a changing page number, so exact matching never catches
# them. Folding digits to '#' makes "page 28" and "page 29" the same line.
_DIGITS_RE = re.compile(r"\d+")
# Ligatures PyMuPDF sometimes emits verbatim; they break BM25 tokenisation
# ("inflammation" vs "in{ligature}ammation").
_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}


class Vocabulary:
    """Word counts for one document, used to resolve line-break hyphens.

    A hyphen at a line break is ambiguous: "inflamma-\\ntion" is a syllable
    break that must be joined, while "side-\\neffects" is a real compound that
    must keep its hyphen. Measured across this corpus the split is not even
    close to consistent -- the Oxford Handbook is ~89% syllable breaks, the BNF
    ~80% real compounds -- so a fixed rule is wrong for one book or the other.

    The document itself holds the evidence: whichever form it uses elsewhere,
    away from a line break, is the form intended here.
    """

    __slots__ = ("words", "hyphenated")

    def __init__(self, words: Counter, hyphenated: Counter) -> None:
        self.words = words
        self.hyphenated = hyphenated

    def resolve(self, left: str, right: str) -> str:
        low_l, low_r = left.lower(), right.lower()
        # Strongest evidence first: the document writes it that way elsewhere.
        if self.hyphenated.get(f"{low_l}-{low_r}", 0) >= 1:
            return f"{left}-{right}"
        if self.words.get(f"{low_l}{low_r}", 0) >= 1:
            return f"{left}{right}"
        # Otherwise: two words that both stand alone are a compound.
        if self.words.get(low_l, 0) >= 2 and self.words.get(low_r, 0) >= 2:
            return f"{left}-{right}"
        return f"{left}{right}"


_VOCAB_TOKEN_RE = re.compile(r"[A-Za-z]{2,}(?:-[A-Za-z]{2,})+|[A-Za-z]{2,}")


def build_vocabulary(raw_text: str) -> Vocabulary:
    """Count words that are NOT truncated by a line-break hyphen.

    Tokens adjacent to a line-break hyphen are excluded, so the ambiguous cases
    cannot vote on their own resolution.
    """
    words: Counter[str] = Counter()
    hyphenated: Counter[str] = Counter()
    skip_first = False

    for line in raw_text.split("\n"):
        stripped = line.rstrip()
        ends_hyphenated = stripped.endswith("-")
        if ends_hyphenated:
            stripped = stripped[:-1]

        tokens = _VOCAB_TOKEN_RE.findall(stripped)
        if skip_first and tokens:
            tokens = tokens[1:]  # second half of a hyphenated word
        if ends_hyphenated and tokens:
            tokens = tokens[:-1]  # first half of a hyphenated word

        for token in tokens:
            low = token.lower()
            if "-" in low:
                hyphenated[low] += 1
            else:
                words[low] += 1

        skip_first = ends_hyphenated
    return Vocabulary(words, hyphenated)


_EMPTY_VOCAB = Vocabulary(Counter(), Counter())


def _clean(text: str, vocab: Vocabulary | None = None) -> str:
    for ligature, replacement in _LIGATURES.items():
        if ligature in text:
            text = text.replace(ligature, replacement)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    resolver = vocab or _EMPTY_VOCAB
    text = _HYPHEN_BREAK_RE.sub(lambda m: resolver.resolve(m.group(1), m.group(2)), text)

    # Order matters: collapse real paragraph breaks first, then unwrap the line
    # breaks PDF typesetting leaves mid-sentence. Without this every chunk (and
    # every quoted citation) is riddled with newlines mid-clause.
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = _LINE_WRAP_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def _boilerplate_key(text: str) -> str:
    return _DIGITS_RE.sub("#", text.strip().lower())


# Artefacts of how these files were named, stripped so citations read cleanly.
_LIBGEN_SUFFIX_RE = re.compile(r"\s*-\s*libgen\.\w+\s*$", re.IGNORECASE)
_PUBLISHER_PAREN_RE = re.compile(r"\s*\(\d{4}\s*,[^)]*\)\s*")
_LEADING_SERIES_RE = re.compile(r"^\s*\[[^\]]{0,60}\]\s*")


def _title_from_filename(path: Path) -> str:
    stem = path.stem
    stem = _LIBGEN_SUFFIX_RE.sub("", stem)
    stem = _PUBLISHER_PAREN_RE.sub(" ", stem)
    stem = _LEADING_SERIES_RE.sub("", stem)
    stem = stem.replace("_", " ")
    stem = re.sub(r"\s+", " ", stem).strip(" -,")
    return stem[:150] or path.stem


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
        use_layout_analysis: bool = False,
        drop_repeating_headers: bool = True,
        min_block_chars: int = 40,
        min_recurring_heading_count: int = 5,
    ) -> None:
        self.extract_tables = extract_tables
        self.use_layout_analysis = use_layout_analysis
        self.drop_repeating_headers = drop_repeating_headers
        self.min_block_chars = min_block_chars
        self.min_recurring_heading_count = min_recurring_heading_count

    # -- public ---------------------------------------------------------

    def parse(
        self,
        path: str | Path,
        *,
        max_pages: int | None = None,
        page_indices: Sequence[int] | None = None,
    ) -> ParsedDocument:
        path = Path(path)
        doc = pymupdf.open(path)
        try:
            if self.extract_tables and self.use_layout_analysis:
                from pymupdf import layout

                layout.activate()

            if max_pages is not None and page_indices is not None:
                raise ValueError("max_pages and page_indices cannot be used together")
            if page_indices is not None:
                selected_pages = list(page_indices)
                if any(index < 0 or index >= doc.page_count for index in selected_pages):
                    raise ValueError(f"page_indices must be between 0 and {doc.page_count - 1}")
                if len(set(selected_pages)) != len(selected_pages):
                    raise ValueError("page_indices cannot contain duplicates")
            else:
                page_count = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
                selected_pages = list(range(page_count))

            identity = f"{path.name} {(doc.metadata or {}).get('title') or ''}".casefold()
            is_bnf = "british national formulary" in identity or "bnf" in identity
            is_ohcm = "oxford handbook" in identity
            is_katzung = "katzung" in identity
            if is_bnf:
                return self._parse_bnf(doc, path, selected_pages)
            if is_ohcm:
                return self._parse_ohcm(doc, path, selected_pages)
            if is_katzung:
                return self._parse_katzung(doc, path, selected_pages)

            # First pass: the document's own vocabulary, which decides whether
            # each line-break hyphen is a compound or a syllable break. This is
            # the cheap text extraction; table detection is the expensive part
            # and happens once, in the second pass.
            vocab = build_vocabulary(
                "\n".join(doc.load_page(i).get_text() for i in range(doc.page_count))
            )

            raw_pages = [
                (index, self._page_blocks(doc, index, vocab)) for index in selected_pages
            ]
            boilerplate = (
                self._find_boilerplate([blocks for _, blocks in raw_pages])
                if self.drop_repeating_headers
                else set()
            )
            recurring_headings = self._find_recurring_headings(
                [blocks for _, blocks in raw_pages], min_count=self.min_recurring_heading_count
            )

            blocks: list[PageBlock] = []
            current_section: str | None = None
            for page_index, page_blocks in raw_pages:
                page_no = page_index + 1
                for text, kind in page_blocks:
                    if kind == "text":
                        text = _BNF_ABOVE_MARKER_RE.sub("", text, count=1)
                    if kind == "text":
                        if _boilerplate_key(text) in boilerplate:
                            continue
                        if _looks_like_heading(text):
                            # Headings are kept: they carry real clinical meaning
                            # ("Contraindications", "Adverse effects"). But a
                            # heading-shaped line that recurs many times verbatim
                            # is structural, not a fresh topic -- a chapter name
                            # reprinted as a running page header, or a generic
                            # subsection label ("Dose", "Interactions") reused
                            # under dozens of unrelated drugs. Letting either
                            # overwrite current_section wipes out the actual
                            # drug/topic name on every page it recurs on, which
                            # is how a multi-page monograph loses its own name
                            # partway through. Keep the existing, more specific
                            # section instead; only a heading below the
                            # recurrence threshold (a genuinely one-off topic
                            # name) replaces it.
                            if current_section is None or _boilerplate_key(text) not in recurring_headings:
                                current_section = text.strip()
                        elif len(text) < self.min_block_chars:
                            continue
                    blocks.append(
                        PageBlock(
                            text=text,
                            page=page_no,
                            block_type=kind,  # type: ignore[arg-type]
                            section=current_section,
                        )
                    )

            return ParsedDocument(
                doc_id=self.doc_id_for(path),
                source_path=str(path.resolve()),
                title=self._title(doc, path, blocks),
                n_pages=doc.page_count,
                blocks=blocks,
                metadata={
                    "pdf_title": (doc.metadata or {}).get("title") or "",
                    "author": (doc.metadata or {}).get("author") or "",
                    "n_blocks": len(blocks),
                    "pages_parsed": len(selected_pages),
                    "boilerplate_dropped": len(boilerplate),
                    "layout_analysis": self.use_layout_analysis and self.extract_tables,
                },
            )
        finally:
            doc.close()

    def _parse_ohcm(self, doc, path: Path, selected_pages: Sequence[int]) -> ParsedDocument:
        from . import ohcm_layout

        tags = ohcm_layout.parse_ohcm_pages(doc, selected_pages)
        blocks = []
        for item in tags:
            blocks.append(
                PageBlock(
                    text=item["text"],
                    page=item["page"],
                    block_type="text",
                    section=item["ctx"].label(),
                    metadata={"source": str(path.resolve())},
                )
            )
        return ParsedDocument(
            doc_id=self.doc_id_for(path),
            source_path=str(path.resolve()),
            title=self._title(doc, path, blocks),
            n_pages=doc.page_count,
            blocks=blocks,
            metadata={
                "parser_profile": "ohcm",
                "parser_version": ohcm_layout.OHCM_PARSER_VERSION,
                "boilerplate_dropped": 0,
            },
        )


    def _parse_katzung(self, doc, path: Path, selected_pages: Sequence[int]) -> ParsedDocument:
        from . import katzung_layout

        tags = katzung_layout.parse_katzung_pages(doc, selected_pages)
        blocks = []
        for item in tags:
            ctx = item["ctx"]
            meta = {"source": str(path.resolve())}
            # Carry the structural context into metadata so the chunker can
            # rebuild the breadcrumb header from it (see chunk_header()).
            for key in ("chapter_title", "section", "subsection", "table_title"):
                value = getattr(ctx, key, None)
                if value:
                    meta[key] = value
            if item.get("kind") != "text":
                meta["kind"] = item["kind"]
            blocks.append(
                PageBlock(
                    text=item["text"],
                    page=item["page"],
                    block_type="table" if item.get("kind") == "table" else "text",
                    section=ctx.label(),
                    metadata=meta,
                )
            )
        return ParsedDocument(
            doc_id=self.doc_id_for(path),
            source_path=str(path.resolve()),
            title=self._title(doc, path, blocks),
            n_pages=doc.page_count,
            blocks=blocks,
            metadata={
                "parser_profile": "katzung",
                "parser_version": katzung_layout.KATZUNG_PARSER_VERSION,
            },
        )


    def _parse_bnf(self, doc, path: Path, selected_pages: Sequence[int]) -> ParsedDocument:
        from . import bnf_layout

        def tables_for_page(page):
            if not self.extract_tables:
                return []
            try:
                if self.use_layout_analysis:
                    tables = page.find_tables().tables
                else:
                    tables = page.find_tables(use_layout=False).tables
                if self.use_layout_analysis and not tables:
                    tables = page.find_tables(use_layout=False).tables
                out = []
                for table in tables:
                    markdown = self._table_to_markdown(table)
                    if markdown:
                        out.append((pymupdf.Rect(table.bbox), markdown))
                return out
            except Exception as exc:
                logger.warning("BNF table extraction failed: %s", exc)
                return []

        tags = bnf_layout.bnf_pages(
            doc,
            selected_pages,
            tables_fn=tables_for_page if self.extract_tables else None,
        )
        blocks = []
        for item in tags:
            metadata = bnf_layout.ctx_meta(item)
            metadata["source"] = str(path.resolve())
            blocks.append(
                PageBlock(
                    text=item.text,
                    page=item.page,
                    block_type="table" if item.kind == "table" else "text",
                    section=item.ctx.label() or item.ctx.field,
                    metadata=metadata,
                )
            )

        pdf_metadata = doc.metadata or {}
        return ParsedDocument(
            doc_id=self.doc_id_for(path),
            source_path=str(path.resolve()),
            title=self._title(doc, path, blocks),
            n_pages=doc.page_count,
            blocks=blocks,
            metadata={
                "pdf_title": pdf_metadata.get("title") or "",
                "author": pdf_metadata.get("author") or "",
                "n_blocks": len(blocks),
                "pages_parsed": len(selected_pages),
                "layout_analysis": self.use_layout_analysis and self.extract_tables,
                "parser_profile": "bnf",
                "parser_version": bnf_layout.BNF_PARSER_VERSION,
            },
        )

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
                logger.exception("Failed to parse %s, skipping", p)
        return docs

    # -- internals ------------------------------------------------------

    @staticmethod
    def _dose_box_rects(page, raw_blocks) -> list:
        """Find BNF dose panels by their pale-blue PDF drawing fill.

        The fill is a candidate cue, not sufficient by itself: other BNF
        callouts can use the same color, so the panel must also contain the
        printed "INDICATIONS AND DOSE" label.
        """
        target = (224, 231, 245)  # #E0E7F5
        if not hasattr(page, "rect") or not hasattr(page, "get_drawings"):
            return []
        page_rect = page.rect
        min_width = page_rect.width * 0.22
        min_height = 30.0
        min_area = page_rect.width * page_rect.height * 0.01
        candidates = []

        for drawing in page.get_drawings():
            fill = drawing.get("fill")
            if not fill or len(fill) < 3:
                continue
            rgb = tuple(round(float(channel) * 255) for channel in fill[:3])
            if any(abs(actual - expected) > 3 for actual, expected in zip(rgb, target)):
                continue

            rect = pymupdf.Rect(drawing.get("rect"))
            if rect.width < min_width or rect.height < min_height:
                continue
            if rect.width * rect.height < min_area:
                continue

            center_text = []
            for block in raw_blocks:
                x0, y0, x1, y1, text, *_ = block
                center = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
                if rect.contains(center):
                    center_text.append(text)
            if re.search(r"\bINDICATIONS\s+AND\s+DOSE\b", "\n".join(center_text), re.I):
                candidates.append(rect)

        # A page can encode one filled panel as an outer rectangle plus nested
        # rectangles. Keep the largest candidate when their overlap is almost
        # complete, while preserving distinct dose panels on the same page.
        candidates.sort(key=lambda rect: rect.width * rect.height, reverse=True)
        unique = []
        for rect in candidates:
            area = rect.width * rect.height
            duplicate = False
            for kept in unique:
                intersection = rect & kept
                overlap = max(0.0, intersection.width) * max(0.0, intersection.height)
                smaller = min(area, kept.width * kept.height)
                if smaller and overlap / smaller >= 0.92:
                    duplicate = True
                    break
            if not duplicate:
                unique.append(rect)
        return unique

    def _page_blocks(self, doc, index: int, vocab: Vocabulary | None = None) -> list[tuple[str, str]]:
        page = doc.load_page(index)
        positioned: list[tuple[float, float, float, str, str]] = []
        table_boxes = []
        if self.extract_tables:
            try:
                if self.use_layout_analysis:
                    tables = page.find_tables().tables
                else:
                    tables = page.find_tables(use_layout=False).tables
                if self.use_layout_analysis and not tables:
                    # The layout model can classify dense BNF tables as text or
                    # list items; retain the line-based detector as a fallback.
                    tables = page.find_tables(use_layout=False).tables
                for table in tables:
                    md = self._table_to_markdown(table)
                    if md:
                        box = pymupdf.Rect(table.bbox)
                        positioned.append((box.y0, box.x0, box.x1, md, "table"))
                        table_boxes.append(box)
            except Exception as exc:
                logger.warning(
                    "Table/layout extraction failed on page %d: %s", index + 1, exc
                )

        raw = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        raw = [b for b in raw if len(b) >= 5 and isinstance(b[4], str) and b[4].strip()]

        # Collect text inside a verified BNF dose panel as one structural block.
        # This keeps the blue box together and prevents adjacent prose from
        # being absorbed into the dose passage.
        dose_boxes = self._dose_box_rects(page, raw)
        dose_members: dict[int, list[int]] = {i: [] for i in range(len(dose_boxes))}
        consumed: set[int] = set()
        for block_index, block in enumerate(raw):
            x0, y0, x1, y1, *_ = block
            center = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
            for box_index, box in enumerate(dose_boxes):
                if box.contains(center):
                    dose_members[box_index].append(block_index)
                    consumed.add(block_index)
                    break

        for box_index, box in enumerate(dose_boxes):
            members = []
            for block_index in dose_members[box_index]:
                x0, y0, x1, y1, text, *_ = raw[block_index]
                if any(pymupdf.Rect(x0, y0, x1, y1).intersects(tb) for tb in table_boxes):
                    continue
                cleaned = _clean(text, vocab)
                if cleaned:
                    members.append((y0, x0, cleaned))
            if members:
                members.sort(key=lambda item: (round(item[0] / 2), item[1]))
                text = "\n\n".join(item[2] for item in members)
                positioned.append((box.y0, box.x0, box.x1, text, "dose_box"))

        # Keep raw geometry until tables and text can be sorted together.
        for block_index, (x0, y0, x1, y1, text, *_) in enumerate(raw):
            if block_index in consumed:
                continue
            if any(pymupdf.Rect(x0, y0, x1, y1).intersects(tb) for tb in table_boxes):
                continue  # already captured as a table
            cleaned = _clean(text, vocab)
            if cleaned:
                positioned.append((y0, x0, x1, cleaned, "text"))

        # Use column-major order when a clear gutter separates two columns.


        page_rect = getattr(page, "rect", None)
        page_width = float(getattr(page_rect, "width", 0.0))
        positioned = self._sort_positioned(positioned, page_width)
        return [(text, kind) for _, _, _, text, kind in positioned]

    @staticmethod
    def _sort_positioned(items, page_width: float):
        def row_order(item):
            return (round(item[0] / 30.0), item[1])
        if len(items) < 6 or page_width <= 0:
            return sorted(items, key=row_order)
        starts = sorted(item[1] for item in items)
        gaps = [(right - left, left, right) for left, right in zip(starts, starts[1:])]
        if not gaps:
            return sorted(items, key=row_order)
        gap, left_edge, right_edge = max(gaps)
        gutter = (left_edge + right_edge) / 2
        left_count = sum(item[1] < gutter for item in items)
        right_count = len(items) - left_count
        if gap < page_width * 0.12 or min(left_count, right_count) < 2:
            return sorted(items, key=row_order)
        return sorted(
            items,
            key=lambda item: (item[1] >= gutter, round(item[0] / 30.0), item[1]),
        )

    @staticmethod
    def _table_to_markdown(table) -> str:
        try:
            rows = table.extract()
        except Exception:
            return ""
        rows = [[(c or "").strip().replace("\n", " ") for c in row] for row in rows if any(row)]
        if len(rows) < 2:
            return ""
        first_row = " ".join(rows[0]).casefold()
        first_cells = {cell.strip().casefold() for cell in rows[0] if cell.strip()}
        has_header = (
            any(
                marker in first_row
                for marker in ("indications and dose", "dose equivalence", "equivalent doses")
            )
            or (
                len(first_cells) >= 2
                and bool(first_cells & {"indication", "dose", "route", "form"})
            )
        )
        width = max(len(r) for r in rows)
        if has_header:
            header, *body = rows
        else:
            # Continuation pages can start with a dose row, not a column heading.
            header, body = ["Content"], rows
        header = header + [""] * (width - len(header))
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * width]
        for r in body:
            r = r + [""] * (width - len(r))
            lines.append("| " + " | ".join(r) + " |")
        return "\n".join(lines)

    @staticmethod
    def _find_boilerplate(pages: list[list[tuple[str, str]]], min_ratio: float = 0.5) -> set[str]:
        """Short lines repeated on >=50% of pages are headers/footers, not content.

        Without this, "British National Formulary 85" lands in a thousand chunks
        and skews BM25 for every query mentioning a formulary.
        """
        if len(pages) < 4:
            return set()
        counts: Counter[str] = Counter()
        for blocks in pages:
            seen_on_page = {
                _boilerplate_key(t)
                for t, kind in blocks
                if kind == "text" and len(t.strip()) < 120
            }
            counts.update(seen_on_page)
        cutoff = max(3, int(len(pages) * min_ratio))
        return {text for text, n in counts.items() if n >= cutoff}

    @staticmethod
    def _find_recurring_headings(
        pages: list[list[tuple[str, str]]], min_count: int = 5
    ) -> set[str]:
        """Heading-shaped lines that recur verbatim are structural, not topical.

        `_find_boilerplate` only catches lines repeated on >=50% of ALL pages,
        which misses a chapter name printed as a running header: it repeats on
        every page of *that chapter* but that is a fraction of the whole book.
        Measured on the BNF: "Nervous system" (a running header) occurred 218
        times and "CAUTIONS, FURTHER INFORMATION" (a generic subsection label
        reused across unrelated drugs) occurred 48 times, while a real,
        specific topic heading (a drug name) occurs once, maybe a handful of
        times at most (e.g. once in the body, once in an index). `min_count=5`
        sits well below the smallest structural repeat-offender observed and
        well above how often a genuine one-off heading legitimately recurs.
        """
        counts: Counter[str] = Counter()
        for blocks in pages:
            for text, kind in blocks:
                if kind == "text" and _looks_like_heading(text):
                    counts[_boilerplate_key(text)] += 1
        return {key for key, n in counts.items() if n >= min_count}

    @staticmethod
    def _title(doc, path: Path, blocks: list[PageBlock]) -> str:
        """Title used as the citation label, so it must identify the book.

        Falls back to the filename rather than a page-1 text block: cover pages
        yield things like "March - September 2023", which cites nothing. The
        BNF's embedded metadata title is empty, so this path is load-bearing.
        """
        meta_title = ((doc.metadata or {}).get("title") or "").strip()
        if len(meta_title) > 3:
            return meta_title[:150]
        return _title_from_filename(path)

    @staticmethod
    def doc_id_for(path: str | Path) -> str:
        """Stable id for a source file, so re-runs address the same document.

        Derived from the file *name*, not its absolute path, so two machines
        agree on document ids even with different folder layouts.
        """
        return hashlib.sha1(Path(path).name.encode("utf-8")).hexdigest()[:16]

    _doc_id = doc_id_for
