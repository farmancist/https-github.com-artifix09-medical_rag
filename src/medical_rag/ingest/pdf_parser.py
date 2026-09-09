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
        drop_repeating_headers: bool = True,
        min_block_chars: int = 40,
    ) -> None:
        self.extract_tables = extract_tables
        self.drop_repeating_headers = drop_repeating_headers
        self.min_block_chars = min_block_chars

    # -- public ---------------------------------------------------------

    def parse(self, path: str | Path, *, max_pages: int | None = None) -> ParsedDocument:
        path = Path(path)
        doc = pymupdf.open(path)
        try:
            n_pages = doc.page_count if max_pages is None else min(doc.page_count, max_pages)

            # First pass: the document's own vocabulary, which decides whether
            # each line-break hyphen is a compound or a syllable break. This is
            # the cheap text extraction; table detection is the expensive part
            # and happens once, in the second pass.
            vocab = build_vocabulary(
                "\n".join(doc.load_page(i).get_text() for i in range(n_pages))
            )

            raw_pages = [self._page_blocks(doc, i, vocab) for i in range(n_pages)]
            boilerplate = (
                self._find_boilerplate(raw_pages) if self.drop_repeating_headers else set()
            )

            blocks: list[PageBlock] = []
            current_section: str | None = None
            for page_no, page_blocks in enumerate(raw_pages, start=1):
                for text, kind in page_blocks:
                    if kind == "text":
                        if _boilerplate_key(text) in boilerplate:
                            continue
                        if _looks_like_heading(text):
                            # Headings are kept: they carry real clinical meaning
                            # ("Contraindications", "Adverse effects").
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
                    "pages_parsed": n_pages,
                    "boilerplate_dropped": len(boilerplate),
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
                logger.exception("Failed to parse %s, skipping", p)
        return docs

    # -- internals ------------------------------------------------------

    def _page_blocks(self, doc, index: int, vocab: Vocabulary | None = None) -> list[tuple[str, str]]:
        page = doc.load_page(index)
        out: list[tuple[str, str]] = []

        table_boxes = []
        if self.extract_tables:
            try:
                for table in page.find_tables():
                    md = self._table_to_markdown(table)
                    if md:
                        out.append((md, "table"))
                        table_boxes.append(pymupdf.Rect(table.bbox))
            except Exception:
                logger.debug("Table extraction failed on page %d", index + 1)

        raw = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        raw = [b for b in raw if len(b) >= 5 and isinstance(b[4], str) and b[4].strip()]
        # Reading order: coarse vertical bands, then left-to-right. The 30pt band
        # tolerates baseline jitter between columns without merging separate rows.
        raw.sort(key=lambda b: (round(b[1] / 30.0), b[0]))

        for x0, y0, x1, y1, text, *_ in raw:
            if any(pymupdf.Rect(x0, y0, x1, y1).intersects(tb) for tb in table_boxes):
                continue  # already captured as a table
            cleaned = _clean(text, vocab)
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
