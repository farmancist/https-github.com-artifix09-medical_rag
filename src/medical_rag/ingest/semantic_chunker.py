"""Semantic chunker — breakpoint percentile 85, overlap 20%.

How it works:
  1. Split each document into sentences (medical-aware: we do not break on
     "Dr.", "mg.", "e.g.", "vs.", or decimal numbers like "0.5 mg").
  2. Embed every sentence with the *same* PubMedBERT encoder used for retrieval.
  3. Compute cosine distance between consecutive sentences. A distance above the
     85th percentile is a topic shift -> cut there.
  4. Enforce min/max token budgets, then add 20% overlap between neighbours.

Why 85 and not the book-RAG's 95: clinical text shifts topic far more often per
page (indications -> dosage -> contraindications -> interactions). A 95th
percentile threshold cuts too rarely and glues distinct clinical facts into one
chunk, which is exactly the failure mode that produces confidently wrong dosage
answers. 20% overlap (vs 15%) keeps a dosage line attached to the drug name that
qualifies it.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

import numpy as np

from ..config import Settings, get_settings
from ..embeddings.encoder import DenseEncoder, get_encoder
from ..models import Chunk, PageBlock, ParsedDocument

logger = logging.getLogger(__name__)
SEMANTIC_CHUNKER_VERSION = "semantic-1.0.0"

# Abbreviations that are almost never sentence-final, so a following capital
# letter does NOT mean a new sentence ("Dr. Smith", "e.g. Ibuprofen", "et al.").
#
# Units (mg, kg, mL) and dosing abbreviations (b.i.d., p.o.) are deliberately
# absent: they frequently DO end a sentence ("...metformin 500 mg. The patient
# improved."), and the candidate rule below already requires a following capital,
# which rules out the mid-sentence "500 mg. of drug X" case.
_ABBREVIATIONS = frozenset(
    """
    dr mr mrs ms prof sr jr st fig figs no nos vs v approx est etc e.g i.e cf al
    inc ltd co corp dept univ ed eds vol chap pp ca viz ibid
    """.split()
)

# A split candidate: terminal punctuation, optional closing quotes/brackets,
# whitespace, then something that starts a sentence.
#   group 1 = the punctuation, group 2 = the closers
# Python's `re` only supports fixed-width lookbehind, so the abbreviation guard
# cannot live in the pattern -- candidates are filtered in `_ends_sentence`.
_SPLIT_CANDIDATE_RE = re.compile(r"([.!?])([\"')\]]*)\s+(?=[A-Z(\[\"'])")
# The token immediately before a candidate's punctuation.
_PRECEDING_TOKEN_RE = re.compile(r"([A-Za-z][A-Za-z.]*)$")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """Cheap word-ish token count. Good enough for budgeting; the encoder does
    the real tokenisation and truncates at `embedding_max_seq_length`."""
    return len(_TOKEN_RE.findall(text))


def _ends_sentence(text: str, dot_index: int) -> bool:
    """Does the '.' at `dot_index` actually terminate a sentence?"""
    match = _PRECEDING_TOKEN_RE.search(text[:dot_index])
    if not match:
        return True
    token = match.group(1)
    # A lone capital is an initial: "J. Smith", "A. Patel".
    if len(token) == 1 and token.isupper():
        return False
    return token.lower().rstrip(".") not in _ABBREVIATIONS


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        start = 0
        for match in _SPLIT_CANDIDATE_RE.finditer(paragraph):
            if match.group(1) == "." and not _ends_sentence(paragraph, match.start(1)):
                continue
            sentence = paragraph[start : match.end(2)].strip()
            if sentence:
                parts.append(sentence)
            start = match.end()
        tail = paragraph[start:].strip()
        if tail:
            parts.append(tail)
    return parts


class _Unit:
    """A sentence plus the page/section it came from."""

    __slots__ = ("text", "page", "section", "is_table", "is_dose_box", "meta")

    def __init__(
        self, text: str, page: int, section: str | None, is_table: bool,
        meta: dict | None = None, is_dose_box: bool = False,
    ) -> None:
        self.text = text
        self.page = page
        self.section = section
        self.is_table = is_table
        self.meta = dict(meta or {})
        self.is_dose_box = is_dose_box or bool(self.meta.get("dose_box"))


class SemanticChunker:
    def __init__(
        self,
        settings: Settings | None = None,
        encoder: DenseEncoder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.encoder = encoder or get_encoder()
        self.percentile = self.settings.chunk_breakpoint_percentile
        self.overlap_ratio = self.settings.chunk_overlap_ratio
        self.min_tokens = self.settings.chunk_min_tokens
        self.max_tokens = self.settings.chunk_max_tokens

    # -- public ---------------------------------------------------------

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        units = self._to_units(doc.blocks)
        if not units:
            return []

        title = doc.title.casefold()
        is_bnf = (
            "british national formulary" in title
            or bool(re.search(r"\bbnf\b", title))
        )
        bnf_profile = doc.metadata.get("parser_profile") == "bnf"
        katzung_profile = doc.metadata.get("parser_profile") == "katzung"
        if bnf_profile:
            from .bnf_chunking import structural_groups
            groups = structural_groups(units, self.max_tokens, count_tokens)
        elif katzung_profile:
            from .katzung_chunking import structural_groups
            groups = structural_groups(units, self.max_tokens, count_tokens)
        else:
            groups = self._structural_groups(units) if is_bnf else self._semantic_groups(units)
        groups = self._enforce_budgets(groups, preserve_meta=bnf_profile or katzung_profile)
        groups = self._apply_overlap(groups, preserve_meta=bnf_profile or katzung_profile)
        return self._materialise(doc, groups)

    def chunk_documents(self, docs: Sequence[ParsedDocument]) -> list[Chunk]:
        out: list[Chunk] = []
        for doc in docs:
            chunks = self.chunk_document(doc)
            logger.info("%s -> %d chunks", doc.title[:60], len(chunks))
            out.extend(chunks)
        return out

    # -- steps ----------------------------------------------------------

    def _to_units(self, blocks: list[PageBlock]) -> list[_Unit]:
        units: list[_Unit] = []
        for block in blocks:
            meta = dict(block.metadata)
            if block.block_type == "dose_box":
                units.append(_Unit(block.text, block.page, block.section, False, meta, True))
                continue
            if block.block_type == "table":
                # Markdown tables (BNF) are split row-wise; Katzung's tables are
                # plain text spans with no row structure to split on, so keep the
                # whole block atomic and let the chunker carry its heading.
                is_katzung_table = meta.get("kind") == "table"
                if is_katzung_table:
                    units.append(
                        _Unit(block.text, block.page, block.section, True, meta, False)
                    )
                    continue
                units.extend(self._table_units(block))
                continue
            for sent in split_sentences(block.text):
                units.append(_Unit(sent, block.page, block.section, False, meta))
        return units

    def _table_units(self, block: PageBlock) -> list[_Unit]:
        """Split an oversized table by rows, repeating the header on each part.

        Tables are otherwise atomic, because splitting mid-row destroys the
        drug-to-dose correspondence. But a whole table can be far larger than
        the model's context: measured on the WHO Essential Medicines List, 45%
        of chunks exceeded the 512-subword limit and the worst lost 94% of its
        text to silent truncation. A truncated dose table is worse than a split
        one -- the rows that fall off the end simply cease to exist for
        retrieval. Repeating the header keeps every part self-describing.
        """
        text = block.text
        if count_tokens(text) <= self.max_tokens:
            return [_Unit(text, block.page, block.section, True, block.metadata)]

        lines = text.splitlines()
        if len(lines) < 3:  # not a markdown table we can split on rows
            return [
                _Unit(part, block.page, block.section, True, block.metadata, bool(block.metadata.get("dose_box")))
                for part in self._split_by_words(text)
            ]

        header, separator, *rows = lines
        head_cost = count_tokens(header) + count_tokens(separator)
        units: list[_Unit] = []
        current: list[str] = []
        cost = 0

        def flush() -> None:
            if current:
                body = "\n".join([header, separator, *current])
                units.append(_Unit(body, block.page, block.section, True, block.metadata, bool(block.metadata.get("dose_box"))))

        for row in rows:
            row_cost = count_tokens(row)
            if current and head_cost + cost + row_cost > self.max_tokens:
                flush()
                current, cost = [], 0
            if head_cost + row_cost > self.max_tokens:
                # A single row that cannot fit even alone: split its text.
                flush()
                current, cost = [], 0
                for part in self._split_by_words(row):
                    units.append(
                        _Unit(
                            "\n".join([header, separator, part]),
                            block.page,
                            block.section,
                            True,
                            block.metadata,
                        )
                    )
                continue
            current.append(row)
            cost += row_cost
        flush()
        return units or [_Unit(text, block.page, block.section, True, block.metadata, bool(block.metadata.get("dose_box")))]

    def _fit_units(self, group: list[_Unit]) -> list[_Unit]:
        """Guarantee no single unit exceeds the budget.

        Tables are already handled at parse time; this catches the remaining
        case of one enormous "sentence" (an un-punctuated list, a wall of
        references) that would otherwise become an oversized chunk on its own.
        """
        out: list[_Unit] = []
        for unit in group:
            if count_tokens(unit.text) <= self.max_tokens:
                out.append(unit)
                continue
            out.extend(
                _Unit(part, unit.page, unit.section, unit.is_table, unit.meta, unit.is_dose_box)
                for part in self._split_by_words(unit.text)
            )
        return out

    def _split_by_words(self, text: str) -> list[str]:
        """Last-resort hard split so no unit can exceed the budget."""
        words = text.split()
        if not words:
            return [text]
        parts: list[str] = []
        current: list[str] = []
        for word in words:
            current.append(word)
            if count_tokens(" ".join(current)) >= self.max_tokens:
                parts.append(" ".join(current))
                current = []
        if current:
            parts.append(" ".join(current))
        return parts


    def _structural_groups(self, units: list[_Unit]) -> list[list[_Unit]]:
        """Chunk BNF by printed sections and tables without sentence embeddings.

        BNF monographs already expose strong boundaries in the PDF. Embedding
        every sentence on CPU made each forced rechunk take hours without
        improving those boundaries; section changes, table boundaries, and the
        token budget provide cheaper, deterministic cuts for this reference.
        """
        groups: list[list[_Unit]] = []
        current: list[_Unit] = []

        def flush() -> None:
            nonlocal current
            if current:
                groups.append(current)
                current = []

        for unit in units:
            if unit.is_table or unit.is_dose_box:
                flush()
                groups.append([unit])
                continue
            if current and current[-1].section != unit.section:
                flush()
            if current and self._tokens(current) + count_tokens(unit.text) > self.max_tokens:
                flush()
            current.append(unit)
        flush()
        return groups
    def _semantic_groups(self, units: list[_Unit]) -> list[list[_Unit]]:
        """Cut wherever consecutive-sentence distance exceeds the percentile."""
        if len(units) <= 1:
            return [units]

        vectors = self.encoder.encode([u.text for u in units], show_progress=False)
        # Vectors are L2-normalised, so dot product == cosine similarity.
        sims = np.sum(vectors[:-1] * vectors[1:], axis=1)
        distances = 1.0 - sims

        threshold = float(np.percentile(distances, self.percentile))

        groups: list[list[_Unit]] = []
        current: list[_Unit] = [units[0]]
        for i, unit in enumerate(units[1:]):
            prev = units[i]
            hard_break = (
                unit.is_table
                or prev.is_table
                or unit.is_dose_box
                or prev.is_dose_box
                # A new section heading is a real boundary regardless of distance.
                or (unit.section is not None and unit.section != prev.section)
            )
            if hard_break or distances[i] > threshold:
                groups.append(current)
                current = [unit]
            else:
                current.append(unit)
        if current:
            groups.append(current)
        return groups

    def _enforce_budgets(
        self, groups: list[list[_Unit]], *, preserve_meta: bool = False
    ) -> list[list[_Unit]]:
        """Split oversized groups, merge undersized ones into their neighbour."""
        sized: list[list[_Unit]] = []
        for group in groups:
            if self._tokens(group) <= self.max_tokens:
                sized.append(group)
                continue
            current: list[_Unit] = []
            for unit in self._fit_units(group):
                unit_tokens = count_tokens(unit.text)
                if current and self._tokens(current) + unit_tokens > self.max_tokens:
                    sized.append(current)
                    current = []
                current.append(unit)
            if current:
                sized.append(current)

        merged: list[list[_Unit]] = []
        for group in sized:
            too_small = self._tokens(group) < self.min_tokens
            fits = merged and self._tokens(merged[-1]) + self._tokens(group) <= self.max_tokens
            atomic = any(u.is_table or u.is_dose_box for u in group)
            atomic = atomic or bool(merged and any(u.is_table or u.is_dose_box for u in merged[-1]))
            # Merging across a section boundary just because the tail group is
            # small would glue two unrelated topics into one chunk (e.g. the
            # last, short "Overdose" paragraph of one drug fused onto the next
            # drug's opening chunk). A slightly-undersized chunk is a smaller
            # problem than a mislabelled one, especially now that every chunk
            # carries its own section anchor (see `_with_section_anchor`).
            same_section = bool(merged) and merged[-1][-1].section == group[0].section
            same_context = (
                bool(merged) and merged[-1][-1].meta == group[0].meta
                if preserve_meta else True
            )
            if too_small and fits and not atomic and same_section and same_context:
                merged[-1].extend(group)
            else:
                merged.append(group)
        return merged

    def _apply_overlap(
        self, groups: list[list[_Unit]], *, preserve_meta: bool = False
    ) -> list[list[_Unit]]:
        """Prepend the tail of the previous group (20% of this group's budget)."""
        if self.overlap_ratio <= 0 or len(groups) < 2:
            return groups

        out: list[list[_Unit]] = [groups[0]]
        for i in range(1, len(groups)):
            group = groups[i]
            prev = groups[i - 1]
            if (
                any(u.is_table or u.is_dose_box for u in group)
                or any(u.is_table or u.is_dose_box for u in prev)
                or prev[-1].section != group[0].section
                or (preserve_meta and prev[-1].meta != group[0].meta)
            ):
                out.append(group)
                continue

            # Overlap must never push a chunk past the budget. Previously a
            # single oversized sentence was prepended whole (the "and tail"
            # escape hatch), which is how chunks reached 897 tokens against a
            # 384 limit -- and anything past ~512 subwords is silently dropped
            # by the encoder, so the overlap was actively destroying content.
            group_tokens = self._tokens(group)
            room = self.max_tokens - group_tokens
            if room <= 0:
                out.append(group)
                continue
            budget = min(int(group_tokens * self.overlap_ratio), room)

            tail: list[_Unit] = []
            used = 0
            for unit in reversed(prev):
                if unit.is_table:
                    break
                cost = count_tokens(unit.text)
                if used + cost > budget:
                    break  # no overlap is better than an oversized chunk
                tail.insert(0, unit)
                used += cost
            out.append(tail + group)
        return out

    def _materialise(self, doc: ParsedDocument, groups: list[list[_Unit]]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for index, group in enumerate(groups):
            if not group:
                continue
            body = " ".join(u.text for u in group).strip()
            if not body:
                continue
            pages = [u.page for u in group]
            section = next((u.section for u in group if u.section), None)
            metadata = {
                "has_table": any(u.is_table for u in group),
                "has_dose_box": any(u.is_dose_box for u in group),
            }
            bnf_profile = doc.metadata.get("parser_profile") == "bnf"
            katzung_profile = doc.metadata.get("parser_profile") == "katzung"
            if bnf_profile:
                from .bnf_chunking import BNF_CHUNKER_VERSION, chunk_header

                context = dict(group[0].meta)
                header = chunk_header(context)
                text = f"{header}\n{body}" if header else body
                section = (
                    context.get("entity") or context.get("h3") or context.get("h2")
                    or context.get("h1") or section
                )
                printed_pages = list(dict.fromkeys(
                    str(u.meta["printed_page"])
                    for u in group if u.meta.get("printed_page") is not None
                ))
                metadata.update(context)
                metadata.update({
                    "pdf_page_start": min(pages),
                    "pdf_page_end": max(pages),
                    "printed_pages": printed_pages,
                    "source": doc.source_path,
                    "parser_profile": doc.metadata.get("parser_profile"),
                    "parser_version": doc.metadata.get("parser_version", "bnf-layout-1.0.0"),
                    "chunker_version": BNF_CHUNKER_VERSION,
                    "ingestion_schema_version": "1.0",
                })
                if printed_pages:
                    metadata["printed_page_start"] = printed_pages[0]
                    metadata["printed_page_end"] = printed_pages[-1]
                    if len(printed_pages) == 1:
                        metadata["printed_page"] = printed_pages[0]
            elif katzung_profile:
                from .katzung_chunking import KATZUNG_CHUNKER_VERSION, chunk_header

                context = dict(group[0].meta)
                header = chunk_header(context)
                text = f"{header}\n{body}" if header else body
                section = (
                    context.get("chapter_title") or context.get("section") or context.get("subsection") or section
                )
                metadata.update({
                    "pdf_page_start": min(pages),
                    "pdf_page_end": max(pages),
                    "source": doc.source_path,
                    "parser_profile": doc.metadata.get("parser_profile"),
                    "parser_version": doc.metadata.get("parser_version", "katzung-layout-1.0.0"),
                    "chunker_version": KATZUNG_CHUNKER_VERSION,
                    "ingestion_schema_version": "1.0",
                })
            else:
                text = self._with_section_anchor(body, section)
            chunks.append(
                Chunk(
                    chunk_id=Chunk.make_id(doc.doc_id, index, text),
                    doc_id=doc.doc_id,
                    text=text,
                    page_start=min(pages),
                    page_end=max(pages),
                    section=section,
                    title=doc.title,
                    source_path=doc.source_path,
                    chunk_index=index,
                    n_tokens=count_tokens(text),
                    metadata=metadata,
                )
            )
        return chunks

    @staticmethod
    def _with_section_anchor(body: str, section: str | None) -> str:
        """Prepend the section/topic label so every chunk is self-describing.

        A long section (a multi-page drug monograph) gets split across several
        chunks by `_enforce_budgets`; only the first naturally contains the
        heading text, since it is the unit that triggered the hard break. Every
        later sub-chunk would otherwise carry no drug/topic name anywhere in
        its text -- and only `Chunk.text` is ever embedded or BM25-indexed
        (`section` is stored as metadata alongside it, never inside it), so a
        query naming the drug would not lexically or semantically match a
        sub-chunk holding e.g. just "Adults: 10mg once daily". Skipped when the
        body already opens with the label, so the natural first sub-chunk is
        not doubled up.

        Not re-checked against the token budget: a label is a handful of
        tokens, and `chunk_max_tokens` is already set with headroom under
        `embedding_max_seq_length` for exactly this kind of small overshoot
        (see config.py).
        """
        if not section:
            return body
        label = section.strip()
        if not label or body.lower().startswith(label.lower()[: min(len(label), 30)]):
            return body
        return f"{label}\n{body}"

    @staticmethod
    def _tokens(group: list[_Unit]) -> int:
        return sum(count_tokens(u.text) for u in group)
