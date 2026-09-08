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

    __slots__ = ("text", "page", "section", "is_table")

    def __init__(self, text: str, page: int, section: str | None, is_table: bool) -> None:
        self.text = text
        self.page = page
        self.section = section
        self.is_table = is_table


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

        groups = self._semantic_groups(units)
        groups = self._enforce_budgets(groups)
        groups = self._apply_overlap(groups)
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
            if block.block_type == "table":
                # A table is atomic — splitting it mid-row destroys its meaning.
                units.append(_Unit(block.text, block.page, block.section, True))
                continue
            for sent in split_sentences(block.text):
                units.append(_Unit(sent, block.page, block.section, False))
        return units

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

    def _enforce_budgets(self, groups: list[list[_Unit]]) -> list[list[_Unit]]:
        """Split oversized groups, merge undersized ones into their neighbour."""
        sized: list[list[_Unit]] = []
        for group in groups:
            if self._tokens(group) <= self.max_tokens:
                sized.append(group)
                continue
            current: list[_Unit] = []
            for unit in group:
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
            atomic = any(u.is_table for u in group)
            if too_small and fits and not atomic:
                merged[-1].extend(group)
            else:
                merged.append(group)
        return merged

    def _apply_overlap(self, groups: list[list[_Unit]]) -> list[list[_Unit]]:
        """Prepend the tail of the previous group (20% of this group's budget)."""
        if self.overlap_ratio <= 0 or len(groups) < 2:
            return groups

        out: list[list[_Unit]] = [groups[0]]
        for i in range(1, len(groups)):
            group = groups[i]
            prev = groups[i - 1]
            if any(u.is_table for u in group):
                out.append(group)
                continue

            budget = int(self._tokens(group) * self.overlap_ratio)
            tail: list[_Unit] = []
            used = 0
            for unit in reversed(prev):
                if unit.is_table:
                    break
                cost = count_tokens(unit.text)
                if used + cost > budget and tail:
                    break
                tail.insert(0, unit)
                used += cost
                if used >= budget:
                    break
            out.append(tail + group)
        return out

    def _materialise(self, doc: ParsedDocument, groups: list[list[_Unit]]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for index, group in enumerate(groups):
            if not group:
                continue
            text = " ".join(u.text for u in group).strip()
            if not text:
                continue
            pages = [u.page for u in group]
            section = next((u.section for u in group if u.section), None)
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
                    metadata={"has_table": any(u.is_table for u in group)},
                )
            )
        return chunks

    @staticmethod
    def _tokens(group: list[_Unit]) -> int:
        return sum(count_tokens(u.text) for u in group)
