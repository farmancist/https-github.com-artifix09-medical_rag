"""Typed objects that flow through the pipeline.

Keeping these in one place means the ingest side and the serving side agree on
the exact payload shape stored in Qdrant, even when they are different machines.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------


class PageBlock(BaseModel):
    """One contiguous run of text pulled off a single PDF page."""

    text: str
    page: int
    block_type: Literal["text", "table", "dose_box"] = "text"
    section: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """A whole PDF after parsing, before chunking."""

    doc_id: str
    source_path: str
    title: str
    n_pages: int
    blocks: list[PageBlock] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


class Chunk(BaseModel):
    """A semantic chunk — the unit that gets embedded and stored."""

    chunk_id: str
    doc_id: str
    text: str
    # Citation anchors: medical answers must point at a page.
    page_start: int
    page_end: int
    section: str | None = None
    title: str = ""
    source_path: str = ""
    chunk_index: int = 0
    n_tokens: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def make_id(doc_id: str, index: int, text: str) -> str:
        """Deterministic id so re-ingesting the same PDF overwrites, not duplicates."""
        digest = hashlib.sha1(f"{doc_id}:{index}:{text}".encode("utf-8")).hexdigest()
        return digest

    def page_label(self) -> str:
        printed_start = self.metadata.get("printed_page_start") or self.metadata.get("printed_page")
        printed_end = self.metadata.get("printed_page_end") or printed_start
        if printed_start:
            return f"p.{printed_start}" if printed_start == printed_end else f"pp.{printed_start}-{printed_end}"
        return f"p.{self.page_start}" if self.page_start == self.page_end else f"pp.{self.page_start}-{self.page_end}"

    def citation_label(self) -> str:
        return f"{self.title or self.doc_id}, {self.page_label()}"


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


class RetrievedChunk(BaseModel):
    """A chunk coming back from search, with every score we accumulated."""

    chunk: Chunk
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    # Which query variant(s) surfaced this chunk — useful for debugging expansion.
    matched_queries: list[str] = Field(default_factory=list)

    @property
    def final_score(self) -> float:
        if self.rerank_score is not None:
            return self.rerank_score
        if self.rrf_score is not None:
            return self.rrf_score
        return self.dense_score or 0.0


class ExpandedQuery(BaseModel):
    original: str
    variants: list[str] = Field(default_factory=list)
    lexicon_hits: list[tuple[str, str]] = Field(default_factory=list)
    method: str = "none"

    @property
    def all_queries(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for q in [self.original, *self.variants]:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(q.strip())
        return out


# --------------------------------------------------------------------------
# Generation / guardrails
# --------------------------------------------------------------------------


class GuardrailVerdict(BaseModel):
    passed: bool
    reason: str = ""
    code: str = "ok"
    severity: Literal["info", "warn", "block"] = "info"


class ConfidenceReport(BaseModel):
    score: float
    threshold: float
    passed: bool
    components: dict[str, float] = Field(default_factory=dict)
    explanation: str = ""


class Citation(BaseModel):
    marker: str  # e.g. "[1]"
    chunk_id: str
    title: str
    page_start: int
    page_end: int
    printed_page: str | None = None
    source_path: str
    snippet: str = ""


class RAGAnswer(BaseModel):
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: ConfidenceReport | None = None
    guardrails: list[GuardrailVerdict] = Field(default_factory=list)
    refused: bool = False
    disclaimer: str = ""
    expanded_query: ExpandedQuery | None = None
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
