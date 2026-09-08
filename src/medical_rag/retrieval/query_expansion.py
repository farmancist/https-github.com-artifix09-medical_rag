"""Query expansion: layman terms -> medical terms.

This is the component the book-RAG does not have, and it is the one that makes
or breaks recall here. The corpus is written in clinical vocabulary
("myocardial infarction", "dyspnea", "contraindication"); users type
"heart attack", "can't breathe", "when not to use". Dense retrieval alone
bridges some of that gap, but BM25 bridges none of it — the exact clinical token
is simply absent from the query.

Two tiers, run in order:
  1. lexicon  — deterministic, zero-latency, auditable dictionary lookup.
  2. llm      — Groq (small, fast model) for phrasings the lexicon misses.

`hybrid` (the default) runs both and merges. If Groq is unreachable the LLM tier
degrades silently to the lexicon result; expansion must never fail a request.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

from ..config import Settings, get_settings
from ..generation.llm import GroqLLM, get_llm
from ..models import ExpandedQuery

logger = logging.getLogger(__name__)

LEXICON_PATH = Path(__file__).resolve().parents[1] / "resources" / "medical_lexicon.json"

_EXPANSION_SYSTEM = (
    "You are a medical terminology normaliser for a clinical document retrieval system. "
    "Rewrite a lay-person's question into the vocabulary a medical textbook or clinical "
    "guideline would use.\n"
    "Rules:\n"
    "- Return ONLY a JSON array of strings, no prose.\n"
    "- Each string is a complete, standalone search query.\n"
    "- Replace lay terms with their clinical equivalents (e.g. 'heart attack' -> "
    "'myocardial infarction'). Keep drug names, numbers, doses and units exactly as given.\n"
    "- Do NOT answer the question, diagnose, or add facts not implied by the question.\n"
    "- Do NOT invent conditions the user did not mention.\n"
    "- If the question is already in clinical language, return an empty array."
)


@lru_cache(maxsize=1)
def load_lexicon(path: str | None = None) -> dict[str, list[str]]:
    p = Path(path) if path else LEXICON_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    return {k.lower(): v for k, v in data.items() if not k.startswith("_")}


@lru_cache(maxsize=1)
def _compiled_lexicon() -> list[tuple[re.Pattern[str], str, list[str]]]:
    """Longest-first so 'high blood pressure' wins over 'blood'."""
    lexicon = load_lexicon()
    entries = sorted(lexicon.items(), key=lambda kv: len(kv[0]), reverse=True)
    return [
        (re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE), term, synonyms)
        for term, synonyms in entries
    ]


class QueryExpander:
    def __init__(self, settings: Settings | None = None, llm: GroqLLM | None = None) -> None:
        self.settings = settings or get_settings()
        self.mode = self.settings.query_expansion_mode
        self.max_variants = self.settings.query_expansion_max_variants
        self._llm = llm

    @property
    def llm(self) -> GroqLLM:
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    # -- public ---------------------------------------------------------

    def expand(self, query: str) -> ExpandedQuery:
        query = query.strip()
        if not self.settings.query_expansion_enabled or not query:
            return ExpandedQuery(original=query, method="disabled")

        variants: list[str] = []
        hits: list[tuple[str, str]] = []
        methods: list[str] = []

        if self.mode in ("lexicon", "hybrid"):
            lex_variants, hits = self._expand_lexicon(query)
            if lex_variants:
                methods.append("lexicon")
            variants.extend(lex_variants)

        if self.mode in ("llm", "hybrid"):
            llm_variants = self._expand_llm(query)
            if llm_variants:
                methods.append("llm")
            variants.extend(llm_variants)

        return ExpandedQuery(
            original=query,
            variants=self._dedupe(query, variants)[: self.max_variants],
            lexicon_hits=hits,
            method="+".join(methods) or "none",
        )

    # -- tiers ----------------------------------------------------------

    def _expand_lexicon(self, query: str) -> tuple[list[str], list[tuple[str, str]]]:
        """Substitute matched lay terms with their clinical synonyms.

        We emit one variant per *synonym rank* rather than one per term, so a
        query with two lay terms still produces a small, coherent set of
        rewrites instead of a combinatorial explosion.
        """
        matches: list[tuple[re.Pattern[str], str, list[str]]] = []
        for pattern, term, synonyms in _compiled_lexicon():
            if pattern.search(query):
                matches.append((pattern, term, synonyms))
        if not matches:
            return [], []

        # Drop matches contained inside a longer match already taken.
        kept: list[tuple[re.Pattern[str], str, list[str]]] = []
        for pattern, term, synonyms in matches:
            if any(term in other_term and term != other_term for _, other_term, _ in kept):
                continue
            kept.append((pattern, term, synonyms))

        depth = max(len(s) for _, _, s in kept)
        variants: list[str] = []
        for rank in range(min(depth, self.max_variants)):
            rewritten = query
            for pattern, _term, synonyms in kept:
                replacement = synonyms[min(rank, len(synonyms) - 1)]
                rewritten = pattern.sub(replacement, rewritten)
            variants.append(rewritten)

        hits = [(term, synonyms[0]) for _, term, synonyms in kept]
        return variants, hits

    def _expand_llm(self, query: str) -> list[str]:
        payload = self.llm.chat_json(
            [
                {"role": "system", "content": _EXPANSION_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\n\n"
                        f"Return at most {self.max_variants} clinical rewrites as a JSON array."
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=256,
            model=self.settings.groq_expansion_model,
            default=[],
        )
        if isinstance(payload, dict):  # model wrapped it, e.g. {"queries": [...]}
            payload = next((v for v in payload.values() if isinstance(v, list)), [])
        if not isinstance(payload, list):
            return []
        return [str(v).strip() for v in payload if isinstance(v, (str, int, float)) and str(v).strip()]

    @staticmethod
    def _dedupe(original: str, variants: list[str]) -> list[str]:
        seen = {original.strip().lower()}
        out: list[str] = []
        for v in variants:
            key = v.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(v.strip())
        return out
