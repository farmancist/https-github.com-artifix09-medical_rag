"""Chunk-boundary rules for hierarchy-tagged BNF text.

Kept free of project imports so they can be unit-tested alone; SemanticChunker
delegates to these.  A *unit* is anything with .text, .is_table and .meta (dict).
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

BNF_CHUNKER_VERSION = "bnf-structural-1.1.0"

# One chunk never spans a change in any of these.
KEY_FIELDS = (
    "kind", "h1", "h2", "h3", "entity", "preparation", "field", "indication",
    "route", "freq", "dose_box", "table_title", "interaction_source", "interaction_target",
    "interaction_id",
)


def group_key(meta: dict | None, fallback: object = None) -> tuple:
    if not meta:
        return ("section", fallback)
    return tuple(meta.get(k) for k in KEY_FIELDS)


def chunk_header(meta: dict | None) -> str:
    """Compact context line prepended to chunk text (it is what gets embedded).

    Drug and preparation first, because those are the entities a query names;
    then the field and dose qualifiers. The classification
    ('ANTIVIRALS > NUCLEOSIDE ANALOGUES') is included once, since it carries
    class vocabulary a query may use.
    """
    if not meta:
        return ""
    lead = " > ".join(
        p for p in (
            meta.get("h1"), meta.get("subchapter"), meta.get("h2"), meta.get("h3"),
            meta.get("entity"), meta.get("preparation"), meta.get("classification"),
            meta.get("table_title"),
        ) if p
    )
    tail = " | ".join(
        p for p in (
            meta.get("field"), meta.get("indication"), meta.get("route"), meta.get("freq"),
            meta.get("interaction_source"), meta.get("interaction_target"),
        )
        if p
    )
    return " | ".join(p for p in (lead, tail) if p)


def split_list(text: str, max_tokens: int, count: Callable[[str], int]) -> list[str]:
    """Split a '·'-separated list (side-effects, cautions) at item boundaries,
    never mid-item. Falls back to [text] when it is not a list."""
    items = [i.strip() for i in re.split(r"\s*\u00b7\s*", text) if i.strip()]
    if len(items) < 5:
        return [text]
    parts, cur, cost = [], [], 0
    for it in items:
        c = count(it) + 1
        if cur and cost + c > max_tokens:
            parts.append(" \u00b7 ".join(cur))
            cur, cost = [], 0
        cur.append(it)
        cost += c
    if cur:
        parts.append(" \u00b7 ".join(cur))
    return parts


def structural_groups(units: Sequence, max_tokens: int, count: Callable[[str], int]) -> list[list]:
    """Cut on any change of (kind, drug, field, indication, route, freq),
    keep tables atomic, and respect the token budget."""
    groups: list[list] = []
    cur: list = []
    cost = 0
    key = lambda u: group_key(getattr(u, "meta", None), getattr(u, "section", None))  # noqa: E731

    def flush():
        nonlocal cur, cost
        if cur:
            groups.append(cur)
        cur, cost = [], 0

    for u in units:
        if u.is_table:
            flush()
            groups.append([u])
            continue
        c = count(u.text)
        if cur and key(cur[-1]) != key(u):
            flush()
        if cur and cost + c > max_tokens:
            flush()
        cur.append(u)
        cost += c
    flush()
    return groups
