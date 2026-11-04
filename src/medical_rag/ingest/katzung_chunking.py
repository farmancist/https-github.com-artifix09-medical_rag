"""Chunk-boundary rules for Katzung pharmacology text.

Kept free of project imports so it can be unit-tested alone; SemanticChunker
delegates to these.  A *unit* is anything with .text, .is_table, .section and
.meta (dict).
"""

from __future__ import annotations

from typing import Callable, Sequence

KATZUNG_CHUNKER_VERSION = "katzung-structural-1.1.0"

# One chunk never spans a change in any of these.
KEY_FIELDS = ("chapter_title", "section", "subsection", "table_title")


def group_key(meta: dict | None, fallback: object = None) -> tuple:
    if not meta:
        return ("section", fallback)
    return tuple(meta.get(k) for k in KEY_FIELDS)


def chunk_header(meta: dict | None) -> str:
    """Context line prepended to chunk text (it is what gets embedded).

    Chapter > section > subsection first, because those are the boundaries a
    query names; the table heading last, since it only qualifies what follows.
    """
    if not meta:
        return ""
    return " > ".join(
        p for p in (
            meta.get("chapter_title"),
            meta.get("section"),
            meta.get("subsection"),
            meta.get("table_title"),
        ) if p
    )


def structural_groups(units: Sequence, max_tokens: int, count: Callable[[str], int]) -> list[list]:
    """Cut on any change of (chapter, section, subsection, table),
    keep tables atomic, and respect the token budget.

    Katzung is a board-review book: its headings *are* the topic boundaries, so
    a structural cut is a semantic cut, and embedding every sentence on CPU
    would not improve them.
    """
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