"""Chunk-boundary rules for hierarchy-tagged OHCM text."""
from __future__ import annotations
from typing import Callable, Sequence

OHCM_CHUNKER_VERSION = "ohcm-structural-1.0.0"

KEY_FIELDS = ("chapter_title", "topic_title", "section")

def group_key(meta: dict | None) -> tuple:
    if not meta: return ("default",)
    return tuple(meta.get(k) for k in KEY_FIELDS)

def chunk_header(meta: dict | None) -> str:
    if not meta: return ""
    return " > ".join(p for p in (meta.get("chapter_title"), meta.get("topic_title"), meta.get("section")) if p)
