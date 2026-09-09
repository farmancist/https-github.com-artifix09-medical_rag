r"""BM25 encoded as Qdrant sparse vectors.

Design note — why not `rank_bm25` in-process:
    A local BM25 index would be a pickle that has to be rebuilt wherever the
    ingest runs and then shipped to every other machine, and the copies would
    silently drift apart.
    Instead we store BM25 term weights *in Qdrant* as sparse vectors and let the
    collection's `Modifier.IDF` supply the IDF factor at query time. One index,
    one source of truth, both machines agree.

Split of the BM25 formula:
    score = IDF(t) * [ tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl/avgdl)) ]
            \_____/   \_________________________________________________/
            Qdrant                     this module (document side)

Query vectors carry weight 1.0 per term, so the dot product reproduces BM25.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)

K1 = 1.5
B = 0.75

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-']*", re.IGNORECASE)

# Deliberately small: aggressive stopword lists hurt medical queries
# ("type 1", "no reflow", "in vitro" all carry meaning).
#
# Note "a" and "i" are absent on purpose -- see the single-character rule in
# `tokenize`. Their commonness is handled by IDF, not by a stoplist.
_STOPWORDS = frozenset(
    """
    an the and or but if then than that this these those of in on at to for from by with
    as is are was were be been being do does did doing have has had having you he she it we
    they them his her its their our your me my what which who whom when where why how
    can could should would may might will shall about into over under again further once
    here there all any both each few more most other some such only own same so too very
    """.split()
)

# Light, rule-based normalisation. A full stemmer (Porter) over-stems medical
# terms ("dosing"/"dose" is fine, but "arteries"/"arterial" should stay apart),
# so we only fold the plural/gerund endings that are safe.
_SUFFIXES = ("iesis", "ies", "ses", "es", "s")


def normalize_token(token: str) -> str:
    t = token.lower().strip("-'")
    if len(t) <= 3 or any(ch.isdigit() for ch in t):
        return t
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("sses") or t.endswith("shes") or t.endswith("ches"):
        return t[:-2]
    if t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    """Tokenise for BM25.

    Single characters are KEPT. In general text they are noise, but in medical
    text they are frequently the whole point: "vitamin D", "vitamin A",
    "hepatitis B", "T cell", "type 2". Dropping them silently turns
    "vitamin D deficiency" into "vitamin deficiency" on both the document and
    the query side, which is a different clinical concept.

    Their commonness is not a problem: Qdrant applies IDF server-side, so a
    letter that appears in nearly every chunk earns almost no weight anyway.
    """
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        low = raw.lower()
        if low in _STOPWORDS:
            continue
        out.append(normalize_token(low))
        # "co-trimoxazole" is also indexed as "co" + "trimoxazole" so a query
        # using either surface form still hits.
        if "-" in low:
            for part in low.split("-"):
                if len(part) >= 3 and part not in _STOPWORDS:
                    out.append(normalize_token(part))
    return out


def token_id(token: str) -> int:
    """Stable 32-bit index for a term. Qdrant sparse indices are uint32.

    Collisions at 2^31 buckets over a medical vocabulary of ~10^5 terms are
    negligible (birthday probability < 1e-5) and would only add a little noise
    to the lexical channel, which the reranker then filters.
    """
    import hashlib

    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(), "big") >> 1


class SparseVector:
    __slots__ = ("indices", "values")

    def __init__(self, indices: list[int], values: list[float]) -> None:
        self.indices = indices
        self.values = values

    def __len__(self) -> int:
        return len(self.indices)

    def as_dict(self) -> dict[str, list]:
        return {"indices": self.indices, "values": self.values}


class BM25Encoder:
    """Produces document-side BM25 sparse vectors and query-side term vectors.

    `avgdl` is corpus-level state. It is persisted next to the processed data so
    an incremental ingest continues from the previous statistics instead of
    silently re-scaling every weight.
    """

    def __init__(self, *, avgdl: float = 0.0, n_docs: int = 0, k1: float = K1, b: float = B):
        self.avgdl = avgdl
        self.n_docs = n_docs
        self.k1 = k1
        self.b = b

    # -- corpus stats ---------------------------------------------------

    def fit(self, texts: Sequence[str]) -> "BM25Encoder":
        lengths = [len(tokenize(t)) for t in texts]
        total = sum(lengths)
        self.n_docs = len(lengths)
        self.avgdl = (total / self.n_docs) if self.n_docs else 0.0
        logger.info("BM25 stats: n_docs=%d avgdl=%.1f", self.n_docs, self.avgdl)
        return self

    def update(self, texts: Sequence[str]) -> "BM25Encoder":
        """Fold a new batch into the running average (incremental ingest)."""
        lengths = [len(tokenize(t)) for t in texts]
        if not lengths:
            return self
        prev_total = self.avgdl * self.n_docs
        self.n_docs += len(lengths)
        self.avgdl = (prev_total + sum(lengths)) / self.n_docs
        return self

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"avgdl": self.avgdl, "n_docs": self.n_docs, "k1": self.k1, "b": self.b}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BM25Encoder":
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            avgdl=data.get("avgdl", 0.0),
            n_docs=data.get("n_docs", 0),
            k1=data.get("k1", K1),
            b=data.get("b", B),
        )

    # -- encoding -------------------------------------------------------

    def encode_document(self, text: str) -> SparseVector:
        tokens = tokenize(text)
        if not tokens:
            return SparseVector([], [])
        counts = Counter(tokens)
        dl = len(tokens)
        avgdl = self.avgdl or float(dl)
        norm = self.k1 * (1.0 - self.b + self.b * dl / avgdl)

        indices: list[int] = []
        values: list[float] = []
        for term, tf in counts.items():
            weight = tf * (self.k1 + 1.0) / (tf + norm)
            indices.append(token_id(term))
            values.append(float(weight))
        return SparseVector(indices, values)

    def encode_documents(self, texts: Iterable[str]) -> list[SparseVector]:
        return [self.encode_document(t) for t in texts]

    def encode_query(self, query: str) -> SparseVector:
        """Query side is presence-weighted; Qdrant applies IDF."""
        tokens = tokenize(query)
        if not tokens:
            return SparseVector([], [])
        counts = Counter(tokens)
        indices: list[int] = []
        values: list[float] = []
        for term, tf in counts.items():
            indices.append(token_id(term))
            # Repeating a term in a query is a weak signal; damp it.
            values.append(1.0 + math.log(tf))
        return SparseVector(indices, values)
