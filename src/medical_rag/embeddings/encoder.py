"""PubMedBERT dense encoder.

One class, two machines: `DEVICE=auto` resolves to `cuda` on your desktop and
`cpu` on the laptop, and the batch size is scaled accordingly. Nothing else in
the codebase touches torch directly.

The model is loaded lazily so that importing the package (e.g. to run the API's
health check, or the tests) does not pull ~450 MB of weights.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)


class DenseEncoder:
    """Wraps a sentence-transformers bi-encoder with L2-normalised output.

    Normalising at encode time means cosine similarity == dot product, which is
    what the Qdrant collection is configured for.
    """

    def __init__(self, settings: Settings | None = None, *, model_name: str | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.embedding_model
        self.device = self.settings.resolve_device()
        self._model = None

    # -- lifecycle ------------------------------------------------------

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is not installed, so no text can be embedded.\n"
                    "Run `python setup_env.py` to set the machine up, or install manually:\n"
                    "  pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
                    "  pip install sentence-transformers\n"
                    "To inspect PDF parsing without any model, use:\n"
                    "  python scripts/ingest.py --parse-only"
                ) from exc

            logger.info("Loading embedding model %s on %s", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.max_seq_length = self.settings.embedding_max_seq_length
        return self._model

    @property
    def dim(self) -> int:
        """Real dimension from the loaded model; falls back to config if not loaded."""
        if self._model is None:
            return self.settings.embedding_dim
        return int(self._model.get_sentence_embedding_dimension())

    def unload(self) -> None:
        self._model = None
        try:
            import torch

            if self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    # -- encoding -------------------------------------------------------

    def _batch_size(self, override: int | None) -> int:
        if override:
            return override
        base = self.settings.embedding_batch_size
        # The desktop can take much bigger batches than the laptop.
        return base * 4 if self.device == "cuda" else base

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode documents. Returns (n, dim) float32, L2-normalised."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self.model.encode(
            list(texts),
            batch_size=self._batch_size(batch_size),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query. Kept separate so an asymmetric model can be
        swapped in later without touching call sites."""
        return self.encode([query])[0]

    def encode_iter(
        self, texts: Iterable[str], *, chunk: int = 2048
    ) -> Iterable[np.ndarray]:
        """Stream large corpora so 15k+ chunks never sit in RAM twice."""
        buf: list[str] = []
        for t in texts:
            buf.append(t)
            if len(buf) >= chunk:
                yield self.encode(buf, show_progress=True)
                buf = []
        if buf:
            yield self.encode(buf, show_progress=True)


_encoder: DenseEncoder | None = None


def get_encoder() -> DenseEncoder:
    """Process-wide singleton — loading PubMedBERT twice is pure waste."""
    global _encoder
    if _encoder is None:
        _encoder = DenseEncoder()
    return _encoder
