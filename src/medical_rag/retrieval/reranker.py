"""Cross-encoder reranking — mandatory in this pipeline.

In the book RAG the reranker was optional. Here it is not, for two reasons:

  1. Bi-encoder retrieval on medical text returns *topically* correct but
     *clinically* wrong passages with high similarity — the paediatric dose of a
     drug and the adult dose look nearly identical to a bi-encoder. Only a
     cross-encoder, which reads query and passage jointly, separates them.
  2. The confidence gate downstream (>= 0.8) needs a calibrated relevance
     signal. RRF scores are rank-derived and have no absolute meaning; the
     cross-encoder logit does.

MedCPT's cross-encoder is domain-trained on PubMed query-article pairs, which is
why it is the default. Any `cross-encoder/*` model works as a drop-in via
`RERANKER_MODEL`.
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

from ..config import Settings, get_settings
from ..device import DeviceInfo, device_info_for, get_device
from ..models import RetrievedChunk

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class CrossEncoderReranker:
    def __init__(self, settings: Settings | None = None, *, model_name: str | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.reranker_model
        self.device_info: DeviceInfo = (
            get_device() if self.settings.device == "auto" else device_info_for(self.settings.device)
        )
        self.device = self.device_info.backend  # "cuda" | "dml" | "cpu" -- kept for logging/comparisons
        self._model = None
        self._max_length = 512

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info(
                "Loading reranker %s | backend=%s torch_device=%s provider=%s",
                self.model_name,
                self.device_info.st_backend,
                self.device_info.torch_device,
                self.device_info.provider,
            )
            onnx_kwargs = (
                {"backend": "onnx", "model_kwargs": {"provider": self.device_info.provider}}
                if self.device_info.st_backend == "onnx"
                else {}
            )
            self._model = CrossEncoder(
                self.model_name,
                device=self.device_info.torch_device,
                max_length=self._max_length,
                **onnx_kwargs,
            )
        return self._model

    def unload(self) -> None:
        self._model = None
        try:
            import torch

            if self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    # -- public ---------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Score candidates against the *original* query and return the top_k.

        The original query is used deliberately, not an expanded variant:
        expansion exists to widen recall, but precision must be judged against
        what the user actually asked.
        """
        if not candidates:
            return []

        top_k = top_k or self.settings.final_top_k
        pairs = [(query, c.chunk.text) for c in candidates]

        try:
            raw_scores = self.model.predict(
                pairs,
                batch_size=self.settings.reranker_batch_size,
                show_progress_bar=False,
            )
        except Exception:
            # A reranker failure must not silently degrade a medical answer to
            # un-reranked results: surface it, and let the caller decide.
            logger.exception("Reranker failed for query %r", query)
            raise

        for candidate, score in zip(candidates, raw_scores):
            value = float(score)
            # MedCPT and ms-marco cross-encoders emit unbounded logits; squash
            # to [0,1] so the confidence gate has a stable scale.
            candidate.rerank_score = value if 0.0 <= value <= 1.0 else _sigmoid(value)

        ranked = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)
        return ranked[:top_k]


_reranker: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker
