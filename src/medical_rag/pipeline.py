"""The Medical RAG pipeline, end to end.

    query
      -> INPUT guardrail        (emergency / injection / personal-directive)
      -> query expansion        (layman -> medical)
      -> hybrid search          (BM25 + dense, RRF over variants x channels)
      -> cross-encoder rerank   (mandatory)
      -> EVIDENCE guardrail     (confidence >= 0.8, else refuse)
      -> Groq generation        (context-only prompt)
      -> OUTPUT guardrail       (citations resolve, no unsupported numbers)
      -> disclaimer             (mandatory)

Everything is lazily constructed, so importing this module is cheap and the API
can report health before any model weight is touched.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator, Sequence

from qdrant_client import models as qmodels

from .config import Settings, get_settings
from .generation.guardrails import ConfidenceScorer, InputGuardrail, OutputGuardrail
from .generation.llm import GroqLLM, LLMError, get_llm
from .generation.prompts import DISCLAIMER, REFUSAL_INSUFFICIENT, build_messages
from .models import ConfidenceReport, GuardrailVerdict, RAGAnswer, RetrievedChunk
from .retrieval.hybrid import HybridRetriever
from .retrieval.query_expansion import QueryExpander
from .retrieval.reranker import CrossEncoderReranker, get_reranker
from .vectorstore.qdrant_store import QdrantStore, build_filter

logger = logging.getLogger(__name__)


class MedicalRAGPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        llm: GroqLLM | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.input_guard = InputGuardrail()
        self.expander = QueryExpander(self.settings)
        self._retriever = retriever
        self._reranker = reranker
        self._llm = llm
        self.confidence = ConfidenceScorer(self.settings)
        self.output_guard = OutputGuardrail(self.settings)

    # -- lazy components ------------------------------------------------

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(self.settings)
        return self._retriever

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = get_reranker()
        return self._reranker

    @property
    def llm(self) -> GroqLLM:
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def warmup(self) -> None:
        """Force model loads up front so the first real query is not slow."""
        self.retriever.encoder.encode(["warmup"])
        self.reranker.model.predict([("warmup", "warmup")])
        logger.info("Warmup complete")

    # -- main entry point -----------------------------------------------

    def answer(
        self,
        query: str,
        *,
        doc_ids: Sequence[str] | None = None,
        sections: Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> RAGAnswer:
        timings: dict[str, float] = {}
        query = (query or "").strip()

        # ---- 1. input guardrail ----
        with self._timed(timings, "input_guard"):
            input_verdicts = self.input_guard.check(query)
        blocking = [v for v in input_verdicts if v.severity == "block"]
        if blocking:
            return self._blocked(query, input_verdicts, blocking[0], timings)

        # ---- 2. query expansion ----
        with self._timed(timings, "expansion"):
            expanded = self.expander.expand(query)
        if expanded.variants:
            logger.info("Expanded via %s -> %s", expanded.method, expanded.variants)

        # ---- 3. hybrid retrieval ----
        query_filter: qmodels.Filter | None = build_filter(doc_ids=doc_ids, sections=sections)
        with self._timed(timings, "retrieval"):
            candidates = self.retriever.retrieve(expanded, query_filter=query_filter)

        if not candidates:
            return self._insufficient(
                query,
                expanded,
                [],
                self.confidence.score([]),
                input_verdicts,
                timings,
                "Retrieval returned no candidates.",
            )

        # ---- 4. mandatory rerank ----
        with self._timed(timings, "rerank"):
            try:
                top = self.reranker.rerank(query, candidates, top_k=top_k or self.settings.final_top_k)
            except Exception:
                # The reranker is mandatory: without it there is no calibrated
                # score to gate on, so we refuse instead of answering blind.
                return self._blocked(
                    query,
                    input_verdicts
                    + [
                        GuardrailVerdict(
                            passed=False,
                            code="reranker_unavailable",
                            severity="block",
                            reason="The relevance reranker failed. This pipeline will not "
                            "answer medical questions without it.",
                        )
                    ],
                    GuardrailVerdict(
                        passed=False,
                        code="reranker_unavailable",
                        severity="block",
                        reason="The relevance reranker is unavailable, so the answer could not "
                        "be quality-gated. No answer was generated.",
                    ),
                    timings,
                    expanded=expanded,
                )

        # ---- 5. evidence gate ----
        with self._timed(timings, "confidence"):
            report = self.confidence.score(top)
        logger.info(report.explanation)
        if not report.passed:
            return self._insufficient(
                query, expanded, top, report, input_verdicts, timings, report.explanation
            )

        # ---- 6. generation ----
        with self._timed(timings, "generation"):
            try:
                raw_answer = self.llm.chat(build_messages(query, top))
            except LLMError as exc:
                return self._blocked(
                    query,
                    input_verdicts,
                    GuardrailVerdict(
                        passed=False,
                        code="llm_unavailable",
                        severity="block",
                        reason=f"The language model could not be reached: {exc}",
                    ),
                    timings,
                    expanded=expanded,
                    retrieved=top,
                    confidence=report,
                )

        # ---- 7. output gate ----
        with self._timed(timings, "output_guard"):
            output_verdicts, citations = self.output_guard.check(raw_answer, top)

        all_verdicts = input_verdicts + output_verdicts
        output_block = next((v for v in output_verdicts if v.severity == "block"), None)
        if output_block:
            logger.warning("Output guardrail blocked answer: %s", output_block.reason)
            return RAGAnswer(
                query=query,
                answer=self.output_guard.apply_disclaimer(
                    REFUSAL_INSUFFICIENT
                    + "\n\n_(An answer was drafted but failed verification against the "
                    f"sources: {output_block.reason})_"
                ),
                citations=[],
                confidence=report,
                guardrails=all_verdicts,
                refused=True,
                disclaimer=DISCLAIMER,
                expanded_query=expanded,
                retrieved=top,
                timings_ms=timings,
            )

        # ---- 8. disclaimer ----
        return RAGAnswer(
            query=query,
            answer=self.output_guard.apply_disclaimer(raw_answer),
            citations=citations,
            confidence=report,
            guardrails=all_verdicts,
            refused=False,
            disclaimer=DISCLAIMER,
            expanded_query=expanded,
            retrieved=top,
            timings_ms=timings,
        )

    # -- retrieval-only (debugging / evaluation) ------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        rerank: bool = True,
    ) -> list[RetrievedChunk]:
        expanded = self.expander.expand(query)
        candidates = self.retriever.retrieve(expanded)
        if rerank and candidates:
            return self.reranker.rerank(query, candidates, top_k=top_k)
        return candidates[: (top_k or self.settings.final_top_k)]

    # -- helpers --------------------------------------------------------

    @contextmanager
    def _timed(self, sink: dict[str, float], name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            sink[name] = round((time.perf_counter() - start) * 1000, 1)

    def _blocked(
        self,
        query: str,
        verdicts: list[GuardrailVerdict],
        blocking: GuardrailVerdict,
        timings: dict[str, float],
        *,
        expanded=None,
        retrieved: Sequence[RetrievedChunk] | None = None,
        confidence: ConfidenceReport | None = None,
    ) -> RAGAnswer:
        # Blocking verdicts carry their own user-facing text (the emergency
        # notice is already a complete, self-contained response).
        answer = blocking.reason
        return RAGAnswer(
            query=query,
            answer=self.output_guard.apply_disclaimer(answer),
            citations=[],
            confidence=confidence,
            guardrails=verdicts,
            refused=True,
            disclaimer=DISCLAIMER,
            expanded_query=expanded,
            retrieved=list(retrieved or []),
            timings_ms=timings,
        )

    def _insufficient(
        self,
        query: str,
        expanded,
        retrieved: Sequence[RetrievedChunk],
        report: ConfidenceReport,
        verdicts: list[GuardrailVerdict],
        timings: dict[str, float],
        reason: str,
    ) -> RAGAnswer:
        verdicts = verdicts + [
            GuardrailVerdict(
                passed=False,
                code="low_confidence",
                severity="block",
                reason=reason,
            )
        ]
        return RAGAnswer(
            query=query,
            answer=self.output_guard.apply_disclaimer(REFUSAL_INSUFFICIENT),
            citations=[],
            confidence=report,
            guardrails=verdicts,
            refused=True,
            disclaimer=DISCLAIMER,
            expanded_query=expanded,
            retrieved=list(retrieved),
            timings_ms=timings,
        )


_pipeline: MedicalRAGPipeline | None = None


def get_pipeline() -> MedicalRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MedicalRAGPipeline()
    return _pipeline


__all__ = ["MedicalRAGPipeline", "get_pipeline", "QdrantStore"]
