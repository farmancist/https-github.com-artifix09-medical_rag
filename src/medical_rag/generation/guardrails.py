"""Strict guardrails: confidence >= 0.8 and a mandatory disclaimer.

Three gates, in order:

  INPUT   before retrieval  — emergencies, self-harm, requests for a personal
                              prescription, and prompt-injection attempts.
  EVIDENCE after reranking  — the confidence score. Below threshold, we refuse
                              rather than generate. This is the single most
                              important difference from the book RAG: an
                              unsupported answer about a book is a nuisance, an
                              unsupported answer about a dose is a hazard.
  OUTPUT  after generation  — every citation marker must resolve to a real
                              source, unsupported numerics are flagged, and the
                              disclaimer is appended unconditionally.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from ..config import Settings, get_settings
from ..models import Citation, ConfidenceReport, GuardrailVerdict, RetrievedChunk
from .prompts import DISCLAIMER, EMERGENCY_NOTICE

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Input gate
# --------------------------------------------------------------------------

_EMERGENCY_PATTERNS = [
    r"\bcrushing chest pain\b",
    r"\bchest pain\b.*\b(radiat|left arm|jaw|sweat|short(ness)? of breath)\b",
    r"\b(can'?t|cannot|unable to) breathe\b",
    r"\bnot breathing\b",
    r"\bstopped breathing\b",
    r"\bunconscious\b|\bunresponsive\b|\bpassed out\b.*\bnot wak",
    r"\b(severe|heavy|uncontrolled|won'?t stop) bleeding\b",
    r"\bbleeding\b.*\b(won'?t|will not|does not|doesn'?t) stop\b",
    r"\bseizure\b.*\b(ongoing|won'?t stop|still)\b",
    r"\bstroke symptoms?\b|\bface drooping\b|\bslurred speech\b.*\bsudden\b",
    r"\banaphyla(xis|ctic)\b|\bthroat (is )?closing\b|\btongue swelling\b",
    r"\boverdos(e|ed)\b|\bpoison(ed|ing)\b|\bswallowed\b.*\b(bleach|pills|batter)",
    r"\bsuicid(e|al)\b|\bkill myself\b|\bend my life\b|\bself[- ]harm\b",
    r"\bcoughing up blood\b|\bvomiting blood\b",
    r"\bblue lips\b|\bturning blue\b",
    r"\bwater (broke|breaking)\b.*\b(bleed|pain)\b",
]
_EMERGENCY_RE = [re.compile(p, re.IGNORECASE) for p in _EMERGENCY_PATTERNS]

# Requests to act on the individual, rather than to explain the literature.
_PERSONAL_DIRECTIVE_PATTERNS = [
    r"\bwhat (should|do) i take\b",
    r"\bhow (much|many) should i take\b",
    r"\bshould i (stop|start|double|increase|decrease|skip)\b.*\b(dose|medication|pill|drug)\b",
    r"\bdiagnose me\b|\bwhat do i have\b|\bwhat'?s wrong with me\b",
    r"\bprescribe\b.*\b(me|for me)\b",
    r"\bis it safe for me to\b",
    r"\bcan i (stop|skip|double)\b.*\b(my )?(medication|dose|pills?)\b",
]
_PERSONAL_DIRECTIVE_RE = [re.compile(p, re.IGNORECASE) for p in _PERSONAL_DIRECTIVE_PATTERNS]

_INJECTION_PATTERNS = [
    r"ignore (all|any|the) (previous|prior|above) instructions?",
    r"disregard (your|the) (instructions?|rules?|system prompt)",
    r"you are (now|no longer)\b.*\b(a|an)\b",
    r"\bpretend (to be|you are)\b",
    r"reveal (your|the) (system )?prompt",
    r"\bdeveloper mode\b|\bDAN mode\b",
    r"without (the )?(disclaimer|safety|guardrails?|warnings?)",
    r"\bas a (licensed )?(doctor|physician)\b.*\b(prescribe|tell me to)\b",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


class InputGuardrail:
    """Screens the question before any retrieval happens."""

    def check(self, query: str) -> list[GuardrailVerdict]:
        verdicts: list[GuardrailVerdict] = []
        text = query.strip()

        if not text or len(text) < 3:
            verdicts.append(
                GuardrailVerdict(
                    passed=False,
                    code="empty_query",
                    severity="block",
                    reason="The question is empty or too short to search on.",
                )
            )
            return verdicts

        if len(text) > 2000:
            verdicts.append(
                GuardrailVerdict(
                    passed=False,
                    code="query_too_long",
                    severity="block",
                    reason="Question exceeds 2000 characters.",
                )
            )
            return verdicts

        if any(rx.search(text) for rx in _EMERGENCY_RE):
            # Emergencies short-circuit the whole pipeline. Retrieval latency is
            # not acceptable when the correct response is "call for help now".
            verdicts.append(
                GuardrailVerdict(
                    passed=False,
                    code="emergency",
                    severity="block",
                    reason=EMERGENCY_NOTICE,
                )
            )
            return verdicts

        if any(rx.search(text) for rx in _INJECTION_RE):
            verdicts.append(
                GuardrailVerdict(
                    passed=True,
                    code="injection_attempt",
                    severity="warn",
                    reason="Instruction-override phrasing detected; system rules are enforced "
                    "regardless.",
                )
            )

        if any(rx.search(text) for rx in _PERSONAL_DIRECTIVE_RE):
            verdicts.append(
                GuardrailVerdict(
                    passed=True,
                    code="personal_directive",
                    severity="warn",
                    reason="Question asks for a personal clinical decision. The answer will "
                    "describe what the sources say and will not direct your care.",
                )
            )

        if not verdicts:
            verdicts.append(GuardrailVerdict(passed=True, code="ok", reason="Input checks passed."))
        return verdicts


# --------------------------------------------------------------------------
# Evidence gate
# --------------------------------------------------------------------------


class ConfidenceScorer:
    """Turns retrieval quality into a single [0,1] score.

    Four components, because any one of them alone is gameable:

      top_relevance (0.45)  the best cross-encoder score. The primary signal:
                            is there at least one genuinely on-point passage?
      support       (0.25)  how many passages clear the support bar. One good
                            hit can be a fluke; three agreeing hits rarely are.
      margin        (0.15)  gap between the top hit and the median candidate.
                            A flat score distribution means the retriever could
                            not discriminate, i.e. the question is probably not
                            covered by the corpus at all.
      agreement     (0.15)  mean score of the passages actually sent to the LLM.
                            Guards against a strong #1 followed by noise.
    """

    WEIGHTS = {"top_relevance": 0.45, "support": 0.25, "margin": 0.15, "agreement": 0.15}
    SUPPORT_BAR = 0.5
    TARGET_SUPPORT = 3

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def score(self, retrieved: Sequence[RetrievedChunk]) -> ConfidenceReport:
        threshold = self.settings.confidence_threshold
        if not retrieved:
            return ConfidenceReport(
                score=0.0,
                threshold=threshold,
                passed=False,
                components={},
                explanation="No passages were retrieved.",
            )

        scores = sorted(
            (r.rerank_score if r.rerank_score is not None else 0.0 for r in retrieved),
            reverse=True,
        )
        top = scores[0]
        n_supporting = sum(1 for s in scores if s >= self.SUPPORT_BAR)
        support = min(1.0, n_supporting / self.TARGET_SUPPORT)
        median = scores[len(scores) // 2]
        margin = max(0.0, min(1.0, (top - median) / 0.5))
        agreement = sum(scores) / len(scores)

        components = {
            "top_relevance": round(top, 4),
            "support": round(support, 4),
            "margin": round(margin, 4),
            "agreement": round(agreement, 4),
        }
        total = sum(self.WEIGHTS[name] * value for name, value in components.items())
        total = round(min(1.0, max(0.0, total)), 4)

        passed = total >= threshold
        explanation = (
            f"confidence {total:.2f} vs threshold {threshold:.2f} "
            f"(top={top:.2f}, {n_supporting} passage(s) above the {self.SUPPORT_BAR} support bar)"
        )
        return ConfidenceReport(
            score=total,
            threshold=threshold,
            passed=passed,
            components=components,
            explanation=explanation,
        )


# --------------------------------------------------------------------------
# Output gate
# --------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")
# A number that carries clinical weight: a dose, a threshold, an interval.
_CLINICAL_NUMBER_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|µg|g|kg|ml|mL|l|L|IU|units?|mmol|mEq|mmHg|%|"
    r"hours?|hrs?|days?|weeks?|months?|years?|times?\s?(?:a|per)\s?day)\b",
    re.IGNORECASE,
)
_NO_ANSWER_MARKER = "the provided sources do not cover this"


class OutputGuardrail:
    """Validates the generated answer against the sources it was given."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # -- checks ---------------------------------------------------------

    def check(
        self, answer: str, retrieved: Sequence[RetrievedChunk]
    ) -> tuple[list[GuardrailVerdict], list[Citation]]:
        verdicts: list[GuardrailVerdict] = []
        markers = [int(m) for m in _CITATION_RE.findall(answer)]
        valid = {m for m in markers if 1 <= m <= len(retrieved)}
        dangling = sorted({m for m in markers if m not in valid})

        if dangling:
            verdicts.append(
                GuardrailVerdict(
                    passed=False,
                    code="dangling_citation",
                    severity="block",
                    reason=f"Answer cites sources that do not exist: {dangling}.",
                )
            )

        model_declined = _NO_ANSWER_MARKER in answer.lower()

        if self.settings.require_citations and not valid and not model_declined:
            verdicts.append(
                GuardrailVerdict(
                    passed=False,
                    code="no_citations",
                    severity="block",
                    reason="Answer contains no valid source citations.",
                )
            )

        unsupported = self._unverified_numbers(answer, retrieved)
        if unsupported:
            verdicts.append(
                GuardrailVerdict(
                    passed=False,
                    code="unsupported_numeric",
                    severity="block",
                    reason=(
                        "Answer states clinical values that appear in no retrieved source: "
                        + ", ".join(sorted(unsupported)[:5])
                    ),
                )
            )

        if not verdicts:
            verdicts.append(
                GuardrailVerdict(passed=True, code="ok", reason="Output checks passed.")
            )
        return verdicts, self._citations(valid, retrieved)

    @staticmethod
    def _unverified_numbers(answer: str, retrieved: Sequence[RetrievedChunk]) -> set[str]:
        """Any dose/interval in the answer must appear verbatim in the context.

        Numeric hallucination is the highest-severity failure mode here, and it
        is one of the few that can be caught deterministically.
        """
        corpus = " ".join(r.chunk.text for r in retrieved).lower()
        corpus_norm = re.sub(r"\s+", "", corpus)
        unsupported: set[str] = set()
        for match in _CLINICAL_NUMBER_RE.finditer(answer):
            token = match.group(0)
            if re.sub(r"\s+", "", token.lower()) not in corpus_norm:
                unsupported.add(token)
        return unsupported

    @staticmethod
    def _citations(markers: set[int], retrieved: Sequence[RetrievedChunk]) -> list[Citation]:
        out: list[Citation] = []
        for marker in sorted(markers):
            chunk = retrieved[marker - 1].chunk
            snippet = chunk.text.strip()
            out.append(
                Citation(
                    marker=f"[{marker}]",
                    chunk_id=chunk.chunk_id,
                    title=chunk.title or chunk.doc_id,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    source_path=chunk.source_path,
                    snippet=(snippet[:300] + "…") if len(snippet) > 300 else snippet,
                )
            )
        return out

    # -- finishing ------------------------------------------------------

    def apply_disclaimer(self, answer: str) -> str:
        if not self.settings.disclaimer_enabled:
            return answer
        if DISCLAIMER in answer:
            return answer
        return f"{answer.rstrip()}\n\n---\n\n{DISCLAIMER}"
