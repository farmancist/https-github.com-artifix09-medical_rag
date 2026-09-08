"""Prompts.

The system prompt is the first guardrail layer. It is deliberately restrictive:
the model is an *extractor and summariser of the retrieved context*, not a
clinician. Anything it cannot support from context must become an explicit
"the provided sources do not cover this".
"""

from __future__ import annotations

from typing import Sequence

from ..models import RetrievedChunk

# Deliberately ASCII. These strings are printed to terminals, and a Windows
# console on a legacy code page raises UnicodeEncodeError on emoji -- which
# silently truncated the disclaimer and made the CLI exit non-zero.
DISCLAIMER = (
    "**Medical disclaimer** - This information is drawn from the indexed medical "
    "literature for general educational purposes only. It is not a diagnosis, "
    "prescription, or medical advice, and it may not reflect your individual "
    "circumstances. Consult a qualified healthcare professional before acting on "
    "anything here. In an emergency, contact your local emergency services immediately."
)

EMERGENCY_NOTICE = (
    "**THIS MAY BE A MEDICAL EMERGENCY.** Stop reading and contact your local emergency "
    "services now (for example 911 in the US, 112 in the EU, 999 in the UK, 1122 in Pakistan), "
    "or go to the nearest emergency department. Do not wait for information from this system."
)

SYSTEM_PROMPT = """You are a careful medical information assistant answering ONLY from the \
numbered source excerpts provided to you.

ABSOLUTE RULES:
1. Use ONLY the numbered sources below. You have no other knowledge for this task. If the \
sources do not contain the answer, say exactly: "The provided sources do not cover this." \
Then stop. Never fill a gap from memory.
2. Cite every factual sentence with its source marker, like [1] or [2][3]. A sentence with a \
clinical claim and no citation is a failure.
3. Reproduce doses, units, frequencies, thresholds, ages and durations EXACTLY as written in \
the sources. Never round, convert, average, or infer a number that is not printed there.
4. Never diagnose the user, never tell them which drug to take, and never adjust a dose for \
them. Describe what the literature states; do not direct the individual.
5. If sources disagree, say so and present both positions with their citations.
6. Preserve the qualifiers that come with a claim: population (adult/paediatric/pregnant/renal \
impairment), route, severity, and contraindications. A dose without its population is unsafe.
7. Do not speculate about prognosis, and do not reassure. State what the sources state.
8. Write in plain language. When you use a clinical term, gloss it once in parentheses.

FORMAT:
- Lead with a direct 1-2 sentence answer.
- Then the supporting detail, as short paragraphs or bullets, each carrying its citation.
- If relevant to the question, add a brief "Important cautions" section listing \
contraindications, warnings, or red flags that the sources mention.
- Do not append a disclaimer; the system adds one.
"""

REFUSAL_INSUFFICIENT = (
    "I could not find enough reliable support in the indexed medical sources to answer this "
    "safely.\n\n"
    "Rather than guess — which is dangerous with clinical information — here is what I suggest:\n"
    "- Rephrase with more specifics (the condition, drug name, patient group, or route).\n"
    "- Check whether the relevant document has actually been ingested into the index.\n"
    "- For anything affecting your own care, speak to a qualified healthcare professional."
)


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Render retrieved chunks as numbered, attributable sources."""
    blocks: list[str] = []
    for i, retrieved in enumerate(chunks, start=1):
        chunk = retrieved.chunk
        header = f"[{i}] {chunk.title or chunk.doc_id}"
        if chunk.section:
            header += f" — {chunk.section}"
        pages = (
            f"p.{chunk.page_start}"
            if chunk.page_start == chunk.page_end
            else f"pp.{chunk.page_start}-{chunk.page_end}"
        )
        blocks.append(f"{header} ({pages})\n{chunk.text.strip()}")
    return "\n\n---\n\n".join(blocks)


def build_messages(query: str, chunks: Sequence[RetrievedChunk]) -> list[dict[str, str]]:
    context = format_context(chunks)
    user = (
        f"SOURCES:\n\n{context}\n\n"
        f"{'=' * 60}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer using only the sources above, citing each factual sentence with its "
        "[n] marker."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
