"""Groq chat client.

Thin, deliberately: retries, timeouts and JSON coercion live here so the
guardrail and expansion modules stay about medicine, not HTTP.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Iterator, Sequence

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


class LLMError(RuntimeError):
    pass


class GroqLLM:
    def __init__(self, settings: Settings | None = None, *, model: str | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = model or self.settings.groq_model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from groq import Groq

            if not self.settings.groq_api_key:
                raise LLMError("GROQ_API_KEY is not set — put it in .env")
            self._client = Groq(api_key=self.settings.groq_api_key, timeout=60.0)
        return self._client

    # -- completion -----------------------------------------------------

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        retries: int = 2,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=model or self.model,
                    messages=list(messages),
                    temperature=(
                        self.settings.groq_temperature if temperature is None else temperature
                    ),
                    max_completion_tokens=(
                        self.settings.groq_max_tokens if max_tokens is None else max_tokens
                    ),
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:  # rate limit / transient network
                last_error = exc
                if attempt < retries:
                    backoff = 1.5 * (2**attempt)
                    logger.warning("Groq call failed (%s); retrying in %.1fs", exc, backoff)
                    time.sleep(backoff)
        raise LLMError(f"Groq request failed after {retries + 1} attempts: {last_error}")

    def stream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=model or self.model,
            messages=list(messages),
            temperature=(self.settings.groq_temperature if temperature is None else temperature),
            max_completion_tokens=(
                self.settings.groq_max_tokens if max_tokens is None else max_tokens
            ),
            stream=True,
        )
        for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield delta

    def chat_json(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        model: str | None = None,
        default: Any = None,
    ) -> Any:
        """Ask for JSON and parse defensively.

        Never raises on malformed output: query expansion is an optimisation,
        and a failure there must degrade to the original query rather than take
        the whole request down.
        """
        raw = ""
        try:
            raw = self.chat(
                messages, temperature=temperature, max_tokens=max_tokens, model=model, retries=1
            )
            return json.loads(raw)
        except json.JSONDecodeError:
            match = _JSON_BLOCK_RE.search(raw)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            logger.warning("Could not parse JSON from model output: %r", raw[:200])
            return default
        except LLMError as exc:
            logger.warning("LLM JSON call failed: %s", exc)
            return default


_llm: GroqLLM | None = None


def get_llm() -> GroqLLM:
    global _llm
    if _llm is None:
        _llm = GroqLLM()
    return _llm
