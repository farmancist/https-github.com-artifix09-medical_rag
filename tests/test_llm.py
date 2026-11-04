"""Groq generation settings tests — no network or API key required."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from medical_rag.config import Settings
from medical_rag.generation.llm import GroqLLM


def make_llm(settings: Settings) -> tuple[GroqLLM, Mock]:
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
    )
    llm = GroqLLM(settings)
    llm._client = client
    return llm, client


def test_default_generation_budget_is_expanded(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_MAX_TOKENS", raising=False)

    assert Settings(_env_file=None).groq_max_tokens == 32768


def test_chat_uses_completion_budget_and_respects_override() -> None:
    llm, client = make_llm(Settings(_env_file=None, groq_max_tokens=32768))

    assert llm.chat([{"role": "user", "content": "question"}]) == "answer"
    call = client.chat.completions.create.call_args.kwargs
    assert call["max_completion_tokens"] == 32768
    assert "max_tokens" not in call

    llm.chat([{"role": "user", "content": "question"}], max_tokens=768)
    assert client.chat.completions.create.call_args.kwargs["max_completion_tokens"] == 768


def test_stream_uses_expanded_completion_budget() -> None:
    llm, client = make_llm(Settings(_env_file=None, groq_max_tokens=32768))
    client.chat.completions.create.return_value = []

    assert list(llm.stream([{"role": "user", "content": "question"}])) == []
    call = client.chat.completions.create.call_args.kwargs
    assert call["max_completion_tokens"] == 32768
    assert call["stream"] is True
