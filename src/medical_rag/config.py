"""Single source of truth for configuration.

Every tunable in the architecture diagram lives here and is overridable via
`.env`, so the *same code* runs on any machine -- CPU or CUDA -- with only the
env file differing. `DEVICE=auto` resolves the difference at runtime.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = PROJECT_ROOT / "RAW_DATA"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        # utf-8-sig, not utf-8: an editor (or PowerShell's `Set-Content
        # -Encoding utf8`) can save .env with a BOM, which would otherwise make
        # the first variable parse as "﻿GROK" and silently go missing.
        env_file_encoding="utf-8-sig",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,  # aliased fields stay constructible by field name
    )

    # ---------- runtime ----------
    # Default is "cpu", not "auto": this machine has been losing power
    # unexpectedly 1-7 times a day for at least a month, including days with
    # no GPU load at all -- a general power-delivery/hardware reliability
    # problem, NOT something DirectML/GPU load specifically triggers (see
    # src/medical_rag/device.py docstring for the full correction and the
    # event-log evidence). The RX590 draws ~225W the PSU may not be specced
    # for, so avoiding it is still a reasonable precaution while the
    # hardware gets diagnosed -- but a precaution against a suspected
    # contributing factor, not a fix for a confirmed root cause.
    # "auto"/"cuda"/"dml" remain available as an explicit, deliberate opt-in
    # (e.g. `DEVICE=dml` for a one-off benchmark or a supervised rebuild) --
    # "auto" goes through device.get_device(), which raises rather than
    # silently landing on CPU; any explicit value here is honoured as-is.
    device: Literal["auto", "cpu", "cuda", "dml"] = "cpu"
    log_level: str = "INFO"
    # Layout model can improve detection on some pages but was less reliable
    # than line-based detection on a BNF sample; keep opt-in pending validation.
    pdf_layout_analysis: bool = False

    # ---------- qdrant ----------
    # Two modes, no server to run either way:
    #   QDRANT_URL set    -> Qdrant Cloud (or any reachable Qdrant), the real target
    #   QDRANT_URL empty  -> embedded on-disk store at QDRANT_PATH, zero setup
    qdrant_url: str = ""
    qdrant_path: Path = Field(default=DATA_DIR / "qdrant")
    # `QDRANT` is accepted as an alias so an existing .env keeps working.
    qdrant_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("QDRANT_API_KEY", "QDRANT"),
    )
    qdrant_collection: str = "medical_rag"

    @property
    def qdrant_is_embedded(self) -> bool:
        return not self.qdrant_url.strip()
    qdrant_hnsw_m: int = 32
    qdrant_hnsw_ef_construct: int = 200
    qdrant_hnsw_ef_search: int = 200

    # ---------- groq ----------
    # `GROK` is accepted as an alias (the key itself is a Groq `gsk_...` key).
    groq_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GROQ_API_KEY", "GROK", "GROQ"),
    )
    # Groq retired the Llama 3.x line; these are current as of the catalogue
    # returned by `client.models.list()`. Check that if a 404 shows up here.
    groq_model: str = "openai/gpt-oss-120b"
    groq_temperature: float = 0.1
    # Reasoning models (gpt-oss) spend part of this budget on hidden reasoning
    # tokens before emitting any content. Keep a generous completion budget;
    # the model's own context/output limits still apply.
    groq_max_tokens: int = 32768
    groq_expansion_model: str = "openai/gpt-oss-20b"

    # ---------- embeddings ----------
    embedding_model: str = "NeuML/pubmedbert-base-embeddings"
    embedding_dim: int = 768
    embedding_batch_size: int = 32
    embedding_max_seq_length: int = 512

    # ---------- chunking ----------
    chunk_breakpoint_percentile: float = 85.0
    chunk_overlap_ratio: float = 0.20
    chunk_min_tokens: int = 64
    # Word-ish tokens, NOT the model's WordPiece subwords. Measured on this
    # corpus the subword/word ratio is 1.10 average and 1.21 worst-case, so 512
    # here would produce ~620 subwords and silently lose the tail of every long
    # chunk. 384 * 1.21 ~= 465, comfortably inside embedding_max_seq_length.
    chunk_max_tokens: int = 384

    # ---------- retrieval ----------
    dense_top_k: int = 80
    sparse_top_k: int = 80
    rrf_k: int = 60
    rerank_candidates: int = 60
    final_top_k: int = 8

    # ---------- reranker ----------
    reranker_model: str = "ncbi/MedCPT-Cross-Encoder"
    reranker_batch_size: int = 16

    # ---------- query expansion ----------
    query_expansion_enabled: bool = True
    query_expansion_mode: Literal["lexicon", "llm", "hybrid"] = "hybrid"
    query_expansion_max_variants: int = 3

    # ---------- guardrails ----------
    confidence_threshold: float = 0.8
    require_citations: bool = True
    disclaimer_enabled: bool = True

    # ---------- api ----------
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ---------- paths ----------
    raw_dir: Path = Field(default=RAW_DIR)
    processed_dir: Path = Field(default=PROCESSED_DIR)
    cache_dir: Path = Field(default=CACHE_DIR)

    @field_validator("raw_dir", "processed_dir", "cache_dir", "qdrant_path", mode="after")
    @classmethod
    def _anchor_to_project(cls, v: Path) -> Path:
        """Resolve relative paths against the repo, not the shell's cwd.

        `.env` naturally holds relative values like `QDRANT_PATH=data/qdrant`.
        Without this, running a script from a different directory would quietly
        create a second, empty index instead of finding the real one.
        """
        return v if v.is_absolute() else (PROJECT_ROOT / v)

    @field_validator("chunk_overlap_ratio")
    @classmethod
    def _check_overlap(cls, v: float) -> float:
        if not 0.0 <= v < 0.9:
            raise ValueError("chunk_overlap_ratio must be in [0.0, 0.9)")
        return v

    @field_validator("confidence_threshold")
    @classmethod
    def _check_conf(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence_threshold must be in [0.0, 1.0]")
        return v

    def resolve_device(self) -> str:
        """Turn `auto` into a concrete backend name: "cuda", "dml", or "cpu".

        `auto` is resolved by `device.get_device()`, which raises rather than
        silently landing on CPU (unless ALLOW_CPU=1) -- see that module for
        why. An explicit `DEVICE=` value is a deliberate override and is
        honoured as-is, unchecked.
        """
        if self.device != "auto":
            return self.device
        from .device import get_device

        return get_device().backend

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def configure_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=(level or get_settings().log_level).upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # These libraries are chatty at INFO.
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
