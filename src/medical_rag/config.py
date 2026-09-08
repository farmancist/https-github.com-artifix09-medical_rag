"""Single source of truth for configuration.

Every tunable in the architecture diagram lives here and is overridable via
`.env`, so the *same code* runs on the CPU laptop and the CUDA desktop with
only the env file differing.
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
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,  # aliased fields stay constructible by field name
    )

    # ---------- runtime ----------
    device: Literal["auto", "cpu", "cuda"] = "auto"
    log_level: str = "INFO"

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
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.1
    groq_max_tokens: int = 1024
    groq_expansion_model: str = "llama-3.1-8b-instant"

    # ---------- embeddings ----------
    embedding_model: str = "NeuML/pubmedbert-base-embeddings"
    embedding_dim: int = 768
    embedding_batch_size: int = 32
    embedding_max_seq_length: int = 512

    # ---------- chunking ----------
    chunk_breakpoint_percentile: float = 85.0
    chunk_overlap_ratio: float = 0.20
    chunk_min_tokens: int = 64
    chunk_max_tokens: int = 512

    # ---------- retrieval ----------
    dense_top_k: int = 50
    sparse_top_k: int = 50
    rrf_k: int = 60
    rerank_candidates: int = 40
    final_top_k: int = 6

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
        """Turn `auto` into a concrete torch device string."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # torch missing or broken -> stay on CPU
            return "cpu"

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
