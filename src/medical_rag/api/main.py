"""FastAPI service.

Run:  uvicorn medical_rag.api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import configure_logging, get_settings
from ..models import RAGAnswer
from ..pipeline import MedicalRAGPipeline, get_pipeline
from ..vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    logger.info(
        "Starting Medical RAG API | device=%s | collection=%s",
        settings.resolve_device(),
        settings.qdrant_collection,
    )
    # Models load lazily; warm them so the first user request is not the one
    # that pays a 30s model download.
    try:
        get_pipeline().warmup()
    except Exception:
        logger.exception("Warmup failed — models will load on first request")
    yield


app = FastAPI(
    title="Medical RAG",
    version="0.1.0",
    description=(
        "Retrieval-augmented medical question answering. "
        "PubMedBERT embeddings, Qdrant hybrid search (BM25 + dense, RRF), "
        "mandatory cross-encoder reranking, confidence-gated generation via Groq."
    ),
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, examples=["What causes a heart attack?"])
    doc_ids: list[str] | None = None
    sections: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    include_retrieved: bool = False


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    rerank: bool = True


class SearchHit(BaseModel):
    chunk_id: str
    title: str
    section: str | None
    page_start: int
    page_end: int
    text: str
    dense_score: float | None
    sparse_score: float | None
    rrf_score: float | None
    rerank_score: float | None


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


def _pipeline() -> MedicalRAGPipeline:
    return get_pipeline()


@app.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    store = QdrantStore(settings)
    info = store.info()
    return {
        "status": "ok" if info.get("exists") else "degraded",
        "device": settings.resolve_device(),
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "llm_model": settings.groq_model,
        "groq_configured": bool(settings.groq_api_key),
        "confidence_threshold": settings.confidence_threshold,
        "qdrant": info,
    }


@app.post("/ask", response_model=RAGAnswer, response_model_exclude_none=True)
def ask(request: AskRequest) -> RAGAnswer:
    try:
        answer = _pipeline().answer(
            request.query,
            doc_ids=request.doc_ids,
            sections=request.sections,
            top_k=request.top_k,
        )
    except Exception as exc:
        logger.exception("Pipeline failure")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not request.include_retrieved:
        # The full retrieved set is large; callers opt in explicitly.
        answer.retrieved = []
    return answer


@app.post("/search", response_model=list[SearchHit])
def search(request: SearchRequest) -> list[SearchHit]:
    """Retrieval only — no generation, no guardrails. For tuning and evaluation."""
    try:
        hits = _pipeline().search(request.query, top_k=request.top_k, rerank=request.rerank)
    except Exception as exc:
        logger.exception("Search failure")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return [
        SearchHit(
            chunk_id=h.chunk.chunk_id,
            title=h.chunk.title,
            section=h.chunk.section,
            page_start=h.chunk.page_start,
            page_end=h.chunk.page_end,
            text=h.chunk.text,
            dense_score=h.dense_score,
            sparse_score=h.sparse_score,
            rrf_score=h.rrf_score,
            rerank_score=h.rerank_score,
        )
        for h in hits
    ]


@app.post("/expand")
def expand(request: SearchRequest) -> dict[str, Any]:
    """Inspect query expansion in isolation — useful when tuning the lexicon."""
    expanded = _pipeline().expander.expand(request.query)
    return expanded.model_dump()


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "medical_rag.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    run()
