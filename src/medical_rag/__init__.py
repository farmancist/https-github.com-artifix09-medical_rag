"""Medical RAG.

Pipeline (see docs/architecture.md):
    PDF parser -> semantic chunker (85 / 20%) -> PubMedBERT -> Qdrant HNSW (m=32,
    ef=200) -> query expansion -> hybrid BM25+dense RRF -> cross-encoder rerank
    -> strict guardrails (conf >= 0.8) -> Groq
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
