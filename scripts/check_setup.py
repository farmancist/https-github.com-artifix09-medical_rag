"""Preflight check — run this first on each machine.

Verifies: dependencies, device, .env keys, Qdrant reachability, Groq
reachability, and the state of the collection. Everything is checked
independently so one failure does not hide the others.

    python scripts/check_setup.py
    python scripts/check_setup.py --models   # also downloads/loads the models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OK = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"

failures = 0
warnings = 0


def report(status: str, label: str, detail: str = "") -> None:
    global failures, warnings
    if status == FAIL:
        failures += 1
    elif status == WARN:
        warnings += 1
    line = f"{status} {label}"
    if detail:
        line += f" — {detail}"
    print(line)


def check_dependencies() -> None:
    required = {
        "pydantic": "pydantic",
        "pydantic_settings": "pydantic-settings",
        "numpy": "numpy",
        "fitz": "PyMuPDF",
        "torch": "torch",
        "sentence_transformers": "sentence-transformers",
        "qdrant_client": "qdrant-client",
        "groq": "groq",
        "fastapi": "fastapi",
        "rich": "rich",
    }
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        report(FAIL, "dependencies", f"missing: {', '.join(missing)} (pip install -r requirements.txt)")
    else:
        report(OK, "dependencies", "all imports resolve")


def check_device() -> str:
    from medical_rag.config import get_settings

    device = get_settings().resolve_device()
    try:
        import torch

        if device == "cuda":
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            report(OK, "device", f"cuda — {name} ({mem:.1f} GB), torch {torch.__version__}")
        else:
            report(
                WARN,
                "device",
                f"cpu (torch {torch.__version__}) — fine for serving, slow for full ingestion",
            )
    except Exception as exc:
        report(FAIL, "device", str(exc))
    return device


def check_env() -> None:
    from medical_rag.config import PROJECT_ROOT, get_settings

    if not (PROJECT_ROOT / ".env").exists():
        report(FAIL, ".env", "not found — copy .env.example to .env and fill it in")
        return
    settings = get_settings()
    report(OK, ".env", "found")

    if settings.qdrant_is_embedded:
        report(
            OK,
            "QDRANT_URL",
            f"empty -> embedded on-disk mode at {settings.qdrant_path}",
        )
    elif settings.qdrant_api_key:
        report(OK, "QDRANT_URL", f"remote: {settings.qdrant_url}")
    else:
        report(FAIL, "QDRANT_API_KEY", "empty but QDRANT_URL points at a remote cluster")

    if settings.groq_api_key:
        report(OK, "GROQ_API_KEY", f"set (model {settings.groq_model})")
    else:
        report(FAIL, "GROQ_API_KEY", "empty — generation and LLM expansion will fail")


def check_qdrant() -> None:
    from medical_rag.config import get_settings
    from medical_rag.vectorstore.qdrant_store import QdrantStore

    settings = get_settings()
    where = (
        f"embedded ({settings.qdrant_path})"
        if settings.qdrant_is_embedded
        else settings.qdrant_url
    )
    try:
        store = QdrantStore(settings)
        collections = [c.name for c in store.client.get_collections().collections]
        report(OK, "qdrant", f"connected: {where}")
        if settings.qdrant_collection in collections:
            info = store.info()
            level = OK if info["points"] else WARN
            report(level, "collection", f"{settings.qdrant_collection}: {info['points']} points")
        else:
            report(
                WARN,
                "collection",
                f"{settings.qdrant_collection} does not exist yet — run scripts/ingest.py",
            )
    except Exception as exc:
        report(FAIL, "qdrant", f"{type(exc).__name__}: {exc}")


def check_groq() -> None:
    from medical_rag.config import get_settings
    from medical_rag.generation.llm import GroqLLM

    settings = get_settings()
    if not settings.groq_api_key:
        report(WARN, "groq call", "skipped, no API key")
        return
    try:
        reply = GroqLLM(settings).chat(
            [{"role": "user", "content": "Reply with the single word: ready"}],
            max_tokens=8,
            retries=0,
        )
        report(OK, "groq call", f"{settings.groq_model} -> {reply[:40]!r}")
    except Exception as exc:
        report(FAIL, "groq call", f"{type(exc).__name__}: {exc}")


def check_models() -> None:
    from medical_rag.config import get_settings
    from medical_rag.embeddings.encoder import DenseEncoder
    from medical_rag.retrieval.reranker import CrossEncoderReranker

    settings = get_settings()
    try:
        encoder = DenseEncoder(settings)
        vector = encoder.encode(["myocardial infarction management"])
        expected = settings.embedding_dim
        detail = f"{settings.embedding_model} -> dim {vector.shape[1]}"
        if vector.shape[1] != expected:
            report(
                WARN,
                "embedding model",
                detail + f" but EMBEDDING_DIM={expected}; update .env before ingesting",
            )
        else:
            report(OK, "embedding model", detail)
    except Exception as exc:
        report(FAIL, "embedding model", f"{type(exc).__name__}: {exc}")

    try:
        reranker = CrossEncoderReranker(settings)
        score = reranker.model.predict([("chest pain", "Myocardial infarction presents with ...")])
        report(OK, "reranker", f"{settings.reranker_model} -> score {float(score[0]):.3f}")
    except Exception as exc:
        report(FAIL, "reranker", f"{type(exc).__name__}: {exc}")


def check_corpus() -> None:
    from medical_rag.config import get_settings
    from medical_rag.ingest.pipeline import IngestionPipeline, MANIFEST_FILE

    settings = get_settings()
    raw = settings.raw_dir
    if not raw.exists():
        report(FAIL, "RAW_DATA", f"{raw} does not exist")
        return
    pdfs = list(raw.rglob("*.pdf"))
    if not pdfs:
        report(WARN, "RAW_DATA", f"{raw} contains no PDFs")
    else:
        size_mb = sum(p.stat().st_size for p in pdfs) / 1024**2
        report(OK, "RAW_DATA", f"{len(pdfs)} PDF(s), {size_mb:.0f} MB")

    manifest = settings.processed_dir / MANIFEST_FILE
    if manifest.exists():
        try:
            import json

            entries = json.loads(manifest.read_text(encoding="utf-8"))
            done = sum(1 for e in entries.values() if e.get("indexed_at"))
            report(OK, "manifest", f"{done}/{len(entries)} document(s) indexed")
        except Exception as exc:
            report(WARN, "manifest", str(exc))
    else:
        report(WARN, "manifest", "no ingestion run yet — python scripts/ingest.py --recreate")


def check_lexicon() -> None:
    from medical_rag.retrieval.query_expansion import load_lexicon

    try:
        lexicon = load_lexicon()
        report(OK, "lexicon", f"{len(lexicon)} layman->medical entries")
    except Exception as exc:
        report(FAIL, "lexicon", str(exc))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", action="store_true", help="Load the embedding and reranker models (downloads weights)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Medical RAG — setup check")
    print("=" * 70)

    check_dependencies()
    if failures:
        print("\nInstall dependencies first, then re-run.")
        return 1

    check_device()
    check_env()
    check_lexicon()
    check_corpus()
    check_qdrant()
    check_groq()
    if args.models:
        check_models()
    else:
        print("[SKIP] models — re-run with --models to download and verify weights")

    print("=" * 70)
    print(f"{failures} failure(s), {warnings} warning(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
