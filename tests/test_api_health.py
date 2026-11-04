from types import SimpleNamespace
from unittest.mock import Mock

from medical_rag.api import main


def test_health_uses_pipeline_qdrant_store(monkeypatch) -> None:
    settings = SimpleNamespace(
        resolve_device=lambda: "cpu",
        embedding_model="embedding-model",
        reranker_model="reranker-model",
        groq_model="llm-model",
        groq_api_key="configured",
        confidence_threshold=0.8,
    )
    store = SimpleNamespace(
        info=Mock(return_value={"exists": True, "collection": "active", "points": 10})
    )
    pipeline = SimpleNamespace(retriever=SimpleNamespace(store=store))
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "get_pipeline", lambda: pipeline)

    result = main.health()

    store.info.assert_called_once_with()
    assert result["status"] == "ok"
    assert result["qdrant"]["collection"] == "active"
