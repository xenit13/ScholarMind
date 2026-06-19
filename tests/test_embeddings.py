from __future__ import annotations

import httpx

from scholar_mind.config.settings import Settings
from scholar_mind.models.factory import build_embedding_service
from scholar_mind.vector.embeddings import OpenAICompatibleEmbeddingService, RemoteEmbeddingService


def test_remote_embedding_service_uses_http_embeddings_payload(monkeypatch):
    def fake_post(url, json, headers, timeout):
        assert url == "http://embedding.local/v1/embeddings"
        assert json == {"model": "bge-m3", "input": ["query text", "doc text"]}
        assert headers["Content-Type"] == "application/json"
        assert timeout == 4.0
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={"model": "bge-m3", "embeddings": [[0.1, 0.2], [0.3, 0.4]]},
            request=request,
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    embedder = RemoteEmbeddingService(
        model="bge-m3",
        base_url="http://embedding.local/v1",
        dimension=2,
        timeout_seconds=4.0,
    )

    embeddings = embedder.embed_documents(["query text", "doc text"])

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]


def test_remote_embedding_service_includes_dimensions_when_configured(monkeypatch):
    def fake_post(url, json, headers, timeout):
        assert url == "https://open.bigmodel.cn/api/paas/v4/embeddings"
        assert json == {
            "model": "embedding-3",
            "input": ["query text"],
            "dimensions": 1024,
        }
        assert headers["Authorization"] == "Bearer secret"
        assert timeout == 20.0
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={"model": "embedding-3", "data": [{"embedding": [0.1, 0.2]}]},
            request=request,
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    embedder = RemoteEmbeddingService(
        model="embedding-3",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key="secret",
        dimension=1024,
        request_dimensions=1024,
    )

    embeddings = embedder.embed_documents(["query text"])

    assert embeddings == [[0.1, 0.2]]


def test_build_embedding_service_uses_bge_m3_remote_endpoint():
    settings = Settings(
        embedding_model="bge-m3",
        embedding_base_url="https://embedding.local/v1",
    )

    embedder = build_embedding_service(settings)

    assert isinstance(embedder, RemoteEmbeddingService)
    assert embedder.base_url == "https://embedding.local/v1"
    assert embedder.model == "bge-m3"
    assert embedder.dimension == 1024


def test_remote_embedding_service_omits_dimensions_for_openrouter_bge_m3(monkeypatch):
    def fake_post(url, json, headers, timeout):
        assert url == "https://openrouter.ai/api/v1/embeddings"
        assert json == {
            "model": "baai/bge-m3",
            "input": ["query text"],
        }
        assert "dimensions" not in json
        assert headers["Authorization"] == "Bearer secret"
        assert timeout == 20.0
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={"model": "baai/bge-m3", "data": [{"embedding": [0.1, 0.2]}]},
            request=request,
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    embedder = RemoteEmbeddingService(
        model="baai/bge-m3",
        base_url="https://openrouter.ai/api/v1",
        api_key="secret",
        dimension=1024,
    )

    embeddings = embedder.embed_documents(["query text"])

    assert embeddings == [[0.1, 0.2]]


def test_build_embedding_service_uses_openrouter_bge_m3_without_request_dimensions():
    settings = Settings(
        embedding_model="baai/bge-m3",
        embedding_base_url="https://openrouter.ai/api/v1",
        embedding_api_key="secret",
    )

    embedder = build_embedding_service(settings)

    assert isinstance(embedder, RemoteEmbeddingService)
    assert embedder.base_url == "https://openrouter.ai/api/v1"
    assert embedder.model == "baai/bge-m3"
    assert embedder.dimension == 1024
    assert embedder.request_dimensions is None


def test_build_embedding_service_uses_embedding_3_remote_endpoint():
    settings = Settings(
        embedding_model="embedding-3",
        embedding_base_url="https://open.bigmodel.cn/api/paas/v4",
        embedding_api_key="secret",
    )

    embedder = build_embedding_service(settings)

    assert isinstance(embedder, RemoteEmbeddingService)
    assert embedder.base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert embedder.model == "embedding-3"
    assert embedder.dimension == 1024
    assert embedder.request_dimensions == 1024


def test_build_embedding_service_uses_text_embedding_with_llm_credentials():
    settings = Settings(
        embedding_model="text-embedding-3-small",
        llm_base_url="https://api.openai-compatible.local/v1",
        llm_api_key="secret",
    )

    embedder = build_embedding_service(settings)

    assert isinstance(embedder, OpenAICompatibleEmbeddingService)
    assert embedder.dimension == 1536


def test_build_embedding_service_rejects_bge_m3_without_base_url():
    import pytest

    settings = Settings(embedding_model="bge-m3")

    with pytest.raises(RuntimeError, match="bge-m3 embedding requires"):
        build_embedding_service(settings)


def test_build_embedding_service_rejects_embedding_3_without_api_key():
    import pytest

    settings = Settings(
        embedding_model="embedding-3",
        embedding_base_url="https://open.bigmodel.cn/api/paas/v4",
    )

    with pytest.raises(RuntimeError, match="embedding-3 requires SCHOLARMIND_EMBEDDING_API_KEY"):
        build_embedding_service(settings)


def test_build_embedding_service_rejects_text_embedding_without_credentials(monkeypatch):
    import pytest

    monkeypatch.delenv("SCHOLARMIND_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("SCHOLARMIND_LLM_API_KEY", raising=False)
    monkeypatch.delenv("ZAI_BASE_URL", raising=False)
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.delenv("SCHOLARMIND_EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("SCHOLARMIND_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    settings = Settings(
        embedding_model="text-embedding-3-small",
        llm_base_url=None,
        llm_api_key=None,
        embedding_base_url=None,
        embedding_api_key=None,
    )

    with pytest.raises(RuntimeError, match="text-embedding-3-small requires"):
        build_embedding_service(settings)
