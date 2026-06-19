from __future__ import annotations

from pathlib import Path

import anyio
import pytest
from langchain_core.messages import AIMessage

import scholar_mind.app as app_module
from scholar_mind.app import get_container
from scholar_mind.config.settings import get_settings


class _TestEmbeddingService:
    dimension = 16

    @staticmethod
    def _vectorize(text: str) -> list[float]:
        vector = [0.0] * 16
        for token in text.lower().split():
            vector[hash(token) % 16] += 1.0
        return vector

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


class _FakeStructuredRunnable:
    def invoke(self, _prompt):
        return {
            "parsed": None,
            "raw": AIMessage(content=""),
            "parsing_error": None,
        }


class _FakeChatModel:
    def bind_tools(self, _tools):
        return self

    def with_structured_output(self, _schema, include_raw: bool = False):
        assert include_raw is True
        return _FakeStructuredRunnable()

    def invoke(self, _prompt):
        return AIMessage(content="已生成一份简明结果。")


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.delenv("ZAI_BASE_URL", raising=False)
    monkeypatch.delenv("SCHOLARMIND_LLM_API_KEY", raising=False)
    monkeypatch.delenv("SCHOLARMIND_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("SCHOLARMIND_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("SCHOLARMIND_EMBEDDING_BASE_URL", raising=False)
    monkeypatch.setenv("SCHOLARMIND_ENVIRONMENT", "test")
    monkeypatch.setenv("SCHOLARMIND_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv(
        "SCHOLARMIND_CHECKPOINT_DATABASE_URL",
        f"sqlite:///{tmp_path / 'checkpoints.db'}",
    )
    monkeypatch.setenv("SCHOLARMIND_QDRANT_LOCATION", ":memory:")
    monkeypatch.setenv("SCHOLARMIND_LOG_DIR", str(tmp_path / "message_logs"))
    monkeypatch.setenv("SCHOLARMIND_MEMORY_ROOT_DIR", str(tmp_path / "memory"))
    monkeypatch.setenv("SCHOLARMIND_EVAL_ROOT_DIR", str(tmp_path / "eval"))
    monkeypatch.setattr(
        app_module,
        "build_chat_models",
        lambda _settings: {"reasoning": _FakeChatModel(), "light": _FakeChatModel()},
    )
    monkeypatch.setattr(
        app_module,
        "build_embedding_service",
        lambda _settings: _TestEmbeddingService(),
    )
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    get_settings.cache_clear()
    get_container.cache_clear()
    yield
    if get_container.cache_info().currsize:
        anyio.run(get_container().aclose)
    get_settings.cache_clear()
    get_container.cache_clear()
