from __future__ import annotations

from typing import Any

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - optional at runtime
    ChatOpenAI = None

from scholar_mind.config.settings import Settings
from scholar_mind.models.providers import build_provider_bundle
from scholar_mind.vector.embeddings import (
    EmbeddingService,
    OpenAICompatibleEmbeddingService,
    RemoteEmbeddingService,
)


def build_chat_models(settings: Settings) -> dict[str, Any]:
    providers = build_provider_bundle(settings)
    if ChatOpenAI is None:
        return {"reasoning": None, "light": None}
    if not providers.reasoning.enabled:
        return {"reasoning": None, "light": None}
    common = {
        "base_url": providers.reasoning.base_url,
        "api_key": providers.reasoning.api_key,
        "temperature": 0.1,
        "request_timeout": settings.llm_request_timeout_seconds,
        "max_retries": settings.llm_max_retries,
    }
    return {
        "reasoning": ChatOpenAI(model=providers.reasoning.model, **common),
        "light": ChatOpenAI(model=providers.light.model, **common),
    }


def build_embedding_service(settings: Settings) -> EmbeddingService:
    dimension = settings.resolved_embedding_dimension
    if settings.embedding_model in {"bge-m3", "baai/bge-m3"}:
        if not settings.embedding_base_url:
            raise RuntimeError(
                f"{settings.embedding_model} embedding requires SCHOLARMIND_EMBEDDING_BASE_URL"
            )
        return RemoteEmbeddingService(
            model=settings.embedding_model,
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            dimension=dimension,
        )
    if settings.embedding_model == "embedding-3":
        if not settings.embedding_base_url:
            raise RuntimeError("embedding-3 requires SCHOLARMIND_EMBEDDING_BASE_URL")
        if not settings.embedding_api_key:
            raise RuntimeError("embedding-3 requires SCHOLARMIND_EMBEDDING_API_KEY")
        return RemoteEmbeddingService(
            model=settings.embedding_model,
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            dimension=dimension,
            request_dimensions=dimension,
        )
    if settings.embedding_model == "text-embedding-3-small":
        base_url = settings.embedding_base_url or settings.llm_base_url
        api_key = settings.embedding_api_key or settings.llm_api_key
        if not base_url or not api_key:
            raise RuntimeError(
                "text-embedding-3-small requires embedding or LLM base URL and API key"
            )
        return OpenAICompatibleEmbeddingService(
            model=settings.embedding_model,
            base_url=base_url,
            api_key=api_key,
            dimension=dimension,
        )
    raise RuntimeError(f"Unsupported embedding model: {settings.embedding_model}")
