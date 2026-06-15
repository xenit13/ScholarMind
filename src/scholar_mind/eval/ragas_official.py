"""Thin adapter around official RAGAS metric classes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ragas.embeddings.base import BaseRagasEmbedding

from scholar_mind.models.rag_eval_models import OfficialRagasScores


class ProjectRagasEmbeddings(BaseRagasEmbedding):
    """Adapter from ScholarMind's embedding service to ragas 0.4 embeddings."""

    def __init__(self, embedding_service):
        super().__init__()
        self.embedding_service = embedding_service

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        return list(self.embedding_service.embed_query(text))

    def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return [list(item) for item in self.embedding_service.embed_documents(texts)]

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return list(await self.embedding_service.aembed_query(text))

    async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return [list(item) for item in await self.embedding_service.aembed_documents(texts)]


def build_ragas_llm(settings):
    """Build the InstructorLLM required by ragas collection metrics."""
    try:
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("openai and ragas are required for official RAG evaluation") from exc

    model = (
        settings.rag_eval_llm_model
        or getattr(settings, "llm_light_model", "")
        or getattr(settings, "llm_reasoning_model", "")
    )
    if not model:
        raise RuntimeError("RAGAS LLM model is required")
    if not settings.llm_api_key:
        raise RuntimeError("SCHOLARMIND_LLM_API_KEY is required for RAGAS metrics")

    client_kwargs: dict[str, Any] = {
        "api_key": settings.llm_api_key,
        "timeout": settings.llm_request_timeout_seconds,
        "max_retries": settings.llm_max_retries,
    }
    if settings.llm_base_url:
        client_kwargs["base_url"] = settings.llm_base_url

    client = AsyncOpenAI(**client_kwargs)
    return llm_factory(
        model,
        provider="openai",
        client=client,
        temperature=0.0,
        max_tokens=getattr(settings, "rag_eval_llm_max_tokens", 4096),
    )


class OfficialRagasEvaluator:
    def __init__(
        self,
        *,
        metrics: Mapping[str, Any] | None = None,
        llm=None,
        embeddings=None,
    ):
        self.llm = llm
        self.embeddings = embeddings
        self.metrics = dict(metrics) if metrics is not None else None

    async def score(
        self,
        *,
        user_input: str,
        response: str,
        reference: str,
        retrieved_contexts: list[str],
        metric_names: list[str],
    ) -> OfficialRagasScores:
        scores: dict[str, float | None] = {}
        errors: dict[str, str] = {}
        try:
            metrics = self._metrics()
        except Exception as exc:
            message = str(exc)
            return OfficialRagasScores(
                scores={name: None for name in metric_names},
                errors={name: message for name in metric_names},
            )
        payload: dict[str, Any] = {
            "user_input": user_input,
            "response": response,
            "reference": reference,
            "retrieved_contexts": retrieved_contexts,
        }
        for name in metric_names:
            metric = metrics.get(name)
            if metric is None:
                continue
            try:
                result = await metric.ascore(**_payload_for_metric(name, payload))
                value = getattr(result, "value", result)
                scores[name] = None if value is None else float(value)
            except Exception as exc:
                scores[name] = None
                errors[name] = str(exc)
        return OfficialRagasScores(scores=scores, errors=errors)

    def _metrics(self) -> dict[str, Any]:
        if self.metrics is None:
            self.metrics = self._build_metrics(self.llm, self.embeddings)
        return self.metrics

    @staticmethod
    def _build_metrics(llm, embeddings) -> dict[str, Any]:
        try:
            from ragas.metrics.collections import (  # type: ignore
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                Faithfulness,
                NoiseSensitivity,
                SemanticSimilarity,
            )
        except ImportError as exc:  # pragma: no cover - exercised when dependency missing
            raise RuntimeError("ragas is required for official RAG evaluation") from exc

        return {
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
            "context_precision": ContextPrecision(llm=llm),
            "context_recall": ContextRecall(llm=llm),
            "noise_sensitivity": NoiseSensitivity(llm=llm),
            "semantic_similarity": SemanticSimilarity(embeddings=embeddings),
        }


def _payload_for_metric(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    keys_by_metric = {
        "faithfulness": ("user_input", "response", "retrieved_contexts"),
        "answer_relevancy": ("user_input", "response"),
        "context_precision": ("user_input", "reference", "retrieved_contexts"),
        "context_recall": ("user_input", "retrieved_contexts", "reference"),
        "noise_sensitivity": ("user_input", "response", "reference", "retrieved_contexts"),
        "semantic_similarity": ("reference", "response"),
    }
    keys = keys_by_metric.get(name, tuple(payload))
    return {key: payload[key] for key in keys}
