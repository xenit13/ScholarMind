"""Pydantic contracts for Document 23 RAG evaluation runs and results."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from scholar_mind.config.settings import get_settings
from scholar_mind.models.domain import RetrievalStrategyName

OFFICIAL_RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "noise_sensitivity",
    "semantic_similarity",
]

CUSTOM_RAG_METRICS = [
    "retrieval_latency",
    "strategy",
    "redundancy",
    "completeness",
    "rag_score",
]

DEFAULT_RAG_EVAL_METRICS = OFFICIAL_RAGAS_METRICS + CUSTOM_RAG_METRICS


def utcnow() -> datetime:
    return datetime.now(UTC)


def uid(prefix: str) -> str:
    return f"{prefix}{uuid4().hex}"


class RagEvalCase(BaseModel):
    case_id: str
    user_input: str
    reference: str
    required_points: list[str]
    tags: dict[str, Any] = Field(default_factory=dict)
    expected_source_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @field_validator("case_id", "user_input", "reference")
    @classmethod
    def _require_non_empty_string(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be non-empty")
        return value.strip()

    @field_validator("required_points")
    @classmethod
    def _require_points(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("required_points must be non-empty")
        return cleaned


class RagEvalRunRequest(BaseModel):
    dataset_name: str = Field(default_factory=lambda: get_settings().rag_eval_default_dataset)
    case_limit: int | None = Field(default=None, ge=1, le=1000)
    strategies: list[RetrievalStrategyName] = Field(
        default_factory=lambda: [
            RetrievalStrategyName.DENSE,
            RetrievalStrategyName.SPARSE,
            RetrievalStrategyName.HYBRID,
            RetrievalStrategyName.RERANKED_HYBRID,
        ]
    )
    metrics: list[str] = Field(default_factory=lambda: list(DEFAULT_RAG_EVAL_METRICS))
    top_k: int = Field(default=5, ge=1, le=50)

    @field_validator("dataset_name")
    @classmethod
    def _dataset_name_required(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("dataset_name must be non-empty")
        return value.strip()

    @model_validator(mode="after")
    def _validate_metrics(self):
        allowed = set(DEFAULT_RAG_EVAL_METRICS)
        unknown = sorted(set(self.metrics) - allowed)
        if unknown:
            raise ValueError(f"Unsupported RAG eval metrics: {', '.join(unknown)}")
        if not self.metrics:
            raise ValueError("metrics must be non-empty")
        if not self.strategies:
            raise ValueError("strategies must be non-empty")
        return self


class RagMetricStats(BaseModel):
    avg: float | None = None
    min: float | None = None
    max: float | None = None


class RagLatencyStats(BaseModel):
    avg: float | None = None
    p50: float | None = None
    p95: float | None = None
    max: int | None = None


class RagScoreStats(RagMetricStats):
    scored_count: int = 0
    missing_score_count: int = 0


class RagEvalAggregate(BaseModel):
    sample_count: int = 0
    faithfulness: RagMetricStats = Field(default_factory=RagMetricStats)
    answer_relevancy: RagMetricStats = Field(default_factory=RagMetricStats)
    context_precision: RagMetricStats = Field(default_factory=RagMetricStats)
    context_recall: RagMetricStats = Field(default_factory=RagMetricStats)
    noise_sensitivity: RagMetricStats = Field(default_factory=RagMetricStats)
    semantic_similarity: RagMetricStats = Field(default_factory=RagMetricStats)
    redundancy: RagMetricStats = Field(default_factory=RagMetricStats)
    completeness: RagMetricStats = Field(default_factory=RagMetricStats)
    rag_score: RagScoreStats = Field(default_factory=RagScoreStats)
    retrieval_latency_ms: RagLatencyStats = Field(default_factory=RagLatencyStats)


class RagEvalRunSummary(BaseModel):
    run_id: str
    dataset_name: str
    strategies: list[str]
    metrics: list[str]
    ragas_model: str
    embedding_model: str
    status: str
    sample_count: int
    started_at: datetime
    finished_at: datetime | None = None
    error_summary: str | None = None
    aggregates: dict[str, RagEvalAggregate] = Field(default_factory=dict)


class RagEvalResult(BaseModel):
    result_id: str = Field(default_factory=lambda: uid("ragres_"))
    run_id: str
    case_id: str
    strategy: str
    user_input: str
    response: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_contexts: list[str] = Field(default_factory=list)
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    noise_sensitivity: float | None = None
    semantic_similarity: float | None = None
    retrieval_latency_ms: int = 0
    redundancy: float = 0.0
    completeness: float | None = None
    rag_score: float | None = None
    metric_errors: dict[str, str] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=utcnow)
    created_at: datetime = Field(default_factory=utcnow)


class OfficialRagasScores(BaseModel):
    scores: dict[str, float | None] = Field(default_factory=dict)
    errors: dict[str, str] = Field(default_factory=dict)
