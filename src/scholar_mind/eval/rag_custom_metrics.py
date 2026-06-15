"""Custom deterministic metrics used alongside official RAGAS metrics."""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel

from scholar_mind.agents.common import try_structured_output
from scholar_mind.utils.text import cosine_similarity

RAG_SCORE_WEIGHTS = {
    "faithfulness": 0.22,
    "answer_relevancy": 0.12,
    "semantic_similarity": 0.08,
    "context_recall": 0.18,
    "context_precision": 0.13,
    "completeness": 0.13,
    "noise_resistance": 0.08,
    "redundancy_resistance": 0.06,
}

EMPTY_RETRIEVAL_DEFAULTS = {
    "faithfulness": 0.0,
    "context_recall": 0.0,
    "context_precision": 0.0,
    "completeness": 0.0,
    "noise_sensitivity": 1.0,
}


class RequiredPointCoverageOutput(BaseModel):
    covered: bool = False


class RequiredPointCoverageJudge:
    def __init__(self, llm):
        if llm is None:
            raise RuntimeError("LLM is required for completeness evaluation")
        self.llm = llm

    def covers_required_point(self, point: str, context_text: str) -> bool:
        prompt = (
            "Judge whether the retrieved context covers the required point for RAG "
            "evaluation. Return covered=true only when the context explicitly supports "
            "the point. Do not infer missing facts.\n\n"
            f"Required point:\n{point}\n\n"
            f"Retrieved context:\n{context_text}"
        )
        result = try_structured_output(self.llm, prompt, RequiredPointCoverageOutput)
        if result is None:
            raise RuntimeError("standard completeness judge failed")
        return bool(result.covered)


def normalize_context(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def compute_retrieval_latency(latency_ms: int | float | None) -> int:
    return max(int(latency_ms or 0), 0)


def extract_strategy(strategy: str) -> str:
    return strategy.strip() or "unknown"


def compute_redundancy(
    contexts: Sequence[str],
    chunk_ids: Sequence[str],
    *,
    embeddings: Sequence[Sequence[float]] | None,
    threshold: float,
) -> float:
    if not contexts:
        return 0.0
    if embeddings is None:
        raise ValueError("embeddings are required for redundancy")
    if len(embeddings) != len(contexts):
        raise ValueError("embeddings length must match contexts length")

    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    previous_vectors: list[list[float]] = []
    vectors = [_normalize_vector([float(value) for value in row]) for row in embeddings]
    redundant = 0

    for index, context in enumerate(contexts):
        chunk_id = chunk_ids[index] if index < len(chunk_ids) else ""
        normalized = normalize_context(context)
        vector = vectors[index]
        is_redundant = False
        if chunk_id and chunk_id in seen_ids:
            is_redundant = True
        elif normalized and normalized in seen_texts:
            is_redundant = True
        elif any(
            cosine_similarity(vector, previous) >= threshold for previous in previous_vectors
        ):
            is_redundant = True

        redundant += int(is_redundant)
        if chunk_id:
            seen_ids.add(chunk_id)
        if normalized:
            seen_texts.add(normalized)
        previous_vectors.append(vector)

    return round(redundant / len(contexts), 4)


def compute_completeness(
    required_points: Sequence[str],
    retrieved_contexts: Sequence[str],
    *,
    llm=None,
) -> float:
    points = [item.strip() for item in required_points if item and item.strip()]
    if not points:
        raise ValueError("required_points must be non-empty")
    if llm is None or not hasattr(llm, "covers_required_point"):
        raise ValueError("standard completeness judge is required")
    if not retrieved_contexts:
        return 0.0
    context_text = "\n".join(retrieved_contexts)
    covered = 0
    for point in points:
        if _point_covered(point, context_text, llm=llm):
            covered += 1
    return round(covered / len(points), 4)


def compute_rag_score(metrics: dict[str, float | int | None]) -> tuple[float | None, list[str]]:
    required = [
        "faithfulness",
        "answer_relevancy",
        "semantic_similarity",
        "context_recall",
        "context_precision",
        "completeness",
        "noise_sensitivity",
        "redundancy",
    ]
    missing = [field for field in required if metrics.get(field) is None]
    if missing:
        return None, missing

    values = {field: _clamp(float(metrics[field])) for field in required}
    score = (
        RAG_SCORE_WEIGHTS["faithfulness"] * values["faithfulness"]
        + RAG_SCORE_WEIGHTS["answer_relevancy"] * values["answer_relevancy"]
        + RAG_SCORE_WEIGHTS["semantic_similarity"] * values["semantic_similarity"]
        + RAG_SCORE_WEIGHTS["context_recall"] * values["context_recall"]
        + RAG_SCORE_WEIGHTS["context_precision"] * values["context_precision"]
        + RAG_SCORE_WEIGHTS["completeness"] * values["completeness"]
        + RAG_SCORE_WEIGHTS["noise_resistance"] * (1.0 - values["noise_sensitivity"])
        + RAG_SCORE_WEIGHTS["redundancy_resistance"] * (1.0 - values["redundancy"])
    )
    return round(_clamp(score), 4), []


def apply_empty_retrieval_defaults(
    metrics: dict[str, float | int | None],
    retrieved_contexts: Sequence[str],
) -> dict[str, float | int | None]:
    """Use explicit failure values when RAG was evaluated but retrieved no context."""
    if retrieved_contexts:
        return dict(metrics)

    adjusted = dict(metrics)
    for field, value in EMPTY_RETRIEVAL_DEFAULTS.items():
        if adjusted.get(field) is None:
            adjusted[field] = value
    return adjusted


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _point_covered(point: str, context_text: str, *, llm=None) -> bool:
    return bool(llm.covers_required_point(point, context_text))


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)
