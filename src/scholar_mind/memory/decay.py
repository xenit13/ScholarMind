from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

from scholar_mind.models.domain import MemoryType, StructuredMemoryRecord


@dataclass(frozen=True)
class MemoryScoreInput:
    record: StructuredMemoryRecord
    semantic_score: float


@dataclass(frozen=True)
class MemoryScoreBreakdown:
    record: StructuredMemoryRecord
    semantic_score: float
    importance: float
    decay_factor: float
    access_boost: float
    final_score: float

    def to_trace_dict(self) -> dict[str, object]:
        return {
            "memory_id": self.record.memory_id,
            "semantic_score": round(float(self.semantic_score), 4),
            "importance": round(float(self.importance), 4),
            "decay_factor": round(float(self.decay_factor), 4),
            "access_boost": round(float(self.access_boost), 4),
            "final_score": round(float(self.final_score), 4),
            "status": self.record.status.value
            if hasattr(self.record.status, "value")
            else str(self.record.status),
        }


def decay_factor(record: StructuredMemoryRecord, *, now: datetime | None = None) -> float:
    current = _aware(now or datetime.now(UTC))
    access_anchor = (
        _aware(record.last_accessed_at)
        if record.last_accessed_at is not None
        else _aware(record.created_at)
    )
    anchor = max(_aware(record.created_at), _aware(record.updated_at), access_anchor)
    days = max(0.0, (current - anchor).total_seconds() / 86400)
    return max(float(record.decay_floor), math.exp(-float(record.decay_rate) * days))


def access_boost(
    record: StructuredMemoryRecord,
    *,
    boost_factor: float = 0.2,
    cap: float = 1.5,
) -> float:
    raw = 1 + float(boost_factor) * math.log(1 + max(0, int(record.access_count_30d)))
    return min(float(cap), raw)


def final_memory_score(
    record: StructuredMemoryRecord,
    semantic_score: float,
    *,
    now: datetime | None = None,
    enabled: bool = True,
    access_boost_factor: float = 0.2,
    access_boost_cap: float = 1.5,
) -> MemoryScoreBreakdown:
    if not enabled:
        return MemoryScoreBreakdown(
            record=record,
            semantic_score=float(semantic_score),
            importance=float(record.importance),
            decay_factor=1.0,
            access_boost=1.0,
            final_score=float(semantic_score),
        )
    record_decay = decay_factor(record, now=now)
    record_access_boost = access_boost(
        record,
        boost_factor=access_boost_factor,
        cap=access_boost_cap,
    )
    final_score = (
        float(semantic_score)
        * (0.65 + 0.35 * float(record.importance))
        * record_decay
        * record_access_boost
    )
    return MemoryScoreBreakdown(
        record=record,
        semantic_score=float(semantic_score),
        importance=float(record.importance),
        decay_factor=record_decay,
        access_boost=record_access_boost,
        final_score=final_score,
    )


def rank_memory_candidates(
    candidates: list[MemoryScoreInput],
    *,
    now: datetime | None = None,
    top_k: int,
    min_final_score: float,
    enabled: bool = True,
    access_boost_factor: float = 0.2,
    access_boost_cap: float = 1.5,
) -> list[MemoryScoreBreakdown]:
    scored = [
        final_memory_score(
            item.record,
            item.semantic_score,
            now=now,
            enabled=enabled,
            access_boost_factor=access_boost_factor,
            access_boost_cap=access_boost_cap,
        )
        for item in candidates
    ]
    filtered = [item for item in scored if item.final_score >= min_final_score]
    return sorted(filtered, key=lambda item: item.final_score, reverse=True)[:top_k]


def archive_score(record: StructuredMemoryRecord, *, now: datetime | None = None) -> float:
    return float(record.importance) * decay_factor(record, now=now)


def default_decay_parameters(memory_type: str | MemoryType, source: str) -> tuple[float, float]:
    memory_type_value = memory_type.value if hasattr(memory_type, "value") else str(memory_type)
    if source == "explicit" and memory_type_value in {
        MemoryType.PREFERENCE.value,
        MemoryType.PROJECT_CONSTRAINT.value,
    }:
        return 0.005, 0.7
    if memory_type_value in {
        MemoryType.PREFERENCE.value,
        MemoryType.PROJECT_CONSTRAINT.value,
        MemoryType.KNOWLEDGE_LEVEL.value,
    }:
        return 0.01, 0.5
    if memory_type_value in {
        MemoryType.RESEARCH_INTEREST.value,
        MemoryType.GOAL.value,
        MemoryType.WORKFLOW.value,
        MemoryType.PAPER_READ.value,
    }:
        return 0.03, 0.3
    return 0.06, 0.2


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
