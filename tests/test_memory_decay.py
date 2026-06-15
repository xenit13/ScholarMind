from __future__ import annotations

from datetime import UTC, datetime, timedelta

from scholar_mind.memory.decay import (
    MemoryScoreInput,
    access_boost,
    decay_factor,
    final_memory_score,
    rank_memory_candidates,
)
from scholar_mind.models.domain import StructuredMemoryRecord


def _record(
    memory_id: str,
    *,
    now: datetime,
    days_old: int = 0,
    importance: float = 0.6,
    decay_rate: float = 0.03,
    decay_floor: float = 0.3,
    access_count_30d: int = 0,
) -> StructuredMemoryRecord:
    created_at = now - timedelta(days=days_old)
    return StructuredMemoryRecord(
        memory_id=memory_id,
        user_id="u1",
        scope="user",
        memory_type="interaction_summary",
        content=f"记忆 {memory_id}",
        source="conversation",
        importance=importance,
        confidence=0.8,
        status="active",
        created_at=created_at,
        updated_at=created_at,
        decay_rate=decay_rate,
        decay_floor=decay_floor,
        access_count_30d=access_count_30d,
    )


def test_decay_factor_respects_rate_and_floor():
    now = datetime(2026, 5, 19, tzinfo=UTC)
    old = _record("old", now=now, days_old=120, decay_rate=0.06, decay_floor=0.2)
    fresh = _record("fresh", now=now, days_old=0, decay_rate=0.06, decay_floor=0.2)

    assert decay_factor(fresh, now=now) == 1.0
    assert decay_factor(old, now=now) == 0.2


def test_access_boost_grows_but_is_capped():
    now = datetime(2026, 5, 19, tzinfo=UTC)
    low = _record("low", now=now, access_count_30d=1)
    high = _record("high", now=now, access_count_30d=500)

    assert access_boost(low) > 1.0
    assert access_boost(high) == 1.5


def test_final_score_can_promote_recent_important_memory():
    now = datetime(2026, 5, 19, tzinfo=UTC)
    old_high_semantic = _record(
        "old",
        now=now,
        days_old=90,
        importance=0.2,
        decay_rate=0.06,
        decay_floor=0.2,
    )
    recent_important = _record(
        "recent",
        now=now,
        days_old=1,
        importance=0.95,
        decay_rate=0.01,
        decay_floor=0.5,
        access_count_30d=5,
    )

    ranked = rank_memory_candidates(
        [
            MemoryScoreInput(record=old_high_semantic, semantic_score=0.9),
            MemoryScoreInput(record=recent_important, semantic_score=0.72),
        ],
        now=now,
        top_k=2,
        min_final_score=0.0,
    )

    assert ranked[0].record.memory_id == "recent"
    assert ranked[0].final_score > ranked[1].final_score


def test_disabled_decay_uses_semantic_score_order():
    now = datetime(2026, 5, 19, tzinfo=UTC)
    old_high_semantic = _record("old", now=now, days_old=90, importance=0.2)
    recent_important = _record("recent", now=now, days_old=1, importance=0.95)

    ranked = rank_memory_candidates(
        [
            MemoryScoreInput(record=old_high_semantic, semantic_score=0.9),
            MemoryScoreInput(record=recent_important, semantic_score=0.72),
        ],
        now=now,
        top_k=2,
        min_final_score=0.0,
        enabled=False,
    )

    assert [item.record.memory_id for item in ranked] == ["old", "recent"]
    assert final_memory_score(old_high_semantic, 0.9, now=now, enabled=False).final_score == 0.9
