from __future__ import annotations

import dataclasses
import random

import pytest

from scholar_mind.eval.locomo_build.seeds import (
    CASES_PER_PERSONA,
    MEMORY_TYPES,
    PAPER_CATEGORIES,
    PERSONAS,
    RESEARCH_TASKS,
    SEEDS_PER_PERSONA,
    SESSIONS_PER_PERSONA,
    PaperRecord,
    build_persona_case_topic,
    get_distractor_case_id,
    sample_papers_for_persona,
)


def test_five_personas_defined():
    assert len(PERSONAS) == 5
    assert {p.persona_id for p in PERSONAS} == {"p01", "p02", "p03", "p04", "p05"}


def test_personas_have_distinct_backgrounds():
    backgrounds = {p.background for p in PERSONAS}
    assert len(backgrounds) == 5


def test_six_memory_types():
    assert set(MEMORY_TYPES) == {
        "paper_read",
        "workflow",
        "preference",
        "feedback",
        "knowledge_level",
        "project_constraint",
    }


def test_six_research_tasks_cover_core_capabilities():
    assert len(RESEARCH_TASKS) == 6
    for task in RESEARCH_TASKS:
        assert isinstance(task, str) and task


def test_cases_per_persona_is_six():
    assert CASES_PER_PERSONA == 6


def test_sessions_per_persona_is_six():
    assert SESSIONS_PER_PERSONA == 6


def test_seeds_per_persona_value():
    assert SEEDS_PER_PERSONA == 36


def test_seeds_per_persona_relationship():
    assert SEEDS_PER_PERSONA == CASES_PER_PERSONA * len(MEMORY_TYPES)


def test_distractor_case_cycles_within_persona():
    assert get_distractor_case_id("case_001") == "case_002"
    assert get_distractor_case_id("case_006") == "case_001"


def test_distractor_case_id_never_equals_case_id():
    for n in range(1, CASES_PER_PERSONA + 1):
        case_id = f"case_{n:03d}"
        assert get_distractor_case_id(case_id) != case_id


def test_get_distractor_case_id_rejects_case_000():
    with pytest.raises(ValueError, match="out of range"):
        get_distractor_case_id("case_000")


def test_get_distractor_case_id_rejects_case_007():
    with pytest.raises(ValueError, match="out of range"):
        get_distractor_case_id("case_007")


def test_get_distractor_case_id_rejects_bad_prefix():
    with pytest.raises(ValueError, match="unexpected case_id format"):
        get_distractor_case_id("foo_001")


def test_get_distractor_case_id_rejects_wrong_case():
    with pytest.raises(ValueError, match="unexpected case_id format"):
        get_distractor_case_id("Case_001")


def test_case_topic_format():
    topic = build_persona_case_topic("cs.AI", "memory evaluation")
    assert topic == "cs.AI memory evaluation"


def test_paper_record_is_frozen_dataclass():
    record = PaperRecord(arxiv_id="2401.00001", title="Memory eval", category="cs.AI")
    assert record.arxiv_id == "2401.00001"
    assert record.title == "Memory eval"
    assert record.category == "cs.AI"
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.arxiv_id = "x"


def _make_fake_papers() -> list[PaperRecord]:
    out: list[PaperRecord] = []
    for cat_idx, cat in enumerate(PAPER_CATEGORIES):
        # 30 per category supports 5 personas × 5 per category with cross-persona
        # distinctness (5 × 5 = 25 ≤ 30).
        for n in range(30):
            out.append(
                PaperRecord(
                    arxiv_id=f"2604.{cat_idx}{n:03d}",
                    title=f"Sample paper {cat}-{n}",
                    category=cat,
                )
            )
    return out


def test_sample_papers_returns_distinct_papers():
    rng = random.Random(42)
    pool = _make_fake_papers()
    chosen = sample_papers_for_persona(pool, "p01", papers_needed=30, rng=rng)
    assert len(chosen) == 30
    arxiv_ids = [p.arxiv_id for p in chosen]
    assert len(set(arxiv_ids)) == 30


def test_sample_papers_covers_all_categories():
    rng = random.Random(42)
    pool = _make_fake_papers()
    chosen = sample_papers_for_persona(pool, "p01", papers_needed=30, rng=rng)
    cats = {p.category for p in chosen}
    assert cats == set(PAPER_CATEGORIES)


def test_sample_papers_does_not_repeat_across_personas():
    rng = random.Random(42)
    pool = _make_fake_papers()
    used: set[str] = set()
    for persona_id in ("p01", "p02", "p03", "p04", "p05"):
        chosen = sample_papers_for_persona(
            pool, persona_id, papers_needed=30, rng=rng, used_arxiv_ids=used
        )
        for p in chosen:
            assert p.arxiv_id not in used
            used.add(p.arxiv_id)


def test_sample_papers_raises_when_pool_too_small():
    rng = random.Random(42)
    pool = _make_fake_papers()[:5]

    with pytest.raises(ValueError, match="paper pool exhausted"):
        sample_papers_for_persona(pool, "p01", papers_needed=30, rng=rng)


def test_sample_papers_raises_when_pool_exhausted_mid_iteration():
    """Pool with enough cs.AI/cs.CL papers but nothing else should fail on cs.CV."""
    rng = random.Random(42)
    pool = [
        PaperRecord(arxiv_id=f"2604.{cat_idx}000", title=f"t{i}", category=cat)
        for cat_idx, cat in enumerate(PAPER_CATEGORIES)
        for i in range(5)
        if cat in ("cs.AI", "cs.CL")  # only 2 of 6 categories have any papers
    ]
    with pytest.raises(ValueError, match="paper pool exhausted"):
        sample_papers_for_persona(pool, "p01", papers_needed=30, rng=rng)
