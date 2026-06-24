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
    build_all_seeds,
    build_persona_case_topic,
    build_seeds_for_persona,
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


def test_build_seeds_for_persona_returns_36_seeds():
    rng = random.Random(42)
    pool = _make_fake_papers()
    persona = PERSONAS[0]
    seeds = build_seeds_for_persona(persona, pool, rng=rng)
    assert len(seeds) == 36


def test_each_case_has_six_memory_types():
    rng = random.Random(42)
    pool = _make_fake_papers()
    seeds = build_seeds_for_persona(PERSONAS[0], pool, rng=rng)
    by_case: dict[str, set[str]] = {}
    for seed in seeds:
        by_case.setdefault(seed.case_id, set()).add(seed.memory_type)
    assert len(by_case) == 6
    for case_id, types in by_case.items():
        missing = set(MEMORY_TYPES) - types
        assert types == set(MEMORY_TYPES), f"case {case_id} missing types: {missing}"


def test_distractor_case_id_set_on_every_seed():
    rng = random.Random(42)
    pool = _make_fake_papers()
    seeds = build_seeds_for_persona(PERSONAS[0], pool, rng=rng)
    for seed in seeds:
        assert seed.distractor_case_id
        assert seed.distractor_case_id != seed.case_id


def test_temporal_seeds_have_consistent_dates():
    rng = random.Random(42)
    pool = _make_fake_papers()
    seeds = build_seeds_for_persona(PERSONAS[0], pool, rng=rng)
    temporal_seeds = [s for s in seeds if s.temporal is not None]
    for s in temporal_seeds:
        assert s.temporal.old_date < s.temporal.new_date


def test_maybe_temporal_returns_update_when_rng_forces_it():
    """Force temporal=True via rng that returns 0.0 from rng.random()."""
    from scholar_mind.eval.locomo_build.seeds import _maybe_temporal

    rng = random.Random(0)
    # Force rng.random() to return 0.0 (≤ TEMPORAL_FRACTION) by overriding
    rng.random = lambda: 0.0  # type: ignore[assignment]
    content = {"default_depth": "survey-first overview"}
    result = _maybe_temporal("preference", content, rng, case_index=1)
    assert result is not None
    assert result.old_date < result.new_date
    assert result.new == content
    assert result.old != content  # alt content must differ


def test_maybe_temporal_returns_none_for_non_temporal_types():
    from scholar_mind.eval.locomo_build.seeds import _maybe_temporal

    rng = random.Random(0)
    rng.random = lambda: 0.0  # type: ignore[assignment]
    for memory_type in ("paper_read", "workflow", "knowledge_level", "project_constraint"):
        assert _maybe_temporal(memory_type, {}, rng, case_index=1) is None


def test_maybe_temporal_returns_none_when_rng_above_threshold():
    from scholar_mind.eval.locomo_build.seeds import _maybe_temporal

    rng = random.Random(0)
    rng.random = lambda: 0.99  # type: ignore[assignment]
    assert _maybe_temporal("preference", {"default_depth": "x"}, rng, case_index=1) is None


def test_temporal_seeds_only_for_preference_or_feedback():
    rng = random.Random(42)
    pool = _make_fake_papers()
    seeds = build_seeds_for_persona(PERSONAS[0], pool, rng=rng)
    for s in seeds:
        if s.temporal is not None:
            assert s.memory_type in {"preference", "feedback"}


def test_build_all_seeds_yields_5_personas_x_36():
    rng = random.Random(42)
    pool = _make_fake_papers()
    by_persona = build_all_seeds(pool, rng=rng)
    assert set(by_persona.keys()) == {p.persona_id for p in PERSONAS}
    for persona_id, seeds in by_persona.items():
        assert len(seeds) == 36, f"{persona_id} has {len(seeds)} seeds"
        for seed in seeds:
            assert seed.persona_id == persona_id


def test_papers_do_not_repeat_across_personas_in_build_all():
    rng = random.Random(42)
    pool = _make_fake_papers()
    by_persona = build_all_seeds(pool, rng=rng)
    seen: set[str] = set()
    for _persona_id, seeds in by_persona.items():
        persona_paper_ids: set[str] = set()
        for seed in seeds:
            for paper in seed.papers:
                persona_paper_ids.add(paper.arxiv_id)
        for arxiv_id in persona_paper_ids:
            assert arxiv_id not in seen, (
                f"paper {arxiv_id} reused across personas"
            )
            seen.add(arxiv_id)


def test_knowledge_level_seed_uses_persona_background():
    """M1 regression: knowledge_level content must reflect persona's actual background."""
    rng = random.Random(42)
    pool = _make_fake_papers()
    for persona in PERSONAS:
        seeds = build_seeds_for_persona(persona, pool, rng=rng)
        kl_seeds = [s for s in seeds if s.memory_type == "knowledge_level"]
        assert len(kl_seeds) == 6  # one per case
        for s in kl_seeds:
            assert s.content["background"] == persona.background, (
                f"{persona.persona_id} knowledge_level seed has wrong background"
            )


# ---------------------------------------------------------------------------
# Task 5: load_paper_pool
# ---------------------------------------------------------------------------

from datetime import date  # noqa: E402

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from scholar_mind.db.models import Base, PaperModel  # noqa: E402
from scholar_mind.eval.locomo_build.seeds import load_paper_pool  # noqa: E402


def _seed_sqlite_with_papers(tmp_path, count_per_cat=20):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    cats = PAPER_CATEGORIES
    with Session() as session:
        for cat_idx, cat in enumerate(cats):
            for n in range(count_per_cat):
                session.add(
                    PaperModel(
                        paper_id=f"2604.{cat_idx}{n:03d}",
                        title=f"Test paper {cat}-{n}",
                        authors_json="[]",
                        abstract="...",
                        categories_json=f'["{cat}"]',
                        publish_date=date(2026, 4, 1),
                        citation_count=0,
                        has_source=True,
                    )
                )
        session.commit()
    return db_path


def test_load_paper_pool_returns_records(tmp_path):
    db_path = _seed_sqlite_with_papers(tmp_path)
    pool = load_paper_pool(f"sqlite:///{db_path}")
    assert len(pool) == 120  # 6 cats * 20 papers
    cats = {p.category for p in pool}
    assert cats == set(PAPER_CATEGORIES)


def test_load_paper_pool_filters_to_six_categories(tmp_path):
    db_path = _seed_sqlite_with_papers(tmp_path, count_per_cat=20)
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.add(
            PaperModel(
                paper_id="2604.9999",
                title="Out-of-scope paper",
                authors_json="[]",
                abstract="...",
                categories_json='["physics.gen-ph"]',
                publish_date=date(2026, 4, 1),
                citation_count=0,
                has_source=True,
            )
        )
        session.commit()
    pool = load_paper_pool(f"sqlite:///{db_path}")
    assert all(p.category in set(PAPER_CATEGORIES) for p in pool)


# ---------------------------------------------------------------------------
# Task 6: write_seeds_json (Stage 1 serialization)
# ---------------------------------------------------------------------------

import json  # noqa: E402

from scholar_mind.eval.locomo_build.seeds import write_seeds_json  # noqa: E402


def test_write_seeds_json_roundtrip(tmp_path):
    rng = random.Random(42)
    pool = _make_fake_papers()
    by_persona = build_all_seeds(pool, rng=rng)
    out_file = tmp_path / "seeds.json"
    write_seeds_json(by_persona, out_file)
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {"p01", "p02", "p03", "p04", "p05"}
    assert len(payload["p01"]) == 36
    first = payload["p01"][0]
    for key in (
        "seed_id",
        "persona_id",
        "case_id",
        "case_topic",
        "papers",
        "memory_type",
        "content",
        "temporal",
        "distractor_case_id",
    ):
        assert key in first


def test_stage1_byte_identical_on_rerun_with_same_seed(tmp_path):
    """Spec § 9 success criterion 7: Stage 1 must be byte-identical on rerun."""
    pool = _make_fake_papers()
    out_a = tmp_path / "seeds_a.json"
    out_b = tmp_path / "seeds_b.json"
    write_seeds_json(build_all_seeds(pool, rng=random.Random(42)), out_a)
    write_seeds_json(build_all_seeds(pool, rng=random.Random(42)), out_b)
    assert out_a.read_bytes() == out_b.read_bytes()


def test_stage1_differs_with_different_seed(tmp_path):
    """Different seeds should (very likely) produce different output."""
    pool = _make_fake_papers()
    out_a = tmp_path / "seeds_a.json"
    out_b = tmp_path / "seeds_b.json"
    write_seeds_json(build_all_seeds(pool, rng=random.Random(42)), out_a)
    write_seeds_json(build_all_seeds(pool, rng=random.Random(7)), out_b)
    assert out_a.read_bytes() != out_b.read_bytes()
