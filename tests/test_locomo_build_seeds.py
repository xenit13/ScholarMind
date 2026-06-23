from __future__ import annotations

from scholar_mind.eval.locomo_build.seeds import (
    CASES_PER_PERSONA,
    MEMORY_TYPES,
    PERSONAS,
    RESEARCH_TASKS,
    SESSIONS_PER_PERSONA,
    PaperRecord,
    build_persona_case_topic,
    get_distractor_case_id,
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


def test_distractor_case_cycles_within_persona():
    assert get_distractor_case_id("case_001") == "case_002"
    assert get_distractor_case_id("case_006") == "case_001"


def test_distractor_case_id_never_equals_case_id():
    for n in range(1, CASES_PER_PERSONA + 1):
        case_id = f"case_{n:03d}"
        assert get_distractor_case_id(case_id) != case_id


def test_case_topic_format():
    topic = build_persona_case_topic("cs.AI", "memory evaluation")
    assert topic == "cs.AI memory evaluation"


def test_paper_record_is_frozen_dataclass():
    record = PaperRecord(arxiv_id="2401.00001", title="Memory eval", category="cs.AI")
    assert record.arxiv_id == "2401.00001"
    assert record.title == "Memory eval"
    assert record.category == "cs.AI"
