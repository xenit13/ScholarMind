from __future__ import annotations

import pytest
from pydantic import ValidationError

from scholar_mind.eval.locomo_build.schema import (
    QA,
    Conversation,
    PaperRef,
    Persona,
    Sample,
    Seed,
    Turn,
)


def test_turn_with_seed_id_round_trip():
    turn = Turn(
        speaker="user",
        dia_id="s1:3",
        text="顺便提一下，这篇论文我标为 anchor paper。",
        metadata={
            "seed_id": "p01_case_001_paper_read",
            "memory_type": "paper_read",
            "is_distractor": False,
        },
    )
    assert turn.speaker == "user"
    assert turn.metadata["seed_id"] == "p01_case_001_paper_read"


def test_distractor_turn_allows_null_seed():
    turn = Turn(
        speaker="assistant",
        dia_id="s1:4",
        text="好的，明白了。",
        metadata={"seed_id": None, "memory_type": None, "is_distractor": True},
    )
    assert turn.metadata["is_distractor"] is True


def test_invalid_speaker_rejected():
    with pytest.raises(ValidationError):
        Turn(
            speaker="narrator",
            dia_id="s1:1",
            text="...",
            metadata={"seed_id": None, "memory_type": None, "is_distractor": True},
        )


def test_seed_construction_with_content_only():
    Seed(
        seed_id="p01_case_001_paper_read",
        persona_id="p01",
        case_id="case_001",
        case_topic="cs.AI memory evaluation",
        papers=[PaperRef(arxiv_id="2604.20779", title="SWE-chat", category="cs.AI")],
        memory_type="paper_read",
        content={"role": "anchor paper"},
        temporal=None,
        distractor_case_id="case_002",
    )


def test_qa_minimal_fields():
    qa = QA(
        question="这个项目的 paper role 是什么?",
        answer="anchor paper",
        category=1,
        evidence=["s1:3"],
        metadata={
            "question_kind": "memory_single_hop",
            "template_id": "single_hop_role",
            "memory_focus": ["paper_read"],
            "case_id": "case_001",
            "distractor_case_id": None,
        },
    )
    assert qa.category == 1


def test_qa_adversarial_uses_adversarial_answer_field():
    qa = QA(
        question="是否为论文 X 设置过 Y 标签?",
        answer="no information available",
        category=5,
        evidence=["s1:3"],
        metadata={
            "question_kind": "memory_adversarial",
            "template_id": "adversarial_cross_paper",
            "memory_focus": ["confusable_memory"],
            "case_id": "case_001",
            "distractor_case_id": "case_002",
        },
    )
    assert qa.category == 5
    assert qa.metadata["distractor_case_id"] == "case_002"


def test_sample_round_trip():
    persona = Persona(
        persona_id="p01",
        user_id="locomo_v2_p01",
        background="ML 工程师,强工程弱理论",
    )
    conversation = Conversation(
        speaker_a="user",
        speaker_b="assistant",
        session_1_date_time="2026-05-03",
        session_1=[],
    )
    sample = Sample(
        sample_id="scholarmind_locomo_v2_p01",
        persona=persona,
        conversation=conversation,
        qa=[],
    )
    assert sample.sample_id == "scholarmind_locomo_v2_p01"
    assert sample.conversation.session_1 == []


def test_sample_serializes_to_locomo_json_shape():
    sample = Sample(
        sample_id="scholarmind_locomo_v2_p01",
        persona=Persona(persona_id="p01", user_id="u", background="b"),
        conversation=Conversation(
            speaker_a="user",
            speaker_b="assistant",
            session_1_date_time="2026-05-03",
            session_1=[
                Turn(
                    speaker="user",
                    dia_id="s1:1",
                    text="hi",
                    metadata={"seed_id": None, "memory_type": None, "is_distractor": True},
                )
            ],
        ),
        qa=[
            QA(
                question="q",
                answer="a",
                category=1,
                evidence=["s1:1"],
                metadata={
                    "question_kind": "memory_single_hop",
                    "template_id": "t",
                    "memory_focus": ["paper_read"],
                    "case_id": "case_001",
                    "distractor_case_id": None,
                },
            )
        ],
    )
    payload = sample.model_dump()
    assert payload["conversation"]["speaker_a"] == "user"
    assert payload["qa"][0]["category"] == 1


def test_conversation_supports_session_2_through_6_extension():
    conversation = Conversation(
        speaker_a="user",
        speaker_b="assistant",
        session_1_date_time="2026-05-03",
        session_1=[],
        session_2_date_time="2026-05-06",
        session_2=[Turn(speaker="user", dia_id="s2:1", text="x")],
        session_3_date_time="2026-05-09",
        session_3=[Turn(speaker="assistant", dia_id="s3:1", text="y")],
        session_4_date_time="2026-05-12",
        session_4=[],
        session_5_date_time="2026-05-15",
        session_5=[Turn(speaker="user", dia_id="s5:1", text="z")],
        session_6_date_time="2026-05-18",
        session_6=[],
    )
    payload = conversation.model_dump()
    for n in range(1, 7):
        assert f"session_{n}" in payload
        assert f"session_{n}_date_time" in payload
        assert isinstance(payload[f"session_{n}"], list)


def test_conversation_rejects_invalid_speaker_in_session_2():
    with pytest.raises(ValidationError):
        Conversation(
            speaker_a="user",
            speaker_b="assistant",
            session_1_date_time="2026-05-03",
            session_2=[{"speaker": "narrator", "dia_id": "s2:1", "text": "x", "metadata": {}}],
        )


def test_conversation_rejects_unknown_extra_field():
    with pytest.raises(ValidationError):
        Conversation(
            speaker_a="user",
            speaker_b="assistant",
            session_1_date_time="2026-05-03",
            random_garbage="x",
        )


def test_paper_ref_rejects_extra_field():
    with pytest.raises(ValidationError):
        PaperRef(arxiv_id="x", title="y", category="z", oops=1)


def test_seed_rejects_invalid_memory_type():
    with pytest.raises(ValidationError):
        Seed(
            seed_id="p01_case_001_paper_read",
            persona_id="p01",
            case_id="case_001",
            case_topic="cs.AI memory evaluation",
            papers=[PaperRef(arxiv_id="2604.20779", title="SWE-chat", category="cs.AI")],
            memory_type="typo",
            content={"role": "anchor paper"},
            temporal=None,
            distractor_case_id="case_002",
        )


def test_turn_metadata_default_factory_populates_keys():
    turn = Turn(speaker="user", dia_id="s1:1", text="x")
    assert turn.metadata["seed_id"] is None
    assert turn.metadata["is_distractor"] is True


def test_persona_rejects_extra_field():
    with pytest.raises(ValidationError):
        Persona(
            persona_id="p01",
            user_id="u",
            background="b",
            oops=1,
        )


def test_qa_rejects_invalid_category():
    base = {
        "question": "q",
        "answer": "a",
        "evidence": ["s1:1"],
        "metadata": {"question_kind": "memory_single_hop"},
    }
    with pytest.raises(ValidationError):
        QA(**base, category=0)
    with pytest.raises(ValidationError):
        QA(**base, category=6)
