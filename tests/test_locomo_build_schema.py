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


def test_seed_requires_content_or_temporal():
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
