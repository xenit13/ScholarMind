from __future__ import annotations

import json

import pytest

from scholar_mind.eval.locomo_build.questions import (
    check_qa_distribution,
    generate_category_qas,
    is_answer_in_dialogue,
    parse_llm_qa_response,
    seed_ids_to_dia_ids,
)


def test_parse_llm_qa_response_accepts_clean_json():
    raw = (
        '[{"question":"q1","answer":"a1","evidence_seed_ids":["s1"],'
        '"case_id":"case_001","distractor_case_id":null,"template_id":"t1"}]'
    )
    parsed = parse_llm_qa_response(raw)
    assert len(parsed) == 1
    assert parsed[0]["question"] == "q1"


def test_parse_llm_qa_response_strips_markdown_fences():
    raw = (
        '```json\n[{"question":"q","answer":"a","evidence_seed_ids":["s"],'
        '"case_id":"c","distractor_case_id":null,"template_id":"t"}]\n```'
    )
    parsed = parse_llm_qa_response(raw)
    assert len(parsed) == 1


def test_parse_llm_qa_response_rejects_partial_payload():
    raw = '[{"question":"q1"}]'
    with pytest.raises(ValueError, match="missing keys"):
        parse_llm_qa_response(raw)


def test_seed_ids_to_dia_ids_translates_using_lookup():
    lookup = {
        "p01_case_001_paper_read": ["s1:3", "s1:5"],
        "p01_case_001_workflow": ["s1:7"],
    }
    evidence = seed_ids_to_dia_ids(["p01_case_001_paper_read"], lookup)
    assert evidence == ["s1:3", "s1:5"]


def test_seed_ids_to_dia_ids_raises_on_missing_seed():
    lookup = {"a": ["s1:1"]}
    with pytest.raises(ValueError, match="missing seed_id in dialogue lookup"):
        seed_ids_to_dia_ids(["b"], lookup)


def test_is_answer_in_dialogue_detects_exact_substring():
    dialogue_text = " ".join([
        "user mentioned this paper is the anchor paper for the project.",
    ])
    assert is_answer_in_dialogue("anchor paper", [dialogue_text]) is True


def test_is_answer_in_dialogue_case_insensitive():
    dialogue_text = "user: Anchor Paper is the role here."
    assert is_answer_in_dialogue("anchor paper", [dialogue_text]) is True


def test_is_answer_in_dialogue_returns_false_when_absent():
    assert is_answer_in_dialogue("nonexistent answer", ["random text"]) is False


def test_check_qa_distribution_passes_with_12_each():
    qas = []
    for cat in range(1, 6):
        for i in range(12):
            qas.append(
                {
                    "category": cat,
                    "metadata": {"template_id": f"t{i}"},
                    "answer": f"ans{i}",
                    "question": f"q{cat}-{i}",
                }
            )
    check_qa_distribution(qas, expected_per_category=12)


def test_check_qa_distribution_fails_when_undersized():
    qas = [
        {"category": 1, "question": "same q", "metadata": {"template_id": "t"}, "answer": "a"}
    ] * 5
    with pytest.raises(ValueError, match="expected 12 QAs for category 1, got 5"):
        check_qa_distribution(qas, expected_per_category=12)


def test_check_qa_distribution_fails_on_dup_question():
    qas = [
        {"category": 1, "question": "same q", "metadata": {"template_id": "t1"}, "answer": "a1"},
        {"category": 1, "question": "same q", "metadata": {"template_id": "t2"}, "answer": "a2"},
    ] + [
        {
            "category": c,
            "question": f"unique {c} {i}",
            "metadata": {"template_id": f"t{c}{i}"},
            "answer": "a",
        }
        for c in range(1, 6)
        for i in range(12)
        if not (c == 1 and i < 2)
    ]
    with pytest.raises(ValueError, match="duplicate question"):
        check_qa_distribution(qas, expected_per_category=12)


class _FakeChatModel:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls = 0

    def invoke(self, _prompt):
        idx = self.calls
        self.calls += 1
        return type("Resp", (), {"content": self.responses[idx]})()


def _fake_qa_payload(category: int, count: int = 12) -> str:
    items = [
        {
            "question": f"问题 {category} {i}",
            "answer": f"answer{i}" if category != 5 or i < 6 else "no information available",
            "evidence_seed_ids": ["p01_case_001_paper_read"],
            "case_id": "case_001",
            "distractor_case_id": "case_002" if category == 5 else None,
            "template_id": f"t{category}_{i}",
        }
        for i in range(count)
    ]
    return json.dumps(items)


def test_generate_category_qas_returns_list_of_dicts():
    fake = _FakeChatModel([_fake_qa_payload(1)])
    seeds_per_case = [
        {
            "case_id": "case_001",
            "case_topic": "cs.AI memory evaluation",
            "seeds": [
                {
                    "seed_id": "p01_case_001_paper_read",
                    "memory_type": "paper_read",
                    "content": {"role": "anchor paper", "paper_title": "SWE-chat"},
                }
            ],
        }
    ]
    out = generate_category_qas(
        chat_model=fake,
        persona_id="p01",
        category=1,
        seeds_per_case=seeds_per_case,
        max_retries=1,
    )
    assert len(out) == 12
    assert all(item["category"] == 1 for item in out)


def test_generate_category_qas_raises_when_distribution_invalid():
    fake = _FakeChatModel([_fake_qa_payload(1, count=5)])
    with pytest.raises(RuntimeError, match="QA generation failed"):
        generate_category_qas(
            chat_model=fake,
            persona_id="p01",
            category=1,
            seeds_per_case=[],
            max_retries=0,
        )
