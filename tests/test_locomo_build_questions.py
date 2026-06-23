from __future__ import annotations

import json

import pytest

from scholar_mind.eval.locomo_build.questions import (
    build_persona_qas,
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
    # Long answer (>6 words) — verbatim copy should be detected as leak
    long_answer = "the anchor paper for the project is swe chat coding agents"
    dialogue_text = (
        "user mentioned that the anchor paper for the project is swe chat coding agents "
        "because it has good method assumptions."
    )
    assert is_answer_in_dialogue(long_answer, [dialogue_text]) is True


def test_is_answer_in_dialogue_case_insensitive():
    long_answer = "The Anchor Paper For The Project Is SWE Chat"
    dialogue_text = (
        "user: the anchor paper for the project is swe chat is what we discussed."
    )
    assert is_answer_in_dialogue(long_answer, [dialogue_text]) is True


def test_is_answer_in_dialogue_exempts_short_answers():
    """Short answers (≤6 words) like paper titles are exempt — they legitimately appear in both."""
    assert is_answer_in_dialogue("anchor paper", ["the anchor paper is here"]) is False
    assert is_answer_in_dialogue("method assumptions failure modes", ["method assumptions failure modes"]) is False


def test_is_answer_in_dialogue_returns_false_when_absent():
    assert is_answer_in_dialogue("this answer does not appear anywhere in the dialogue text", ["random text"]) is False


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


def test_build_persona_qas_translates_evidence_to_dia_ids():
    fake = _FakeChatModel([_fake_qa_payload(c) for c in range(1, 6)])
    seeds_per_case = [
        {
            "case_id": "case_001",
            "case_topic": "cs.AI memory evaluation",
            "seeds": [
                {
                    "seed_id": "p01_case_001_paper_read",
                    "memory_type": "paper_read",
                    "content": {"role": "anchor paper"},
                }
            ],
        }
    ]
    seed_to_dia = {"p01_case_001_paper_read": ["s1:3", "s1:5"]}
    dialogue_texts = ["unrelated text only"]

    qas = build_persona_qas(
        chat_model=fake,
        persona_id="p01",
        seeds_per_case=seeds_per_case,
        seed_to_dia_lookup=seed_to_dia,
        dialogue_texts=dialogue_texts,
    )
    assert len(qas) == 60
    assert all("evidence" in qa and qa["evidence"] for qa in qas)
    assert qas[0]["evidence"] == ["s1:3", "s1:5"]


def test_build_persona_qas_rejects_answer_in_dialogue():
    # Long answer (>6 words) that appears verbatim in dialogue triggers leak detection
    leaked_answer = "the user mentioned the anchor paper is swe chat in our discussion"
    fake = _FakeChatModel(
        [
            json.dumps(
                [
                    {
                        "question": f"q{i}",
                        "answer": leaked_answer,
                        "evidence_seed_ids": ["p01_case_001_paper_read"],
                        "case_id": "case_001",
                        "distractor_case_id": None,
                        "template_id": f"t{i}",
                    }
                    for i in range(12)
                ]
            )
        ]
        + [_fake_qa_payload(c) for c in range(2, 6)]
    )
    seed_to_dia = {"p01_case_001_paper_read": ["s1:1"]}
    dialogue_texts = ["the user mentioned the anchor paper is swe chat in our discussion today."]

    with pytest.raises(RuntimeError, match="answer leaked from dialogue"):
        build_persona_qas(
            chat_model=fake,
            persona_id="p01",
            seeds_per_case=[
                {
                    "case_id": "case_001",
                    "case_topic": "t",
                    "seeds": [
                        {
                            "seed_id": "p01_case_001_paper_read",
                            "memory_type": "paper_read",
                            "content": {},
                        }
                    ],
                }
            ],
            seed_to_dia_lookup=seed_to_dia,
            dialogue_texts=dialogue_texts,
            max_retries=0,
        )
