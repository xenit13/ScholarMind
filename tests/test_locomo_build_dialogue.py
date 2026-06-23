from __future__ import annotations

import pytest

from scholar_mind.eval.locomo_build.dialogue import (
    assign_dia_ids,
    check_chinese_ratio,
    check_distractor_ratio,
    check_seed_coverage,
    check_speaker_balance,
    find_missing_seeds,
)
from scholar_mind.eval.locomo_build.prompts import (
    build_dialogue_expansion_prompt,
    build_qa_generation_prompt,
)


def test_dialogue_prompt_includes_persona_background():
    prompt = build_dialogue_expansion_prompt(
        persona_background="ML 工程师,强工程弱理论",
        session_index=1,
        session_date="2026-05-03",
        seeds=[
            {
                "seed_id": "p01_case_001_paper_read",
                "memory_type": "paper_read",
                "content": {"role": "anchor paper", "paper_title": "SWE-chat"},
            }
        ],
    )
    assert "ML 工程师" in prompt
    assert "session 1" in prompt or "session_1" in prompt
    assert "2026-05-03" in prompt
    assert "p01_case_001_paper_read" in prompt


def test_dialogue_prompt_forbids_explicit_remember_instruction():
    prompt = build_dialogue_expansion_prompt(
        persona_background="...",
        session_index=1,
        session_date="2026-05-03",
        seeds=[],
    )
    assert "请记住" not in prompt
    assert "remember that" not in prompt.lower()


def test_dialogue_prompt_requests_distractor_turns():
    prompt = build_dialogue_expansion_prompt(
        persona_background="...",
        session_index=1,
        session_date="2026-05-03",
        seeds=[],
    )
    assert "distractor" in prompt.lower() or "off-topic" in prompt.lower()


def test_qa_prompt_includes_category_name():
    prompt = build_qa_generation_prompt(
        persona_id="p01",
        category=2,
        category_name="memory_multi_hop",
        category_description="combine 2-3 memories",
        seeds_per_case=[
            {
                "case_id": "case_001",
                "case_topic": "cs.AI memory evaluation",
                "seeds": [],
            }
        ],
    )
    assert "memory_multi_hop" in prompt
    assert "12" in prompt  # asked to generate 12 questions


def test_qa_prompt_forbids_copying_dialogue_text():
    prompt = build_qa_generation_prompt(
        persona_id="p01",
        category=1,
        category_name="memory_single_hop",
        category_description="...",
        seeds_per_case=[],
    )
    assert "do not copy" in prompt.lower() or "禁止复制" in prompt


def test_qa_prompt_specifies_max_answer_length():
    prompt = build_qa_generation_prompt(
        persona_id="p01",
        category=1,
        category_name="memory_single_hop",
        category_description="...",
        seeds_per_case=[],
    )
    assert "10 words" in prompt or "10 个词" in prompt


def _fake_llm_output(seed_ids: list) -> list:
    return [
        {
            "speaker": "user" if i % 2 == 0 else "assistant",
            "text": "中文 中文 hello world",
            "seed_id": sid,
        }
        for i, sid in enumerate(seed_ids)
    ]


def test_assign_dia_ids_formats_session_turn():
    turns = _fake_llm_output([None, None, "seed_a", None])
    out = assign_dia_ids(turns, session_index=1)
    assert [t["dia_id"] for t in out] == ["s1:1", "s1:2", "s1:3", "s1:4"]


def test_assign_dia_ids_preserves_metadata():
    turns = [{"speaker": "user", "text": "...", "seed_id": "s"}]
    out = assign_dia_ids(turns, session_index=2)
    assert out[0]["metadata"]["seed_id"] == "s"
    assert out[0]["metadata"]["is_distractor"] is False
    assert out[0]["metadata"]["memory_type"] is None  # filled later when joining with seeds


def test_assign_dia_ids_marks_distractor_when_seed_id_null():
    turns = [{"speaker": "user", "text": "...", "seed_id": None}]
    out = assign_dia_ids(turns, session_index=1)
    assert out[0]["metadata"]["is_distractor"] is True


def test_check_seed_coverage_flags_missing():
    required = {"a", "b", "c"}
    turns = [
        {"metadata": {"seed_id": "a"}},
        {"metadata": {"seed_id": "a"}},
        {"metadata": {"seed_id": None}},
    ]
    missing = find_missing_seeds(turns, required)
    assert missing == {"b", "c"}


def test_check_seed_coverage_passes_when_all_present():
    required = {"a", "b"}
    turns = [
        {"metadata": {"seed_id": "a"}},
        {"metadata": {"seed_id": "b"}},
    ]
    assert find_missing_seeds(turns, required) == set()


def test_check_distractor_ratio_passes_in_range():
    turns = [{"metadata": {"is_distractor": True}} for _ in range(12)] + [
        {"metadata": {"is_distractor": False}} for _ in range(18)
    ]
    check_distractor_ratio(turns, low=0.40, high=0.50)


def test_check_distractor_ratio_fails_below_low():
    turns = [{"metadata": {"is_distractor": True}} for _ in range(5)] + [
        {"metadata": {"is_distractor": False}} for _ in range(25)
    ]
    with pytest.raises(ValueError, match="distractor ratio"):
        check_distractor_ratio(turns, low=0.40, high=0.50)


def test_check_speaker_balance_passes_with_enough_assistant():
    turns = [{"speaker": "user"} for _ in range(15)] + [{"speaker": "assistant"} for _ in range(15)]
    check_speaker_balance(turns, min_assistant_ratio=0.35)


def test_check_speaker_balance_fails_when_assistant_underrepresented():
    turns = [{"speaker": "user"} for _ in range(25)] + [{"speaker": "assistant"} for _ in range(5)]
    with pytest.raises(ValueError, match="assistant ratio"):
        check_speaker_balance(turns, min_assistant_ratio=0.35)


def test_check_chinese_ratio_passes_with_chinese_dominant_text():
    turns = [{"text": "中文为主的一段对话，里面有一些 English words 但不主导。"} for _ in range(30)]
    check_chinese_ratio(turns, min_ratio=0.50)


def test_check_chinese_ratio_fails_when_english_dominant():
    turns = [{"text": "this is mostly English text with very little Chinese"} for _ in range(30)]
    with pytest.raises(ValueError, match="chinese ratio"):
        check_chinese_ratio(turns, min_ratio=0.50)


def test_check_seed_coverage_full_runs_without_error():
    required = {"a", "b"}
    turns = [
        {"metadata": {"seed_id": "a"}},
        {"metadata": {"seed_id": "b"}},
        {"metadata": {"seed_id": None}},
    ]
    check_seed_coverage(turns, required)
