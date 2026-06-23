from __future__ import annotations

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
