from __future__ import annotations

import json
from typing import Any

_DIALOGUE_PROMPT_TEMPLATE = """You are writing a multi-turn conversation between a ScholarMind
user and the ScholarMind research assistant.

User persona background: {persona_background}
Conversation date: {session_date}
This is session {session_index} of 6 in the user's project history.

You will embed the following memory seeds naturally into the conversation.
Each seed represents something the user has revealed about their research
workflow. Embed them in passing — user mentions the memory while discussing
papers/cases. Do NOT include explicit memory-instruction phrases (forbidden
patterns: any Chinese phrase starting with 请 followed by 记, or any English
phrase instructing the assistant to memorize a fact). The user is having a
working conversation with the assistant, not giving memory commands.

The assistant must respond substantively — ask clarifying questions, suggest
approaches, or comment on the work. Do not just say "got it" or "noted".

# CRITICAL: MEMORY vs DISTRACTOR RATIO

You must produce exactly {target_turns} turns. Of those:
- **{seed_turns_min}-{seed_turns_max} turns must reference a seed_id** (memory-bearing turns)
- **{distractor_turns_min}-{distractor_turns_max} turns must have seed_id=null** (distractor turns)

Each provided seed must appear in AT LEAST 3 different turns. With {seed_count}
seeds × 3 references = {seed_turns_min} memory-bearing turns minimum.

Distractor turns cover topics unrelated to memory: weather, scheduling,
paper formatting quirks, tooling issues, LaTeX questions, weekend plans, etc.
Distractor text must NOT contain any seed facts.

# EXAMPLE TURN DISTRIBUTION (for {target_turns} turns, {seed_count} seeds)

If you produce 80% distractors (only 6 memory turns), the output is BROKEN and will be rejected.
Aim for ~50% memory / ~50% distractor.

Memory seeds for this session (JSON):
{seeds_json}

Output a JSON array of exactly {target_turns} turn objects. Each turn:
{{"speaker": "user" | "assistant", "text": "...", "seed_id": "<seed_id>" | null}}

Rules:
- seed_id MUST be null for distractor turns, and MUST match one of the
  seed ids above for memory-bearing turns
- Each seed_id appears in at least 3 turns
- Alternate speakers naturally; user-to-assistant ratio around 50/50
- Each turn text is 1-3 sentences in Chinese (with English technical labels
  preserved, e.g., role names, paper titles)
- Total Chinese characters per turn should be ≥ 50% of the turn (technical
  terms can be English)

Return only the JSON array. No prose, no markdown fences.
"""

_QA_PROMPT_TEMPLATE = """You are generating LOCOMO-style evaluation questions for the
ScholarMind memory benchmark.

Persona: {persona_id}
Category: {category} — {category_name}
Description: {category_description}

Generate exactly 12 questions for this category, grounded on the memory
seeds below. Each question must:
- Reference a specific case_topic, paper title, or memory content from the
  seeds
- Have a SHORT answer (≤ 10 words), extracted directly from seed content
- Be phrased in Chinese (English technical labels preserved)

Do NOT copy dialogue text verbatim — answers must come from the structured
seed content, not from any conversation. Do not include the answer as a
substring of the question.

For category 5 (memory_adversarial), 6 of the 12 questions should have the
correct answer "no information available" (because the question mixes in a
distractor case), and 6 should have an actual short answer (cross-case
comparison).

Memory seeds grouped by case:
{seeds_per_case_json}

Output a JSON array of exactly 12 objects:
{{"question": "...", "answer": "...", "evidence_seed_ids": ["seed_id", ...],
"case_id": "case_00X", "distractor_case_id": "case_00Y" | null,
"template_id": "<short-snake-case-id>"}}

For category 5: distractor_case_id must be set. For other categories:
distractor_case_id is null.

Return only the JSON array. No prose, no markdown fences.
"""

_CATEGORY_DESCRIPTIONS = {
    1: (
        "memory_single_hop",
        "single memory direct recall — ask about one fact from one seed",
    ),
    2: (
        "memory_multi_hop",
        "combine 2-3 memories — answer requires info from multiple seeds "
        "in the same case",
    ),
    3: (
        "memory_temporal",
        "temporal reasoning — when a preference or feedback seed has "
        "temporal updates, ask for the LATEST value",
    ),
    4: (
        "memory_personalization",
        "personalization — ask how the user's recorded background/preference "
        "should shape the assistant's output style",
    ),
    5: (
        "memory_adversarial",
        "adversarial confusable — mix a target case with its distractor "
        "case, 6 with real cross-case answer + 6 with 'no information "
        "available'",
    ),
}


def build_dialogue_expansion_prompt(
    *,
    persona_background: str,
    session_index: int,
    session_date: str,
    seeds: list[dict[str, Any]],
    target_turns: int = 30,
) -> str:
    seed_count = len(seeds)
    seed_turns_min = max(seed_count * 3, int(target_turns * 0.40))
    seed_turns_max = int(target_turns * 0.60)
    distractor_turns_min = target_turns - seed_turns_max
    distractor_turns_max = target_turns - seed_turns_min
    return _DIALOGUE_PROMPT_TEMPLATE.format(
        persona_background=persona_background,
        session_index=session_index,
        session_date=session_date,
        seeds_json=json.dumps(seeds, ensure_ascii=False, indent=2),
        target_turns=target_turns,
        seed_count=seed_count,
        seed_turns_min=seed_turns_min,
        seed_turns_max=seed_turns_max,
        distractor_turns_min=distractor_turns_min,
        distractor_turns_max=distractor_turns_max,
    )


def build_qa_generation_prompt(
    *,
    persona_id: str,
    category: int,
    category_name: str,
    category_description: str,
    seeds_per_case: list[dict[str, Any]],
) -> str:
    return _QA_PROMPT_TEMPLATE.format(
        persona_id=persona_id,
        category=category,
        category_name=category_name,
        category_description=category_description,
        seeds_per_case_json=json.dumps(seeds_per_case, ensure_ascii=False, indent=2),
    )


def get_category_description(category: int) -> tuple[str, str]:
    if category not in _CATEGORY_DESCRIPTIONS:
        raise ValueError(f"unsupported category {category}")
    return _CATEGORY_DESCRIPTIONS[category]
