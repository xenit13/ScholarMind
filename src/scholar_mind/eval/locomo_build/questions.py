from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any, Protocol

from scholar_mind.eval.locomo_build.prompts import (
    build_qa_generation_prompt,
    get_category_description,
)

_REQUIRED_QA_KEYS = {
    "question",
    "answer",
    "evidence_seed_ids",
    "case_id",
    "distractor_case_id",
    "template_id",
}

logger = logging.getLogger(__name__)


class ChatModel(Protocol):
    def invoke(self, prompt: str) -> Any: ...


def parse_llm_qa_response(raw: str) -> list[dict[str, Any]]:
    """Parse LLM response into list of QA dicts; strip markdown fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid QA payload: {exc}") from exc
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("invalid QA payload: expected non-empty JSON array")
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("invalid QA payload: each item must be an object")
        missing = _REQUIRED_QA_KEYS - set(item)
        if missing:
            raise ValueError(f"invalid QA payload: item missing keys {sorted(missing)}")
    return parsed


def seed_ids_to_dia_ids(
    seed_ids: list[str], lookup: dict[str, list[str]]
) -> list[str]:
    """Translate seed_ids to dialogue dia_ids using a seed→dia_ids lookup."""
    out: list[str] = []
    for sid in seed_ids:
        if sid not in lookup:
            raise ValueError(f"missing seed_id in dialogue lookup: {sid}")
        out.extend(lookup[sid])
    return out


def is_answer_in_dialogue(answer: str, dialogue_texts: list[str]) -> bool:
    """Return True if answer appears as a case-insensitive substring in any dialogue text."""
    needle = re.sub(r"\s+", " ", answer.strip().lower())
    if not needle:
        return False
    for hay in dialogue_texts:
        if needle in re.sub(r"\s+", " ", hay.lower()):
            return True
    return False


def check_qa_distribution(
    qas: list[dict[str, Any]], *, expected_per_category: int = 12
) -> None:
    """Raise ValueError if any category 1-5 doesn't have exactly expected_per_category QAs,
    or if any question is duplicated.
    """
    by_cat: dict[int, list[dict[str, Any]]] = {}
    for qa in qas:
        by_cat.setdefault(int(qa["category"]), []).append(qa)
    for cat in range(1, 6):
        items = by_cat.get(cat, [])
        if len(items) != expected_per_category:
            raise ValueError(
                f"expected {expected_per_category} QAs for category {cat}, got {len(items)}"
            )
    seen: Counter = Counter(qa.get("question", "") for qa in qas)
    duplicates = [q for q, count in seen.items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate question: {duplicates[:3]}")


def generate_category_qas(
    *,
    chat_model: ChatModel,
    persona_id: str,
    category: int,
    seeds_per_case: list[dict[str, Any]],
    expected_count: int = 12,
    max_retries: int = 2,
) -> list[dict[str, Any]]:
    """Call chat_model to generate QAs for one category, retry on distribution check failure."""
    category_name, description = get_category_description(category)
    prompt = build_qa_generation_prompt(
        persona_id=persona_id,
        category=category,
        category_name=category_name,
        category_description=description,
        seeds_per_case=seeds_per_case,
    )
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        raw = chat_model.invoke(prompt).content
        try:
            parsed = parse_llm_qa_response(raw)
        except ValueError as exc:
            last_error = exc
            logger.warning("qa parse failed (attempt %d): %s", attempt, exc)
            continue
        for item in parsed:
            item["category"] = category
        try:
            _check_single_category_distribution(parsed, category, expected_count)
        except ValueError as exc:
            last_error = exc
            logger.warning("qa distribution check failed (attempt %d): %s", attempt, exc)
            continue
        return parsed
    raise RuntimeError(
        f"QA generation failed for category {category} "
        f"after {max_retries + 1} attempts: {last_error}"
    )


def _check_single_category_distribution(
    qas: list[dict[str, Any]], category: int, expected_count: int
) -> None:
    """Validate QAs for a single category: exact count and no duplicate questions.

    Unlike ``check_qa_distribution``, this only inspects the items belonging to
    ``category`` since ``generate_category_qas`` produces one category at a time.
    """
    by_cat: dict[int, list[dict[str, Any]]] = {}
    for qa in qas:
        by_cat.setdefault(int(qa["category"]), []).append(qa)
    items = by_cat.get(category, [])
    if len(items) != expected_count:
        raise ValueError(
            f"expected {expected_count} QAs for category {category}, got {len(items)}"
        )
    seen: Counter = Counter(qa.get("question", "") for qa in items)
    duplicates = [q for q, count in seen.items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate question: {duplicates[:3]}")
