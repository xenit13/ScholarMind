from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from scholar_mind.eval.locomo_build.prompts import build_dialogue_expansion_prompt

logger = logging.getLogger(__name__)

_REQUIRED_TURN_KEYS = {"speaker", "text", "seed_id"}


class ChatModel(Protocol):
    def invoke(self, prompt: str) -> Any: ...


def assign_dia_ids(turns: list[dict[str, Any]], *, session_index: int) -> list[dict[str, Any]]:
    """Assign session-relative dia_ids (s{N}:{idx}) and populate metadata."""
    out: list[dict[str, Any]] = []
    prefix = f"s{session_index}"
    for idx, turn in enumerate(turns, start=1):
        seed_id = turn.get("seed_id")
        out.append(
            {
                "speaker": turn["speaker"],
                "dia_id": f"{prefix}:{idx}",
                "text": turn["text"],
                "metadata": {
                    "seed_id": seed_id,
                    "memory_type": None,
                    "is_distractor": seed_id is None,
                },
            }
        )
    return out


def find_missing_seeds(
    turns: list[dict[str, Any]], required_seed_ids: set[str]
) -> set[str]:
    """Return the set of required_seed_ids not referenced by any turn."""
    present: set[str] = set()
    for turn in turns:
        sid = turn.get("metadata", {}).get("seed_id")
        if sid:
            present.add(sid)
    return set(required_seed_ids) - present


def check_seed_coverage(
    turns: list[dict[str, Any]], required_seed_ids: set[str]
) -> None:
    """Raise ValueError if any required_seed_ids is missing from turns."""
    missing = find_missing_seeds(turns, required_seed_ids)
    if missing:
        raise ValueError(f"seeds missing from dialogue: {sorted(missing)}")


def check_distractor_ratio(
    turns: list[dict[str, Any]], *, low: float = 0.40, high: float = 0.50
) -> None:
    """Raise ValueError if distractor turn ratio is outside [low, high]."""
    if not turns:
        raise ValueError("distractor ratio undefined for empty turn list")
    distractor = sum(1 for t in turns if t.get("metadata", {}).get("is_distractor"))
    ratio = distractor / len(turns)
    if not (low <= ratio <= high):
        raise ValueError(
            f"distractor ratio out of range: got {ratio:.3f}, expected [{low}, {high}]"
        )


def check_speaker_balance(
    turns: list[dict[str, Any]], *, min_assistant_ratio: float = 0.35
) -> None:
    """Raise ValueError if assistant turns < min_assistant_ratio of total."""
    if not turns:
        raise ValueError("speaker balance undefined for empty turn list")
    assistant = sum(1 for t in turns if t.get("speaker") == "assistant")
    ratio = assistant / len(turns)
    if ratio < min_assistant_ratio:
        raise ValueError(
            f"assistant ratio too low: got {ratio:.3f}, expected ≥ {min_assistant_ratio}"
        )


def _is_chinese_char(ch: str) -> bool:
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF


def check_chinese_ratio(
    turns: list[dict[str, Any]],
    *,
    min_ratio: float = 0.50,
    max_violations_fraction: float = 0.20,
) -> None:
    """Raise ValueError if too many turns have Chinese char ratio below min_ratio."""
    if not turns:
        raise ValueError("chinese ratio undefined for empty turn list")
    violations = 0
    for turn in turns:
        text = turn.get("text", "")
        chars = [c for c in text if not c.isspace()]
        if not chars:
            continue
        chinese_count = sum(1 for c in chars if _is_chinese_char(c))
        if chinese_count / len(chars) < min_ratio:
            violations += 1
    if violations / len(turns) > max_violations_fraction:
        raise ValueError(
            f"chinese ratio violated in {violations}/{len(turns)} turns "
            f"(threshold: {max_violations_fraction:.0%})"
        )


def parse_llm_dialogue_response(raw: str) -> list[dict[str, Any]]:
    """Parse LLM response into list of turn dicts; strip markdown fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid dialogue payload: {exc}") from exc
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("invalid dialogue payload: expected non-empty JSON array")
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("invalid dialogue payload: each item must be an object")
        missing = _REQUIRED_TURN_KEYS - set(item)
        if missing:
            raise ValueError(f"invalid dialogue payload: turn missing keys {missing}")
        if item["speaker"] not in {"user", "assistant"}:
            raise ValueError("invalid dialogue payload: speaker must be user/assistant")
    return parsed


def expand_session(
    *,
    chat_model: ChatModel,
    persona_background: str,
    session_index: int,
    session_date: str,
    seeds: list[dict[str, Any]],
    target_turns: int = 30,
    max_retries: int = 1,
) -> list[dict[str, Any]]:
    """Call chat_model to expand seeds into a session, with retry on validation failure."""
    required_seed_ids = {s["seed_id"] for s in seeds}
    prompt = build_dialogue_expansion_prompt(
        persona_background=persona_background,
        session_index=session_index,
        session_date=session_date,
        seeds=seeds,
        target_turns=target_turns,
    )
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        raw = chat_model.invoke(prompt).content
        try:
            parsed = parse_llm_dialogue_response(raw)
        except ValueError as exc:
            last_error = exc
            logger.warning("dialogue parse failed (attempt %d): %s", attempt, exc)
            continue
        turns = assign_dia_ids(parsed, session_index=session_index)
        try:
            check_seed_coverage(turns, required_seed_ids)
            check_speaker_balance(turns)
        except ValueError as exc:
            last_error = exc
            logger.warning("dialogue coverage check failed (attempt %d): %s", attempt, exc)
            if attempt < max_retries:
                continue
            turns = _inject_missing_seed_fallback(turns, required_seed_ids, session_index)
        return turns
    raise RuntimeError(f"dialogue expansion failed after {max_retries + 1} attempts: {last_error}")


def _inject_missing_seed_fallback(
    turns: list[dict[str, Any]],
    required_seed_ids: set[str],
    session_index: int,
) -> list[dict[str, Any]]:
    """Last-resort: append 'by the way, ...' turns for any missing seeds."""
    missing = find_missing_seeds(turns, required_seed_ids)
    if not missing:
        return turns
    next_idx = len(turns) + 1
    for sid in sorted(missing):
        turns.append(
            {
                "speaker": "user",
                "dia_id": f"s{session_index}:{next_idx}",
                "text": f"顺便提一下，{sid} 的内容我之前也提过。",
                "metadata": {"seed_id": sid, "memory_type": None, "is_distractor": False},
            }
        )
        next_idx += 1
    return turns
