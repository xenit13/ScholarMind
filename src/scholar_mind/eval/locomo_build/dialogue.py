from __future__ import annotations

from typing import Any


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
