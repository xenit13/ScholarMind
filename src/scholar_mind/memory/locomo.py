from __future__ import annotations

import json
import re
from typing import Any

LOCOMO_EVENT_SCHEMA_VERSION = "locomo_event_v1"
LOCOMO_SOURCE_MODE = "locomo_benchmark"
DIALOG_ID_RE = re.compile(r"\bD\d+:\d+\b")
_DIALOG_ID_PARTS_RE = re.compile(r"^D(?P<session>\d+):(?P<turn>\d+)$")


def dialog_ids_from_text(text: str) -> list[str]:
    seen: set[str] = set()
    dialog_ids: list[str] = []
    for dialog_id in DIALOG_ID_RE.findall(text):
        if dialog_id in seen:
            continue
        seen.add(dialog_id)
        dialog_ids.append(dialog_id)
    return dialog_ids


def dialog_ids_from_payload(*parts: Any) -> list[str]:
    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, str):
            text_parts.append(part)
        elif part:
            text_parts.append(json.dumps(part, ensure_ascii=False, sort_keys=True))
    return dialog_ids_from_text("\n".join(text_parts))


def is_locomo_event_structured(structured: dict[str, Any] | None) -> bool:
    if not isinstance(structured, dict):
        return False
    return (
        structured.get("schema_version") == LOCOMO_EVENT_SCHEMA_VERSION
        or structured.get("source_mode") == LOCOMO_SOURCE_MODE
    )


def is_locomo_event_memory(value: Any) -> bool:
    structured = getattr(value, "structured", None)
    return is_locomo_event_structured(structured)


def locomo_dialog_ids(value: Any) -> list[str]:
    structured = getattr(value, "structured", None)
    content = getattr(value, "content", "")
    evidence = getattr(value, "evidence", [])
    structured_ids = structured.get("dialog_ids", []) if isinstance(structured, dict) else []
    return dialog_ids_from_payload(structured_ids, content, evidence)


def locomo_dialog_range(value: Any) -> tuple[int, int, int] | None:
    parsed = []
    for dialog_id in locomo_dialog_ids(value):
        match = _DIALOG_ID_PARTS_RE.match(dialog_id)
        if match is None:
            continue
        parsed.append((int(match.group("session")), int(match.group("turn"))))
    if not parsed:
        return None
    sessions = {session for session, _turn in parsed}
    if len(sessions) != 1:
        return None
    turns = [turn for _session, turn in parsed]
    return parsed[0][0], min(turns), max(turns)


def is_locomo_chunk_memory(value: Any) -> bool:
    return len(locomo_dialog_ids(value)) > 1


def mark_locomo_event_candidate(candidate: Any) -> Any:
    structured = dict(getattr(candidate, "structured", {}) or {})
    dialog_ids = dialog_ids_from_payload(
        structured.get("dialog_ids", []),
        getattr(candidate, "content", ""),
        getattr(candidate, "evidence", []),
    )
    structured["schema_version"] = LOCOMO_EVENT_SCHEMA_VERSION
    structured["source_mode"] = LOCOMO_SOURCE_MODE
    if dialog_ids:
        structured["dialog_ids"] = dialog_ids
    return candidate.model_copy(update={"structured": structured})
