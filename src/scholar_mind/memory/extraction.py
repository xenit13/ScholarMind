from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from scholar_mind.agents.common import (
    extract_json_candidate,
    invoke_structured_output,
    merge_usage,
    raw_output_text,
)
from scholar_mind.models.domain import MemoryCandidate, MemoryCandidateExtractionOutput
from scholar_mind.utils.messages import deserialize_messages

MAX_CANDIDATES_PER_ROUND = 5


def extract_memory_candidates_from_round(
    llm,
    round_messages: list[dict],
    explicit_memories: list[str] | None = None,
) -> tuple[list[MemoryCandidate], dict[str, float], bool]:
    explicit_candidates = explicit_memories_to_candidates(explicit_memories or [])
    if _is_memory_application_only_round(round_messages):
        return explicit_candidates, merge_usage(), bool(explicit_candidates)
    if llm is None:
        return explicit_candidates, merge_usage(), bool(explicit_candidates)

    prompt = _build_candidate_extraction_prompt(round_messages)
    structured, usage = invoke_structured_output(
        llm,
        prompt,
        MemoryCandidateExtractionOutput,
        recover=_recover_memory_candidate_output,
    )
    if structured is None:
        return explicit_candidates, usage, bool(explicit_candidates)

    candidates = explicit_candidates + structured.candidates
    candidates = _dedupe_candidates(candidates)
    return candidates[:MAX_CANDIDATES_PER_ROUND], usage, True


def explicit_memories_to_candidates(memories: list[str]) -> list[MemoryCandidate]:
    candidates: list[MemoryCandidate] = []
    for memory in memories:
        content = memory.strip() if isinstance(memory, str) else ""
        if not content:
            continue
        candidates.append(
            MemoryCandidate(
                memory_type="interaction_summary",
                content=content,
                structured={"explicit": True},
                keywords=[],
                importance=0.8,
                confidence=1.0,
                source="explicit",
                evidence=[],
            )
        )
    return candidates


def _is_memory_application_only_round(round_messages: list[dict]) -> bool:
    user_text = _latest_human_text(round_messages)
    if not user_text:
        return False
    if _has_memory_update_intent(user_text):
        return False
    return _has_memory_application_intent(user_text)


def _latest_human_text(round_messages: list[dict]) -> str:
    messages = deserialize_messages(
        [item["message"] for item in round_messages if isinstance(item, dict) and "message" in item]
    )
    for message in reversed(messages):
        if getattr(message, "type", "") == "human":
            return str(message.content).strip().lower()
    return ""


def _has_memory_update_intent(text: str) -> bool:
    return _mentions_any(
        text,
        [
            "请记住",
            "记住，",
            "记住:",
            "记住：",
            "以后",
            "补充长期偏好",
            "补充偏好",
            "更正一下",
            "默认希望",
            "固定输出项",
            "最关心",
        ],
    )


def _has_memory_application_intent(text: str) -> bool:
    return _mentions_any(
        text,
        [
            "基于刚才这些偏好",
            "基于这些偏好",
            "按这些偏好",
            "按更正后的偏好",
            "用你记住的",
            "你现在记住了哪些",
            "总结我对",
            "按默认标准",
            "回到默认设置",
        ],
    )


def _build_candidate_extraction_prompt(round_messages: list[dict]) -> str:
    messages = deserialize_messages(
        [item["message"] for item in round_messages if isinstance(item, dict) and "message" in item]
    )
    prompt_messages = []
    for idx, message in enumerate(messages):
        payload: dict[str, object] = {
            "message_id": _message_id(round_messages, idx),
            "type": getattr(message, "type", "system"),
            "content": str(message.content),
        }
        if isinstance(message, AIMessage) and message.tool_calls:
            payload["tool_calls"] = message.tool_calls
        if isinstance(message, ToolMessage):
            payload["tool_call_id"] = message.tool_call_id
            payload["name"] = message.name
            payload["status"] = message.status
        prompt_messages.append(payload)

    return (
        "# Role\n"
        "You are the structured memory extraction agent for ScholarMind.\n\n"
        "# Goal\n"
        "Extract durable user memory candidates from one conversation round.\n\n"
        "# Rules\n"
        "- Extract only stable facts useful in future interactions.\n"
        "- Prefer explicit user-stated preferences, research interests, knowledge level, "
        "goals, workflows, project constraints, read papers, and feedback.\n"
        "- Do not extract one-off requests, temporary task state, tool traces, "
        "or assistant plans.\n"
        "- Do not infer facts the user did not clearly express.\n"
        "- Sensitive personal data is allowed only when the user explicitly asks to remember it.\n"
        "- Return at most 5 candidates.\n\n"
        "# Memory management operations\n"
        "When the user asks to forget, delete, remove, or no longer remember a durable fact, "
        "return the matching candidate and set `structured.operation` to `DELETE`.\n"
        "When the user asks to temporarily archive, suspend, hide, or stop using a memory, "
        "set `structured.operation` to `ARCHIVE`.\n"
        "When the user asks to restore, recover, or use an archived memory again, "
        "set `structured.operation` to `RESTORE`.\n"
        "For normal new or changed facts, omit `structured.operation`; the memory manager "
        "will decide ADD, UPDATE, or NONE by memory type, content, and semantic match.\n\n"
        "# Output\n"
        "Return valid JSON only, with this top-level field:\n"
        "`candidates`: array of objects matching "
        "`memory_type,content,structured,keywords,importance,confidence,source,evidence`.\n\n"
        f"Messages: {json.dumps(prompt_messages, ensure_ascii=False)}"
    )


def _recover_memory_candidate_output(raw) -> MemoryCandidateExtractionOutput | None:
    payload = extract_json_candidate(raw_output_text(raw).strip())
    if payload is None:
        return None
    raw_candidates = _extract_candidate_items(payload)
    candidates = []
    for item in raw_candidates:
        candidate = _candidate_from_payload(item)
        if candidate is not None:
            candidates.append(candidate)
    return MemoryCandidateExtractionOutput(candidates=candidates)


def _extract_candidate_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("candidates", "memories", "facts", "items"):
        items = payload.get(key)
        if isinstance(items, list):
            return items
        if isinstance(items, str):
            return [items]
    return []


def _candidate_from_payload(item: Any) -> MemoryCandidate | None:
    if isinstance(item, str):
        content = item.strip()
        if not content:
            return None
        return MemoryCandidate(
            memory_type="interaction_summary",
            content=content,
            source="conversation",
        )
    if not isinstance(item, dict):
        return None
    payload = dict(item)
    content = payload.get("content") or payload.get("memory") or payload.get("text")
    if not content:
        return None
    payload["content"] = str(content).strip()
    payload.setdefault("memory_type", payload.pop("type", "interaction_summary"))
    payload.setdefault("structured", {})
    payload.setdefault("keywords", [])
    payload.setdefault("importance", 0.6)
    payload.setdefault("confidence", 0.7)
    payload.setdefault("source", "conversation")
    payload.setdefault("evidence", [])
    try:
        return MemoryCandidate.model_validate(payload)
    except Exception:
        return None


def _dedupe_candidates(candidates: list[MemoryCandidate]) -> list[MemoryCandidate]:
    deduped: list[MemoryCandidate] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        fingerprint = (_enum_value(candidate.memory_type), _normalize_text(candidate.content))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(candidate)
    return deduped


def _mentions_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _message_id(round_messages: list[dict], idx: int) -> str:
    if idx >= len(round_messages):
        return f"message-{idx}"
    raw = round_messages[idx]
    if isinstance(raw, dict):
        return str(raw.get("message_id") or f"message-{idx}")
    return f"message-{idx}"


def _normalize_text(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = re.sub(r"\s+", "", cleaned)
    return re.sub(r"[，。！？!?,.:：;；、\"'“”‘’()（）\[\]{}<>《》]", "", cleaned)


def _enum_value(value) -> str:
    return value.value if hasattr(value, "value") else str(value)
