"""Helpers for `memory_fact_v1` data stored in `StructuredMemoryRecord.structured`."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from scholar_mind.models.domain import StructuredMemoryRecord

DISCRETE_MEMORY_SCHEMA_VERSION = "memory_fact_v1"
DISCRETE_FACT_KIND = "discrete_fact"
DISCRETE_CONFLICT_MIN_CONFIDENCE = 0.75


@dataclass(frozen=True)
class DiscreteMemoryFact:
    subject: dict[str, Any]
    entity: dict[str, Any]
    attribute: str
    value: dict[str, Any]
    polarity: str
    certainty: str
    temporal: dict[str, Any]
    conflict_key: str
    source_mode: str


def normalize_discrete_structured(structured: dict[str, Any]) -> dict[str, Any]:
    payload = _copy_mapping(structured)
    if not _looks_like_discrete_fact(payload):
        return payload

    subject = _normalize_node(payload.get("subject"))
    entity = _normalize_node(payload.get("entity"))
    value = _normalize_node(payload.get("value"))
    temporal = _normalize_node(payload.get("temporal"))
    attribute = _normalize_text(payload.get("attribute"))

    payload["schema_version"] = DISCRETE_MEMORY_SCHEMA_VERSION
    payload["fact_kind"] = DISCRETE_FACT_KIND
    payload["subject"] = subject
    payload["entity"] = entity
    payload["attribute"] = attribute
    payload["value"] = value
    payload["polarity"] = _normalize_text(payload.get("polarity")) or "unknown"
    payload["certainty"] = _normalize_text(payload.get("certainty")) or "inferred"
    payload["temporal"] = temporal
    payload["conflict_key"] = _build_conflict_key(
        subject, entity, attribute
    ) or _normalize_conflict_key(payload.get("conflict_key"))
    payload["source_mode"] = _normalize_text(payload.get("source_mode"))
    return payload


def parse_discrete_fact(structured: Mapping[str, Any] | None) -> DiscreteMemoryFact | None:
    if not isinstance(structured, Mapping):
        return None
    payload = normalize_discrete_structured(dict(structured))
    if payload.get("schema_version") != DISCRETE_MEMORY_SCHEMA_VERSION:
        return None
    if payload.get("fact_kind") != DISCRETE_FACT_KIND:
        return None
    attribute = _normalize_text(payload.get("attribute"))
    conflict_key = _normalize_conflict_key(payload.get("conflict_key"))
    if not attribute or not conflict_key:
        return None
    return DiscreteMemoryFact(
        subject=_normalize_node(payload.get("subject")),
        entity=_normalize_node(payload.get("entity")),
        attribute=attribute,
        value=_normalize_node(payload.get("value")),
        polarity=_normalize_text(payload.get("polarity")) or "unknown",
        certainty=_normalize_text(payload.get("certainty")) or "inferred",
        temporal=_normalize_node(payload.get("temporal")),
        conflict_key=conflict_key,
        source_mode=_normalize_text(payload.get("source_mode")),
    )


def discrete_value_token(fact: DiscreteMemoryFact) -> tuple[str, str]:
    value = (
        fact.value.get("canonical")
        or fact.value.get("id")
        or fact.value.get("text")
        or fact.value.get("label")
        or json.dumps(fact.value, ensure_ascii=False, sort_keys=True)
    )
    return _normalize_token(value), _normalize_token(fact.polarity)


def skips_temporal_conflict(
    left: DiscreteMemoryFact,
    right: DiscreteMemoryFact,
) -> bool:
    left_tense = _normalize_text(left.temporal.get("tense"))
    right_tense = _normalize_text(right.temporal.get("tense"))
    return (left_tense == "past") != (right_tense == "past")


def format_discrete_memory(record: StructuredMemoryRecord) -> str | None:
    fact = parse_discrete_fact(record.structured)
    if fact is None:
        return None
    entity = _display_value(fact.entity, preferred=("label", "canonical", "id", "text"))
    value = _display_value(fact.value, preferred=("canonical", "label", "text", "id"))
    return (
        f"[{DISCRETE_MEMORY_SCHEMA_VERSION}] "
        f"attribute={fact.attribute}; "
        f"entity={entity}; "
        f"value={value}; "
        f"polarity={fact.polarity}; "
        f"confidence={record.confidence:.2f}; "
        f"content={record.content}"
    )


def _looks_like_discrete_fact(payload: Mapping[str, Any]) -> bool:
    if payload.get("schema_version") == DISCRETE_MEMORY_SCHEMA_VERSION:
        return True
    if payload.get("fact_kind") == DISCRETE_FACT_KIND:
        return True
    return "attribute" in payload and "value" in payload and (
        "subject" in payload or "entity" in payload
    )


def _build_conflict_key(
    subject: Mapping[str, Any],
    entity: Mapping[str, Any],
    attribute: str,
) -> str:
    subject_key = _node_key(subject)
    entity_key = _node_key(entity)
    parts = []
    if subject_key:
        parts.append(f"subject:{subject_key}")
    if entity_key:
        parts.append(f"entity:{entity_key}")
    if attribute:
        parts.append(f"attribute:{attribute}")
    return "|".join(parts)


def _node_key(node: Mapping[str, Any]) -> str:
    node_type = _normalize_token(node.get("type") or node.get("kind"))
    raw_id = node.get("id") or node.get("canonical") or node.get("label") or node.get("text")
    if node_type == "user":
        return "user"
    if isinstance(raw_id, str) and node_type:
        prefix = f"{node_type}:"
        if raw_id.strip().lower().startswith(prefix):
            raw_id = raw_id.strip()[len(prefix) :]
    node_id = _normalize_token(raw_id)
    if node_type and node_id:
        return f"{node_type}:{node_id}"
    return node_id


def _normalize_node(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_nested(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, str):
        text = value.strip()
        return {"label": text} if text else {}
    return {}


def _normalize_nested(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _normalize_node(value)
    if isinstance(value, list):
        return [_normalize_nested(item) for item in value]
    if isinstance(value, str):
        return value.strip()
    return value


def _display_value(
    payload: Mapping[str, Any],
    *,
    preferred: tuple[str, ...],
) -> str:
    for key in preferred:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _copy_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _normalize_nested(value) for key, value in payload.items()}


def _normalize_conflict_key(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", "", value.strip().lower())


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value.strip().lower())


def _normalize_token(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.strip().lower()
    cleaned = re.sub(r"\s+", "", cleaned)
    return re.sub(r"[，。！？!?,.:：;；、\"'“”‘’()（）\[\]{}<>《》]", "", cleaned)
