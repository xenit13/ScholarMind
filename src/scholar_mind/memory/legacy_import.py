from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from scholar_mind.memory.repository import MemoryRepository
from scholar_mind.models.domain import StructuredMemoryRecord


def import_legacy_memory_file(
    path: Path,
    repository: MemoryRepository,
    *,
    user_id: str,
) -> int:
    if not path.exists():
        return 0
    imported = 0
    for payload in _parse_legacy_memory_file(path):
        memory_id = payload.get("memory_id", "").strip()
        content = payload.get("content", "").strip()
        if not memory_id or not content:
            continue
        if repository.get(user_id, memory_id) is not None:
            continue
        created_at = _parse_datetime(payload.get("created_at"))
        record = StructuredMemoryRecord(
            memory_id=memory_id,
            user_id=user_id,
            scope="user",
            memory_type="interaction_summary",
            content=content,
            source=payload.get("source", "conversation").strip() or "conversation",
            importance=0.6,
            confidence=0.7,
            status="active",
            created_at=created_at,
            updated_at=created_at,
            decay_rate=0.03,
            decay_floor=0.3,
            version=1,
        )
        repository.upsert(record)
        imported += 1
    return imported


def _parse_legacy_memory_file(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            if current:
                records.append(current)
            current = {"memory_id": line.removeprefix("## ").strip()}
            continue
        if current is None or not line.startswith("- ") or ":" not in line:
            continue
        key, value = line.removeprefix("- ").split(":", 1)
        current[key.strip()] = value.strip()
    if current:
        records.append(current)
    return records


def _parse_datetime(raw: str | None) -> datetime:
    if not raw:
        return datetime.now(UTC)
    value = raw.strip()
    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed
