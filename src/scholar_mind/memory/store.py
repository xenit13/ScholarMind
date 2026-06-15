from __future__ import annotations

from pathlib import Path

from scholar_mind.models.domain import MemoryRecord


class MemoryStore:
    def __init__(self, root: Path):
        self.root = root

    def append(self, record: MemoryRecord) -> Path:
        path = self.root / record.user_id / "MEMORY.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("# Memory\n\n", encoding="utf-8")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"## {record.record_id}\n"
                f"- created_at: {record.created_at.isoformat()}\n"
                f"- source: {record.source}\n"
                f"- content: {record.content}\n\n"
            )
        return path
