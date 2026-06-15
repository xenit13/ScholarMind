from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import uuid4

from scholar_mind.memory.decay import archive_score
from scholar_mind.memory.repository import MemoryRepository
from scholar_mind.models.domain import (
    MemoryOperationEvent,
    MemoryOperationName,
    MemoryStatus,
    StructuredMemoryRecord,
)

logger = logging.getLogger(__name__)


class MemoryManagementService:
    def __init__(
        self,
        repository: MemoryRepository,
        index,
        embedder,
        *,
        archive_threshold: float = 0.01,
        explicit_keep_importance_threshold: float = 0.85,
    ):
        self.repository = repository
        self.index = index
        self.embedder = embedder
        self.archive_threshold = archive_threshold
        self.explicit_keep_importance_threshold = explicit_keep_importance_threshold

    def list_memories(
        self,
        user_id: str,
        status: str | MemoryStatus | None = MemoryStatus.ACTIVE,
    ) -> list[StructuredMemoryRecord]:
        return self.repository.list_memories(user_id, status)

    def edit_memory(
        self,
        user_id: str,
        memory_id: str,
        new_content: str,
    ) -> StructuredMemoryRecord | None:
        old_record = self.repository.get(user_id, memory_id)
        content = new_content.strip()
        if old_record is None or _status(old_record) == MemoryStatus.DELETED.value or not content:
            return None
        updated = old_record.model_copy(
            update={
                "content": content,
                "source": "user_edited",
                "updated_at": datetime.now(UTC),
                "version": old_record.version + 1,
            }
        )
        self.repository.upsert(updated)
        self._upsert_index(updated)
        self._record_event(
            user_id=user_id,
            operation=MemoryOperationName.UPDATE,
            memory_id=memory_id,
            old_record=old_record,
            new_record=updated,
            reason="user edited memory",
        )
        return updated

    def delete_memory(self, user_id: str, memory_id: str) -> StructuredMemoryRecord | None:
        old_record = self.repository.get(user_id, memory_id)
        if old_record is None or _status(old_record) == MemoryStatus.DELETED.value:
            return None
        deleted = self.repository.set_status(user_id, memory_id, MemoryStatus.DELETED)
        if deleted is None:
            return None
        self._upsert_index(deleted)
        self._record_event(
            user_id=user_id,
            operation=MemoryOperationName.DELETE,
            memory_id=memory_id,
            old_record=old_record,
            new_record=deleted,
            reason="user deleted memory",
        )
        return deleted

    def archive_memory(self, user_id: str, memory_id: str) -> StructuredMemoryRecord | None:
        old_record = self.repository.get(user_id, memory_id)
        if old_record is None or _status(old_record) != MemoryStatus.ACTIVE.value:
            return None
        archived = self.repository.set_status(user_id, memory_id, MemoryStatus.ARCHIVED)
        if archived is None:
            return None
        self._upsert_index(archived)
        self._record_event(
            user_id=user_id,
            operation=MemoryOperationName.ARCHIVE,
            memory_id=memory_id,
            old_record=old_record,
            new_record=archived,
            reason="user archived memory",
        )
        return archived

    def restore_memory(self, user_id: str, memory_id: str) -> StructuredMemoryRecord | None:
        old_record = self.repository.get(user_id, memory_id)
        if old_record is None or _status(old_record) != MemoryStatus.ARCHIVED.value:
            return None
        restored = self.repository.set_status(user_id, memory_id, MemoryStatus.ACTIVE)
        if restored is None:
            return None
        self._upsert_index(restored)
        self._record_event(
            user_id=user_id,
            operation=MemoryOperationName.RESTORE,
            memory_id=memory_id,
            old_record=old_record,
            new_record=restored,
            reason="user restored memory",
        )
        return restored

    def archive_low_value_memories(
        self,
        *,
        user_id: str | None = None,
        now: datetime | None = None,
    ) -> int:
        archived_count = 0
        for record in self.repository.list_active_records(user_id=user_id):
            score = archive_score(record, now=now)
            if not self._should_archive(record, score):
                continue
            old_record = record
            archived = self.repository.set_status(
                record.user_id,
                record.memory_id,
                MemoryStatus.ARCHIVED,
                updated_at=now,
                last_decay_score=score,
            )
            if archived is None:
                continue
            self._upsert_index(archived)
            self._record_event(
                user_id=record.user_id,
                operation=MemoryOperationName.ARCHIVE,
                memory_id=record.memory_id,
                old_record=old_record,
                new_record=archived,
                reason="memory archive_score fell below threshold",
            )
            archived_count += 1
        return archived_count

    def _should_archive(self, record: StructuredMemoryRecord, score: float) -> bool:
        return (
            score < self.archive_threshold
            and record.source != "explicit"
            and float(record.importance) < self.explicit_keep_importance_threshold
        )

    def _record_event(
        self,
        *,
        user_id: str,
        operation: MemoryOperationName,
        memory_id: str,
        old_record: StructuredMemoryRecord | None,
        new_record: StructuredMemoryRecord | None,
        reason: str,
    ) -> None:
        self.repository.record_operation(
            MemoryOperationEvent(
                event_id=f"memop_{uuid4().hex}",
                user_id=user_id,
                operation=operation,
                memory_id=memory_id,
                candidate={"source": "memory_management"},
                old_record=old_record.model_dump(mode="json") if old_record is not None else None,
                new_record=new_record.model_dump(mode="json") if new_record is not None else None,
                reason=reason,
                model="rule",
                created_at=datetime.now(UTC),
            )
        )

    def _upsert_index(self, record: StructuredMemoryRecord) -> None:
        try:
            self.index.upsert_memory(record, self.embedder.embed_query(record.content))
        except Exception:
            logger.exception(
                "Derived memory index upsert failed: memory_id=%s",
                record.memory_id,
            )


def _status(record: StructuredMemoryRecord) -> str:
    return record.status.value if hasattr(record.status, "value") else str(record.status)
