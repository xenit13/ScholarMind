from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session, sessionmaker

from scholar_mind.db.models import MemoryOperationEventModel, MemoryRecordModel
from scholar_mind.models.domain import (
    MemoryOperationEvent,
    MemoryStatus,
    StructuredMemoryRecord,
)


class MemoryRepository:
    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    def upsert(self, record: StructuredMemoryRecord) -> StructuredMemoryRecord:
        with self.session_factory() as session:
            row = session.get(MemoryRecordModel, record.memory_id)
            if row is None:
                row = MemoryRecordModel(memory_id=record.memory_id)
                session.add(row)
            _apply_record(row, record)
            session.commit()
        return record

    def get(self, user_id: str, memory_id: str) -> StructuredMemoryRecord | None:
        with self.session_factory() as session:
            row = session.get(MemoryRecordModel, memory_id)
            if row is None or row.user_id != user_id:
                return None
            return _row_to_record(row)

    def list_active(self, user_id: str) -> list[StructuredMemoryRecord]:
        return self.list_by_status(user_id, MemoryStatus.ACTIVE)

    def list_memories(
        self,
        user_id: str,
        status: str | MemoryStatus | None = MemoryStatus.ACTIVE,
    ) -> list[StructuredMemoryRecord]:
        if status is None:
            with self.session_factory() as session:
                rows = (
                    session.query(MemoryRecordModel)
                    .filter(MemoryRecordModel.user_id == user_id)
                    .order_by(MemoryRecordModel.created_at, MemoryRecordModel.memory_id)
                    .all()
                )
                return [_row_to_record(row) for row in rows]
        return self.list_by_status(user_id, status)

    def list_by_status(
        self,
        user_id: str,
        status: str | MemoryStatus,
    ) -> list[StructuredMemoryRecord]:
        status_value = _enum_value(status)
        with self.session_factory() as session:
            rows = (
                session.query(MemoryRecordModel)
                .filter(
                    MemoryRecordModel.user_id == user_id,
                    MemoryRecordModel.status == status_value,
                )
                .order_by(MemoryRecordModel.created_at, MemoryRecordModel.memory_id)
                .all()
            )
            return [_row_to_record(row) for row in rows]

    def list_active_records(
        self,
        user_id: str | None = None,
    ) -> list[StructuredMemoryRecord]:
        with self.session_factory() as session:
            query = session.query(MemoryRecordModel).filter(
                MemoryRecordModel.status == MemoryStatus.ACTIVE.value
            )
            if user_id is not None:
                query = query.filter(MemoryRecordModel.user_id == user_id)
            rows = query.order_by(MemoryRecordModel.created_at, MemoryRecordModel.memory_id).all()
            return [_row_to_record(row) for row in rows]

    def list_by_ids(
        self,
        user_id: str,
        memory_ids: Iterable[str],
    ) -> dict[str, StructuredMemoryRecord]:
        ids = [memory_id for memory_id in memory_ids if memory_id]
        if not ids:
            return {}
        with self.session_factory() as session:
            rows = (
                session.query(MemoryRecordModel)
                .filter(
                    MemoryRecordModel.user_id == user_id,
                    MemoryRecordModel.memory_id.in_(ids),
                )
                .all()
            )
            return {row.memory_id: _row_to_record(row) for row in rows}

    def record_access(
        self,
        user_id: str,
        memory_ids: Iterable[str],
        accessed_at: datetime | None = None,
    ) -> list[StructuredMemoryRecord]:
        ids = [memory_id for memory_id in memory_ids if memory_id]
        if not ids:
            return []
        timestamp = accessed_at or datetime.now(UTC)
        updated: list[StructuredMemoryRecord] = []
        with self.session_factory() as session:
            rows = (
                session.query(MemoryRecordModel)
                .filter(
                    MemoryRecordModel.user_id == user_id,
                    MemoryRecordModel.memory_id.in_(ids),
                    MemoryRecordModel.status == MemoryStatus.ACTIVE.value,
                )
                .all()
            )
            for row in rows:
                previous_access = row.last_accessed_at
                row.last_accessed_at = timestamp
                row.access_count = int(row.access_count or 0) + 1
                if previous_access is None or _aware(timestamp) - _aware(
                    previous_access
                ) > timedelta(days=30):
                    row.access_count_30d = 1
                else:
                    row.access_count_30d = int(row.access_count_30d or 0) + 1
                updated.append(_row_to_record(row))
            session.commit()
        return updated

    def set_status(
        self,
        user_id: str,
        memory_id: str,
        status: str | MemoryStatus,
        *,
        updated_at: datetime | None = None,
        last_decay_score: float | None = None,
    ) -> StructuredMemoryRecord | None:
        with self.session_factory() as session:
            row = session.get(MemoryRecordModel, memory_id)
            if row is None or row.user_id != user_id:
                return None
            row.status = _enum_value(status)
            row.updated_at = updated_at or datetime.now(UTC)
            row.version = int(row.version or 1) + 1
            if last_decay_score is not None:
                row.last_decay_score = last_decay_score
            record = _row_to_record(row)
            session.commit()
            return record

    def record_operation(self, event: MemoryOperationEvent) -> MemoryOperationEvent:
        with self.session_factory() as session:
            session.add(_event_to_row(event))
            session.commit()
        return event

    def list_operation_events(self, user_id: str) -> list[MemoryOperationEvent]:
        with self.session_factory() as session:
            rows = (
                session.query(MemoryOperationEventModel)
                .filter(MemoryOperationEventModel.user_id == user_id)
                .order_by(
                    MemoryOperationEventModel.created_at,
                    MemoryOperationEventModel.event_id,
                )
                .all()
            )
            return [_row_to_event(row) for row in rows]


def _apply_record(row: MemoryRecordModel, record: StructuredMemoryRecord) -> None:
    row.user_id = record.user_id
    row.scope = record.scope
    row.session_id = record.session_id
    row.request_id = record.request_id
    row.memory_type = _enum_value(record.memory_type)
    row.content = record.content
    row.structured_json = json.dumps(record.structured, ensure_ascii=False)
    row.keywords_json = json.dumps(record.keywords, ensure_ascii=False)
    row.source = record.source
    row.evidence_json = json.dumps(record.evidence, ensure_ascii=False)
    row.importance = record.importance
    row.confidence = record.confidence
    row.sensitivity = record.sensitivity
    row.status = _enum_value(record.status)
    row.created_at = record.created_at
    row.updated_at = record.updated_at
    row.last_accessed_at = record.last_accessed_at
    row.valid_from = record.valid_from
    row.valid_to = record.valid_to
    row.expires_at = record.expires_at
    row.decay_rate = record.decay_rate
    row.decay_floor = record.decay_floor
    row.access_count = record.access_count
    row.access_count_30d = record.access_count_30d
    row.last_decay_score = record.last_decay_score
    row.supersedes_json = json.dumps(record.supersedes, ensure_ascii=False)
    row.superseded_by = record.superseded_by
    row.version = record.version


def _row_to_record(row: MemoryRecordModel) -> StructuredMemoryRecord:
    return StructuredMemoryRecord(
        memory_id=row.memory_id,
        user_id=row.user_id,
        scope=row.scope,
        session_id=row.session_id,
        request_id=row.request_id,
        memory_type=row.memory_type,
        content=row.content,
        structured=_load_json(row.structured_json, {}),
        keywords=_load_json(row.keywords_json, []),
        source=row.source,
        evidence=_load_json(row.evidence_json, []),
        importance=row.importance,
        confidence=row.confidence,
        sensitivity=row.sensitivity,
        status=row.status,
        created_at=row.created_at,
        updated_at=row.updated_at,
        last_accessed_at=row.last_accessed_at,
        valid_from=row.valid_from,
        valid_to=row.valid_to,
        expires_at=row.expires_at,
        decay_rate=row.decay_rate,
        decay_floor=row.decay_floor,
        access_count=row.access_count,
        access_count_30d=row.access_count_30d,
        last_decay_score=row.last_decay_score,
        supersedes=_load_json(row.supersedes_json, []),
        superseded_by=row.superseded_by,
        version=row.version,
    )


def _event_to_row(event: MemoryOperationEvent) -> MemoryOperationEventModel:
    return MemoryOperationEventModel(
        event_id=event.event_id,
        user_id=event.user_id,
        operation=_enum_value(event.operation),
        memory_id=event.memory_id,
        session_id=event.session_id,
        request_id=event.request_id,
        candidate_json=json.dumps(event.candidate, ensure_ascii=False),
        old_record_json=(
            json.dumps(event.old_record, ensure_ascii=False)
            if event.old_record is not None
            else None
        ),
        new_record_json=(
            json.dumps(event.new_record, ensure_ascii=False)
            if event.new_record is not None
            else None
        ),
        reason=event.reason,
        model=event.model,
        created_at=event.created_at,
    )


def _row_to_event(row: MemoryOperationEventModel) -> MemoryOperationEvent:
    return MemoryOperationEvent(
        event_id=row.event_id,
        user_id=row.user_id,
        operation=row.operation,
        memory_id=row.memory_id,
        session_id=row.session_id,
        request_id=row.request_id,
        candidate=_load_json(row.candidate_json, {}),
        old_record=_load_json(row.old_record_json, None),
        new_record=_load_json(row.new_record_json, None),
        reason=row.reason,
        model=row.model,
        created_at=row.created_at,
    )


def _load_json(raw: str | None, default):
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def _enum_value(value) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
