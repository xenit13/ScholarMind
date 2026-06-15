from __future__ import annotations

from datetime import UTC, datetime, timedelta

from scholar_mind.config.settings import Settings
from scholar_mind.db.session import build_session_factory, init_database
from scholar_mind.memory.repository import MemoryRepository
from scholar_mind.models.domain import StructuredMemoryRecord
from scholar_mind.services.memory_management import MemoryManagementService


class _Index:
    def __init__(self):
        self.upserts: list[object] = []

    def upsert_memory(self, record, _embedding):
        self.upserts.append(record)


class _FailingIndex:
    def upsert_memory(self, _record, _embedding):
        raise RuntimeError("index unavailable")


class _Embedder:
    def embed_query(self, _content: str):
        return [0.1, 0.2]


def _settings(tmp_path):
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'memory.db'}",
        memory_root_dir=str(tmp_path / "memories"),
        log_dir=str(tmp_path / "logs"),
        qdrant_location=":memory:",
        bootstrap_sample_data=False,
    )


def _repository(tmp_path) -> MemoryRepository:
    settings = _settings(tmp_path)
    init_database(settings)
    return MemoryRepository(build_session_factory(settings))


def _record(
    memory_id: str,
    *,
    user_id: str = "u1",
    content: str = "用户偏好中文回答。",
    status: str = "active",
    source: str = "conversation",
    importance: float = 0.2,
    days_old: int = 0,
) -> StructuredMemoryRecord:
    now = datetime(2026, 5, 19, 8, 0, tzinfo=UTC)
    created_at = now - timedelta(days=days_old)
    return StructuredMemoryRecord(
        memory_id=memory_id,
        user_id=user_id,
        scope="user",
        memory_type="interaction_summary",
        content=content,
        source=source,
        importance=importance,
        confidence=0.8,
        status=status,
        created_at=created_at,
        updated_at=created_at,
        decay_rate=0.06,
        decay_floor=0.2,
    )


def test_memory_management_edit_delete_restore_are_user_scoped(tmp_path):
    repository = _repository(tmp_path)
    repository.upsert(_record("mem_1"))
    repository.upsert(_record("mem_other", user_id="other"))
    index = _Index()
    service = MemoryManagementService(repository, index, _Embedder())

    edited = service.edit_memory("u1", "mem_1", "用户偏好中文回答，并需要引用。")
    denied = service.edit_memory("u1", "mem_other", "不应被修改")
    deleted = service.delete_memory("u1", "mem_1")
    restored = service.restore_memory("u1", "mem_1")

    events = repository.list_operation_events("u1")
    assert edited is not None
    assert edited.content == "用户偏好中文回答，并需要引用。"
    assert edited.version == 2
    assert denied is None
    assert deleted is not None
    assert deleted.status == "deleted"
    assert restored is None
    assert [event.operation for event in events] == ["UPDATE", "DELETE"]
    assert index.upserts[-1].status == "deleted"


def test_memory_management_archives_and_restores_memory(tmp_path):
    repository = _repository(tmp_path)
    repository.upsert(_record("mem_1"))
    index = _Index()
    service = MemoryManagementService(repository, index, _Embedder())

    archived = service.archive_memory("u1", "mem_1")
    restored = service.restore_memory("u1", "mem_1")

    events = repository.list_operation_events("u1")
    assert archived is not None
    assert archived.status == "archived"
    assert restored is not None
    assert restored.status == "active"
    assert [event.operation for event in events] == ["ARCHIVE", "RESTORE"]
    assert index.upserts[-1].status == "active"


def test_memory_management_keeps_audit_when_derived_index_upsert_fails(tmp_path):
    repository = _repository(tmp_path)
    repository.upsert(_record("mem_1"))
    service = MemoryManagementService(repository, _FailingIndex(), _Embedder())

    edited = service.edit_memory("u1", "mem_1", "用户偏好中文回答，并需要引用。")

    events = repository.list_operation_events("u1")
    assert edited is not None
    assert edited.version == 2
    assert events[0].operation == "UPDATE"
    assert events[0].new_record["content"] == "用户偏好中文回答，并需要引用。"


def test_archive_low_value_memories_skips_explicit_high_importance_records(tmp_path):
    repository = _repository(tmp_path)
    repository.upsert(_record("old_low", days_old=180, importance=0.01))
    repository.upsert(
        _record("explicit_keep", source="explicit", days_old=180, importance=0.9)
    )
    service = MemoryManagementService(
        repository,
        _Index(),
        _Embedder(),
        archive_threshold=0.01,
        explicit_keep_importance_threshold=0.85,
    )

    archived_count = service.archive_low_value_memories(
        user_id="u1",
        now=datetime(2026, 5, 19, 8, 0, tzinfo=UTC),
    )

    assert archived_count == 1
    assert repository.get("u1", "old_low").status == "archived"
    assert repository.get("u1", "explicit_keep").status == "active"
    assert repository.list_operation_events("u1")[0].operation == "ARCHIVE"
