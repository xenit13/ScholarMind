from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import create_engine, inspect, text

from scholar_mind.config.settings import Settings
from scholar_mind.db.session import build_session_factory, init_database
from scholar_mind.memory.legacy_import import import_legacy_memory_file
from scholar_mind.memory.rebuild import rebuild_memory_index
from scholar_mind.memory.repository import MemoryRepository
from scholar_mind.models.domain import StructuredMemoryRecord
from scholar_mind.rag.index import QdrantIndex


def _settings(tmp_path):
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'memory.db'}",
        memory_root_dir=str(tmp_path / "memories"),
        log_dir=str(tmp_path / "logs"),
        qdrant_location=":memory:",
        bootstrap_sample_data=False,
    )


def _record(memory_id: str, *, status: str = "active") -> StructuredMemoryRecord:
    now = datetime(2026, 5, 19, 8, 0, tzinfo=UTC)
    return StructuredMemoryRecord(
        memory_id=memory_id,
        user_id="u1",
        scope="user",
        memory_type="interaction_summary",
        content=f"记忆 {memory_id}",
        source="conversation",
        importance=0.6,
        confidence=0.7,
        status=status,
        created_at=now,
        updated_at=now,
        decay_rate=0.03,
        decay_floor=0.3,
    )


def test_memory_repository_persists_and_filters_active_records(tmp_path):
    settings = _settings(tmp_path)
    init_database(settings)
    repository = MemoryRepository(build_session_factory(settings))

    repository.upsert(_record("mem_active"))
    repository.upsert(_record("mem_archived", status="archived"))

    active = repository.list_active("u1")
    stored = repository.get("u1", "mem_active")

    assert [record.memory_id for record in active] == ["mem_active"]
    assert stored is not None
    assert stored.content == "记忆 mem_active"
    assert stored.importance == 0.6


def test_init_database_migrates_memory_records_without_state_key(tmp_path):
    db_path = tmp_path / "legacy_memory.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    now = "2026-05-20 08:00:00"
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE memory_records (
                    memory_id VARCHAR(64) NOT NULL PRIMARY KEY,
                    user_id VARCHAR(64) NOT NULL,
                    scope VARCHAR(32) NOT NULL,
                    session_id VARCHAR(64),
                    request_id VARCHAR(64),
                    memory_type VARCHAR(64) NOT NULL,
                    state_key VARCHAR(255) NOT NULL,
                    content TEXT NOT NULL,
                    structured_json TEXT NOT NULL,
                    keywords_json TEXT NOT NULL,
                    source VARCHAR(64) NOT NULL,
                    evidence_json TEXT NOT NULL,
                    importance FLOAT NOT NULL,
                    confidence FLOAT NOT NULL,
                    sensitivity VARCHAR(32) NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    last_accessed_at DATETIME,
                    valid_from DATETIME,
                    valid_to DATETIME,
                    expires_at DATETIME,
                    decay_rate FLOAT NOT NULL,
                    decay_floor FLOAT NOT NULL,
                    access_count INTEGER NOT NULL,
                    access_count_30d INTEGER NOT NULL,
                    last_decay_score FLOAT,
                    supersedes_json TEXT NOT NULL,
                    superseded_by VARCHAR(64),
                    version INTEGER NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO memory_records (
                    memory_id, user_id, scope, session_id, request_id, memory_type,
                    state_key, content, structured_json, keywords_json, source,
                    evidence_json, importance, confidence, sensitivity, status,
                    created_at, updated_at, decay_rate, decay_floor, access_count,
                    access_count_30d, supersedes_json, version
                ) VALUES (
                    'mem_legacy', 'u1', 'user', 's1', 'r1', 'interaction_summary',
                    'legacy-state', '旧记忆', '{}', '[]', 'conversation',
                    '[]', 0.6, 0.7, 'normal', 'active',
                    :now, :now, 0.06, 0.2, 0, 0, '[]', 1
                )
                """
            ),
            {"now": now},
        )

    settings = _settings(tmp_path)
    settings.database_url = f"sqlite:///{db_path}"
    init_database(settings)
    repository = MemoryRepository(build_session_factory(settings))

    columns = {column["name"] for column in inspect(engine).get_columns("memory_records")}
    legacy = repository.get("u1", "mem_legacy")
    repository.upsert(_record("mem_new"))
    stored = repository.get("u1", "mem_new")

    assert "state_key" not in columns
    assert legacy is not None
    assert legacy.content == "旧记忆"
    assert stored is not None
    assert stored.content == "记忆 mem_new"


def test_import_legacy_memory_file_is_idempotent(tmp_path):
    settings = _settings(tmp_path)
    init_database(settings)
    repository = MemoryRepository(build_session_factory(settings))
    memory_file = tmp_path / "memories" / "u1" / "MEMORY.md"
    memory_file.parent.mkdir(parents=True)
    memory_file.write_text(
        "# Memory\n\n"
        "## mem_001\n"
        "- created_at: 2026-04-14T10:32:11+00:00\n"
        "- source: conversation\n"
        "- content: 用户偏好中文回答。\n\n"
        "## mem_002\n"
        "- created_at: 2026-04-14T10:40:25+00:00\n"
        "- source: explicit\n"
        "- content: 用户关注 RAG 评测。\n\n",
        encoding="utf-8",
    )

    first_count = import_legacy_memory_file(memory_file, repository, user_id="u1")
    second_count = import_legacy_memory_file(memory_file, repository, user_id="u1")

    active = repository.list_active("u1")
    assert first_count == 2
    assert second_count == 0
    assert [record.memory_id for record in active] == ["mem_001", "mem_002"]
    assert active[1].source == "explicit"


def test_rebuild_memory_index_uses_structured_payload(tmp_path):
    settings = _settings(tmp_path)
    init_database(settings)
    repository = MemoryRepository(build_session_factory(settings))
    repository.upsert(_record("mem_active"))
    repository.upsert(_record("mem_deleted", status="deleted"))
    index = QdrantIndex(settings, dimension=2)

    class _Embedder:
        def embed_query(self, _text: str):
            return [0.1, 0.2]

    rebuilt_count = rebuild_memory_index(
        repository,
        index,
        _Embedder(),
        user_id="u1",
    )

    hits = index.search_memory("u1", [0.1, 0.2], limit=5)
    assert rebuilt_count == 1
    assert len(hits) == 1
    assert hits[0].payload["memory_id"] == "mem_active"
    assert hits[0].payload["record_id"] == "mem_active"
    assert hits[0].payload["status"] == "active"
    assert hits[0].payload["importance"] == 0.6
