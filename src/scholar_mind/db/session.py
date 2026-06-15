from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from scholar_mind.config.settings import Settings, get_settings
from scholar_mind.db.models import (
    Base,
    ConversationMetricModel,
    MemoryEvalAnnotationBatchModel,
    MemoryEvalAnnotationV2Model,
    MemoryEvalReportV2Model,
    MemoryEvalRunV2Model,
    MemoryExtractionEventV2Model,
    MemoryLibraryAuditBatchModel,
    MemoryLibraryAuditReportModel,
    MemoryRecordModel,
    MemoryRetrievalEventV2Model,
    RagEvalCaseModel,
    RagEvalResultV2Model,
    RagEvalRunV2Model,
    RagRetrievalEventV2Model,
    RequestRagEvalAnnotationModel,
    RequestRunModel,
)


def build_engine(settings: Settings | None = None):
    app_settings = settings or get_settings()
    return create_engine(app_settings.database_url, future=True)


def build_session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=build_engine(settings), autoflush=False, autocommit=False, future=True)


def init_database(settings: Settings | None = None) -> None:
    engine = build_engine(settings)
    Base.metadata.create_all(engine)
    _ensure_metric_columns(engine)
    _ensure_online_eval_schema(engine)
    _cleanup_unused_schema(engine)


@contextmanager
def session_scope(settings: Settings | None = None):
    factory = build_session_factory(settings)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _ensure_metric_columns(engine) -> None:
    inspector = inspect(engine)
    if "conversation_metrics" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("conversation_metrics")}
    if "cost" in existing:
        _rebuild_conversation_metrics_table(engine, existing)
        existing = {
            column["name"] for column in inspect(engine).get_columns("conversation_metrics")
        }
    required = {
        "prompt_tokens": "INTEGER DEFAULT 0",
        "completion_tokens": "INTEGER DEFAULT 0",
        "total_tokens": "INTEGER DEFAULT 0",
    }
    with engine.begin() as connection:
        for column_name, column_type in required.items():
            if column_name in existing:
                continue
            connection.execute(
                text(
                    f"ALTER TABLE conversation_metrics "
                    f"ADD COLUMN {column_name} {column_type}"
                )
            )


def _ensure_online_eval_schema(engine) -> None:
    RequestRunModel.__table__.create(bind=engine, checkfirst=True)
    RagRetrievalEventV2Model.__table__.create(bind=engine, checkfirst=True)
    RequestRagEvalAnnotationModel.__table__.create(bind=engine, checkfirst=True)
    RagEvalCaseModel.__table__.create(bind=engine, checkfirst=True)
    RagEvalRunV2Model.__table__.create(bind=engine, checkfirst=True)
    RagEvalResultV2Model.__table__.create(bind=engine, checkfirst=True)
    MemoryRetrievalEventV2Model.__table__.create(bind=engine, checkfirst=True)
    MemoryExtractionEventV2Model.__table__.create(bind=engine, checkfirst=True)
    MemoryEvalAnnotationBatchModel.__table__.create(bind=engine, checkfirst=True)
    MemoryEvalAnnotationV2Model.__table__.create(bind=engine, checkfirst=True)
    MemoryEvalRunV2Model.__table__.create(bind=engine, checkfirst=True)
    MemoryEvalReportV2Model.__table__.create(bind=engine, checkfirst=True)
    MemoryLibraryAuditBatchModel.__table__.create(bind=engine, checkfirst=True)
    MemoryLibraryAuditReportModel.__table__.create(bind=engine, checkfirst=True)
    _ensure_table_columns(
        engine,
        "request_runs",
        {
            "rag_score": "FLOAT",
            "faithfulness": "FLOAT",
            "answer_relevancy": "FLOAT",
            "context_precision": "FLOAT",
            "context_recall": "FLOAT",
            "noise_sensitivity": "FLOAT",
            "semantic_similarity": "FLOAT",
            "redundancy": "FLOAT",
            "completeness": "FLOAT",
            "rag_eval_status": "VARCHAR(32) DEFAULT 'pending'",
            "rag_scored_at": "DATETIME",
        },
    )
    _ensure_table_columns(
        engine,
        "rag_retrieval_events_v2",
        {
            "rag_score": "FLOAT",
            "faithfulness": "FLOAT",
            "answer_relevancy": "FLOAT",
            "context_precision": "FLOAT",
            "context_recall": "FLOAT",
            "noise_sensitivity": "FLOAT",
            "semantic_similarity": "FLOAT",
            "redundancy": "FLOAT",
            "completeness": "FLOAT",
        },
    )
    _ensure_table_columns(
        engine,
        "rag_eval_results_v2",
        {
            "generated_at": "DATETIME",
        },
    )


def _ensure_table_columns(engine, table_name: str, required: dict[str, str]) -> None:
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns(table_name)}
    with engine.begin() as connection:
        for column_name, column_type in required.items():
            if column_name in existing:
                continue
            connection.execute(
                text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
            )


def _rebuild_conversation_metrics_table(engine, existing_columns: set[str]) -> None:
    current_columns = [column.name for column in ConversationMetricModel.__table__.columns]
    transferable = [column for column in current_columns if column in existing_columns]
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE conversation_metrics_new (
                    request_id VARCHAR(64) NOT NULL PRIMARY KEY,
                    user_id VARCHAR(64) NOT NULL,
                    session_id VARCHAR(64) NOT NULL,
                    query_type VARCHAR(32) NOT NULL,
                    success BOOLEAN NOT NULL,
                    retrieval_latency_ms INTEGER NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    citations_count INTEGER NOT NULL,
                    retrieved_chunks_count INTEGER NOT NULL,
                    output_length INTEGER NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    agent_path_json TEXT NOT NULL,
                    error_summary TEXT,
                    created_at DATETIME NOT NULL
                )
                """
            )
        )
        if transferable:
            columns_csv = ", ".join(transferable)
            connection.execute(
                text(
                    f"INSERT INTO conversation_metrics_new ({columns_csv}) "
                    f"SELECT {columns_csv} FROM conversation_metrics"
                )
            )
        connection.execute(text("DROP TABLE conversation_metrics"))
        connection.execute(
            text("ALTER TABLE conversation_metrics_new RENAME TO conversation_metrics")
        )
        connection.execute(
            text("CREATE INDEX ix_conversation_metrics_user_id ON conversation_metrics (user_id)")
        )
        connection.execute(
            text(
                "CREATE INDEX ix_conversation_metrics_session_id "
                "ON conversation_metrics (session_id)"
            )
        )


def _cleanup_unused_schema(engine) -> None:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    retired_tables = {
        "memory_call_events",
        "request_eval_runs",
        "rag_call_events",
        "request_eval_judgements",
    }
    with engine.begin() as connection:
        for table_name in sorted(retired_tables & tables):
            connection.execute(text(f"DROP TABLE {table_name}"))
    if "memory_records" in tables:
        columns = {column["name"] for column in inspect(engine).get_columns("memory_records")}
        if "state_key" in columns:
            _rebuild_memory_records_table(engine, columns)
    if "memory_eval_annotations_v2" in tables:
        columns = {
            column["name"] for column in inspect(engine).get_columns("memory_eval_annotations_v2")
        }
        if "critical_memory_ids_json" in columns:
            _rebuild_memory_eval_annotations_v2_table(engine, columns)


def _rebuild_memory_records_table(engine, existing_columns: set[str]) -> None:
    current_columns = [column.name for column in MemoryRecordModel.__table__.columns]
    transferable = [column for column in current_columns if column in existing_columns]
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE memory_records_new (
                    memory_id VARCHAR(64) NOT NULL PRIMARY KEY,
                    user_id VARCHAR(64) NOT NULL,
                    scope VARCHAR(32) NOT NULL,
                    session_id VARCHAR(64),
                    request_id VARCHAR(64),
                    memory_type VARCHAR(64) NOT NULL,
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
        if transferable:
            columns_csv = ", ".join(transferable)
            connection.execute(
                text(
                    f"INSERT INTO memory_records_new ({columns_csv}) "
                    f"SELECT {columns_csv} FROM memory_records"
                )
            )
        connection.execute(text("DROP TABLE memory_records"))
        connection.execute(text("ALTER TABLE memory_records_new RENAME TO memory_records"))
        connection.execute(
            text("CREATE INDEX ix_memory_records_user_id ON memory_records (user_id)")
        )
        connection.execute(
            text("CREATE INDEX ix_memory_records_session_id ON memory_records (session_id)")
        )
        connection.execute(
            text("CREATE INDEX ix_memory_records_request_id ON memory_records (request_id)")
        )
        connection.execute(
            text("CREATE INDEX ix_memory_records_memory_type ON memory_records (memory_type)")
        )
        connection.execute(
            text("CREATE INDEX ix_memory_records_status ON memory_records (status)")
        )


def _rebuild_memory_eval_annotations_v2_table(engine, existing_columns: set[str]) -> None:
    current_columns = [column.name for column in MemoryEvalAnnotationV2Model.__table__.columns]
    transferable = [column for column in current_columns if column in existing_columns]
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE memory_eval_annotations_v2_new (
                    annotation_id VARCHAR(64) NOT NULL PRIMARY KEY,
                    batch_id VARCHAR(64) NOT NULL,
                    request_id VARCHAR(64) NOT NULL,
                    relevant_memory_ids_json TEXT NOT NULL DEFAULT '[]',
                    stale_memory_ids_json TEXT NOT NULL DEFAULT '[]',
                    claims_json TEXT NOT NULL DEFAULT '[]',
                    expected_extracted_memories_json TEXT NOT NULL DEFAULT '[]',
                    annotator VARCHAR(64) NOT NULL DEFAULT '',
                    created_at DATETIME NOT NULL
                )
                """
            )
        )
        if transferable:
            columns_csv = ", ".join(transferable)
            connection.execute(
                text(
                    f"INSERT INTO memory_eval_annotations_v2_new ({columns_csv}) "
                    f"SELECT {columns_csv} FROM memory_eval_annotations_v2"
                )
            )
        connection.execute(text("DROP TABLE memory_eval_annotations_v2"))
        connection.execute(
            text(
                "ALTER TABLE memory_eval_annotations_v2_new "
                "RENAME TO memory_eval_annotations_v2"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX ix_memory_eval_annotations_v2_batch_id "
                "ON memory_eval_annotations_v2 (batch_id)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX ix_memory_eval_annotations_v2_request_id "
                "ON memory_eval_annotations_v2 (request_id)"
            )
        )
