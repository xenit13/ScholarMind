from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SessionModel(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    topics_json: Mapped[str] = mapped_column(Text, default="[]")
    memory_context_loaded: Mapped[bool] = mapped_column(Boolean, default=False)
    last_state_json: Mapped[str] = mapped_column(Text, default="{}")


class EvalReportModel(Base):
    __tablename__ = "eval_reports"

    report_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    type: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    config_json: Mapped[str] = mapped_column(Text)
    results_json: Mapped[str] = mapped_column(Text)


class ConversationMetricModel(Base):
    __tablename__ = "conversation_metrics"

    request_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    query_type: Mapped[str] = mapped_column(String(32))
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    retrieval_latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    citations_count: Mapped[int] = mapped_column(Integer, default=0)
    retrieved_chunks_count: Mapped[int] = mapped_column(Integer, default=0)
    output_length: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    agent_path_json: Mapped[str] = mapped_column(Text, default="[]")
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class MemoryMetricModel(Base):
    __tablename__ = "memory_metrics"

    metric_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    extracted_count: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class MemoryRecordModel(Base):
    __tablename__ = "memory_records"

    memory_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    scope: Mapped[str] = mapped_column(String(32), default="user")
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    memory_type: Mapped[str] = mapped_column(String(64), default="interaction_summary", index=True)
    content: Mapped[str] = mapped_column(Text)
    structured_json: Mapped[str] = mapped_column(Text, default="{}")
    keywords_json: Mapped[str] = mapped_column(Text, default="[]")
    source: Mapped[str] = mapped_column(String(64), default="conversation")
    evidence_json: Mapped[str] = mapped_column(Text, default="[]")
    importance: Mapped[float] = mapped_column(Float, default=0.6)
    confidence: Mapped[float] = mapped_column(Float, default=0.7)
    sensitivity: Mapped[str] = mapped_column(String(32), default="normal")
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_accessed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    decay_rate: Mapped[float] = mapped_column(Float, default=0.03)
    decay_floor: Mapped[float] = mapped_column(Float, default=0.3)
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    access_count_30d: Mapped[int] = mapped_column(Integer, default=0)
    last_decay_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    supersedes_json: Mapped[str] = mapped_column(Text, default="[]")
    superseded_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)


class MemoryOperationEventModel(Base):
    __tablename__ = "memory_operation_events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    operation: Mapped[str] = mapped_column(String(32), index=True)
    memory_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    candidate_json: Mapped[str] = mapped_column(Text, default="{}")
    old_record_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    new_record_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    model: Mapped[str] = mapped_column(String(64), default="rule")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


# ---------------------------------------------------------------------------
# Request audit models
# ---------------------------------------------------------------------------


class RequestRunModel(Base):
    __tablename__ = "request_runs"

    request_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    query: Mapped[str] = mapped_column(Text)
    query_type: Mapped[str] = mapped_column(String(32))
    final_answer: Mapped[str] = mapped_column(Text, default="")
    memory_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    execution_health_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    has_retry: Mapped[bool] = mapped_column(Boolean, default=False)
    has_fallback: Mapped[bool] = mapped_column(Boolean, default=False)
    execution_health_json: Mapped[str] = mapped_column(Text, default="{}")
    runtime_metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    agent_trace_json: Mapped[str] = mapped_column(Text, default="[]")
    agent_events_json: Mapped[str] = mapped_column(Text, default="[]")
    answer_event_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class MemoryRetrievalEventV2Model(Base):
    __tablename__ = "memory_retrieval_events_v2"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    query: Mapped[str] = mapped_column(Text, default="")
    embedding_latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    vector_search_latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    retrieved_memory_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    retrieved_scores_json: Mapped[str] = mapped_column(Text, default="[]")
    retrieved_count: Mapped[int] = mapped_column(Integer, default=0)
    injected_memory_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    injected_count: Mapped[int] = mapped_column(Integer, default=0)
    injected_text: Mapped[str] = mapped_column(Text, default="")
    injected_tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class MemoryExtractionEventV2Model(Base):
    __tablename__ = "memory_extraction_events_v2"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True, unique=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    dispatch_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dispatch_success: Mapped[bool] = mapped_column(Boolean, default=False)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    written_memory_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    written_memory_texts_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class MemoryEvalAnnotationBatchModel(Base):
    __tablename__ = "memory_eval_annotation_batches"

    batch_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    version: Mapped[str] = mapped_column(String(32), default="memory_eval_v2")
    status: Mapped[str] = mapped_column(String(32), default="exported")
    k: Mapped[int] = mapped_column(Integer, default=5)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    annotated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    report_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


class MemoryEvalAnnotationV2Model(Base):
    __tablename__ = "memory_eval_annotations_v2"

    annotation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    batch_id: Mapped[str] = mapped_column(String(64), index=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    relevant_memory_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    stale_memory_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    claims_json: Mapped[str] = mapped_column(Text, default="[]")
    expected_extracted_memories_json: Mapped[str] = mapped_column(Text, default="[]")
    annotator: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class MemoryEvalRunV2Model(Base):
    __tablename__ = "memory_eval_runs_v2"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    batch_id: Mapped[str] = mapped_column(String(64), index=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    memory_score: Mapped[float] = mapped_column(default=0.0)
    memory_injected_count: Mapped[int] = mapped_column(Integer, default=0)
    memory_injected_latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    memory_injected_tokens: Mapped[int] = mapped_column(Integer, default=0)
    memory_hit_at_k: Mapped[float | None] = mapped_column(nullable=True)
    memory_relevant_recall: Mapped[float | None] = mapped_column(nullable=True)
    memory_relevant_precision: Mapped[float | None] = mapped_column(nullable=True)
    first_relevant_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    memory_stale_retrieval_rate: Mapped[float | None] = mapped_column(nullable=True)
    memory_answer_relevance: Mapped[float | None] = mapped_column(nullable=True)
    memory_extraction_precision: Mapped[float | None] = mapped_column(nullable=True)
    memory_extraction_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    memory_extraction_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    score_breakdown_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class MemoryEvalReportV2Model(Base):
    __tablename__ = "memory_eval_reports_v2"

    report_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    batch_id: Mapped[str] = mapped_column(String(64), index=True)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_memory_score: Mapped[float] = mapped_column(default=0.0)
    avg_memory_hit_at_k: Mapped[float | None] = mapped_column(nullable=True)
    avg_memory_relevant_recall: Mapped[float | None] = mapped_column(nullable=True)
    avg_memory_relevant_precision: Mapped[float | None] = mapped_column(nullable=True)
    avg_first_relevant_rank: Mapped[float | None] = mapped_column(nullable=True)
    avg_memory_stale_retrieval_rate: Mapped[float | None] = mapped_column(nullable=True)
    avg_memory_answer_relevance: Mapped[float | None] = mapped_column(nullable=True)
    avg_memory_extraction_precision: Mapped[float | None] = mapped_column(nullable=True)
    summary_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class MemoryLibraryAuditBatchModel(Base):
    __tablename__ = "memory_library_audit_batches"

    batch_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), default="exported")
    memory_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    report_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


class MemoryLibraryAuditReportModel(Base):
    __tablename__ = "memory_library_audit_reports"

    report_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    batch_id: Mapped[str] = mapped_column(String(64), index=True)
    memory_count: Mapped[int] = mapped_column(Integer, default=0)
    duplicate_pair_count: Mapped[int] = mapped_column(Integer, default=0)
    duplicate_memory_count: Mapped[int] = mapped_column(Integer, default=0)
    duplicate_memory_ratio: Mapped[float | None] = mapped_column(nullable=True)
    conflict_pair_count: Mapped[int] = mapped_column(Integer, default=0)
    conflict_memory_count: Mapped[int] = mapped_column(Integer, default=0)
    conflict_memory_ratio: Mapped[float | None] = mapped_column(nullable=True)
    summary_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
