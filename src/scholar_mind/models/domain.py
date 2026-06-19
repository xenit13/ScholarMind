from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


class SessionInfo(BaseModel):
    session_id: str
    user_id: str
    created_at: datetime
    closed_at: datetime | None = None
    message_count: int = 0
    topics_discussed: list[str] = Field(default_factory=list)
    memory_context_loaded: bool = False


class MemoryType(StrEnum):
    PREFERENCE = "preference"
    RESEARCH_INTEREST = "research_interest"
    KNOWLEDGE_LEVEL = "knowledge_level"
    GOAL = "goal"
    WORKFLOW = "workflow"
    PROJECT_CONSTRAINT = "project_constraint"
    INTERACTION_SUMMARY = "interaction_summary"
    FEEDBACK = "feedback"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


class MemoryOperationName(StrEnum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NONE = "NONE"
    ARCHIVE = "ARCHIVE"
    RESTORE = "RESTORE"


class MemoryRecord(BaseModel):
    record_id: str
    user_id: str
    created_at: datetime
    source: str
    content: str


class StructuredMemoryRecord(BaseModel):
    memory_id: str
    user_id: str
    scope: Literal["user", "session", "org"] = "user"
    session_id: str | None = None
    request_id: str | None = None
    memory_type: MemoryType = MemoryType.INTERACTION_SUMMARY
    content: str
    structured: dict[str, Any] = Field(default_factory=dict)
    keywords: list[str] = Field(default_factory=list)
    source: Literal["explicit", "conversation", "system_extracted", "user_edited"]
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    importance: float = Field(default=0.6, ge=0.0, le=1.0)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    sensitivity: Literal["normal", "sensitive"] = "normal"
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: datetime
    updated_at: datetime
    last_accessed_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    expires_at: datetime | None = None
    decay_rate: float = 0.03
    decay_floor: float = 0.3
    access_count: int = 0
    access_count_30d: int = 0
    last_decay_score: float | None = None
    supersedes: list[str] = Field(default_factory=list)
    superseded_by: str | None = None
    version: int = 1

    @property
    def record_id(self) -> str:
        return self.memory_id

    def to_memory_record(self) -> MemoryRecord:
        return MemoryRecord(
            record_id=self.memory_id,
            user_id=self.user_id,
            created_at=self.created_at,
            source=self.source,
            content=self.content,
        )


class MemoryCandidate(BaseModel):
    memory_type: MemoryType
    content: str
    structured: dict[str, Any] = Field(default_factory=dict)
    keywords: list[str] = Field(default_factory=list)
    importance: float = Field(default=0.6, ge=0.0, le=1.0)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    source: Literal["explicit", "conversation", "system_extracted"]
    evidence: list[dict[str, Any]] = Field(default_factory=list)


class MemoryCandidateExtractionOutput(BaseModel):
    candidates: list[MemoryCandidate] = Field(default_factory=list)


class MemoryOperationEvent(BaseModel):
    event_id: str
    user_id: str
    operation: MemoryOperationName
    memory_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    candidate: dict[str, Any] = Field(default_factory=dict)
    old_record: dict[str, Any] | None = None
    new_record: dict[str, Any] | None = None
    reason: str = ""
    model: str = "rule"
    created_at: datetime


class MemoryOperationResult(BaseModel):
    operation: MemoryOperationName
    memory_id: str | None = None
    record: StructuredMemoryRecord | None = None
    event_id: str | None = None
    reason: str = ""


class MessageLogEntry(BaseModel):
    message_id: str
    thread_id: str
    user_id: str
    message: dict[str, Any]
    timestamp: datetime
    round_index: int


class MemoryExtractionOutput(BaseModel):
    memories: list[str] = Field(default_factory=list)


class CompressionOutput(BaseModel):
    summary: str = ""


class ReportSummary(BaseModel):
    report_id: str
    type: str
    created_at: datetime
    config: dict[str, Any]
    results: dict[str, Any]


class SessionCreateRequest(BaseModel):
    user_id: str


class DailyChatRequest(BaseModel):
    user_id: str
    query: str = Field(min_length=1)
    session_id: str | None = None


class DailyChatResponse(BaseModel):
    answer: str
    session_id: str
    request_id: str
    memory_hit_count: int = 0
    memory_notices: list[str] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)


T = TypeVar("T")


class ResponseMeta(BaseModel):
    request_id: str
    timestamp: datetime = Field(default_factory=utcnow)
    latency_ms: int | None = None


class ErrorPayload(BaseModel):
    code: str
    message: str
    details: str | None = None


class ApiResponse(BaseModel, Generic[T]):
    success: bool
    data: T | None = None
    error: ErrorPayload | None = None
    meta: ResponseMeta


class StreamEvent(BaseModel):
    event: str
    data: dict[str, Any]
