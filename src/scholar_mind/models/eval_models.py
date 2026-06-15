"""Request audit event models used by the backend.

Document 23 removes the old per-request answer/RAG proxy scoring models. This
module now keeps only neutral request trace structures and v2 retrieval events.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from scholar_mind.rag.top_k import FINAL_CITATION_TOP_K


def utcnow() -> datetime:
    return datetime.now(UTC)


def uid(prefix: str = "") -> str:
    return f"{prefix}{uuid4().hex}"


class RagRetrievalEventV2(BaseModel):
    event_id: str = Field(default_factory=lambda: uid("ragret_"))
    request_id: str = ""
    query: str = ""
    normalized_query: str | None = None
    strategy: str = "hybrid"
    top_k: int = FINAL_CITATION_TOP_K
    filters: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0
    returned_contexts: list[str] = Field(default_factory=list)
    returned_chunk_ids: list[str] = Field(default_factory=list)
    returned_paper_ids: list[str] = Field(default_factory=list)
    caller_agent: str | None = None
    tool_name: str = "rag_retrieve"
    created_at: datetime = Field(default_factory=utcnow)


class MemoryOperation(StrEnum):
    CONTEXT_RETRIEVE = "memory_context_retrieve"
    CONVERSATION_COMPRESS = "conversation_compress"
    MEMORY_INJECTION = "memory_injection"
    MEMORY_WRITE = "memory_write"


class MemoryCallEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uid("mem_"))
    request_id: str = ""
    operation: MemoryOperation = MemoryOperation.CONTEXT_RETRIEVE
    query: str | None = None
    latency_ms: int | None = None
    hit_count: int | None = None
    injected_text: str | None = None
    injected_chars: int | None = None
    source_memory_ids: list[str] = Field(default_factory=list)
    compression_before_tokens: int | None = None
    compression_after_tokens: int | None = None
    created_at: datetime = Field(default_factory=utcnow)


class AgentEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uid("agt_"))
    request_id: str = ""
    agent: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int = 0
    output_summary: str = ""
    created_at: datetime = Field(default_factory=utcnow)


class AnswerEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uid("ans_"))
    request_id: str = ""
    draft: str = ""
    final_answer: str = ""
    citations: list[dict[str, Any]] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


class RequestEvalContext(BaseModel):
    request_id: str = Field(default_factory=lambda: uid("req_"))
    session_id: str = ""
    user_id: str = ""
    query: str = ""
    query_type: str = ""
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    rag_events: list[RagRetrievalEventV2] = Field(default_factory=list)
    memory_events: list[MemoryCallEvent] = Field(default_factory=list)
    agent_events: list[AgentEvent] = Field(default_factory=list)
    answer_events: list[AnswerEvent] = Field(default_factory=list)
    final_state_summary: dict[str, Any] = Field(default_factory=dict)
