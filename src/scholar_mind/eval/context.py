"""RequestEvalContext – per-request evaluation context using contextvars.

This module provides a thread-safe, coroutine-safe evaluation context that
is created at the start of each user request and populated during execution
by instrumented tools, memory operations, and agent nodes.

Usage::

    from scholar_mind.eval.context import init_eval_context, get_eval_context

    # In ResearchService._execute():
    ctx = init_eval_context(
        session_id=session_id,
        user_id=user_id,
        query=query,
        query_type=query_type.value,
    )

    # In instrumented tools/nodes:
    ctx = get_eval_context()
    if ctx is not None:
        ctx.rag_events.append(RagRetrievalEventV2(...))

    # After request completes:
    report = finish_eval_context(ctx, final_state)
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any

from scholar_mind.models.eval_models import (
    AgentEvent,
    AnswerEvent,
    MemoryCallEvent,
    RagRetrievalEventV2,
    RequestEvalContext,
)

_current_eval_context: ContextVar[RequestEvalContext | None] = ContextVar(
    "scholar_mind_eval_context", default=None
)

_token_stack: list[Token] = []


def init_eval_context(
    *,
    session_id: str,
    user_id: str,
    query: str,
    query_type: str,
    request_id: str | None = None,
) -> RequestEvalContext:
    """Create and set a new RequestEvalContext for the current request."""
    ctx = RequestEvalContext(
        session_id=session_id,
        user_id=user_id,
        query=query,
        query_type=query_type,
        **({"request_id": request_id} if request_id else {}),
    )
    token = _current_eval_context.set(ctx)
    _token_stack.append(token)
    return ctx


def get_eval_context() -> RequestEvalContext | None:
    """Return the current RequestEvalContext, or None if not initialised."""
    return _current_eval_context.get()


def record_rag_event(event: RagRetrievalEventV2) -> None:
    """Append a RAG retrieval event to the current context."""
    ctx = _current_eval_context.get()
    if ctx is not None:
        ctx.rag_events.append(event)


def record_memory_event(event: MemoryCallEvent) -> None:
    """Append a Memory call event to the current context."""
    ctx = _current_eval_context.get()
    if ctx is not None:
        ctx.memory_events.append(event)


def record_agent_event(event: AgentEvent) -> None:
    """Append an Agent execution event to the current context."""
    ctx = _current_eval_context.get()
    if ctx is not None:
        ctx.agent_events.append(event)


def record_answer_event(event: AnswerEvent) -> None:
    """Append an Answer generation event to the current context."""
    ctx = _current_eval_context.get()
    if ctx is not None:
        ctx.answer_events.append(event)


def finish_eval_context(
    ctx: RequestEvalContext,
    final_state: dict[str, Any] | None = None,
) -> RequestEvalContext:
    """Finalize the evaluation context after the request completes.

    Sets finished_at and populates final_state_summary.
    """
    ctx.finished_at = datetime.now(UTC)
    if final_state:
        ctx.final_state_summary = {
            "retrieved_chunks_count": len(final_state.get("retrieved_chunks", [])),
            "citations_count": len(final_state.get("citations", [])),
            "rag_latency_ms": final_state.get("rag_latency_ms", 0),
            "has_final_answer": bool(final_state.get("final_answer")),
            "has_draft": bool(final_state.get("draft")),
        }
    # Pop the token to reset context
    if _token_stack:
        token = _token_stack.pop()
        _current_eval_context.reset(token)
    return ctx
