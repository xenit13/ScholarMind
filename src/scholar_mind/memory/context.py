"""Per-request memory evaluation context."""

from __future__ import annotations

from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any

from scholar_mind.models.eval_models import MemoryCallEvent, RequestEvalContext

_current_memory_context: ContextVar[RequestEvalContext | None] = ContextVar(
    "scholar_mind_memory_context", default=None
)
_token_stack: list[Token] = []


def init_memory_context(
    *,
    session_id: str,
    user_id: str,
    query: str,
    query_type: str,
    request_id: str | None = None,
) -> RequestEvalContext:
    ctx = RequestEvalContext(
        session_id=session_id,
        user_id=user_id,
        query=query,
        query_type=query_type,
        **({"request_id": request_id} if request_id else {}),
    )
    token = _current_memory_context.set(ctx)
    _token_stack.append(token)
    return ctx


def get_memory_context() -> RequestEvalContext | None:
    return _current_memory_context.get()


def record_memory_event(event: MemoryCallEvent) -> None:
    ctx = _current_memory_context.get()
    if ctx is not None:
        ctx.memory_events.append(event)


def finish_memory_context(
    ctx: RequestEvalContext,
    final_state: dict[str, Any] | None = None,
) -> RequestEvalContext:
    ctx.finished_at = datetime.now(UTC)
    if final_state:
        ctx.final_state_summary = {
            "has_final_answer": bool(final_state.get("final_answer")),
            "has_memory_context": bool(final_state.get("memory_context")),
        }
    if _token_stack:
        token = _token_stack.pop()
        _current_memory_context.reset(token)
    return ctx
