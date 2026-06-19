from __future__ import annotations

import inspect
import re
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from scholar_mind.memory.context import finish_memory_context, init_memory_context
from scholar_mind.models.domain import DailyChatRequest, DailyChatResponse
from scholar_mind.utils.messages import serialize_messages

_MEMORY_SAVE_PATTERNS = [
    re.compile(
        r"^\s*(?:请|请你|麻烦|麻烦你|帮我|帮忙|请帮我)?\s*(?:记住|记下|记一下|记着|记好)\s*(?:这件事|这点|这个|以下内容)?[\s:：,，]*(?P<content>.+?)\s*$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^\s*remember\s+(?:that\s+)?(?P<content>.+?)\s*$",
        flags=re.IGNORECASE,
    ),
]

_MEMORY_FOLLOWUP_TASK_PATTERN = re.compile(
    r"(?:。|！|!|？|\?|,|，)?\s*(?:顺便|另外|然后|接下来|再帮我|请帮我|帮我|并帮我|并给我|同时帮我|同时给我).*$",
    flags=re.IGNORECASE,
)


class DailyChatService:
    def __init__(self, *, settings, session_repository, memory_manager, llm):
        self.settings = settings
        self.session_repository = session_repository
        self.memory_manager = memory_manager
        self.llm = llm

    async def answer(self, request: DailyChatRequest) -> DailyChatResponse:
        query = request.query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if self.llm is None:
            raise RuntimeError("Chat LLM is not configured")

        request_id = uuid4().hex
        session_id = request.session_id or uuid4().hex
        self.session_repository.create_or_get(user_id=request.user_id, session_id=session_id)
        previous_state = self.session_repository.get_last_state(session_id)
        previous_messages = _state_messages(previous_state)
        eval_ctx = init_memory_context(
            session_id=session_id,
            user_id=request.user_id,
            query=query,
            query_type="daily_chat",
            request_id=request_id,
        )
        final_state: dict[str, Any] = {}
        try:
            context_payload = await self.memory_manager.get_context_payload(
                user_id=request.user_id,
                session_id=session_id,
                current_query=query,
            )
            memory_context = getattr(context_payload, "context", "") or ""
            memory_hit_count = int(getattr(context_payload, "hit_count", 0) or 0)
            memory_notices = list(getattr(context_payload, "notices", []) or [])

            prompt_messages = [
                SystemMessage(content=_build_system_prompt(memory_context)),
                HumanMessage(content=query),
            ]
            response = await _invoke_llm(self.llm, prompt_messages)
            answer = _response_text(response)
            round_messages: list[BaseMessage] = [
                HumanMessage(content=query),
                AIMessage(content=answer),
            ]
            round_index = _next_round_index(previous_messages)
            explicit_memories = _extract_explicit_memory_candidates(query)

            current_messages = previous_messages + round_messages
            final_state = {
                "query": query,
                "messages": serialize_messages(current_messages),
                "memory_context": memory_context,
                "memory_hit_count": memory_hit_count,
                "memory_notices": memory_notices,
                "final_answer": answer,
                "request_id": request_id,
            }
            self.session_repository.update_from_state(
                user_id=request.user_id,
                session_id=session_id,
                state=final_state,
            )
            pending_buffer = getattr(self.memory_manager, "pending_buffer", None)
            if pending_buffer is not None:
                pending_buffer.add_round(
                    user_id=request.user_id,
                    session_id=session_id,
                    request_id=request_id,
                    round_index=round_index,
                    messages=round_messages,
                )
            self.memory_manager.log_round(
                user_id=request.user_id,
                session_id=session_id,
                round_index=round_index,
                messages=round_messages,
                explicit_memories=explicit_memories,
            )
            return DailyChatResponse(
                answer=answer,
                session_id=session_id,
                request_id=request_id,
                memory_hit_count=memory_hit_count,
                memory_notices=memory_notices,
            )
        finally:
            finish_memory_context(eval_ctx, final_state)


def _build_system_prompt(memory_context: str) -> str:
    return (
        "You are a helpful daily chat assistant with long-term memory.\n\n"
        "Use the memory context only when it is relevant to the user's latest message.\n"
        "Do not mention memory IDs or retrieval mechanics.\n"
        "If memory conflicts with the user's latest message, prefer the latest message.\n"
        "If there is no relevant memory, answer normally.\n\n"
        f"Memory context:\n{memory_context or '(none)'}"
    )


async def _invoke_llm(llm, messages: list[BaseMessage]) -> Any:
    ainvoke = getattr(llm, "ainvoke", None)
    if callable(ainvoke):
        result = ainvoke(messages)
        if inspect.isawaitable(result):
            return await result
        return result
    invoke = getattr(llm, "invoke", None)
    if not callable(invoke):
        raise RuntimeError("Chat LLM does not expose invoke or ainvoke")
    return invoke(messages)


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content)


def _state_messages(state: dict[str, Any]) -> list[BaseMessage]:
    messages = state.get("messages") if isinstance(state, dict) else None
    if not messages:
        return []
    return list(messages)


def _next_round_index(previous_messages: list[BaseMessage]) -> int:
    return sum(1 for message in previous_messages if getattr(message, "type", "") == "human") + 1


def _extract_explicit_memory_candidates(query: str) -> list[str]:
    for pattern in _MEMORY_SAVE_PATTERNS:
        match = pattern.match(query.strip())
        if not match:
            continue
        content = _clean_explicit_memory_content(match.group("content"))
        if content:
            return [content]
    return []


def _clean_explicit_memory_content(content: str) -> str:
    content = _MEMORY_FOLLOWUP_TASK_PATTERN.sub("", content.strip())
    return content.strip(" \t\n\r。！!？?,，")
