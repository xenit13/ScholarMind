from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from scholar_mind.api.deps import container_dep
from scholar_mind.asgi import create_app
from scholar_mind.memory.context import get_memory_context
from scholar_mind.models.domain import DailyChatRequest, DailyChatResponse
from scholar_mind.services.chat import DailyChatService


class _FakeLLM:
    def __init__(self, answer: str = "我会按你的偏好简洁回答。"):
        self.answer = answer
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages)
        return AIMessage(content=self.answer)


class _FakePendingBuffer:
    def __init__(self):
        self.added_rounds = []

    def add_round(self, **kwargs):
        self.added_rounds.append(kwargs)


class _FakeMemoryManager:
    def __init__(self):
        self.pending_buffer = _FakePendingBuffer()
        self.context_calls = []
        self.active_contexts = []
        self.logged_rounds = []

    async def get_context_payload(self, **kwargs):
        self.context_calls.append(kwargs)
        self.active_contexts.append(get_memory_context())
        return SimpleNamespace(
            context="Persisted memory:\n- User prefers concise answers.",
            hit_count=1,
            notices=["Pending memory was included."],
        )

    def log_round(self, **kwargs):
        self.logged_rounds.append(kwargs)


class _FakeSessionRepository:
    def __init__(self, initial_state=None):
        self.initial_state = initial_state or {}
        self.created = []
        self.updated = []

    def create_or_get(self, user_id, session_id):
        self.created.append({"user_id": user_id, "session_id": session_id})
        return SimpleNamespace(session_id=session_id, user_id=user_id)

    def get_last_state(self, session_id):
        return self.initial_state

    def update_from_state(self, *, user_id, session_id, state):
        self.updated.append({"user_id": user_id, "session_id": session_id, "state": state})
        return SimpleNamespace(session_id=session_id, user_id=user_id)


@pytest.mark.asyncio
async def test_daily_chat_injects_memory_and_records_round_for_future_memory():
    llm = _FakeLLM()
    memory_manager = _FakeMemoryManager()
    session_repository = _FakeSessionRepository()
    service = DailyChatService(
        settings=SimpleNamespace(),
        session_repository=session_repository,
        memory_manager=memory_manager,
        llm=llm,
    )

    response = await service.answer(
        DailyChatRequest(user_id="u1", session_id="s1", query="我今天应该怎么安排？")
    )

    assert response.answer == "我会按你的偏好简洁回答。"
    assert response.memory_hit_count == 1
    assert response.memory_notices == ["Pending memory was included."]
    assert memory_manager.context_calls == [
        {
            "user_id": "u1",
            "session_id": "s1",
            "current_query": "我今天应该怎么安排？",
        }
    ]
    assert memory_manager.active_contexts[0] is not None
    assert memory_manager.active_contexts[0].query_type == "daily_chat"
    assert memory_manager.active_contexts[0].request_id == response.request_id
    assert get_memory_context() is None
    system_prompt = llm.prompts[0][0].content
    assert "daily chat assistant" in system_prompt
    assert "User prefers concise answers." in system_prompt
    assert "If memory conflicts with the user's latest message" in system_prompt
    assert llm.prompts[0][1] == HumanMessage(content="我今天应该怎么安排？")

    logged = memory_manager.logged_rounds[0]
    assert logged["user_id"] == "u1"
    assert logged["session_id"] == "s1"
    assert logged["round_index"] == 1
    assert [message.type for message in logged["messages"]] == ["human", "ai"]

    pending = memory_manager.pending_buffer.added_rounds[0]
    assert pending["user_id"] == "u1"
    assert pending["session_id"] == "s1"
    assert pending["round_index"] == 1
    assert [message.type for message in pending["messages"]] == ["human", "ai"]

    stored = session_repository.updated[0]["state"]
    assert stored["query"] == "我今天应该怎么安排？"
    assert stored["memory_context"] == "Persisted memory:\n- User prefers concise answers."
    assert stored["memory_hit_count"] == 1
    assert len(stored["messages"]) == 2


@pytest.mark.asyncio
async def test_daily_chat_passes_explicit_memory_candidates_to_log_round():
    memory_manager = _FakeMemoryManager()
    service = DailyChatService(
        settings=SimpleNamespace(),
        session_repository=_FakeSessionRepository(),
        memory_manager=memory_manager,
        llm=_FakeLLM(),
    )

    await service.answer(
        DailyChatRequest(
            user_id="u1",
            session_id="s1",
            query="请记住我喜欢简洁回答，顺便告诉我今天怎么安排",
        )
    )

    assert memory_manager.logged_rounds[0]["explicit_memories"] == ["我喜欢简洁回答"]


def test_chat_stream_route_returns_sse_answer(monkeypatch):
    class _FakeChatService:
        async def answer(self, request):
            assert request.user_id == "u1"
            assert request.session_id == "s1"
            assert request.query == "你好"
            return DailyChatResponse(
                answer="你好，我记得你的偏好。",
                session_id="s1",
                request_id="req_1",
                memory_hit_count=2,
                memory_notices=[],
            )

    app = create_app()
    app.dependency_overrides[container_dep] = lambda: SimpleNamespace(
        chat_service=_FakeChatService()
    )
    client = TestClient(app)

    response = client.post(
        "/api/v1/chat/stream",
        json={"user_id": "u1", "session_id": "s1", "query": "你好"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: message" in response.text
    assert '"answer": "你好，我记得你的偏好。"' in response.text
    assert '"memory_hit_count": 2' in response.text
