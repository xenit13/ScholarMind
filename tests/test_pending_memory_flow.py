from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from scholar_mind.agents.planner import make_planner_node as _make_planner_node
from scholar_mind.agents.reviewer import make_reviewer_node as _make_reviewer_node
from scholar_mind.agents.state import flatten_graph_state
from scholar_mind.config.settings import Settings
from scholar_mind.db.session import build_session_factory, init_database
from scholar_mind.memory.admission import MemoryAdmissionAction, MemoryAdmissionModelOutput
from scholar_mind.memory.manager import MemoryManager
from scholar_mind.memory.pending_buffer import PendingConversationBuffer
from scholar_mind.memory.repository import MemoryRepository
from scholar_mind.models.domain import MemoryCandidate, MemoryCandidateExtractionOutput
from scholar_mind.utils.messages import serialize_messages


class _RawResult:
    def __init__(self, content: str, usage_metadata: dict[str, int]):
        self.content = content
        self.usage_metadata = usage_metadata


class _Runnable:
    def __init__(self, llm, payload):
        self.llm = llm
        self.payload = payload

    def invoke(self, prompt: str):
        self.llm.prompts.append(prompt)
        return self.payload

    async def ainvoke(self, prompt: str):
        return self.invoke(prompt)


class _StructuredOutputLLM:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.prompts: list[str] = []

    def with_structured_output(self, _schema, include_raw: bool = False):
        assert include_raw is True
        return _Runnable(self, self.payloads.pop(0))


class _FailingPlannerLLM:
    def with_structured_output(self, *_args, **_kwargs):
        raise AssertionError("planner should not call the model when query_type_hint is present")


class _PromptCatalog:
    def get(self, name: str) -> str:
        return f"{name} prompt"


class _Index:
    def search_memory(self, *_args, **_kwargs):
        return []

    def upsert_memory(self, *_args, **_kwargs):
        pass


class _Embedder:
    def embed_query(self, _content: str):
        return [0.1, 0.2]

    async def aembed_query(self, _content: str):
        return [0.1, 0.2]


def make_planner_node(*args, **kwargs):
    node = _make_planner_node(*args, **kwargs)

    async def _node(state):
        return flatten_graph_state(await node(state))

    return _node


def make_reviewer_node(*args, **kwargs):
    node = _make_reviewer_node(*args, **kwargs)

    async def _node(state):
        return flatten_graph_state(await node(state))

    return _node


def test_pending_buffer_keeps_only_human_and_assistant_natural_language():
    now = datetime(2026, 6, 17, tzinfo=UTC)
    buffer = PendingConversationBuffer(now_fn=lambda: now)

    buffer.add_round(
        user_id="u1",
        session_id="s1",
        request_id="req1",
        round_index=3,
        messages=[
            HumanMessage(content="请记住我偏好中文回答"),
            AIMessage(
                content="",
                tool_calls=[{"id": "call_1", "name": "search", "args": {"query": "x"}}],
            ),
            ToolMessage(content="tool result", tool_call_id="call_1"),
            AIMessage(content="好的，我会按中文回答。"),
        ],
    )

    payload = buffer.get_context_payload(user_id="u1", session_id="s1")

    assert payload.context == (
        "Pending conversation not yet extracted into memory:\n"
        "[request_id=req1 round=3]\n"
        "Human: 请记住我偏好中文回答\n"
        "Assistant: 好的，我会按中文回答。"
    )
    assert payload.hit_count == 1
    assert "tool result" not in payload.context
    assert "call_1" not in payload.context


def test_pending_buffer_caps_each_session_at_token_limit_by_keeping_tail():
    now = datetime(2026, 6, 17, tzinfo=UTC)
    buffer = PendingConversationBuffer(max_tokens=6, now_fn=lambda: now)

    buffer.add_round(
        user_id="u1",
        session_id="s1",
        request_id="req1",
        messages=[HumanMessage(content="旧内容" * 40)],
    )
    buffer.add_round(
        user_id="u1",
        session_id="s1",
        request_id="req2",
        messages=[HumanMessage(content="新内容" * 40)],
    )

    payload = buffer.get_context_payload(user_id="u1", session_id="s1")

    assert "旧内容" not in payload.context
    assert "新内容" in payload.context
    assert payload.token_estimate <= 6


def test_pending_buffer_expires_rounds_after_ttl():
    now = datetime(2026, 6, 17, tzinfo=UTC)
    current = {"now": now}
    buffer = PendingConversationBuffer(
        ttl=timedelta(hours=1),
        now_fn=lambda: current["now"],
    )
    buffer.add_round(
        user_id="u1",
        session_id="s1",
        request_id="req1",
        messages=[HumanMessage(content="请记住我偏好中文回答")],
    )

    current["now"] = now + timedelta(hours=1, seconds=1)

    assert buffer.get_context_payload(user_id="u1", session_id="s1").context == ""


def test_pending_buffer_rejected_round_final_notice_is_deduped_redacted_and_consumed():
    now = datetime(2026, 6, 17, tzinfo=UTC)
    buffer = PendingConversationBuffer(now_fn=lambda: now)
    buffer.add_round(
        user_id="u1",
        session_id="s1",
        request_id="req1",
        round_index=1,
        messages=[HumanMessage(content="请记住我的 api_key 是 demo_key_123456")],
    )
    for _ in range(2):
        buffer.mark_rejected(
            user_id="u1",
            session_id="s1",
            request_id="req1",
            round_index=1,
            user_question="请记住我的 api_key 是 demo_key_123456",
            reasons=["secrets"],
        )
    buffer.remove_round(user_id="u1", session_id="s1", request_id="req1")

    first = buffer.get_context_payload(user_id="u1", session_id="s1")
    second = buffer.get_context_payload(user_id="u1", session_id="s1")

    assert first.context == ""
    assert first.notices == [
        "问题「请记住我的 api_key 是 [已隐藏]」包含凭证、密钥或令牌类敏感信息，"
        "这类信息不适合保存到长期记忆，我没有将其写入记忆。"
    ]
    assert "demo_key_123456" not in first.notices[0]
    assert second.notices == []


@pytest.mark.asyncio
async def test_pending_memory_rejected_round_notice_reaches_final_answer(tmp_path):
    user_id = "u1"
    session_id = "s1"
    request_id = "req1"
    rejected_question = "请记住我的 api_key 是 demo_key_123456。请用一句话确认。"
    notice = (
        "问题「请记住我的 api_key 是 [已隐藏]。请用一句话确认。」"
        "包含凭证、密钥或令牌类敏感信息，"
        "这类信息不适合保存到长期记忆，我没有将其写入记忆。"
    )
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'memory.db'}",
        memory_root_dir=str(tmp_path / "memories"),
        log_dir=str(tmp_path / "logs"),
        qdrant_location=":memory:",
        bootstrap_sample_data=False,
    )
    init_database(settings)
    llm = _StructuredOutputLLM(
        [
            {
                "parsed": MemoryCandidateExtractionOutput(
                    candidates=[
                        MemoryCandidate(
                            memory_type="preference",
                            content="用户要求记住 api_key 是 demo_key_123456。",
                            structured={},
                            keywords=["api_key"],
                            importance=0.8,
                            confidence=0.9,
                            source="conversation",
                            evidence=[{"message_id": "s1-1-0", "role": "human"}],
                        )
                    ]
                ),
                "raw": _RawResult("", {"input_tokens": 9, "output_tokens": 6, "total_tokens": 15}),
                "parsing_error": None,
            },
            {
                "parsed": MemoryAdmissionModelOutput(
                    action=MemoryAdmissionAction.DROP,
                    reason="contains secret",
                    matched_rules=["secrets"],
                ),
                "raw": _RawResult("", {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}),
                "parsing_error": None,
            },
        ]
    )
    manager = MemoryManager(
        settings,
        _Index(),
        _Embedder(),
        llm=llm,
        memory_repository=MemoryRepository(build_session_factory(settings)),
    )
    messages = [HumanMessage(content=rejected_question)]
    manager.pending_buffer.add_round(
        user_id=user_id,
        session_id=session_id,
        request_id=request_id,
        round_index=1,
        messages=messages,
    )

    extraction = manager.extract_request_memories(
        user_id=user_id,
        request_id=request_id,
        round_messages=[{"message": serialize_messages(messages)[0]}],
    )
    planner = make_planner_node(_FailingPlannerLLM(), manager, _PromptCatalog())
    planned = await planner(
        {
            "query": "解释一下 RAG 是什么。",
            "user_id": user_id,
            "session_id": session_id,
            "query_type_hint": "qa",
        }
    )
    reviewer = make_reviewer_node(
        llm=_StructuredOutputLLM(
            [
                {
                    "parsed": None,
                    "raw": _RawResult(
                        "RAG 通过检索外部资料增强回答。",
                        {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12},
                    ),
                    "parsing_error": ValueError("invalid json"),
                }
            ]
        ),
        prompt_catalog=_PromptCatalog(),
    )
    reviewed = await reviewer(
        {
            "query": "解释一下 RAG 是什么。",
            "query_type": "qa",
            "draft": "RAG 是检索增强生成。",
            "memory_notices": planned["memory_notices"],
            "retrieved_chunks": [],
            "agent_trace": [],
        }
    )

    assert extraction["success"] is True
    assert extraction["written_count"] == 0
    assert planned["memory_context"] == ""
    assert planned["memory_notices"] == [notice]
    assert reviewed["final_answer"].startswith("RAG 通过检索外部资料增强回答。")
    assert reviewed["final_answer"].endswith(notice)
    assert "第 1 轮问题" not in reviewed["final_answer"]
    assert "demo_key_123456" not in reviewed["final_answer"]
