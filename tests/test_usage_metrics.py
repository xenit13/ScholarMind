from __future__ import annotations

import sqlite3
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from scholar_mind.agents.common import merge_usage, usage_from_result
from scholar_mind.agents.state import flatten_graph_state
from scholar_mind.config.settings import Settings, get_settings
from scholar_mind.db.models import ConversationMetricModel, MemoryMetricModel
from scholar_mind.db.session import build_session_factory, init_database
from scholar_mind.memory.compressor import MessageCompressor
from scholar_mind.memory.manager import MemoryManager
from scholar_mind.models.domain import AskRequest, CompressionOutput, MemoryExtractionOutput
from scholar_mind.services.repositories import MetricsRepository
from scholar_mind.services.research import ResearchService, _usage_metrics


class _ResultWithUsage:
    usage_metadata = {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}


class _RawResult:
    def __init__(self, content: str, usage_metadata: dict[str, int]):
        self.content = content
        self.usage_metadata = usage_metadata


class _Runnable:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, _prompt: str):
        return self.payload


class _StructuredLLM:
    def __init__(self, payload):
        self.payload = payload

    def with_structured_output(self, _schema, include_raw: bool = False):
        return _Runnable(self.payload)


class _SequentialStructuredLLM:
    def __init__(self, payloads):
        self.payloads = list(payloads)

    def with_structured_output(self, _schema, include_raw: bool = False):
        return _Runnable(self.payloads.pop(0))


class _RecordingMetricsRepository:
    def __init__(self):
        self.rounds: list[dict] = []

    def record_round(self, **kwargs):
        self.rounds.append(kwargs)


class _StubSessionRepository:
    def create_or_get(self, *, user_id: str, session_id: str):
        return {"user_id": user_id, "session_id": session_id}

    def update_from_state(self, *, user_id: str, session_id: str, state: dict):
        return {"user_id": user_id, "session_id": session_id, "state": state}


class _StubCompressor:
    def __init__(self, usage: dict[str, float]):
        self.usage = usage

    def compress_with_usage(self, messages):
        return messages, self.usage


class _StubMemoryManager:
    def __init__(self, usage: dict[str, float]):
        self.compressor = _StubCompressor(usage)

    def log_round(self, **_kwargs):
        return None

    def extract_pending_memories(self, user_id: str | None = None):
        return 0


class _RecordingMemoryManager:
    def __init__(self, usage: dict[str, float]):
        self.compressor = _StubCompressor(usage)
        self.logged_rounds: list[dict] = []
        self.extracted_user_ids: list[str | None] = []

    def log_round(self, **kwargs):
        self.logged_rounds.append(kwargs)

    def extract_pending_memories(self, user_id: str | None = None):
        self.extracted_user_ids.append(user_id)
        return 0


class _RecordingPendingBuffer:
    def __init__(self):
        self.rounds: list[dict] = []

    def add_round(self, **kwargs):
        self.rounds.append(kwargs)


class _RecordingMemoryManagerWithPending(_RecordingMemoryManager):
    def __init__(self, usage: dict[str, float]):
        super().__init__(usage)
        self.pending_buffer = _RecordingPendingBuffer()


class _StubOrchestrator:
    async def get_state(self, _session_id: str):
        return {}

    async def run(self, state: dict):
        flat_state = flatten_graph_state(state)
        return {
            "session_id": flat_state["session_id"],
            "messages": state["messages"] + [AIMessage(content="answer")],
            "final_answer": "answer",
            "citations": [],
            "related_papers": [],
            "retrieved_chunks": [],
            "rag_latency_ms": 0,
            "agent_trace": [],
            "llm_usage": merge_usage(
                flat_state.get("llm_usage"),
                {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                    "latency_ms": 5,
                },
            ),
        }


class _CapturingOrchestrator(_StubOrchestrator):
    def __init__(self):
        self.states: list[dict] = []

    async def run(self, state: dict):
        self.states.append(state)
        return await super().run(state)


class _FakeEmbeddingService:
    def embed_query(self, _text: str):
        return [0.1, 0.2]

    async def aembed_query(self, _text: str):
        return [0.1, 0.2]


class _FakeIndex:
    def __init__(self):
        self.records = []

    def search_memory(self, **_kwargs):
        return []

    def upsert_memory(self, record, _embedding):
        self.records.append(record)


def test_usage_from_result_tracks_tokens_without_price_estimate():
    usage = usage_from_result(
        prompt="prompt text",
        completion="completion text",
        timecost_s=0.25,
        result=_ResultWithUsage(),
    )

    assert usage == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "latency_ms": 250,
    }


def test_usage_metrics_only_returns_token_counts():
    merged = merge_usage(
        {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5, "latency_ms": 10},
        {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5, "latency_ms": 15},
    )

    assert _usage_metrics({"llm_usage": merged}) == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }


def test_message_compressor_returns_usage_for_summary_call():
    compressor = MessageCompressor(
        context_window_tokens=1000,
        compact_threshold_ratio=0.5,
        llm=_StructuredLLM(
            {
                "parsed": CompressionOutput(summary="compressed summary"),
                "raw": _RawResult(
                    '{"summary":"compressed summary"}',
                    {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12},
                ),
                "parsing_error": None,
            }
        ),
    )
    messages = [
        HumanMessage(content="round 1 " + ("x" * 1000)),
        AIMessage(content="answer 1 " + ("y" * 1000)),
        HumanMessage(content="round 2"),
        AIMessage(content="answer 2"),
        HumanMessage(content="round 3"),
        AIMessage(content="answer 3"),
    ]

    compressed, usage = compressor.compress_with_usage(messages)

    assert compressed[0].content == "compressed summary"
    assert usage["prompt_tokens"] == 8
    assert usage["completion_tokens"] == 4
    assert usage["total_tokens"] == 12
    assert usage["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_research_service_includes_compressor_usage_in_request_metrics():
    metrics_repository = _RecordingMetricsRepository()
    service = ResearchService(
        settings=SimpleNamespace(
            default_top_k=8,
            eval_enabled=False,
        ),
        session_repository=_StubSessionRepository(),
        metrics_repository=metrics_repository,
        memory_manager=_StubMemoryManager(
            {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7, "latency_ms": 11}
        ),
        orchestrator=_StubOrchestrator(),
    )

    await service.ask(
        AskRequest(query="What does hybrid retrieval improve?", user_id="metrics-user")
    )

    recorded = metrics_repository.rounds[0]
    assert recorded["user_id"] == "metrics-user"
    assert recorded["query_type"] == "qa"
    assert recorded["prompt_tokens"] == 8
    assert recorded["completion_tokens"] == 3
    assert recorded["total_tokens"] == 11
    assert recorded["output_length"] == 6


@pytest.mark.asyncio
async def test_research_service_passes_conditional_memory_injection_flag():
    orchestrator = _CapturingOrchestrator()
    service = ResearchService(
        settings=SimpleNamespace(
            default_top_k=8,
            eval_enabled=False,
        ),
        session_repository=_StubSessionRepository(),
        metrics_repository=_RecordingMetricsRepository(),
        memory_manager=_StubMemoryManager(
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        ),
        orchestrator=orchestrator,
    )

    await service.ask(
        AskRequest(
            query="What does hybrid retrieval improve?",
            user_id="metrics-user",
            conditional_memory_injection=True,
        )
    )

    payload = orchestrator.states[0]["request"]["payload"]
    assert payload["conditional_memory_injection"] is True


@pytest.mark.asyncio
async def test_research_service_uses_configured_memory_injection_default():
    orchestrator = _CapturingOrchestrator()
    service = ResearchService(
        settings=SimpleNamespace(
            default_top_k=8,
            eval_enabled=False,
            conditional_memory_injection=True,
        ),
        session_repository=_StubSessionRepository(),
        metrics_repository=_RecordingMetricsRepository(),
        memory_manager=_StubMemoryManager(
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        ),
        orchestrator=orchestrator,
    )

    await service.ask(AskRequest(query="What does hybrid retrieval improve?", user_id="u1"))

    payload = orchestrator.states[0]["request"]["payload"]
    assert payload["conditional_memory_injection"] is True


@pytest.mark.asyncio
async def test_research_service_request_memory_injection_flag_overrides_config():
    orchestrator = _CapturingOrchestrator()
    service = ResearchService(
        settings=SimpleNamespace(
            default_top_k=8,
            eval_enabled=False,
            conditional_memory_injection=True,
        ),
        session_repository=_StubSessionRepository(),
        metrics_repository=_RecordingMetricsRepository(),
        memory_manager=_StubMemoryManager(
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        ),
        orchestrator=orchestrator,
    )

    await service.ask(
        AskRequest(
            query="What does hybrid retrieval improve?",
            user_id="u1",
            conditional_memory_injection=False,
        )
    )

    payload = orchestrator.states[0]["request"]["payload"]
    assert payload["conditional_memory_injection"] is False


@pytest.mark.asyncio
async def test_research_service_skips_request_scoped_memory_extraction_without_v2_repository():
    memory_manager = _RecordingMemoryManager(
        {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7, "latency_ms": 11}
    )
    service = ResearchService(
        settings=SimpleNamespace(
            default_top_k=8,
            eval_enabled=False,
        ),
        session_repository=_StubSessionRepository(),
        metrics_repository=_RecordingMetricsRepository(),
        memory_manager=memory_manager,
        orchestrator=_StubOrchestrator(),
    )

    await service.ask(
        AskRequest(query="What does hybrid retrieval improve?", user_id="fallback-user")
    )

    assert memory_manager.logged_rounds
    assert memory_manager.extracted_user_ids == []


def test_research_service_payload_can_disable_memory_extraction():
    memory_manager = _RecordingMemoryManagerWithPending(
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    dispatched = []
    service = ResearchService(
        settings=SimpleNamespace(default_top_k=8, eval_enabled=False),
        session_repository=_StubSessionRepository(),
        metrics_repository=_RecordingMetricsRepository(),
        memory_manager=memory_manager,
        orchestrator=_StubOrchestrator(),
    )
    service._dispatch_request_memory_extraction = lambda **kwargs: dispatched.append(kwargs)

    service._persist_state(
        user_id="u1",
        session_id="s1",
        request_id="req1",
        query="What should I remember?",
        request_payload={"memory_extraction_enabled": False},
        previous_state={"messages": []},
        result={
            "session_id": "s1",
            "messages": [
                HumanMessage(content="What should I remember?"),
                AIMessage(content="answer"),
            ],
            "final_answer": "answer",
            "citations": [],
            "related_papers": [],
            "retrieved_chunks": [],
            "rag_latency_ms": 0,
            "agent_trace": [],
            "llm_usage": {},
        },
    )

    assert memory_manager.pending_buffer.rounds == []
    assert memory_manager.logged_rounds == []
    assert dispatched == []


def test_research_service_payload_can_disable_request_scoped_memory_extraction_only():
    memory_manager = _RecordingMemoryManagerWithPending(
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    dispatched = []
    service = ResearchService(
        settings=SimpleNamespace(default_top_k=8, eval_enabled=False),
        session_repository=_StubSessionRepository(),
        metrics_repository=_RecordingMetricsRepository(),
        memory_manager=memory_manager,
        orchestrator=_StubOrchestrator(),
    )
    service._dispatch_request_memory_extraction = lambda **kwargs: dispatched.append(kwargs)

    service._persist_state(
        user_id="u1",
        session_id="s1",
        request_id="req1",
        query="Please remember this.",
        request_payload={"request_memory_extraction_enabled": False},
        previous_state={"messages": []},
        result={
            "session_id": "s1",
            "messages": [
                HumanMessage(content="Please remember this."),
                AIMessage(content="answer"),
            ],
            "final_answer": "answer",
            "citations": [],
            "related_papers": [],
            "retrieved_chunks": [],
            "rag_latency_ms": 0,
            "agent_trace": [],
            "llm_usage": {},
        },
    )

    assert len(memory_manager.pending_buffer.rounds) == 1
    assert len(memory_manager.logged_rounds) == 1
    assert dispatched == []


def test_memory_manager_records_memory_dimension_metrics():
    settings = get_settings()
    init_database(settings)
    session_factory = build_session_factory(settings)
    metrics_repository = MetricsRepository(session_factory)
    manager = MemoryManager(
        settings,
        _FakeIndex(),
        _FakeEmbeddingService(),
        llm=_StructuredLLM(
            {
                "parsed": MemoryExtractionOutput(memories=["用户偏好简洁且带引用的回答"]),
                "raw": _RawResult(
                    '{"memories":["用户偏好简洁且带引用的回答"]}',
                    {"input_tokens": 9, "output_tokens": 3, "total_tokens": 12},
                ),
                "parsing_error": None,
            }
        ),
        metrics_repository=metrics_repository,
    )

    manager.log_round(
        user_id="memory-metrics-user",
        session_id="memory-metrics-session",
        round_index=1,
        messages=[
            HumanMessage(content="I prefer concise answers with citations."),
            AIMessage(content="Understood."),
        ],
    )

    extracted = manager.extract_pending_memories(user_id="memory-metrics-user")

    with session_factory() as session:
        rows = session.query(MemoryMetricModel).all()

    assert extracted == 1
    assert len(rows) == 1
    assert rows[0].user_id == "memory-metrics-user"
    assert rows[0].success is True
    assert rows[0].extracted_count == 1
    assert rows[0].prompt_tokens == 9
    assert rows[0].completion_tokens == 3
    assert rows[0].total_tokens == 12
    assert rows[0].latency_ms >= 0


def test_init_database_migrates_legacy_conversation_metrics_cost_column(tmp_path):
    db_path = tmp_path / "legacy-metrics.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE conversation_metrics (
            request_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            query_type TEXT NOT NULL,
            success INTEGER NOT NULL,
            retrieval_latency_ms INTEGER NOT NULL,
            latency_ms INTEGER NOT NULL,
            citations_count INTEGER NOT NULL,
            retrieved_chunks_count INTEGER NOT NULL,
            output_length INTEGER NOT NULL,
            cost REAL NOT NULL,
            agent_path_json TEXT NOT NULL,
            error_summary TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO conversation_metrics (
            request_id, user_id, session_id, query_type, success,
            retrieval_latency_ms, latency_ms, citations_count,
            retrieved_chunks_count, output_length, cost,
            agent_path_json, error_summary, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "legacy-request",
            "legacy-user",
            "legacy-session",
            "qa",
            1,
            10,
            20,
            1,
            2,
            100,
            0.5,
            "[]",
            None,
            "2026-04-16 00:00:00",
        ),
    )
    connection.commit()
    connection.close()

    settings = Settings(database_url=f"sqlite:///{db_path}")
    init_database(settings)
    session_factory = build_session_factory(settings)
    repository = MetricsRepository(session_factory)

    repository.record_round(
        request_id="new-request",
        user_id="new-user",
        session_id="new-session",
        query_type="qa",
        success=True,
        retrieval_latency_ms=5,
        latency_ms=15,
        citations_count=0,
        retrieved_chunks_count=3,
        output_length=42,
        agent_path=["planner", "reviewer"],
        prompt_tokens=11,
        completion_tokens=7,
        total_tokens=18,
    )

    with session_factory() as session:
        rows = session.query(ConversationMetricModel).order_by(
            ConversationMetricModel.request_id
        ).all()

    assert [row.request_id for row in rows] == ["legacy-request", "new-request"]
    assert rows[0].total_tokens == 0
    assert rows[1].total_tokens == 18


def test_memory_manager_retries_failed_round_on_next_extraction():
    settings = get_settings()
    index = _FakeIndex()
    user_id = f"memory-retry-user-{uuid4().hex}"
    manager = MemoryManager(
        settings,
        index,
        _FakeEmbeddingService(),
        llm=_SequentialStructuredLLM(
            [
                {
                    "parsed": None,
                    "raw": None,
                    "parsing_error": RuntimeError("llm unavailable"),
                },
                {
                    "parsed": MemoryExtractionOutput(memories=["用户偏好简洁且带引用的回答"]),
                    "raw": _RawResult(
                        '{"memories":["用户偏好简洁且带引用的回答"]}',
                        {"input_tokens": 9, "output_tokens": 3, "total_tokens": 12},
                    ),
                    "parsing_error": None,
                },
            ]
        ),
    )

    manager.log_round(
        user_id=user_id,
        session_id="memory-retry-session",
        round_index=1,
        messages=[
            HumanMessage(content="I prefer concise answers with citations."),
            AIMessage(content="Understood."),
        ],
    )

    first_extracted = manager.extract_pending_memories(user_id=user_id)
    first_state = manager._load_extraction_state()

    assert first_extracted == 0
    assert index.records == []
    assert first_state[user_id]["offset"] == 0

    second_extracted = manager.extract_pending_memories(user_id=user_id)
    second_state = manager._load_extraction_state()

    assert second_extracted == 1
    assert len(index.records) == 1
    assert index.records[0].content == "用户偏好简洁且带引用的回答"
    assert second_state[user_id]["offset"] > 0
