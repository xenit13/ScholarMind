from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from scholar_mind.config.settings import Settings
from scholar_mind.db.session import build_session_factory, init_database
from scholar_mind.memory.manager import MemoryManager
from scholar_mind.memory.repository import MemoryRepository
from scholar_mind.models.domain import (
    MemoryCandidate,
    MemoryCandidateExtractionOutput,
    StructuredMemoryRecord,
)
from scholar_mind.utils.messages import serialize_messages


class _Settings:
    message_context_window_tokens = 2048
    message_compact_threshold_ratio = 0.75
    memory_top_k = 5
    memory_min_similarity_score = 0.6
    log_dir = "logs"
    memory_root_dir = "memories"

    def __init__(self, base_path):
        self.base_path = base_path

    def resolve_path(self, value: str):
        return self.base_path / value


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


class _StructuredOutputLLM:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.prompts: list[str] = []

    def with_structured_output(self, _schema, include_raw: bool = False):
        assert include_raw is True
        return _Runnable(self, self.payloads.pop(0))


class _Index:
    def __init__(self, search_results=None):
        self.search_results = list(search_results or [])
        self.upserts: list[tuple[object, list[float]]] = []

    def search_memory(self, *_args, **_kwargs):
        return self.search_results

    def upsert_memory(self, record, embedding):
        self.upserts.append((record, embedding))


class _Embedder:
    def embed_query(self, _content: str):
        return [0.1, 0.2]

    async def aembed_query(self, _content: str):
        return [0.1, 0.2]


def _discrete_structured() -> dict:
    return {
        "schema_version": "memory_fact_v1",
        "fact_kind": "discrete_fact",
        "subject": {"type": "user", "id": "u1", "label": "用户"},
        "entity": {"type": "language", "id": "java", "label": "Java"},
        "attribute": "preference",
        "value": {"canonical": "dislike", "text": "不喜欢"},
        "polarity": "negative",
        "certainty": "explicit",
        "temporal": {"tense": "current"},
        "conflict_key": "subject:user:u1|entity:language:java|attribute:preference",
        "source_mode": "conversation",
    }


def test_memory_manager_recovers_alias_memory_payload(tmp_path):
    llm = _StructuredOutputLLM(
        [
            {
                "parsed": None,
                "raw": _RawResult(
                    '```json\n{"user_memories":[{"content":"用户偏好逐段讲解论文"}]}\n```',
                    {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
                ),
                "parsing_error": ValueError("invalid json"),
            }
        ]
    )
    manager = MemoryManager(_Settings(tmp_path), _Index(), _Embedder(), llm=llm)
    round_messages = [{"message": serialize_messages([HumanMessage(content="请逐段讲解论文")])[0]}]

    memories, usage, success = manager._extract_memories_with_llm(round_messages)

    assert success is True
    assert memories == ["用户偏好逐段讲解论文"]
    assert usage["total_tokens"] == 9


@pytest.mark.asyncio
async def test_memory_manager_save_skips_near_duplicate_at_point_nine_five(tmp_path, monkeypatch):
    index = _Index(search_results=[SimpleNamespace(score=0.95)])
    manager = MemoryManager(_Settings(tmp_path), index, _Embedder(), llm=None)
    monkeypatch.setattr(
        asyncio,
        "to_thread",
        lambda func, *args, **kwargs: asyncio.sleep(
            0,
            result=func(*args, **kwargs),
        ),
    )

    record = await manager.save(user_id="u1", content="偏好中文回答")

    assert record is None
    assert index.upserts == []
    assert not (tmp_path / "memories" / "u1" / "MEMORY.md").exists()


def test_memory_manager_save_sync_skips_near_duplicate_at_point_nine_five(tmp_path):
    index = _Index(search_results=[SimpleNamespace(score=0.95)])
    manager = MemoryManager(_Settings(tmp_path), index, _Embedder(), llm=None)

    manager._save_sync("u1", "偏好中文回答")

    assert index.upserts == []
    assert not (tmp_path / "memories" / "u1" / "MEMORY.md").exists()


def test_memory_manager_merges_explicit_and_extracted_memories_without_duplicate_writes(tmp_path):
    class _SemanticEmbedder:
        vectors = {
            "我偏好中文回答": [1.0, 0.0],
            "用户偏好使用中文回答": [0.99, 0.01],
        }

        def embed_query(self, content: str):
            return self.vectors.get(content, [0.0, 1.0])

        async def aembed_query(self, content: str):
            return self.embed_query(content)

    llm = _StructuredOutputLLM(
        [
            {
                "parsed": None,
                "raw": _RawResult(
                    '{"memories":["用户偏好使用中文回答"]}',
                    {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
                ),
                "parsing_error": None,
            }
        ]
    )
    index = _Index()
    manager = MemoryManager(_Settings(tmp_path), index, _SemanticEmbedder(), llm=llm)

    manager.log_round(
        user_id="u1",
        session_id="s1",
        round_index=1,
        messages=[
            HumanMessage(content="记住我偏好中文回答"),
            HumanMessage(content="后续请用中文回复"),
        ],
        explicit_memories=["我偏好中文回答"],
    )

    extracted = manager.extract_pending_memories(user_id="u1")

    assert extracted == 1
    assert len(index.upserts) == 1
    assert index.upserts[0][0].content == "我偏好中文回答"


def test_memory_manager_keeps_explicit_memories_when_llm_extraction_fails(tmp_path):
    llm = _StructuredOutputLLM(
        [
            {
                "parsed": None,
                "raw": None,
                "parsing_error": RuntimeError("llm unavailable"),
            }
        ]
    )
    index = _Index()
    manager = MemoryManager(_Settings(tmp_path), index, _Embedder(), llm=llm)

    manager.log_round(
        user_id="u1",
        session_id="s1",
        round_index=1,
        messages=[HumanMessage(content="记住我偏好中文回答")],
        explicit_memories=["我偏好中文回答"],
    )

    extracted = manager.extract_pending_memories(user_id="u1")

    assert extracted == 1
    assert len(index.upserts) == 1
    assert index.upserts[0][0].content == "我偏好中文回答"


def test_memory_manager_logs_rounds_with_local_timestamp_and_local_date_file(tmp_path, monkeypatch):
    fixed_local_time = datetime(2026, 4, 22, 0, 30, tzinfo=timezone(timedelta(hours=8)))
    monkeypatch.setattr("scholar_mind.memory.manager._local_now", lambda: fixed_local_time)
    manager = MemoryManager(_Settings(tmp_path), _Index(), _Embedder(), llm=None)

    manager.log_round(
        user_id="u1",
        session_id="s1",
        round_index=1,
        messages=[HumanMessage(content="现在几点")],
    )

    log_path = tmp_path / "logs" / "u1" / "session_messages-2026-04-22-1.jsonl"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(next(line for line in lines if line.startswith("{")))

    assert log_path.exists()
    assert payload["timestamp"] == "2026-04-22T00:30:00+08:00"


def test_memory_manager_get_context_filters_by_similarity_threshold(tmp_path):
    index = _Index(
        search_results=[
            SimpleNamespace(
                score=0.92,
                payload={"record_id": "m1", "content": "用户偏好中文回答"},
            ),
            SimpleNamespace(
                score=0.60,
                payload={"record_id": "m2", "content": "用户关注RAG评测"},
            ),
            SimpleNamespace(
                score=0.59,
                payload={"record_id": "m3", "content": "这条不该注入"},
            ),
        ]
    )
    manager = MemoryManager(_Settings(tmp_path), index, _Embedder(), llm=None)

    injected_text, hit_count = manager.get_context_sync("u1", "请结合我的偏好回答")

    assert hit_count == 2
    assert injected_text == "- 用户偏好中文回答\n- 用户关注RAG评测"


def test_memory_manager_get_context_caps_injected_memories_at_top_k(tmp_path):
    settings = _Settings(tmp_path)
    settings.memory_top_k = 5
    index = _Index(
        search_results=[
            SimpleNamespace(
                score=0.99 - idx * 0.01,
                payload={"record_id": f"m{idx}", "content": f"记忆 {idx}"},
            )
            for idx in range(1, 8)
        ]
    )
    manager = MemoryManager(settings, index, _Embedder(), llm=None)

    injected_text, hit_count = manager.get_context_sync("u1", "测试记忆注入上限")

    assert hit_count == 5
    assert injected_text.splitlines() == [
        "- 记忆 1",
        "- 记忆 2",
        "- 记忆 3",
        "- 记忆 4",
        "- 记忆 5",
    ]


def test_memory_manager_reranks_structured_memory_and_records_injected_access(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'memory.db'}",
        memory_root_dir=str(tmp_path / "memories"),
        log_dir=str(tmp_path / "logs"),
        qdrant_location=":memory:",
        memory_top_k=1,
        memory_min_similarity_score=0.0,
        memory_min_final_score=0.0,
        memory_candidate_multiplier=4,
    )
    init_database(settings)
    repository = MemoryRepository(build_session_factory(settings))
    now = datetime.now(UTC)
    old = StructuredMemoryRecord(
        memory_id="mem_old",
        user_id="u1",
        scope="user",
        memory_type="interaction_summary",
        content="旧的低重要度记忆",
        source="conversation",
        importance=0.2,
        confidence=0.8,
        status="active",
        created_at=now - timedelta(days=90),
        updated_at=now - timedelta(days=90),
        decay_rate=0.06,
        decay_floor=0.2,
    )
    recent = old.model_copy(
        update={
            "memory_id": "mem_recent",
            "content": "近期高重要度记忆",
            "importance": 0.95,
            "created_at": now - timedelta(days=1),
            "updated_at": now - timedelta(days=1),
            "decay_rate": 0.01,
            "decay_floor": 0.5,
            "access_count_30d": 5,
        }
    )
    repository.upsert(old)
    repository.upsert(recent)
    index = _Index(
        search_results=[
            SimpleNamespace(score=0.9, payload={"memory_id": "mem_old"}),
            SimpleNamespace(score=0.72, payload={"memory_id": "mem_recent"}),
        ]
    )
    manager = MemoryManager(
        settings,
        index,
        _Embedder(),
        llm=None,
        memory_repository=repository,
    )

    injected_text, hit_count = manager.get_context_sync("u1", "测试重排")

    stored_old = repository.get("u1", "mem_old")
    stored_recent = repository.get("u1", "mem_recent")
    assert hit_count == 1
    assert injected_text == "- 近期高重要度记忆"
    assert stored_old.access_count == 0
    assert stored_recent.access_count == 1
    assert stored_recent.last_accessed_at is not None


def test_memory_manager_decay_disabled_uses_semantic_order(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'memory.db'}",
        memory_root_dir=str(tmp_path / "memories"),
        log_dir=str(tmp_path / "logs"),
        qdrant_location=":memory:",
        memory_top_k=1,
        memory_min_similarity_score=0.0,
        memory_decay_enabled=False,
    )
    init_database(settings)
    repository = MemoryRepository(build_session_factory(settings))
    now = datetime.now(UTC)
    for memory_id, content in [
        ("mem_old", "语义分更高的旧记忆"),
        ("mem_recent", "近期高重要度记忆"),
    ]:
        repository.upsert(
            StructuredMemoryRecord(
                memory_id=memory_id,
                user_id="u1",
                scope="user",
                memory_type="interaction_summary",
                content=content,
                source="conversation",
                importance=0.9,
                confidence=0.8,
                status="active",
                created_at=now,
                updated_at=now,
                decay_rate=0.01,
                decay_floor=0.5,
            )
        )
    index = _Index(
        search_results=[
            SimpleNamespace(score=0.9, payload={"memory_id": "mem_old"}),
            SimpleNamespace(score=0.72, payload={"memory_id": "mem_recent"}),
        ]
    )
    manager = MemoryManager(
        settings,
        index,
        _Embedder(),
        llm=None,
        memory_repository=repository,
    )

    injected_text, hit_count = manager.get_context_sync("u1", "测试关闭衰减")

    assert hit_count == 1
    assert injected_text == "- 语义分更高的旧记忆"


def test_memory_manager_formats_discrete_memory_in_structured_context(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'memory.db'}",
        memory_root_dir=str(tmp_path / "memories"),
        log_dir=str(tmp_path / "logs"),
        qdrant_location=":memory:",
        memory_top_k=1,
        memory_min_similarity_score=0.0,
        memory_min_final_score=0.0,
    )
    init_database(settings)
    repository = MemoryRepository(build_session_factory(settings))
    now = datetime.now(UTC)
    repository.upsert(
        StructuredMemoryRecord(
            memory_id="mem_discrete",
            user_id="u1",
            scope="user",
            memory_type="preference",
            content="用户不喜欢 Java。",
            structured=_discrete_structured(),
            source="conversation",
            importance=0.9,
            confidence=0.9,
            status="active",
            created_at=now,
            updated_at=now,
            decay_rate=0.01,
            decay_floor=0.5,
        )
    )
    index = _Index(
        search_results=[SimpleNamespace(score=0.9, payload={"memory_id": "mem_discrete"})]
    )
    manager = MemoryManager(
        settings,
        index,
        _Embedder(),
        llm=None,
        memory_repository=repository,
    )

    injected_text, hit_count = manager.get_context_sync("u1", "不要推荐 Java 项目")

    assert hit_count == 1
    assert injected_text == (
        "- [memory_fact_v1] attribute=preference; entity=Java; value=dislike; "
        "polarity=negative; confidence=0.90; content=用户不喜欢 Java。"
    )


def test_memory_manager_save_sync_writes_structured_repository_when_available(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'memory.db'}",
        memory_root_dir=str(tmp_path / "memories"),
        log_dir=str(tmp_path / "logs"),
        qdrant_location=":memory:",
    )
    init_database(settings)
    repository = MemoryRepository(build_session_factory(settings))
    index = _Index()
    manager = MemoryManager(
        settings,
        index,
        _Embedder(),
        llm=None,
        memory_repository=repository,
    )

    record = manager._save_sync("u1", "用户偏好中文回答")

    assert record is not None
    stored = repository.get("u1", record.record_id)
    assert stored is not None
    assert stored.content == "用户偏好中文回答"
    assert stored.memory_type == "interaction_summary"
    assert index.upserts[0][0].record_id == record.record_id
    assert not (tmp_path / "memories" / "u1" / "MEMORY.md").exists()


def test_memory_manager_extract_request_memories_uses_structured_operations(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'memory.db'}",
        memory_root_dir=str(tmp_path / "memories"),
        log_dir=str(tmp_path / "logs"),
        qdrant_location=":memory:",
    )
    init_database(settings)
    repository = MemoryRepository(build_session_factory(settings))
    index = _Index()
    llm = _StructuredOutputLLM(
        [
            {
                "parsed": MemoryCandidateExtractionOutput(
                    candidates=[
                        MemoryCandidate(
                            memory_type="preference",
                            content="用户偏好简洁回答，关键结论需要带引用。",
                            structured={"subject": "user", "predicate": "prefers"},
                            keywords=["简洁", "引用"],
                            importance=0.8,
                            confidence=0.9,
                            source="conversation",
                            evidence=[{"message_id": "s1-1-0", "role": "human"}],
                        )
                    ]
                ),
                "raw": _RawResult("", {"input_tokens": 9, "output_tokens": 6, "total_tokens": 15}),
                "parsing_error": None,
            }
        ]
    )
    manager = MemoryManager(
        settings,
        index,
        _Embedder(),
        llm=llm,
        memory_repository=repository,
    )
    round_messages = [
        {
            "message": serialize_messages(
                [HumanMessage(content="以后回答请简洁，但关键结论要带引用")]
            )[0]
        }
    ]

    result = manager.extract_request_memories(
        user_id="u1",
        request_id="req1",
        round_messages=round_messages,
    )

    memories = repository.list_active("u1")
    events = repository.list_operation_events("u1")
    assert result["success"] is True
    assert result["written_count"] == 1
    assert memories[0].content == "用户偏好简洁回答，关键结论需要带引用。"
    assert memories[0].memory_type == "preference"
    assert events[0].operation == "ADD"


def test_memory_manager_drops_prohibited_candidate_before_structured_write(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'memory.db'}",
        memory_root_dir=str(tmp_path / "memories"),
        log_dir=str(tmp_path / "logs"),
        qdrant_location=":memory:",
    )
    init_database(settings)
    repository = MemoryRepository(build_session_factory(settings))
    index = _Index()
    llm = _StructuredOutputLLM(
        [
            {
                "parsed": MemoryCandidateExtractionOutput(
                    candidates=[
                        MemoryCandidate(
                            memory_type="preference",
                            content=(
                                "用户的 OpenAI API key 是 "
                                "sk-abcdefghijklmnopqrstuvwxyz1234567890"
                            ),
                            structured={},
                            keywords=["api key"],
                            importance=0.8,
                            confidence=0.9,
                            source="conversation",
                            evidence=[{"message_id": "s1-1-0", "role": "human"}],
                        )
                    ]
                ),
                "raw": _RawResult("", {"input_tokens": 9, "output_tokens": 6, "total_tokens": 15}),
                "parsing_error": None,
            }
        ]
    )
    manager = MemoryManager(
        settings,
        index,
        _Embedder(),
        llm=llm,
        memory_repository=repository,
    )
    round_messages = [
        {
            "message": serialize_messages(
                [HumanMessage(content="请记住我的 OpenAI API key 是 sk-...")]
            )[0]
        }
    ]

    result = manager.extract_request_memories(
        user_id="u1",
        request_id="req1",
        round_messages=round_messages,
    )

    assert result["success"] is True
    assert result["written_count"] == 0
    assert repository.list_active("u1") == []
    assert index.upserts == []
    events = repository.list_operation_events("u1")
    assert events[0].operation == "NONE"
    assert "admission_drop" in events[0].reason
    assert "secrets" in events[0].reason


def test_memory_manager_uses_model_admission_before_structured_write(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'memory.db'}",
        memory_root_dir=str(tmp_path / "memories"),
        log_dir=str(tmp_path / "logs"),
        qdrant_location=":memory:",
    )
    init_database(settings)
    repository = MemoryRepository(build_session_factory(settings))
    index = _Index()
    llm = _StructuredOutputLLM(
        [
            {
                "parsed": MemoryCandidateExtractionOutput(
                    candidates=[
                        MemoryCandidate(
                            memory_type="preference",
                            content="用户偏好中文、结构化回答。",
                            structured={},
                            keywords=["中文", "结构化"],
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
                "parsed": {
                    "action": "DROP",
                    "reason": "model rejected this memory",
                    "matched_rules": ["model_policy"],
                },
                "raw": _RawResult("", {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}),
                "parsing_error": None,
            },
        ]
    )
    manager = MemoryManager(
        settings,
        index,
        _Embedder(),
        llm=llm,
        memory_repository=repository,
    )
    round_messages = [
        {
            "message": serialize_messages(
                [HumanMessage(content="请记住我偏好中文、结构化回答。")]
            )[0]
        }
    ]

    result = manager.extract_request_memories(
        user_id="u1",
        request_id="req1",
        round_messages=round_messages,
    )

    assert result["success"] is True
    assert result["written_count"] == 0
    assert result["usage"]["total_tokens"] == 20
    assert repository.list_active("u1") == []
    assert index.upserts == []
    events = repository.list_operation_events("u1")
    assert events[0].operation == "NONE"
    assert events[0].reason == "admission_drop:model_policy"


def test_memory_manager_structured_path_keeps_explicit_memories_when_llm_fails(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'memory.db'}",
        memory_root_dir=str(tmp_path / "memories"),
        log_dir=str(tmp_path / "logs"),
        qdrant_location=":memory:",
    )
    init_database(settings)
    repository = MemoryRepository(build_session_factory(settings))
    llm = _StructuredOutputLLM(
        [
            {
                "parsed": None,
                "raw": None,
                "parsing_error": RuntimeError("llm unavailable"),
            }
        ]
    )
    manager = MemoryManager(
        settings,
        _Index(),
        _Embedder(),
        llm=llm,
        memory_repository=repository,
    )
    round_messages = [
        {"message": serialize_messages([HumanMessage(content="记住我偏好中文回答")])[0]}
    ]

    result = manager.extract_request_memories(
        user_id="u1",
        request_id="req1",
        round_messages=round_messages,
        explicit_memories=["我偏好中文回答"],
    )

    memories = repository.list_active("u1")
    events = repository.list_operation_events("u1")
    assert result["success"] is True
    assert result["written_count"] == 1
    assert memories[0].content == "我偏好中文回答"
    assert memories[0].source == "explicit"
    assert events[0].operation == "ADD"
