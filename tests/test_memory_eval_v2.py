from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from typer.testing import CliRunner

from scholar_mind.api.routes.eval import (
    get_memory_eval_request,
    get_online_dashboard,
    get_request_eval,
)
from scholar_mind.db.models import Base
from scholar_mind.db.session import init_database
from scholar_mind.eval.context import finish_eval_context, init_eval_context
from scholar_mind.main import cli_app
from scholar_mind.memory.manager import MemoryManager
from scholar_mind.services.memory_eval_v2 import MemoryEvalServiceV2, MemoryEvalV2Repository
from scholar_mind.services.repositories import OnlineEvalRepository
from scholar_mind.services.research import ResearchService
from scholar_mind.utils.token_estimator import estimate_text_tokens


class _Settings:
    environment = "test"
    eval_root_dir = "data/eval"
    memory_root_dir = "data/memory"
    log_dir = "data/message_logs"
    llm_reasoning_model = "openai/gpt-5.4"
    memory_top_k = 5
    memory_min_similarity_score = 0.6
    message_context_window_tokens = 2048
    message_compact_threshold_ratio = 0.75

    def __init__(self, base_path):
        self.base_path = base_path

    def resolve_path(self, value: str):
        return self.base_path / value


class _Embedder:
    def embed_query(self, _content: str):
        return [0.1, 0.2]

    async def aembed_query(self, _content: str):
        return [0.1, 0.2]


class _Index:
    def __init__(self, search_results=None):
        self.search_results = list(search_results or [])
        self.upserts: list[tuple[object, list[float]]] = []

    def search_memory(self, *_args, **_kwargs):
        return list(self.search_results)

    def upsert_memory(self, record, embedding):
        self.upserts.append((record, embedding))


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


@pytest.fixture
def memory_eval_components(tmp_path):
    settings = _Settings(tmp_path)
    engine = create_engine(f"sqlite:///{tmp_path / 'memory_eval_v2.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return {
        "settings": settings,
        "online_repo": OnlineEvalRepository(factory),
        "memory_repo": MemoryEvalV2Repository(factory),
        "service": MemoryEvalServiceV2(settings, MemoryEvalV2Repository(factory)),
        "factory": factory,
        "tmp_path": tmp_path,
    }


def _write_memory_file(tmp_path, user_id: str, records: list[tuple[str, str]]):
    path = tmp_path / "data" / "memory" / user_id / "MEMORY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Memory", ""]
    for record_id, content in records:
        lines.extend(
            [
                f"## {record_id}",
                "- created_at: 2026-04-21T10:00:00+00:00",
                "- source: conversation",
                f"- content: {content}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _seed_request_with_v2_events(components, request_id: str = "req_mem_v2") -> None:
    components["online_repo"].save_request_run(
        {
            "request_id": request_id,
            "session_id": "sess_1",
            "user_id": "user_1",
            "query": "请结合我之前的偏好回答",
            "query_type": "qa",
            "final_answer": "会按你的中文偏好回答。",
            "memory_score": 0.5,
            "execution_health_score": 0.93,
            "rag_metrics": {"strategy_used": "hybrid"},
            "memory_metrics": {},
            "execution_health": {"execution_health_score": 0.93, "total_latency_ms": 1200},
        }
    )
    components["memory_repo"].save_memory_retrieval_event(
        {
            "request_id": request_id,
            "user_id": "user_1",
            "query": "请结合我之前的偏好回答",
            "embedding_latency_ms": 8,
            "vector_search_latency_ms": 12,
            "retrieved_memory_ids": ["m1", "m2"],
            "retrieved_scores": [0.91, 0.72],
            "retrieved_count": 2,
            "injected_memory_ids": ["m1"],
            "injected_count": 1,
            "injected_text": "- 用户偏好中文回答",
            "injected_tokens": 2,
        }
    )
    components["memory_repo"].save_memory_extraction_dispatch(
        request_id=request_id,
        user_id="user_1",
        dispatch_latency_ms=14,
        dispatch_success=True,
    )
    components["memory_repo"].update_memory_extraction_result(
        request_id=request_id,
        prompt_tokens=20,
        completion_tokens=8,
        total_tokens=28,
        written_memory_ids=["mem_written_1"],
        written_memory_texts=["用户偏好中文回答"],
    )
    _write_memory_file(
        components["tmp_path"],
        "user_1",
        [
            ("m1", "用户偏好中文回答"),
            ("m2", "用户长期关注 RAG 检索评测"),
        ],
    )


def test_memory_extraction_dispatch_preserves_user_id_after_result_created_first(
    memory_eval_components,
):
    repo = memory_eval_components["memory_repo"]

    repo.update_memory_extraction_result(
        request_id="req_extract_order",
        prompt_tokens=20,
        completion_tokens=8,
        total_tokens=28,
        written_memory_ids=["mem_written_1"],
        written_memory_texts=["用户偏好中文回答"],
    )
    repo.save_memory_extraction_dispatch(
        request_id="req_extract_order",
        user_id="user_1",
        dispatch_latency_ms=14,
        dispatch_success=True,
    )

    event = repo.get_memory_extraction_event("req_extract_order")

    assert event is not None
    assert event["user_id"] == "user_1"


def test_init_database_drops_retired_memory_eval_annotation_column(tmp_path):
    db_path = tmp_path / "memory_eval_v2_legacy.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE memory_eval_annotations_v2 (
                    annotation_id VARCHAR(64) NOT NULL PRIMARY KEY,
                    batch_id VARCHAR(64) NOT NULL,
                    request_id VARCHAR(64) NOT NULL,
                    relevant_memory_ids_json TEXT NOT NULL DEFAULT '[]',
                    critical_memory_ids_json TEXT NOT NULL DEFAULT '[]',
                    stale_memory_ids_json TEXT NOT NULL DEFAULT '[]',
                    claims_json TEXT NOT NULL DEFAULT '[]',
                    expected_extracted_memories_json TEXT NOT NULL DEFAULT '[]',
                    annotator VARCHAR(64) NOT NULL DEFAULT '',
                    created_at DATETIME NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO memory_eval_annotations_v2 (
                    annotation_id,
                    batch_id,
                    request_id,
                    relevant_memory_ids_json,
                    critical_memory_ids_json,
                    stale_memory_ids_json,
                    claims_json,
                    expected_extracted_memories_json,
                    annotator,
                    created_at
                ) VALUES (
                    'ann_1',
                    'batch_1',
                    'req_1',
                    '["mem_1"]',
                    '["mem_1"]',
                    '[]',
                    '[]',
                    '[]',
                    'tester',
                    '2026-04-24 00:00:00'
                )
                """
            )
        )

    class _DbSettings:
        database_url = f"sqlite:///{db_path}"

    init_database(_DbSettings())

    inspector = inspect(create_engine(f"sqlite:///{db_path}", future=True))
    columns = {column["name"] for column in inspector.get_columns("memory_eval_annotations_v2")}

    assert "critical_memory_ids_json" not in columns
    assert "relevant_memory_ids_json" in columns


def test_memory_manager_records_v2_retrieval_event(memory_eval_components):
    index = _Index(
        search_results=[
            SimpleNamespace(
                score=0.92,
                payload={"record_id": "m1", "content": "用户偏好中文回答"},
            ),
            SimpleNamespace(
                score=0.58,
                payload={"record_id": "m2", "content": "这条不会注入"},
            ),
        ]
    )
    manager = MemoryManager(
        memory_eval_components["settings"],
        index,
        _Embedder(),
        llm=None,
        memory_eval_v2_repository=memory_eval_components["memory_repo"],
    )
    ctx = init_eval_context(
        request_id="req_mem_event",
        session_id="sess_1",
        user_id="user_1",
        query="结合我的偏好回答",
        query_type="qa",
    )

    injected_text, hit_count = manager.get_context_sync("user_1", "结合我的偏好回答")
    finish_eval_context(ctx, {"final_answer": ""})

    event = memory_eval_components["memory_repo"].get_memory_retrieval_event("req_mem_event")
    assert hit_count == 1
    assert injected_text == "- 用户偏好中文回答"
    assert event is not None
    assert event["retrieved_memory_ids"] == ["m1", "m2"]
    assert event["injected_memory_ids"] == ["m1"]
    assert event["injected_count"] == 1
    assert event["injected_tokens"] == estimate_text_tokens(
        "- 用户偏好中文回答",
        model_name=memory_eval_components["settings"].llm_reasoning_model,
    )


def test_extract_request_memories_updates_v2_extraction_event(memory_eval_components):
    llm = _StructuredOutputLLM(
        [
            {
                "parsed": None,
                "raw": _RawResult(
                    '{"memories":["用户偏好中文回答"]}',
                    {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12},
                ),
                "parsing_error": None,
            }
        ]
    )
    manager = MemoryManager(
        memory_eval_components["settings"],
        _Index(),
        _Embedder(),
        llm=llm,
        memory_eval_v2_repository=memory_eval_components["memory_repo"],
    )
    memory_eval_components["memory_repo"].save_memory_extraction_dispatch(
        request_id="req_extract_v2",
        user_id="user_1",
        dispatch_latency_ms=9,
        dispatch_success=True,
    )

    result = manager.extract_request_memories(
        user_id="user_1",
        request_id="req_extract_v2",
        round_messages=[],
        explicit_memories=[],
    )

    event = memory_eval_components["memory_repo"].get_memory_extraction_event("req_extract_v2")
    assert result["written_count"] == 1
    assert event is not None
    assert event["written_memory_texts"] == ["用户偏好中文回答"]
    assert event["prompt_tokens"] == 8
    assert event["completion_tokens"] == 4
    assert event["total_tokens"] == 12


def test_memory_eval_v2_allows_missing_extraction_component(
    memory_eval_components,
):
    request_id = "req_pending_extract"
    memory_eval_components["online_repo"].save_request_run(
        {
            "request_id": request_id,
            "session_id": "sess_pending",
            "user_id": "user_1",
            "query": "请结合我之前的偏好回答",
            "query_type": "qa",
            "final_answer": "会按你的中文偏好回答。",
            "memory_score": 0.5,
            "execution_health_score": 0.93,
            "rag_metrics": {"strategy_used": "hybrid"},
            "memory_metrics": {},
            "execution_health": {"execution_health_score": 0.93, "total_latency_ms": 1200},
        }
    )
    memory_eval_components["memory_repo"].save_memory_retrieval_event(
        {
            "request_id": request_id,
            "user_id": "user_1",
            "query": "请结合我之前的偏好回答",
            "embedding_latency_ms": 8,
            "vector_search_latency_ms": 12,
            "retrieved_memory_ids": ["m1"],
            "retrieved_scores": [0.91],
            "retrieved_count": 1,
            "injected_memory_ids": ["m1"],
            "injected_count": 1,
            "injected_text": "- 用户偏好中文回答",
            "injected_tokens": 10,
        }
    )
    memory_eval_components["memory_repo"].save_memory_extraction_dispatch(
        request_id=request_id,
        user_id="user_1",
        dispatch_latency_ms=14,
        dispatch_success=True,
    )
    _write_memory_file(
        memory_eval_components["tmp_path"],
        "user_1",
        [("m1", "用户偏好中文回答")],
    )

    exported = memory_eval_components["service"].export_batch(
        from_request_id=request_id,
        limit=1,
    )
    batch_dir = (
        memory_eval_components["tmp_path"]
        / "data"
        / "eval"
        / "memory_batches"
        / exported["batch_id"]
    )
    (batch_dir / "annotations.jsonl").write_text(
        json.dumps(
            {
                "request_id": request_id,
                "relevant_memory_ids": ["m1"],
                "stale_memory_ids": [],
                "claims": [
                    {
                        "claim_id": "c1",
                        "text": "用户偏好中文回答",
                        "supported_by_retrieved_memory": True,
                        "support_memory_ids": ["m1"],
                    }
                ],
                "expected_extracted_memories": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = memory_eval_components["service"].evaluate_batch(batch_id=exported["batch_id"])

    assert result["runs"][0]["memory_score"] > 0
    assert result["runs"][0]["score_breakdown"]["components"] == {
        "retrieval_used": True,
        "injection_used": True,
        "extraction_used": False,
    }


def test_memory_eval_v2_export_and_evaluate_batch(memory_eval_components):
    _seed_request_with_v2_events(memory_eval_components)

    exported = memory_eval_components["service"].export_batch(
        from_request_id="req_mem_v2",
        limit=10,
    )
    batch_id = exported["batch_id"]
    batch_dir = memory_eval_components["tmp_path"] / "data" / "eval" / "memory_batches" / batch_id
    assert (batch_dir / "batch.json").exists()
    assert (batch_dir / "requests.jsonl").exists()
    assert (batch_dir / "memory_catalog.jsonl").exists()

    (batch_dir / "annotations.jsonl").write_text(
        json.dumps(
            {
                "request_id": "req_mem_v2",
                "relevant_memory_ids": ["m1"],
                "stale_memory_ids": [],
                "claims": [
                    {
                        "claim_id": "c1",
                        "text": "用户偏好中文回答",
                        "supported_by_retrieved_memory": True,
                        "support_memory_ids": ["m1"],
                    }
                ],
                "expected_extracted_memories": ["用户偏好中文回答"],
                "annotator": "tester",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = memory_eval_components["service"].evaluate_batch(batch_id=batch_id)
    updated = memory_eval_components["online_repo"].get_request_eval("req_mem_v2")
    request_detail = memory_eval_components["service"].get_request("req_mem_v2")

    assert result["report"]["sample_count"] == 1
    assert result["runs"][0]["memory_score"] > 0.5
    assert updated is not None
    assert updated["memory_score"] == result["runs"][0]["memory_score"]
    assert request_detail is not None
    assert request_detail["run"]["request_id"] == "req_mem_v2"
    assert request_detail["retrieval_event"]["retrieved_memory_ids"] == ["m1", "m2"]


def test_memory_eval_v2_rejects_invalid_annotation(memory_eval_components):
    _seed_request_with_v2_events(memory_eval_components, request_id="req_bad_ann")
    exported = memory_eval_components["service"].export_batch(
        from_request_id="req_bad_ann",
        limit=10,
    )
    batch_id = exported["batch_id"]
    batch_dir = memory_eval_components["tmp_path"] / "data" / "eval" / "memory_batches" / batch_id
    (batch_dir / "annotations.jsonl").write_text(
        json.dumps(
            {
                "request_id": "req_bad_ann",
                "relevant_memory_ids": ["m_not_exist"],
                "stale_memory_ids": [],
                "claims": [
                    {
                        "claim_id": "c1",
                        "text": "bad",
                        "supported_by_retrieved_memory": False,
                        "support_memory_ids": [],
                    }
                ],
                "expected_extracted_memories": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="UNKNOWN_MEMORY_IDS"):
        memory_eval_components["service"].evaluate_batch(batch_id=batch_id)


def test_memory_eval_v2_reweights_when_retrieval_gold_component_missing(memory_eval_components):
    _seed_request_with_v2_events(memory_eval_components, request_id="req_missing_gold")
    exported = memory_eval_components["service"].export_batch(
        from_request_id="req_missing_gold",
        limit=10,
    )
    batch_id = exported["batch_id"]
    batch_dir = memory_eval_components["tmp_path"] / "data" / "eval" / "memory_batches" / batch_id
    (batch_dir / "annotations.jsonl").write_text(
        json.dumps(
            {
                "request_id": "req_missing_gold",
                "relevant_memory_ids": [],
                "stale_memory_ids": [],
                "claims": [
                    {
                        "claim_id": "c1",
                        "text": "用户偏好中文回答",
                        "supported_by_retrieved_memory": True,
                        "support_memory_ids": ["m1"],
                    }
                ],
                "expected_extracted_memories": ["用户偏好中文回答"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = memory_eval_components["service"].evaluate_batch(batch_id=batch_id)
    run = result["runs"][0]
    breakdown = run["score_breakdown"]
    weighted_score = (
        0.15 * breakdown["s_inject"]
        + 0.20 * breakdown["s_extract"]
    ) / 0.35

    assert run["memory_hit_at_k"] is None
    assert run["memory_score"] == pytest.approx(round(weighted_score, 4), abs=1e-4)
    assert breakdown["components"] == {
        "retrieval_used": False,
        "injection_used": True,
        "extraction_used": True,
    }


def test_memory_eval_v2_allows_extraction_only_component(memory_eval_components):
    request_id = "req_extraction_only"
    memory_eval_components["online_repo"].save_request_run(
        {
            "request_id": request_id,
            "session_id": "sess_extract_only",
            "user_id": "user_1",
            "query": "记住我偏好中文回答",
            "query_type": "qa",
            "final_answer": "已记录。",
            "memory_score": None,
            "execution_health_score": 0.93,
            "runtime_metrics": {},
            "execution_health": {"execution_health_score": 0.93},
        }
    )
    memory_eval_components["memory_repo"].save_memory_extraction_dispatch(
        request_id=request_id,
        user_id="user_1",
        dispatch_latency_ms=14,
        dispatch_success=True,
    )
    memory_eval_components["memory_repo"].update_memory_extraction_result(
        request_id=request_id,
        prompt_tokens=20,
        completion_tokens=8,
        total_tokens=28,
        written_memory_ids=["mem_written_1"],
        written_memory_texts=["用户偏好中文回答"],
    )
    _write_memory_file(
        memory_eval_components["tmp_path"],
        "user_1",
        [("mem_written_1", "用户偏好中文回答")],
    )

    exported = memory_eval_components["service"].export_batch(
        from_request_id=request_id,
        limit=1,
    )
    batch_dir = (
        memory_eval_components["tmp_path"]
        / "data"
        / "eval"
        / "memory_batches"
        / exported["batch_id"]
    )
    (batch_dir / "annotations.jsonl").write_text(
        json.dumps(
            {
                "request_id": request_id,
                "relevant_memory_ids": [],
                "stale_memory_ids": [],
                "claims": [],
                "expected_extracted_memories": ["用户偏好中文回答"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = memory_eval_components["service"].evaluate_batch(batch_id=exported["batch_id"])
    run = result["runs"][0]

    assert run["memory_score"] == pytest.approx(run["score_breakdown"]["s_extract"], abs=1e-4)
    assert run["score_breakdown"]["components"] == {
        "retrieval_used": False,
        "injection_used": False,
        "extraction_used": True,
    }


def test_memory_eval_v2_cli_round_trip(memory_eval_components, monkeypatch):
    _seed_request_with_v2_events(memory_eval_components, request_id="req_cli_v2")
    fake_container = SimpleNamespace(memory_eval_v2_service=memory_eval_components["service"])
    monkeypatch.setattr("scholar_mind.main.get_container", lambda: fake_container)
    runner = CliRunner()

    export_result = runner.invoke(
        cli_app,
        ["eval", "memory-export", "--from-request-id", "req_cli_v2", "--limit", "1"],
    )
    assert export_result.exit_code == 0
    exported = json.loads(export_result.stdout)
    batch_id = exported["batch_id"]
    batch_dir = memory_eval_components["tmp_path"] / "data" / "eval" / "memory_batches" / batch_id
    (batch_dir / "annotations.jsonl").write_text(
        json.dumps(
            {
                "request_id": "req_cli_v2",
                "relevant_memory_ids": ["m1"],
                "stale_memory_ids": [],
                "claims": [
                    {
                        "claim_id": "c1",
                        "text": "用户偏好中文回答",
                        "supported_by_retrieved_memory": True,
                        "support_memory_ids": ["m1"],
                    }
                ],
                "expected_extracted_memories": ["用户偏好中文回答"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    eval_result = runner.invoke(cli_app, ["eval", "memory", "--batch-id", batch_id])
    assert eval_result.exit_code == 0
    evaluated = json.loads(eval_result.stdout)

    report_result = runner.invoke(
        cli_app,
        ["eval", "memory-report", "--report-id", evaluated["report_id"]],
    )
    assert report_result.exit_code == 0
    report = json.loads(report_result.stdout)
    assert report["report_id"] == evaluated["report_id"]
    assert report["sample_count"] == 1


def test_memory_library_audit_cli_round_trip(memory_eval_components, monkeypatch):
    _write_memory_file(
        memory_eval_components["tmp_path"],
        "user_cli",
        [
            ("m1", "用户偏好中文回答"),
            ("m2", "用户偏好中文回答"),
        ],
    )
    fake_container = SimpleNamespace(memory_eval_v2_service=memory_eval_components["service"])
    monkeypatch.setattr("scholar_mind.main.get_container", lambda: fake_container)
    runner = CliRunner()

    export_result = runner.invoke(cli_app, ["eval", "memory-library-export"])
    assert export_result.exit_code == 0
    exported = json.loads(export_result.stdout)
    batch_id = exported["batch_id"]
    batch_dir = (
        memory_eval_components["tmp_path"]
        / "data"
        / "eval"
        / "memory_library_audits"
        / batch_id
    )
    (batch_dir / "annotations.json").write_text(
        json.dumps(
            {
                "duplicate_pairs": [
                    {
                        "memory_id_1": "m1",
                        "memory_id_2": "m2",
                        "reason": "重复记忆",
                    }
                ],
                "conflict_pairs": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    eval_result = runner.invoke(cli_app, ["eval", "memory-library", "--batch-id", batch_id])
    assert eval_result.exit_code == 0
    evaluated = json.loads(eval_result.stdout)
    assert evaluated["report"]["duplicate_memory_ratio"] == 1.0

    report_result = runner.invoke(
        cli_app,
        ["eval", "memory-library-report", "--report-id", evaluated["report_id"]],
    )
    assert report_result.exit_code == 0
    report = json.loads(report_result.stdout)
    assert report["report_id"] == evaluated["report_id"]


def test_research_service_schedules_request_scoped_memory_extraction(
    memory_eval_components,
    monkeypatch,
):
    direct_calls = []
    scheduled_calls = []

    class _FakeMemoryManager:
        def extract_request_memories(self, **kwargs):
            direct_calls.append(kwargs)
            raise AssertionError("request-scoped extraction must not run on the request path")

    from scholar_mind.services import research as research_module

    def _fake_enqueue(**kwargs):
        scheduled_calls.append(kwargs)
        return "fake_async_result"  # truthy sentinel so dispatch_success=True

    monkeypatch.setattr(
        research_module,
        "_enqueue_request_memory_extraction",
        _fake_enqueue,
        raising=False,
    )

    service = ResearchService(
        settings=SimpleNamespace(),
        session_repository=SimpleNamespace(),
        metrics_repository=SimpleNamespace(),
        memory_manager=_FakeMemoryManager(),
        orchestrator=SimpleNamespace(),
        online_eval_repository=None,
        memory_eval_v2_repository=memory_eval_components["memory_repo"],
        llm=None,
    )

    service._dispatch_request_memory_extraction(
        user_id="user_1",
        request_id="req_dispatch_v2",
        round_messages=[HumanMessage(content="请记住我偏好中文回答")],
        explicit_memories=["用户偏好中文回答"],
    )

    event = memory_eval_components["memory_repo"].get_memory_extraction_event("req_dispatch_v2")
    assert direct_calls == []
    assert scheduled_calls[0]["request_id"] == "req_dispatch_v2"
    assert scheduled_calls[0]["user_id"] == "user_1"
    assert scheduled_calls[0]["explicit_memories"] == ["用户偏好中文回答"]
    assert (
        scheduled_calls[0]["round_messages"][0]["message"]["data"]["content"]
        == "请记住我偏好中文回答"
    )
    assert event is not None
    assert event["dispatch_success"] is True
    assert event["total_tokens"] is None


def test_research_service_does_not_fallback_to_sync_extraction_when_enqueue_fails(
    memory_eval_components,
    monkeypatch,
):
    direct_calls = []

    class _FakeMemoryManager:
        def extract_request_memories(self, **kwargs):
            direct_calls.append(kwargs)
            raise AssertionError("failed enqueue must not fall back to synchronous extraction")

    from scholar_mind.services import research as research_module

    def _fake_enqueue(**_kwargs):
        raise RuntimeError("broker unavailable")

    monkeypatch.setattr(
        research_module,
        "_enqueue_request_memory_extraction",
        _fake_enqueue,
        raising=False,
    )

    service = ResearchService(
        settings=SimpleNamespace(),
        session_repository=SimpleNamespace(),
        metrics_repository=SimpleNamespace(),
        memory_manager=_FakeMemoryManager(),
        orchestrator=SimpleNamespace(),
        online_eval_repository=None,
        memory_eval_v2_repository=memory_eval_components["memory_repo"],
        llm=None,
    )

    service._dispatch_request_memory_extraction(
        user_id="user_1",
        request_id="req_dispatch_failed",
        round_messages=[HumanMessage(content="请记住我偏好中文回答")],
        explicit_memories=[],
    )

    event = memory_eval_components["memory_repo"].get_memory_extraction_event(
        "req_dispatch_failed"
    )
    assert direct_calls == []
    assert event is not None
    assert event["dispatch_success"] is False


@pytest.mark.asyncio
async def test_memory_eval_v2_api_endpoint():
    class _FakeService:
        def get_request(self, request_id: str):
            return {
                "request_id": request_id,
                "run": {"request_id": request_id, "memory_score": 0.77},
                "batch": {"batch_id": "batch_1"},
                "retrieval_event": {"retrieved_memory_ids": ["m1"]},
                "extraction_event": {"dispatch_success": True},
            }

        def get_batch(self, batch_id: str):
            return {"batch": {"batch_id": batch_id}, "report": None}

        def get_report(self, report_id: str):
            return {"report_id": report_id, "sample_count": 1}

    response = await get_memory_eval_request(
        "req_api_v2",
        container=SimpleNamespace(memory_eval_v2_service=_FakeService()),
    )

    assert response["data"]["run"]["memory_score"] == 0.77


@pytest.mark.asyncio
async def test_memory_eval_v2_request_endpoint_is_ready_for_frontend(memory_eval_components):
    _seed_request_with_v2_events(memory_eval_components, request_id="req_frontend_v2")
    exported = memory_eval_components["service"].export_batch(
        from_request_id="req_frontend_v2",
        limit=1,
    )
    batch_dir = (
        memory_eval_components["tmp_path"]
        / "data"
        / "eval"
        / "memory_batches"
        / exported["batch_id"]
    )
    (batch_dir / "annotations.jsonl").write_text(
        json.dumps(
            {
                "request_id": "req_frontend_v2",
                "relevant_memory_ids": ["m1"],
                "stale_memory_ids": [],
                "claims": [
                    {
                        "claim_id": "c1",
                        "text": "用户偏好中文回答",
                        "supported_by_retrieved_memory": True,
                        "support_memory_ids": ["m1"],
                    }
                ],
                "expected_extracted_memories": ["用户偏好中文回答"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    memory_eval_components["service"].evaluate_batch(batch_id=exported["batch_id"])

    container = SimpleNamespace(
        online_eval_repository=memory_eval_components["online_repo"],
        memory_eval_v2_service=memory_eval_components["service"],
    )
    eval_response = await get_request_eval("req_frontend_v2", container=container)
    memory_response = await get_memory_eval_request("req_frontend_v2", container=container)

    assert eval_response["data"]["memory_metrics"] == {}
    assert memory_response["data"]["run"]["memory_score"] > 0.5
    assert memory_response["data"]["retrieval_event"]["retrieved_memory_ids"] == [
        "m1",
        "m2",
    ]


def test_frontend_memory_panel_uses_v2_endpoint_and_drops_legacy_memory_fields():
    source = Path("static/js/app.js").read_text(encoding="utf-8")

    assert "/eval/memory/requests/" in source
    assert "memory_hit_at_k" in source
    assert "memory_relevant_recall" in source
    assert "memory_relevant_precision" in source
    assert "memory_answer_relevance" in source
    assert "memory_extraction_precision" in source
    assert "Average Memory Score" in source
    assert "Recorded Memories" in source
    assert "Duplicate Ratio" in source
    assert "Conflict Ratio" in source
    assert "avg_memory_score" in source
    assert "label: 'Memory'" in source
    assert "setScoreCard('stat-memory-score', d.avg_memory_score)" in source
    assert "setScoreCard('stat-answer-quality', d.avg_answer_quality_score)" in source
    assert "setScoreCard('stat-memory-score', 0)" not in source

    legacy_fields = [
        "memory_beneficial_rate",
        "avg_memory_relevance_score",
        "avg_memory_usage_score",
        "compression_saved_tokens_total",
        "memory_hit_count",
        "memory_injected_chars",
        "compression_saved_tokens",
        "memory_relevance_score",
        "memory_specificity_score",
        "memory_usage_score",
        "memory_unused_injection_rate",
    ]
    for field_name in legacy_fields:
        assert field_name not in source

    retained_retrieval_labels = [
        "Retrieved Count",
        "Embedding Latency",
    ]
    for label in retained_retrieval_labels:
        assert label in source

    removed_labels = [
        "Dispatch Success",
        "Batch Status",
        "Dispatch Latency",
        "Written IDs",
        "Written Texts",
    ]
    for label in removed_labels:
        assert label not in source


def test_frontend_injected_text_renders_memory_entries_on_separate_lines():
    source = Path("static/js/app.js").read_text(encoding="utf-8")

    assert "function renderInjectedText" in source
    assert ".split(/\\r?\\n/)" in source
    assert "dim-memory-line" in source
    assert "{ key: 'injected_text', label: 'Injected Text', render: renderInjectedText }" in source


def test_memory_library_audit_export_and_evaluate_batch(memory_eval_components):
    _write_memory_file(
        memory_eval_components["tmp_path"],
        "user_a",
        [
            ("m1", "用户偏好中文回答"),
            ("m2", "用户偏好中文回答"),
            ("m3", "用户明确表示不喜欢表格"),
            ("m4", "用户偏好使用表格整理信息"),
        ],
    )

    exported = memory_eval_components["service"].export_library_audit_batch()
    batch_dir = (
        memory_eval_components["tmp_path"]
        / "data"
        / "eval"
        / "memory_library_audits"
        / exported["batch_id"]
    )
    (batch_dir / "annotations.json").write_text(
        json.dumps(
            {
                "duplicate_pairs": [
                    {
                        "memory_id_1": "m1",
                        "memory_id_2": "m2",
                        "reason": "同一偏好重复记录",
                    }
                ],
                "conflict_pairs": [
                    {
                        "memory_id_1": "m3",
                        "memory_id_2": "m4",
                        "reason": "表格偏好互相矛盾",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    result = memory_eval_components["service"].evaluate_library_audit_batch(
        batch_id=exported["batch_id"]
    )

    assert result["report"]["memory_count"] == 4
    assert result["report"]["duplicate_pair_count"] == 1
    assert result["report"]["duplicate_memory_count"] == 2
    assert result["report"]["duplicate_memory_ratio"] == 0.5
    assert result["report"]["conflict_pair_count"] == 1
    assert result["report"]["conflict_memory_count"] == 2
    assert result["report"]["conflict_memory_ratio"] == 0.5


@pytest.mark.asyncio
async def test_dashboard_online_includes_memory_library_stats(memory_eval_components):
    _seed_request_with_v2_events(memory_eval_components, request_id="req_dashboard_memory")
    memory_eval_components["memory_repo"].save_library_audit_report(
        {
            "report_id": "memlibreport_dashboard",
            "batch_id": "memlibaudit_dashboard",
            "memory_count": 2,
            "duplicate_pair_count": 1,
            "duplicate_memory_count": 1,
            "duplicate_memory_ratio": 0.5,
            "conflict_pair_count": 1,
            "conflict_memory_count": 1,
            "conflict_memory_ratio": 0.5,
            "summary": {},
        }
    )
    container = SimpleNamespace(
        online_eval_repository=memory_eval_components["online_repo"],
        memory_eval_v2_service=memory_eval_components["service"],
    )

    response = await get_online_dashboard(
        container=container,
        hours=24,
        query_type=None,
        user_id=None,
    )

    assert response["data"]["avg_memory_score"] == 0.5
    assert response["data"]["recorded_memory_count"] == 2
    assert response["data"]["memory_duplicate_ratio"] == 0.5
    assert response["data"]["memory_conflict_ratio"] == 0.5
    assert "avg_memory_relevance_score" not in response["data"]


def test_memory_eval_v2_hit_at_k_uses_relevant_memory_ids(memory_eval_components):
    _seed_request_with_v2_events(memory_eval_components, request_id="req_hit_relevant")
    exported = memory_eval_components["service"].export_batch(
        from_request_id="req_hit_relevant",
        limit=1,
    )
    batch_dir = (
        memory_eval_components["tmp_path"]
        / "data"
        / "eval"
        / "memory_batches"
        / exported["batch_id"]
    )
    (batch_dir / "annotations.jsonl").write_text(
        json.dumps(
            {
                "request_id": "req_hit_relevant",
                "relevant_memory_ids": ["m1"],
                "stale_memory_ids": [],
                "claims": [
                    {
                        "claim_id": "c1",
                        "text": "用户偏好中文回答",
                        "supported_by_retrieved_memory": True,
                        "support_memory_ids": ["m1"],
                    }
                ],
                "expected_extracted_memories": ["用户偏好中文回答"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = memory_eval_components["service"].evaluate_batch(batch_id=exported["batch_id"])

    assert result["runs"][0]["memory_hit_at_k"] == 1.0
