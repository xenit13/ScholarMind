from __future__ import annotations

from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from scholar_mind.agents.graph import AgentOrchestrator
from scholar_mind.api.deps import container_dep
from scholar_mind.asgi import create_app
from scholar_mind.config.settings import Settings
from scholar_mind.db.session import build_session_factory, init_database
from scholar_mind.memory.manager import MemoryManager
from scholar_mind.memory.repository import MemoryRepository
from scholar_mind.models.domain import (
    MemoryCandidate,
    MemoryCandidateExtractionOutput,
    ReviewerOutput,
)
from scholar_mind.rag.engine import RAGEngine
from scholar_mind.rag.index import QdrantIndex
from scholar_mind.services.memory_eval_v2 import MemoryEvalV2Repository
from scholar_mind.services.repositories import MetricsRepository, PaperRepository, SessionRepository
from scholar_mind.services.research import ResearchService


class _FakeEmbedder:
    dimension = 2

    def embed_query(self, _content: str):
        return [0.1, 0.2]

    async def aembed_query(self, _content: str):
        return [0.1, 0.2]


class _Runnable:
    def __init__(self, llm, schema):
        self.llm = llm
        self.schema = schema

    def invoke(self, prompt):
        self.llm.prompts.append(str(prompt))
        if self.schema is MemoryCandidateExtractionOutput:
            return {
                "parsed": MemoryCandidateExtractionOutput(
                    candidates=_candidates_from_prompt(str(prompt))
                ),
                "raw": AIMessage(content="{}"),
                "parsing_error": None,
            }
        if self.schema is ReviewerOutput:
            return {
                "parsed": ReviewerOutput(
                    final_answer="已根据请求生成学习计划。",
                    review_score=1.0,
                ),
                "raw": AIMessage(content="{}"),
                "parsing_error": None,
            }
        return {"parsed": None, "raw": AIMessage(content="{}"), "parsing_error": None}


class _FakeLLM:
    def __init__(self):
        self.prompts: list[str] = []

    def invoke(self, _prompt):
        return AIMessage(content="已生成学习计划。")

    def with_structured_output(self, schema, include_raw: bool = False):
        assert include_raw is True
        return _Runnable(self, schema)


def _candidates_from_prompt(prompt: str) -> list[MemoryCandidate]:
    if "忘记我的回答风格偏好" in prompt:
        return [_candidate("用户要求忘记回答风格偏好。", operation="DELETE")]
    if "恢复回答风格偏好" in prompt:
        return [_candidate("用户要求恢复回答风格偏好。", operation="RESTORE")]
    if "归档回答风格偏好" in prompt:
        return [_candidate("用户要求归档回答风格偏好。", operation="ARCHIVE")]
    if "仍然简洁" in prompt:
        return [_candidate("用户偏好简洁回答，并优先使用中文。")]
    if "我的偏好是以后回答请简洁" in prompt:
        return [_candidate("用户偏好简洁回答，关键结论需要带引用。")]
    return []


def _candidate(content: str, operation: str | None = None) -> MemoryCandidate:
    structured = {"subject": "user", "predicate": "prefers"}
    if operation:
        structured["operation"] = operation
    return MemoryCandidate(
        memory_type="preference",
        content=content,
        structured=structured,
        keywords=["简洁", "引用"],
        importance=0.9,
        confidence=0.95,
        source="conversation",
        evidence=[{"role": "human"}],
    )


def _settings(tmp_path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'memory.db'}",
        memory_root_dir=str(tmp_path / "memories"),
        log_dir=str(tmp_path / "logs"),
        qdrant_location=":memory:",
        bootstrap_sample_data=False,
        celery_task_always_eager=True,
        memory_top_k=3,
        memory_min_similarity_score=0.0,
        memory_min_final_score=0.0,
        eval_enabled=False,
    )


def _build_real_request_stack(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    init_database(settings)
    session_factory = build_session_factory(settings)
    paper_repository = PaperRepository(session_factory)
    session_repository = SessionRepository(session_factory)
    metrics_repository = MetricsRepository(session_factory)
    memory_eval_repository = MemoryEvalV2Repository(session_factory)
    memory_repository = MemoryRepository(session_factory)
    embedder = _FakeEmbedder()
    index = QdrantIndex(settings, dimension=embedder.dimension)
    llm = _FakeLLM()
    memory_manager = MemoryManager(
        settings,
        index,
        embedder,
        llm=llm,
        metrics_repository=metrics_repository,
        memory_eval_v2_repository=memory_eval_repository,
        memory_repository=memory_repository,
    )
    rag_engine = RAGEngine(paper_repository, index, embedder)
    orchestrator = AgentOrchestrator(
        paper_repository,
        rag_engine,
        memory_manager,
        InMemorySaver(),
        chat_models={"light": llm, "reasoning": llm},
        prompt_root=settings.resolve_path(settings.prompt_dir),
    )
    service = ResearchService(
        settings,
        session_repository,
        metrics_repository,
        memory_manager,
        orchestrator,
        memory_eval_v2_repository=memory_eval_repository,
        llm=llm,
    )
    container = SimpleNamespace(
        settings=settings,
        memory_manager=memory_manager,
        research_service=service,
    )
    from scholar_mind.pipeline import tasks

    monkeypatch.setitem(tasks.celery_app.conf, "task_always_eager", True)
    monkeypatch.setattr(tasks, "get_container", lambda: container)
    return container, memory_repository, memory_manager


@pytest.mark.asyncio
async def test_real_study_plan_api_queries_drive_memory_lifecycle(tmp_path, monkeypatch):
    container, repository, memory_manager = _build_real_request_stack(tmp_path, monkeypatch)
    app = create_app()
    app.dependency_overrides[container_dep] = lambda: container

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async def ask(query: str) -> dict:
            response = await client.post(
                "/api/v1/research/study-plan",
                json={
                    "user_id": "real-user",
                    "session_id": "real-session",
                    "request": query,
                    "conditional_memory_injection": True,
                },
            )
            payload = response.json()
            assert response.status_code == 200
            assert payload["success"] is True
            return payload["data"]

        await ask("我的偏好是以后回答请简洁，关键结论要带引用。")
        added = repository.list_active("real-user")
        assert len(added) == 1

        await ask("以后回答仍然简洁，但请优先使用中文。")
        updated = repository.list_active("real-user")[0]
        assert updated.content == "用户偏好简洁回答，并优先使用中文。"
        assert updated.version == 2

        await ask("以后回答仍然简洁，但请优先使用中文。")
        assert repository.list_active("real-user")[0].version == 2

        injected, hit_count = memory_manager.get_context_sync("real-user", "按我的偏好制定计划")
        accessed = repository.list_active("real-user")[0]
        assert hit_count == 1
        assert "优先使用中文" in injected
        assert accessed.access_count >= 1

        await ask("请暂时归档回答风格偏好。")
        assert repository.list_by_status("real-user", "archived")[0].content == (
            "用户偏好简洁回答，并优先使用中文。"
        )
        archived_context, archived_hits = memory_manager.get_context_sync(
            "real-user",
            "按我的偏好制定计划",
        )
        assert archived_context == ""
        assert archived_hits == 0

        await ask("请恢复回答风格偏好。")
        assert repository.list_active("real-user")[0].content == (
            "用户偏好简洁回答，并优先使用中文。"
        )

        await ask("请忘记我的回答风格偏好。")
        assert repository.list_active("real-user") == []
        assert repository.list_by_status("real-user", "deleted")[0].content == (
            "用户偏好简洁回答，并优先使用中文。"
        )
        assert [event.operation for event in repository.list_operation_events("real-user")] == [
            "ADD",
            "UPDATE",
            "NONE",
            "ARCHIVE",
            "RESTORE",
            "DELETE",
        ]
