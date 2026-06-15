from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from scholar_mind.db.models import Base
from scholar_mind.models.domain import AskRequest
from scholar_mind.services.repositories import OnlineEvalRepository
from scholar_mind.services.research import ResearchService


class _StubSessionRepository:
    def create_or_get(self, *, user_id: str, session_id: str):
        return {"user_id": user_id, "session_id": session_id}


class _StubMetricsRepository:
    def record_round(self, **_kwargs):
        raise AssertionError("failed requests should not record successful round metrics")


class _StubCompressor:
    def compress_with_usage(self, messages):
        return messages, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class _StubMemoryManager:
    compressor = _StubCompressor()


class _FailingOrchestrator:
    async def get_state(self, _session_id: str):
        return {}

    async def run(self, _state: dict):
        raise RuntimeError("hypothesis primary produced no drafts")


@pytest.mark.asyncio
async def test_research_service_persists_failed_request_audit(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'online_eval.db'}", future=True)
    Base.metadata.create_all(engine)
    online_repo = OnlineEvalRepository(
        sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    )
    service = ResearchService(
        settings=SimpleNamespace(default_top_k=8, eval_enabled=True),
        session_repository=_StubSessionRepository(),
        metrics_repository=_StubMetricsRepository(),
        memory_manager=_StubMemoryManager(),
        orchestrator=_FailingOrchestrator(),
        online_eval_repository=online_repo,
    )

    with pytest.raises(RuntimeError, match="hypothesis primary produced no drafts"):
        await service.ask(
            AskRequest(
                query="我想把 LLM Agent 的评测思路用到 信息检索，请分析是否合理。",
                user_id="audit-user",
                session_id="audit-session",
            )
        )

    stats = online_repo.get_dashboard_stats(hours=24, user_id="audit-user")
    requests = online_repo.get_session_evals("audit-session")

    assert stats["total_requests"] == 1
    assert stats["has_error_count"] == 1
    assert requests[0]["execution_health"]["has_error"] is True
    assert requests[0]["execution_health"]["error_type"] == "RuntimeError"
    assert requests[0]["execution_health_score"] < 1.0
