from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from scholar_mind.config.settings import Settings, get_settings
from scholar_mind.db.session import build_session_factory, init_database
from scholar_mind.memory.consistency_audit import MemoryConsistencyAuditor
from scholar_mind.memory.manager import MemoryManager
from scholar_mind.memory.repository import MemoryRepository
from scholar_mind.models.factory import build_chat_models, build_embedding_service
from scholar_mind.services.chat import DailyChatService
from scholar_mind.services.memory_eval_v2 import MemoryEvalServiceV2, MemoryEvalV2Repository
from scholar_mind.services.memory_management import MemoryManagementService
from scholar_mind.services.repositories import (
    MetricsRepository,
    OnlineEvalRepository,
    SessionRepository,
)
from scholar_mind.vector.embeddings import EmbeddingService
from scholar_mind.vector.index import QdrantIndex


@dataclass
class AppContainer:
    settings: Settings
    session_repository: SessionRepository
    metrics_repository: MetricsRepository
    online_eval_repository: OnlineEvalRepository
    memory_eval_v2_repository: MemoryEvalV2Repository
    embedder: EmbeddingService
    index: QdrantIndex
    memory_manager: MemoryManager
    chat_service: DailyChatService
    memory_consistency_auditor: MemoryConsistencyAuditor | None
    memory_management_service: MemoryManagementService | None
    memory_eval_v2_service: MemoryEvalServiceV2

    async def aclose(self) -> None:
        return None


def build_container(settings: Settings | None = None) -> AppContainer:
    app_settings = settings or get_settings()
    init_database(app_settings)
    session_factory = build_session_factory(app_settings)
    session_repository = SessionRepository(session_factory)
    metrics_repository = MetricsRepository(session_factory)
    online_eval_repository = OnlineEvalRepository(session_factory)
    memory_eval_v2_repository = MemoryEvalV2Repository(session_factory)
    memory_repository = (
        MemoryRepository(session_factory)
        if app_settings.memory_structured_storage_enabled
        else None
    )
    chat_models = build_chat_models(app_settings)
    embedder = build_embedding_service(app_settings)
    index = QdrantIndex(app_settings, dimension=embedder.dimension)
    memory_manager = MemoryManager(
        app_settings,
        index,
        embedder,
        llm=chat_models.get("light") or chat_models.get("reasoning"),
        metrics_repository=metrics_repository,
        memory_eval_v2_repository=memory_eval_v2_repository,
        memory_repository=memory_repository,
    )
    chat_service = DailyChatService(
        settings=app_settings,
        session_repository=session_repository,
        memory_manager=memory_manager,
        llm=chat_models.get("light") or chat_models.get("reasoning"),
    )
    memory_management_service = (
        MemoryManagementService(
            memory_repository,
            index,
            embedder,
            archive_threshold=app_settings.memory_archive_threshold,
            explicit_keep_importance_threshold=(
                app_settings.memory_explicit_keep_importance_threshold
            ),
        )
        if memory_repository is not None
        else None
    )
    memory_consistency_auditor = (
        MemoryConsistencyAuditor(
            repository=memory_repository,
            index=index,
            embedder=embedder,
            llm=chat_models.get("light") or chat_models.get("reasoning"),
            min_confidence=app_settings.memory_consistency_audit_min_confidence,
            auto_fix=app_settings.memory_consistency_audit_auto_fix_enabled,
            batch_size=app_settings.memory_consistency_audit_batch_size,
        )
        if memory_repository is not None
        and app_settings.memory_consistency_audit_enabled
        else None
    )
    memory_eval_v2_service = MemoryEvalServiceV2(
        app_settings,
        memory_eval_v2_repository,
    )
    return AppContainer(
        settings=app_settings,
        session_repository=session_repository,
        metrics_repository=metrics_repository,
        online_eval_repository=online_eval_repository,
        memory_eval_v2_repository=memory_eval_v2_repository,
        embedder=embedder,
        index=index,
        memory_manager=memory_manager,
        chat_service=chat_service,
        memory_consistency_auditor=memory_consistency_auditor,
        memory_management_service=memory_management_service,
        memory_eval_v2_service=memory_eval_v2_service,
    )


@lru_cache(maxsize=1)
def get_container() -> AppContainer:
    return build_container(get_settings())
