from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from scholar_mind.agents.graph import AgentOrchestrator
from scholar_mind.config.settings import Settings, get_settings
from scholar_mind.db.session import build_session_factory, init_database
from scholar_mind.eval.datasets import EvalDatasetBuilder
from scholar_mind.eval.rag_eval_service import RagEvalService
from scholar_mind.memory.manager import MemoryManager
from scholar_mind.memory.repository import MemoryRepository
from scholar_mind.models.factory import build_chat_models, build_embedding_service, build_reranker
from scholar_mind.pipeline.indexer import EmbeddingIndexer
from scholar_mind.pipeline.ingestor import ArxivPaperIngestor
from scholar_mind.rag.embeddings import EmbeddingService
from scholar_mind.rag.engine import RAGEngine
from scholar_mind.rag.index import QdrantIndex
from scholar_mind.services.memory_eval_v2 import MemoryEvalServiceV2, MemoryEvalV2Repository
from scholar_mind.services.memory_management import MemoryManagementService
from scholar_mind.services.rag_eval_repository import RagEvalRepository
from scholar_mind.services.repositories import (
    EvalRepository,
    MetricsRepository,
    OnlineEvalRepository,
    PaperRepository,
    SessionRepository,
)
from scholar_mind.services.research import ResearchService
from scholar_mind.utils.sample_data import seed_sample_data


@dataclass
class AppContainer:
    settings: Settings
    paper_repository: PaperRepository
    session_repository: SessionRepository
    eval_repository: EvalRepository
    metrics_repository: MetricsRepository
    online_eval_repository: OnlineEvalRepository
    memory_eval_v2_repository: MemoryEvalV2Repository
    rag_eval_repository: RagEvalRepository
    embedder: EmbeddingService
    index: QdrantIndex
    rag_engine: RAGEngine
    memory_manager: MemoryManager
    memory_management_service: MemoryManagementService | None
    orchestrator: AgentOrchestrator
    research_service: ResearchService
    rag_eval_service: RagEvalService
    memory_eval_v2_service: MemoryEvalServiceV2
    indexer: EmbeddingIndexer
    arxiv_ingestor: ArxivPaperIngestor
    dataset_builder: EvalDatasetBuilder

    async def aclose(self) -> None:
        await self.orchestrator.aclose()


def build_container(settings: Settings | None = None) -> AppContainer:
    app_settings = settings or get_settings()
    init_database(app_settings)
    session_factory = build_session_factory(app_settings)
    with session_factory() as session:
        if app_settings.bootstrap_sample_data:
            seeded = seed_sample_data(session, app_settings)
            if seeded:
                session.commit()
    paper_repository = PaperRepository(session_factory)
    session_repository = SessionRepository(session_factory)
    eval_repository = EvalRepository(session_factory)
    metrics_repository = MetricsRepository(session_factory)
    online_eval_repository = OnlineEvalRepository(session_factory)
    memory_eval_v2_repository = MemoryEvalV2Repository(session_factory)
    memory_repository = (
        MemoryRepository(session_factory)
        if app_settings.memory_structured_storage_enabled
        else None
    )
    rag_eval_repository = RagEvalRepository(session_factory)
    chat_models = build_chat_models(app_settings)
    embedder = build_embedding_service(app_settings)
    index = QdrantIndex(app_settings, dimension=embedder.dimension)
    rag_engine = RAGEngine(paper_repository, index, embedder)
    rag_engine.reranker_service = build_reranker(app_settings)
    indexer = EmbeddingIndexer(rag_engine)
    indexer.build()
    rag_engine.ensure_sparse_stats()
    arxiv_ingestor = ArxivPaperIngestor(
        app_settings,
        paper_repository,
        rag_engine,
    )
    memory_manager = MemoryManager(
        app_settings,
        index,
        embedder,
        llm=chat_models.get("light") or chat_models.get("reasoning"),
        metrics_repository=metrics_repository,
        memory_eval_v2_repository=memory_eval_v2_repository,
        memory_repository=memory_repository,
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
    orchestrator = AgentOrchestrator(
        paper_repository,
        rag_engine,
        memory_manager,
        _build_checkpoint_saver_factory(app_settings),
        chat_models=chat_models,
        prompt_root=app_settings.resolve_path(app_settings.prompt_dir),
    )
    research_service = ResearchService(
        app_settings,
        session_repository,
        metrics_repository,
        memory_manager,
        orchestrator,
        online_eval_repository=online_eval_repository,
        memory_eval_v2_repository=memory_eval_v2_repository,
        llm=chat_models.get("light") or chat_models.get("reasoning"),
    )
    rag_eval_service = RagEvalService(
        app_settings,
        rag_eval_repository,
        rag_engine,
        llm=chat_models.get("light") or chat_models.get("reasoning"),
        embedding_service=embedder,
        online_eval_repository=online_eval_repository,
    )
    memory_eval_v2_service = MemoryEvalServiceV2(
        app_settings,
        memory_eval_v2_repository,
    )
    return AppContainer(
        settings=app_settings,
        paper_repository=paper_repository,
        session_repository=session_repository,
        eval_repository=eval_repository,
        metrics_repository=metrics_repository,
        online_eval_repository=online_eval_repository,
        memory_eval_v2_repository=memory_eval_v2_repository,
        rag_eval_repository=rag_eval_repository,
        embedder=embedder,
        index=index,
        rag_engine=rag_engine,
        memory_manager=memory_manager,
        memory_management_service=memory_management_service,
        orchestrator=orchestrator,
        research_service=research_service,
        rag_eval_service=rag_eval_service,
        memory_eval_v2_service=memory_eval_v2_service,
        indexer=indexer,
        arxiv_ingestor=arxiv_ingestor,
        dataset_builder=EvalDatasetBuilder(paper_repository),
    )


@lru_cache(maxsize=1)
def get_container() -> AppContainer:
    return build_container(get_settings())


def _build_checkpoint_saver_factory(settings: Settings):
    if settings.environment.lower() in {"test", "testing"}:
        def memory_factory() -> InMemorySaver:
            return InMemorySaver()

        return memory_factory

    db_path = _sqlite_path(settings.checkpoint_database_url, settings)
    if db_path != Path(":memory:"):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _backup_legacy_checkpoint_db(db_path)

    async def factory() -> AsyncSqliteSaver:
        manager = AsyncSqliteSaver.from_conn_string(str(db_path))
        saver = await manager.__aenter__()
        saver._context_manager = manager
        await saver.setup()
        return saver

    return factory


def _sqlite_path(database_url: str, settings: Settings) -> Path:
    if database_url.startswith("sqlite:///"):
        return settings.resolve_path(database_url.removeprefix("sqlite:///"))
    if database_url == ":memory:":
        return Path(":memory:")
    return Path(database_url)


def _backup_legacy_checkpoint_db(db_path: Path) -> None:
    if not db_path.exists():
        return
    with sqlite3.connect(db_path) as conn:
        if _is_official_checkpoint_schema(conn):
            return
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    db_path.rename(db_path.with_suffix(f"{db_path.suffix}.legacy-{stamp}"))


def _is_official_checkpoint_schema(conn: sqlite3.Connection) -> bool:
    required = {
        "checkpoints": {
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            "parent_checkpoint_id",
            "type",
            "checkpoint",
            "metadata",
        },
        "writes": {
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            "task_id",
            "idx",
            "channel",
            "type",
            "value",
        },
    }
    for table, columns in required.items():
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        if not rows:
            return False
        available = {str(row[1]) for row in rows}
        if not columns.issubset(available):
            return False
    return True
