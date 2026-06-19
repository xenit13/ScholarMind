from __future__ import annotations

import ast
from pathlib import Path

from typer.testing import CliRunner

from scholar_mind.app import AppContainer
from scholar_mind.asgi import create_app
from scholar_mind.main import cli_app

runner = CliRunner()
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cli_exposes_memory_only_top_level_commands():
    result = runner.invoke(cli_app, ["--help"])

    assert result.exit_code == 0
    assert "eval" in result.stdout

    forbidden_commands = [
        "ask",
        "cross-domain",
        "idea-novelty",
        "paper",
        "paper-reading",
        "study-plan",
        "trend",
    ]
    for command in forbidden_commands:
        assert command not in result.stdout


def test_app_container_exposes_memory_only_services():
    field_names = set(AppContainer.__dataclass_fields__)

    required_fields = {
        "settings",
        "memory_eval_v2_repository",
        "memory_eval_v2_service",
        "memory_manager",
        "memory_management_service",
        "memory_consistency_auditor",
        "embedder",
        "index",
    }
    assert required_fields <= field_names

    forbidden_fields = {
        "arxiv_ingestor",
        "dataset_builder",
        "orchestrator",
        "paper_repository",
        "rag_engine",
        "rag_eval_repository",
        "rag_eval_service",
        "research_service",
    }
    assert field_names.isdisjoint(forbidden_fields)


def test_memory_package_has_no_business_module_imports():
    memory_files = sorted((PROJECT_ROOT / "src" / "scholar_mind" / "memory").glob("*.py"))
    forbidden_prefixes = (
        "scholar_mind.agents",
        "scholar_mind.pipeline",
        "scholar_mind.services.research",
        "scholar_mind.eval",
        "scholar_mind.eval.rag",
        "scholar_mind.eval.ragas",
        "scholar_mind.models.rag_eval",
        "scholar_mind.services.rag_eval",
        "scholar_mind.vector.engine",
    )
    violations: list[str] = []

    for path in memory_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.startswith(forbidden_prefixes):
                    violations.append(f"{path.relative_to(PROJECT_ROOT)} imports {name}")

    assert violations == []


def test_memory_only_branch_has_no_business_source_modules():
    forbidden_paths = [
        "src/scholar_mind/agents",
        "src/scholar_mind/pipeline",
        "src/scholar_mind/api/routes/ingest.py",
        "src/scholar_mind/api/routes/papers.py",
        "src/scholar_mind/api/routes/research.py",
        "src/scholar_mind/eval/rag_custom_metrics.py",
        "src/scholar_mind/eval/rag_dataset.py",
        "src/scholar_mind/eval/rag_eval_service.py",
        "src/scholar_mind/eval/rag_runner.py",
        "src/scholar_mind/eval/ragas_official.py",
        "src/scholar_mind/models/rag_eval_models.py",
        "src/scholar_mind/rag",
        "src/scholar_mind/rag/engine.py",
        "src/scholar_mind/rag/query_transform.py",
        "src/scholar_mind/rag/reranker.py",
        "src/scholar_mind/services/rag_eval_repository.py",
        "src/scholar_mind/services/research.py",
        "src/scholar_mind/utils/sample_data.py",
    ]

    present = []
    for path in forbidden_paths:
        absolute = PROJECT_ROOT / path
        if absolute.is_file():
            present.append(path)
        elif absolute.is_dir() and any(absolute.rglob("*.py")):
            present.append(path)

    assert present == []


def test_database_models_are_memory_only():
    from scholar_mind.db import models

    forbidden_model_names = {
        "PaperModel",
        "PaperSectionModel",
        "PaperChunkModel",
        "RagRetrievalEventV2Model",
        "RequestRagEvalAnnotationModel",
        "RagEvalCaseModel",
        "RagEvalRunV2Model",
        "RagEvalResultV2Model",
    }

    present = sorted(name for name in forbidden_model_names if hasattr(models, name))

    assert present == []


def test_domain_models_are_memory_only():
    from scholar_mind.models import domain

    forbidden_model_names = {
        "AskRequest",
        "BenchmarkStrategyResult",
        "ChatRequest",
        "ChunkType",
        "Citation",
        "CrossDomainOutput",
        "CrossDomainReport",
        "CrossDomainRequest",
        "IdeaNoveltyReport",
        "IdeaNoveltyRequest",
        "PaperChunk",
        "PaperReadingReport",
        "PaperReadingRequest",
        "PlannerOutput",
        "QueryType",
        "RAGEvalSample",
        "ResearchAnswer",
        "RetrievedChunk",
        "RetrievalStrategyName",
        "StructuredPaper",
        "TrendReport",
        "TrendRequest",
    }
    present = sorted(name for name in forbidden_model_names if hasattr(domain, name))

    assert present == []


def test_settings_do_not_expose_rag_or_paper_config():
    from scholar_mind.config.settings import Settings

    settings = Settings()
    forbidden_fields = {
        "cross_domain_candidate_top_k",
        "default_rag_strategy",
        "final_citation_top_k",
        "hybrid_candidate_multiplier",
        "idea_evidence_top_k",
        "papers_seed_path",
        "rag_eval_context_top_k",
        "rag_eval_dataset_path",
        "rag_eval_default_dataset",
        "rag_eval_embedding_model",
        "rag_eval_enabled",
        "rag_eval_llm_max_tokens",
        "rag_eval_llm_model",
        "rag_eval_redundancy_similarity_threshold",
        "reranker_api_key",
        "reranker_base_url",
        "reranker_enabled",
        "reranker_model",
        "reranker_provider",
        "reranker_request_timeout_seconds",
    }

    present = sorted(name for name in forbidden_fields if hasattr(settings, name))

    assert present == []


def test_vector_index_exposes_memory_only_methods():
    from scholar_mind.vector.index import QdrantIndex

    forbidden_methods = {
        "delete_paper_chunks",
        "is_paper_collection_empty",
        "search_chunks_dense",
        "search_chunks_hybrid",
        "search_chunks_sparse",
        "upsert_chunks",
    }
    present = sorted(name for name in forbidden_methods if hasattr(QdrantIndex, name))

    assert present == []


def test_eval_cli_exposes_only_memory_commands():
    result = runner.invoke(cli_app, ["eval", "--help"])

    assert result.exit_code == 0

    required_commands = [
        "memory",
        "memory-consistency-audit",
        "memory-export",
        "memory-library",
        "memory-library-export",
        "memory-library-report",
        "memory-report",
    ]
    for command in required_commands:
        assert command in result.stdout

    forbidden_commands = [
        "rag-report",
        "rag-run",
    ]
    for command in forbidden_commands:
        assert command not in result.stdout


def test_asgi_exposes_memory_only_routes():
    app = create_app()
    route_paths = {getattr(route, "path", "") for route in app.routes}

    required_paths = {
        "/api/v1/health",
        "/api/v1/eval/memory/batches/{batch_id}",
    }
    assert required_paths <= route_paths

    forbidden_prefixes = (
        "/api/v1/research",
        "/api/v1/papers",
        "/api/v1/ingest",
        "/api/v1/eval/rag",
    )
    violations = [
        path for path in sorted(route_paths) if path.startswith(forbidden_prefixes)
    ]
    assert violations == []
