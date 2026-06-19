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
        "scholar_mind.rag.engine",
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
