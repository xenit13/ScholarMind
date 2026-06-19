from __future__ import annotations

from typer.testing import CliRunner

from scholar_mind.app import AppContainer
from scholar_mind.main import cli_app


runner = CliRunner()


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
