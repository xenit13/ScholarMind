from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.mark.slow
def test_e2e_pipeline_runs_end_to_end(tmp_path):
    """Smoke test: requires real LLM credentials and SQLite papers DB.

    Skipped unless SCHOLARMIND_LLM_API_KEY and the default SQLite DB exist.
    """
    if not os.environ.get("SCHOLARMIND_LLM_API_KEY"):
        pytest.skip("set SCHOLARMIND_LLM_API_KEY to run E2E smoke")
    db_path = Path("data/sqlite/scholar_mind.db")
    if not db_path.exists():
        pytest.skip("no scholar_mind.db present")

    from typer.testing import CliRunner

    from scholar_mind.eval.locomo_build.cli import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--database-url",
            f"sqlite:///{db_path}",
            "--out-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    samples_file = tmp_path / "scholarmind_locomo_v2.json"
    assert samples_file.exists()
    report_file = tmp_path / "validation_report.json"
    assert report_file.exists()
