from __future__ import annotations

import pytest
from typer.testing import CliRunner

from scholar_mind.eval.locomo_build.cli import app


def test_cli_help_lists_subcommands():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("seeds", "dialogues", "qas", "validate", "run"):
        assert cmd in result.output


@pytest.mark.slow
def test_cli_seeds_command_errors_on_empty_db(tmp_path):
    """Without real papers in DB this should fail with a clear error."""
    runner = CliRunner()
    out_file = tmp_path / "seeds.json"
    result = runner.invoke(
        app,
        [
            "seeds",
            "--database-url",
            f"sqlite:///{tmp_path / 'empty.db'}",
            "--out",
            str(out_file),
        ],
    )
    assert result.exit_code != 0
