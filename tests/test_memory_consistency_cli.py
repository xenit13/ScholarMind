from __future__ import annotations

import json
from types import SimpleNamespace

from typer.testing import CliRunner

from scholar_mind.main import cli_app


def test_memory_consistency_audit_cli_runs_dry_run(monkeypatch):
    calls = []

    class _Auditor:
        def run(self, *, user_id=None, dry_run=False):
            calls.append({"user_id": user_id, "dry_run": dry_run})
            return {
                "run_id": "memaudit_test",
                "checked_count": 1,
                "repaired_count": 0,
                "would_repair_count": 1,
            }

    monkeypatch.setattr(
        "scholar_mind.main.get_container",
        lambda: SimpleNamespace(memory_consistency_auditor=_Auditor()),
    )
    runner = CliRunner()

    result = runner.invoke(
        cli_app,
        ["eval", "memory-consistency-audit", "--user-id", "u1", "--dry-run"],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "run_id": "memaudit_test",
        "checked_count": 1,
        "repaired_count": 0,
        "would_repair_count": 1,
    }
    assert calls == [{"user_id": "u1", "dry_run": True}]
