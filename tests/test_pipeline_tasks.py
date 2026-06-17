from __future__ import annotations

from types import SimpleNamespace

from scholar_mind.pipeline import tasks


def test_memory_consistency_audit_is_scheduled_for_3am_beijing_time():
    entry = tasks.celery_app.conf.beat_schedule["memory-consistency-audit-daily"]

    assert tasks.celery_app.conf.timezone == "Asia/Shanghai"
    assert entry["task"] == "scholar_mind.memory.audit_consistency"
    assert entry["schedule"]._orig_hour == 3
    assert entry["schedule"]._orig_minute == 0


def test_memory_consistency_audit_task_delegates_to_container(monkeypatch):
    calls = []

    class _Auditor:
        def run(self, *, user_id=None, dry_run=False):
            calls.append({"user_id": user_id, "dry_run": dry_run})
            return {"checked_count": 1, "repaired_count": 0}

    monkeypatch.setattr(
        tasks,
        "get_container",
        lambda: SimpleNamespace(memory_consistency_auditor=_Auditor()),
    )

    result = tasks.audit_memory_consistency(user_id="u1", dry_run=True)

    assert result == {"checked_count": 1, "repaired_count": 0}
    assert calls == [{"user_id": "u1", "dry_run": True}]
