from __future__ import annotations

from celery import Celery
from celery.schedules import crontab

from scholar_mind.app import get_container
from scholar_mind.config.settings import get_settings

settings = get_settings()
celery_app = Celery("scholar_mind", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_always_eager=settings.celery_task_always_eager,
    timezone="Asia/Shanghai",
    beat_schedule={
        "memory-extract-all-users": {
            "task": "scholar_mind.memory.extract",
            "schedule": 300.0,
        },
        "build-index": {
            "task": "scholar_mind.pipeline.build_index",
            "schedule": 3600.0,
        },
        "memory-consistency-audit-daily": {
            "task": "scholar_mind.memory.audit_consistency",
            "schedule": crontab(hour=3, minute=0),
        },
    },
)


@celery_app.task(name="scholar_mind.pipeline.build_index")
def build_index() -> str:
    container = get_container()
    container.indexer.build()
    return "ok"


@celery_app.task(name="scholar_mind.memory.extract")
def extract_memory(user_id: str | None = None) -> int:
    container = get_container()
    return container.memory_manager.extract_pending_memories(user_id=user_id)


@celery_app.task(name="scholar_mind.memory.audit_consistency")
def audit_memory_consistency(
    user_id: str | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    container = get_container()
    auditor = getattr(container, "memory_consistency_auditor", None)
    if auditor is None:
        return {
            "run_id": "",
            "checked_count": 0,
            "inconsistent_count": 0,
            "repaired_count": 0,
            "would_repair_count": 0,
            "skipped_count": 0,
            "repaired_memory_ids": [],
            "skipped": [],
            "status": "disabled",
        }
    return auditor.run(user_id=user_id, dry_run=dry_run)


@celery_app.task(name="scholar_mind.memory.extract_request")
def extract_memory_request(
    *,
    user_id: str,
    request_id: str,
    round_messages: list[dict],
    explicit_memories: list[str] | None = None,
) -> dict[str, object]:
    container = get_container()
    return container.memory_manager.extract_request_memories(
        user_id=user_id,
        request_id=request_id,
        round_messages=round_messages,
        explicit_memories=explicit_memories,
    )
