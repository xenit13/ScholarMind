from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

import typer

from scholar_mind.app import get_container
from scholar_mind.eval.locomo import run_official_locomo

cli_app = typer.Typer(help="ScholarMind CLI")
eval_app = typer.Typer(help="Evaluation operations")
cli_app.add_typer(eval_app, name="eval")


def dump(value: Any) -> None:
    if hasattr(value, "model_dump"):
        payload = value.model_dump(mode="json")
    elif is_dataclass(value):
        payload = asdict(value)
    else:
        payload = value
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@eval_app.command("memory-export")
def eval_memory_export(
    from_request_id: str = typer.Option(..., "--from-request-id"),
    limit: int = typer.Option(50, min=1, max=500),
):
    container = get_container()
    result = container.memory_eval_v2_service.export_batch(
        from_request_id=from_request_id,
        limit=limit,
    )
    dump(result)


@eval_app.command("memory")
def eval_memory(batch_id: str = typer.Option(..., "--batch-id")):
    container = get_container()
    result = container.memory_eval_v2_service.evaluate_batch(batch_id=batch_id)
    dump(result)


@eval_app.command("memory-report")
def eval_memory_report(report_id: str = typer.Option(..., "--report-id")):
    container = get_container()
    dump(container.memory_eval_v2_service.get_report(report_id))


@eval_app.command("locomo")
def eval_locomo(
    data_file: str = typer.Option(..., "--data-file"),
    out_file: str = typer.Option(..., "--out-file"),
    model_key: str = typer.Option("scholarmind_memory", "--model-key"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    no_ingest: bool = typer.Option(False, "--no-ingest"),
):
    container = get_container()
    dump(
        run_official_locomo(
            data_file=data_file,
            out_file=out_file,
            memory_manager=container.memory_manager,
            model_key=model_key,
            limit=limit,
            ingest=not no_ingest,
        )
    )


@eval_app.command("memory-library-export")
def eval_memory_library_export():
    container = get_container()
    dump(container.memory_eval_v2_service.export_library_audit_batch())


@eval_app.command("memory-library")
def eval_memory_library(batch_id: str = typer.Option(..., "--batch-id")):
    container = get_container()
    dump(container.memory_eval_v2_service.evaluate_library_audit_batch(batch_id=batch_id))


@eval_app.command("memory-library-report")
def eval_memory_library_report(report_id: str = typer.Option(..., "--report-id")):
    container = get_container()
    dump(container.memory_eval_v2_service.get_library_audit_report(report_id))


@eval_app.command("memory-consistency-audit")
def eval_memory_consistency_audit(
    user_id: str | None = typer.Option(None, "--user-id"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    container = get_container()
    auditor = getattr(container, "memory_consistency_auditor", None)
    if auditor is None:
        dump(
            {
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
        )
        return
    dump(auditor.run(user_id=user_id, dry_run=dry_run))


if __name__ == "__main__":
    cli_app()
