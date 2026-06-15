from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

import anyio
import typer

from scholar_mind.app import get_container
from scholar_mind.models.domain import (
    AskRequest,
    CrossDomainRequest,
    IdeaNoveltyRequest,
    PaperReadingRequest,
    StudyPlanRequest,
    TrendRequest,
)
from scholar_mind.models.rag_eval_models import RagEvalRunRequest
from scholar_mind.rag.top_k import IDEA_EVIDENCE_TOP_K

cli_app = typer.Typer(help="ScholarMind CLI")
paper_app = typer.Typer(help="Paper operations")
eval_app = typer.Typer(help="Evaluation operations")
cli_app.add_typer(paper_app, name="paper")
cli_app.add_typer(eval_app, name="eval")
PAPER_IDS_OPTION = typer.Option(None, "--paper-ids", "--paper-id")


def dump(value: Any) -> None:
    if hasattr(value, "model_dump"):
        payload = value.model_dump(mode="json")
    elif is_dataclass(value):
        payload = asdict(value)
    else:
        payload = value
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@cli_app.command()
def ask(
    query: str,
    user_id: str = typer.Option("local-user"),
    session_id: str | None = typer.Option(None),
    paper_ids: list[str] = PAPER_IDS_OPTION,
    rag_strategy: str = typer.Option("hybrid"),
):
    container = get_container()
    result = anyio.run(
        container.research_service.ask,
        AskRequest(
            query=query,
            user_id=user_id,
            session_id=session_id,
            paper_ids=paper_ids or [],
            rag_strategy=rag_strategy,
        ),
    )
    dump(result)


@cli_app.command("idea-novelty")
def idea_novelty(
    idea: str,
    user_id: str = typer.Option("local-user"),
    session_id: str | None = typer.Option(None),
    max_papers: int = typer.Option(IDEA_EVIDENCE_TOP_K),
    rag_strategy: str = typer.Option("hybrid"),
):
    container = get_container()
    result = anyio.run(
        container.research_service.idea_novelty,
        IdeaNoveltyRequest(
            idea=idea,
            user_id=user_id,
            session_id=session_id,
            max_papers=max_papers,
            rag_strategy=rag_strategy,
        ),
    )
    dump(result)


@cli_app.command()
def trend(
    topic: str,
    user_id: str = typer.Option("local-user"),
    session_id: str | None = typer.Option(None),
    granularity: str = typer.Option("quarterly"),
):
    container = get_container()
    result = anyio.run(
        container.research_service.trend,
        TrendRequest(topic=topic, user_id=user_id, session_id=session_id, granularity=granularity),
    )
    dump(result)


@cli_app.command("cross-domain")
def cross_domain(
    request: str,
    user_id: str = typer.Option("local-user"),
    session_id: str | None = typer.Option(None),
    max_hypotheses: int = typer.Option(3),
    rag_strategy: str = typer.Option("hybrid"),
):
    container = get_container()
    result = anyio.run(
        container.research_service.cross_domain,
        CrossDomainRequest(
            request=request,
            user_id=user_id,
            session_id=session_id,
            max_hypotheses=max_hypotheses,
            rag_strategy=rag_strategy,
        ),
    )
    dump(result)


@cli_app.command("study-plan")
def study_plan(
    request: str = typer.Argument("帮我制定一个学习计划"),
    user_id: str = typer.Option("local-user"),
    session_id: str | None = typer.Option(None),
    goal: str | None = typer.Option(None),
    timeline_weeks: int | None = typer.Option(None),
    weekly_hours: int | None = typer.Option(None),
):
    container = get_container()
    result = anyio.run(
        container.research_service.study_plan,
        StudyPlanRequest(
            request=request,
            user_id=user_id,
            session_id=session_id,
            goal=goal,
            timeline_weeks=timeline_weeks,
            weekly_hours=weekly_hours,
        ),
    )
    dump(result)


@cli_app.command("paper-reading")
def paper_reading(
    paper_id: str,
    instruction: str = typer.Argument("开始精读"),
    user_id: str = typer.Option("local-user"),
    session_id: str | None = typer.Option(None),
    section: str | None = typer.Option(None),
    paragraph_index: int | None = typer.Option(None),
    depth: str = typer.Option("standard"),
):
    container = get_container()
    result = anyio.run(
        container.research_service.paper_reading,
        PaperReadingRequest(
            paper_id=paper_id,
            instruction=instruction,
            user_id=user_id,
            session_id=session_id,
            section=section,
            paragraph_index=paragraph_index,
            depth=depth,
        ),
    )
    dump(result)


@paper_app.command("search")
def paper_search(query: str):
    container = get_container()
    papers, total = container.paper_repository.search_papers(query)
    dump({"papers": papers, "total": total})


@paper_app.command("get")
def paper_get(paper_id: str):
    container = get_container()
    paper = container.paper_repository.get_paper(paper_id)
    dump(paper or {"error": "PAPER_NOT_FOUND"})


@paper_app.command("ingest-arxiv")
def paper_ingest_arxiv(paper_id: str):
    container = get_container()
    result = anyio.run(container.arxiv_ingestor.ingest_paper, paper_id)
    dump(result)


@eval_app.command("rag-run")
def eval_rag_run():
    container = get_container()
    result = anyio.run(container.rag_eval_service.create_run, RagEvalRunRequest())
    dump(result)


@eval_app.command("rag-report")
def eval_rag_report(run_id: str = typer.Option(..., "--run-id")):
    container = get_container()
    dump(container.rag_eval_service.get_run(run_id))


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


if __name__ == "__main__":
    cli_app()
