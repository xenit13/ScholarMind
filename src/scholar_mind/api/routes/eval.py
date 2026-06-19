from __future__ import annotations

import csv
import io
import json as json_lib
from time import perf_counter
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from scholar_mind.api.deps import container_dep, success_response

router = APIRouter(prefix="/api/v1/eval", tags=["eval"])
CONTAINER_DEP = Depends(container_dep)


@router.get("/requests/{request_id}")
async def get_request_eval(
    request_id: str,
    container=CONTAINER_DEP,
):
    started = perf_counter()
    repo = _get_online_eval_repo(container)
    result = repo.get_request_eval(request_id)
    if result is None:
        raise HTTPException(status_code=404, detail="REQUEST_NOT_FOUND")
    return success_response(result, started)


@router.get("/requests/{request_id}/diagnosis")
async def get_request_diagnosis(
    request_id: str,
    container=CONTAINER_DEP,
):
    started = perf_counter()
    repo = _get_online_eval_repo(container)
    result = repo.get_request_diagnosis(request_id)
    if result is None:
        raise HTTPException(status_code=404, detail="REQUEST_NOT_FOUND")
    return success_response(result, started)


@router.get("/sessions/{session_id}/requests")
async def get_session_evals(
    session_id: str,
    container=CONTAINER_DEP,
    limit: int = Query(default=50, ge=1, le=200),
):
    started = perf_counter()
    repo = _get_online_eval_repo(container)
    results = repo.get_session_evals(session_id)
    return success_response(results[:limit], started)


@router.get("/requests/{request_id}/events")
async def get_request_events(request_id: str, container=CONTAINER_DEP):
    started = perf_counter()
    repo = _get_online_eval_repo(container)
    result = repo.get_request_events(request_id)
    return success_response(result, started)


@router.get("/dashboard/online")
async def get_online_dashboard(
    container=CONTAINER_DEP,
    hours: int = Query(default=24, ge=1, le=720),
    query_type: str | None = Query(default=None),
    user_id: str | None = Query(default=None),
):
    started = perf_counter()
    repo = _get_online_eval_repo(container)
    stats = repo.get_dashboard_stats(hours=hours, query_type=query_type, user_id=user_id)
    memory_service = getattr(container, "memory_eval_v2_service", None)
    if memory_service is not None:
        stats.update(memory_service.get_dashboard_memory_stats())
    return success_response(stats, started)


@router.get("/dashboard/low-scores")
async def get_low_score_requests(
    container=CONTAINER_DEP,
    threshold: float = Query(default=0.4, ge=0.0, le=1.0),
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    started = perf_counter()
    repo = _get_online_eval_repo(container)
    results = repo.get_low_score_requests(threshold=threshold, limit=limit, offset=offset)
    return success_response(results, started)


@router.get("/dashboard/requests")
async def get_all_requests(
    container=CONTAINER_DEP,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    started = perf_counter()
    repo = _get_online_eval_repo(container)
    results = repo.get_all_requests(limit=limit, offset=offset)
    return success_response(results, started)


@router.get("/dashboard/score-trend")
async def get_score_trend(
    container=CONTAINER_DEP,
    hours: int = Query(default=168, ge=1, le=2160),
    granularity: str = Query(default="hourly", pattern="^(hourly|daily|weekly)$"),
    user_id: str | None = Query(default=None),
):
    started = perf_counter()
    repo = _get_online_eval_repo(container)
    trend = repo.get_score_trend(hours=hours, granularity=granularity, user_id=user_id)
    return success_response(trend, started)


@router.get("/dashboard/users")
async def get_dashboard_users(container=CONTAINER_DEP):
    started = perf_counter()
    repo = _get_online_eval_repo(container)
    users = repo.get_distinct_users()
    return success_response(users, started)


@router.get("/dashboard/export")
async def export_dashboard_csv(
    container=CONTAINER_DEP,
    hours: int = Query(default=168, ge=1, le=2160),
    user_id: str | None = Query(default=None),
    format: str = Query(default="csv", pattern="^(csv|json)$"),
):
    repo = _get_online_eval_repo(container)
    rows = repo.get_eval_rows_for_export(hours=hours, user_id=user_id)
    if format == "json":
        return _export_json(rows)
    return _export_csv(rows)


@router.get("/memory/batches/{batch_id}")
async def get_memory_eval_batch(batch_id: str, container=CONTAINER_DEP):
    started = perf_counter()
    service = _get_memory_eval_v2_service(container)
    result = service.get_batch(batch_id)
    if result is None:
        raise HTTPException(status_code=404, detail="MEMORY_EVAL_BATCH_NOT_FOUND")
    return success_response(result, started)


@router.get("/memory/reports/{report_id}")
async def get_memory_eval_report(report_id: str, container=CONTAINER_DEP):
    started = perf_counter()
    service = _get_memory_eval_v2_service(container)
    result = service.get_report(report_id)
    if result is None:
        raise HTTPException(status_code=404, detail="MEMORY_EVAL_REPORT_NOT_FOUND")
    return success_response(result, started)


@router.get("/memory/requests/{request_id}")
async def get_memory_eval_request(request_id: str, container=CONTAINER_DEP):
    started = perf_counter()
    service = _get_memory_eval_v2_service(container)
    result = service.get_request(request_id)
    if result is None:
        raise HTTPException(status_code=404, detail="MEMORY_EVAL_REQUEST_NOT_FOUND")
    return success_response(result, started)


def _get_online_eval_repo(container: Any):
    repo = getattr(container, "online_eval_repository", None)
    if repo is None:
        raise HTTPException(status_code=503, detail="REQUEST_AUDIT_NOT_AVAILABLE")
    return repo


def _get_memory_eval_v2_service(container: Any):
    service = getattr(container, "memory_eval_v2_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="MEMORY_EVAL_V2_NOT_AVAILABLE")
    return service


_CSV_COLUMNS = [
    "request_overview_json",
    "memory_data_json",
    "rag_data_json",
]


def _export_csv(rows: list[dict]) -> Any:
    from fastapi.responses import Response

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(_CSV_COLUMNS)
    for row in rows:
        writer.writerow([row.get(column, "") for column in _CSV_COLUMNS])
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=dashboard_export.csv"},
    )


def _export_json(rows: list[dict]) -> Any:
    from fastapi.responses import Response

    requests = []
    for row in rows:
        requests.append(
            {
                "request_overview": row.get("request_overview", {}),
                "memory_data": row.get(
                    "memory_data",
                    {
                        "run": None,
                        "retrieval_event": None,
                        "extraction_event": None,
                    },
                ),
                "rag_data": row.get(
                    "rag_data",
                    {"metrics": {}, "events": [], "empty_retrieval": False},
                ),
            }
        )
    content = json_lib.dumps({"requests": requests}, indent=2, ensure_ascii=False)
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=dashboard_export.json"},
    )
