from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from scholar_mind.api.deps import container_dep, success_response
from scholar_mind.pipeline.arxiv_storage import list_category_paper_ids
from scholar_mind.pipeline.downloader import (
    ArxivApiMetadataDownloader,
    ArxivMetadataDownloader,
)
from scholar_mind.pipeline.ingestor import ArxivPaperIngestor
from scholar_mind.pipeline.recent_ingest import RecentArxivBatchIngestor
from scholar_mind.utils.streaming import format_sse

router = APIRouter(prefix="/api/v1/ingest", tags=["ingest"])
CONTAINER_DEP = Depends(container_dep)


class IngestPapersRequest(BaseModel):
    paper_ids: list[str]


class IngestLocalPapersRequest(BaseModel):
    paper_ids: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)


def _build_runner(container, proxy: str | None = None):
    container.arxiv_ingestor.proxy = proxy
    metadata_downloader = ArxivMetadataDownloader(proxy=proxy)
    api_metadata_downloader = ArxivApiMetadataDownloader(proxy=proxy)
    paper_ingestor = ArxivPaperIngestor(
        container.settings,
        container.paper_repository,
        container.rag_engine,
        metadata_downloader=metadata_downloader,
        proxy=proxy,
    )
    return RecentArxivBatchIngestor(
        metadata_downloader=metadata_downloader,
        paper_ingestor=paper_ingestor,
        api_metadata_downloader=api_metadata_downloader,
    )


def _resolve_local_paper_requests(
    container,
    request: IngestLocalPapersRequest,
) -> list[tuple[str, str | None]]:
    raw_root = container.settings.resolve_path(container.settings.raw_data_dir) / "arxiv"
    resolved: list[tuple[str, str | None]] = []
    seen_ids: set[str] = set()

    for paper_id in request.paper_ids:
        normalized = paper_id.strip()
        if not normalized or normalized in seen_ids:
            continue
        resolved.append((normalized, None))
        seen_ids.add(normalized)

    for category in request.categories:
        for paper_id in list_category_paper_ids(raw_root, category):
            if paper_id in seen_ids:
                continue
            resolved.append((paper_id, category))
            seen_ids.add(paper_id)

    return resolved


@router.post("/recent")
async def ingest_recent(
    count: int = Query(ge=1, description="Number of recent papers to ingest"),
    category: list[str] = Query(
        default=[], description="arXiv category filter (repeatable). Omit for all categories."
    ),
    proxy: str | None = Header(default=None, alias="X-Arxiv-Proxy"),
    container=CONTAINER_DEP,
):
    started = perf_counter()
    runner = _build_runner(container, proxy=proxy)
    result = await runner.ingest_recent(count=count, categories=category)
    return success_response(result, started)


@router.post("/recent/stream")
async def ingest_recent_stream(
    count: int = Query(ge=1, description="Number of recent papers to ingest"),
    category: list[str] = Query(
        default=[], description="arXiv category filter (repeatable). Omit for all categories."
    ),
    from_date: str | None = Query(default=None, description="Start date YYYY-MM-DD"),
    to_date: str | None = Query(default=None, description="End date YYYY-MM-DD"),
    proxy: str | None = Header(default=None, alias="X-Arxiv-Proxy"),
    container=CONTAINER_DEP,
):
    runner = _build_runner(container, proxy=proxy)

    async def event_stream():
        async for event in runner.ingest_recent_stream(
            count=count, categories=category, from_date=from_date, to_date=to_date,
        ):
            yield format_sse(event["type"], event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/papers/stream")
async def ingest_papers_stream(
    request: IngestPapersRequest,
    proxy: str | None = Header(default=None, alias="X-Arxiv-Proxy"),
    container=CONTAINER_DEP,
):
    runner = _build_runner(container, proxy=proxy)

    async def event_stream():
        async for event in runner.ingest_papers_stream(paper_ids=request.paper_ids):
            yield format_sse(event["type"], event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/local/stream")
async def ingest_local_papers_stream(
    request: IngestLocalPapersRequest,
    container=CONTAINER_DEP,
):
    paper_requests = _resolve_local_paper_requests(container, request)
    if not paper_requests:
        raise HTTPException(status_code=400, detail="specify paper_ids or categories with local files")

    async def event_stream():
        total = len(paper_requests)
        selected_papers = [
            {
                "paper_id": paper_id,
                "title": "",
                "categories": [category] if category else [],
                "created": None,
                "updated": None,
            }
            for paper_id, category in paper_requests
        ]
        yield format_sse(
            "metadata_downloaded",
            {
                "type": "metadata_downloaded",
                "step": "selected",
                "message": f"Selected {total} local papers to ingest",
                "selected_count": total,
                "categories": request.categories,
                "lookback_windows_days": [],
                "selected_papers": selected_papers,
            },
        )

        ingested: list[dict] = []
        for index, (paper_id, category) in enumerate(paper_requests, start=1):
            category_prefix = f"[{category}] " if category else ""
            yield format_sse(
                "paper_start",
                {
                    "type": "paper_start",
                    "step": "ingesting",
                    "index": index,
                    "total": total,
                    "paper_id": paper_id,
                    "title": "",
                    "category": category,
                    "message": f"[{index}/{total}] Ingesting {category_prefix}{paper_id}",
                },
            )
            try:
                result = await container.arxiv_ingestor.ingest_local_paper(paper_id, category=category)
                ingested.append(result)
                yield format_sse(
                    "paper_ingested",
                    {
                        "type": "paper_ingested",
                        "step": "ingesting",
                        "index": index,
                        "total": total,
                        "paper_id": paper_id,
                        "title": result.get("title", ""),
                        "category": category,
                        "paper": result,
                        "message": f"[{index}/{total}] Done: {category_prefix}{paper_id} "
                        f"({result.get('source_format', '?')}, {result.get('chunk_count', 0)} chunks)",
                    },
                )
            except Exception as exc:
                yield format_sse(
                    "paper_failed",
                    {
                        "type": "paper_failed",
                        "step": "ingesting",
                        "index": index,
                        "total": total,
                        "paper_id": paper_id,
                        "title": "",
                        "category": category,
                        "error": str(exc),
                        "message": f"[{index}/{total}] Failed: {category_prefix}{paper_id} — {exc}",
                    },
                )

        yield format_sse(
            "complete",
            {
                "type": "complete",
                "step": "done",
                "result": {
                    "requested_count": total,
                    "selected_count": total,
                    "ingested_count": len(ingested),
                    "categories": request.categories,
                    "lookback_windows_days": [],
                    "selected_papers": selected_papers,
                    "ingested_papers": ingested,
                },
                "message": f"All done: {len(ingested)}/{total} ingested successfully",
            },
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")
