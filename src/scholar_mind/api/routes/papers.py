from __future__ import annotations

from datetime import date
from time import perf_counter

from fastapi import APIRouter, Depends, HTTPException, Query

from scholar_mind.api.deps import container_dep, success_response

router = APIRouter(prefix="/api/v1/papers", tags=["papers"])
CONTAINER_DEP = Depends(container_dep)
TEXT_QUERY = Query(default="")
CATEGORIES_QUERY = Query(default=None)
DATE_QUERY = Query(default=None)
SORT_QUERY = Query(default="relevance")
PAGE_QUERY = Query(default=1, ge=1)
PAGE_SIZE_QUERY = Query(default=20, ge=1, le=100)


@router.get("/search")
async def search_papers(
    q: str = TEXT_QUERY,
    categories: str | None = CATEGORIES_QUERY,
    date_from: date | None = DATE_QUERY,
    date_to: date | None = DATE_QUERY,
    sort_by: str = SORT_QUERY,
    page: int = PAGE_QUERY,
    page_size: int = PAGE_SIZE_QUERY,
    container=CONTAINER_DEP,
):
    started = perf_counter()
    category_list = [item for item in categories.split(",") if item] if categories else []
    papers, total = container.paper_repository.search_papers(
        q,
        categories=category_list,
        date_from=date_from,
        date_to=date_to,
        sort_by=sort_by,
        page=page,
        page_size=page_size,
    )
    return success_response(
        {"papers": papers, "total": total, "page": page, "page_size": page_size},
        started,
    )


@router.get("/{paper_id}")
async def get_paper(paper_id: str, container=CONTAINER_DEP):
    started = perf_counter()
    paper = container.paper_repository.get_paper(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="PAPER_NOT_FOUND")
    return success_response(paper.model_dump(mode="json"), started)
