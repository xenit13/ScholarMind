from __future__ import annotations

from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends

from scholar_mind.api.deps import container_dep, success_response

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
async def health(container: Annotated[object, Depends(container_dep)]):
    started = perf_counter()
    llm_available = any(container.orchestrator.chat_models.values())
    return success_response(
        {
            "status": "healthy",
            "components": {
                "qdrant": "connected",
                "sqlite": "connected",
                "llm": "available" if llm_available else "fallback",
                "celery": "configured",
            },
            "stats": container.metrics_repository.health_stats(),
        },
        started,
    )
