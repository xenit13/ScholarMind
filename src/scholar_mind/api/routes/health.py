from __future__ import annotations

from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends

from scholar_mind.api.deps import container_dep, success_response

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
async def health(container: Annotated[object, Depends(container_dep)]):
    started = perf_counter()
    memory_manager = getattr(container, "memory_manager", None)
    llm_available = getattr(memory_manager, "llm", None) is not None
    return success_response(
        {
            "status": "healthy",
            "components": {
                "qdrant": "connected",
                "sqlite": "connected",
                "llm": "available" if llm_available else "fallback",
                "memory": "configured" if memory_manager is not None else "unavailable",
            },
            "stats": container.metrics_repository.health_stats(),
        },
        started,
    )
