from __future__ import annotations

from threading import Thread
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from scholar_mind.api.deps import container_dep, success_response
from scholar_mind.models.domain import SessionCreateRequest

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])
CONTAINER_DEP = Depends(container_dep)


@router.post("")
async def create_session(request: SessionCreateRequest, container=CONTAINER_DEP):
    started = perf_counter()
    session_id = str(uuid4())
    session = container.session_repository.create_or_get(request.user_id, session_id)
    return success_response(
        {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "memory_context_loaded": False,
        },
        started,
    )


@router.get("/{session_id}")
async def get_session(session_id: str, container=CONTAINER_DEP):
    started = perf_counter()
    session = container.session_repository.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
    return success_response(session.model_dump(mode="json"), started)


@router.delete("/{session_id}")
async def close_session(session_id: str, container=CONTAINER_DEP):
    started = perf_counter()
    session = container.session_repository.close(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
    Thread(
        target=container.memory_manager.extract_pending_memories,
        kwargs={"user_id": session.user_id},
        daemon=True,
    ).start()
    return success_response(
        {
            "session_id": session_id,
            "closed": True,
            "closed_at": session.closed_at,
            "metrics_recorded": True,
            "memory_extraction_scheduled": True,
        },
        started,
    )
