from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from scholar_mind.api.deps import container_dep
from scholar_mind.models.domain import DailyChatRequest
from scholar_mind.utils.streaming import format_sse

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
CONTAINER_DEP = Depends(container_dep)


@router.post("/stream")
async def chat_stream(request: DailyChatRequest, container=CONTAINER_DEP):
    async def event_stream():
        try:
            response = await container.chat_service.answer(request)
            yield format_sse("message", response.model_dump(mode="json"))
            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield format_sse("error", {"message": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
