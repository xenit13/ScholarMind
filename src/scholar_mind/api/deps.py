from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from scholar_mind.app import get_container
from scholar_mind.models.domain import ApiResponse, ErrorPayload, ResponseMeta


async def container_dep():
    return get_container()


def success_response(data, started_at: float) -> dict:
    payload = ApiResponse(
        success=True,
        data=data,
        error=None,
        meta=ResponseMeta(
            request_id=uuid4().hex,
            latency_ms=int((perf_counter() - started_at) * 1000),
        ),
    )
    return payload.model_dump(mode="json")


def error_response(code: str, message: str, started_at: float, details: str | None = None) -> dict:
    payload = ApiResponse(
        success=False,
        data=None,
        error=ErrorPayload(code=code, message=message, details=details),
        meta=ResponseMeta(
            request_id=uuid4().hex,
            latency_ms=int((perf_counter() - started_at) * 1000),
        ),
    )
    return payload.model_dump(mode="json")
