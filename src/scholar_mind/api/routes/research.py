from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from scholar_mind.api.deps import container_dep, success_response
from scholar_mind.models.domain import (
    ASK_QUERY_MIN_LENGTH,
    AskRequest,
    ChatRequest,
    CrossDomainRequest,
    IdeaNoveltyRequest,
    PaperReadingRequest,
    QueryType,
    StudyPlanRequest,
    TranscriptMemoryExtractionRequest,
    TrendRequest,
)
from scholar_mind.rag.top_k import IDEA_EVIDENCE_TOP_K
from scholar_mind.utils.streaming import format_sse

router = APIRouter(prefix="/api/v1/research", tags=["research"])
CONTAINER_DEP = Depends(container_dep)
ASK_QUERY = Query(min_length=ASK_QUERY_MIN_LENGTH)
IDEA_QUERY = Query(min_length=5)
SHORT_TOPIC_QUERY = Query(min_length=3)
USER_ID_QUERY = Query()
PAPER_ID_QUERY = Query()
SESSION_QUERY = Query(default=None)
PAPER_IDS_QUERY = Query(default=None)
CATEGORIES_QUERY = Query(default=None)
MAX_PAPERS_QUERY = Query(default=IDEA_EVIDENCE_TOP_K)
MAX_RESULTS_QUERY = Query(default=10)
RAG_STRATEGY_QUERY = Query(default="hybrid")
GRANULARITY_QUERY = Query(default="quarterly")
MIN_SIMILARITY_QUERY = Query(default=0.1)
GENERATE_HYPOTHESES_QUERY = Query(default=True)
MAX_HYPOTHESES_QUERY = Query(default=3)
CONDITIONAL_MEMORY_INJECTION_QUERY = Query(default=None)
CROSS_DOMAIN_REQUEST_QUERY = Query(min_length=5)
STUDY_REQUEST_QUERY = Query(default="帮我制定一个学习计划")
GOAL_QUERY = Query(default=None)
CURRENT_PROGRESS_QUERY = Query(default=None)
TIMELINE_WEEKS_QUERY = Query(default=None)
WEEKLY_HOURS_QUERY = Query(default=None)
CONSTRAINTS_QUERY = Query(default=None)
INSTRUCTION_QUERY = Query(default="开始精读")
SECTION_QUERY = Query(default=None)
PARAGRAPH_INDEX_QUERY = Query(default=None)
DEPTH_QUERY = Query(default="standard")


def _request_memory_injection_flag(settings, request) -> bool:
    if "conditional_memory_injection" in getattr(request, "model_fields_set", set()):
        return bool(request.conditional_memory_injection)
    return bool(getattr(settings, "conditional_memory_injection", False))


def _memory_injection_flag(settings, value: bool | None) -> bool:
    if value is None:
        return bool(getattr(settings, "conditional_memory_injection", False))
    return bool(value)


def _memory_control_payload(request) -> dict:
    payload = {}
    for key in ("memory_extraction_enabled", "request_memory_extraction_enabled"):
        value = getattr(request, key, None)
        if value is not None:
            payload[key] = value
    if getattr(request, "wait_for_pending_extractions", False):
        payload["wait_for_pending_extractions"] = True
    return payload


@router.post("/ask")
async def ask(request: AskRequest, container=CONTAINER_DEP):
    started = perf_counter()
    result = await container.research_service.ask(request)
    return success_response(result.model_dump(mode="json"), started)


@router.post("/idea-novelty")
async def idea_novelty(request: IdeaNoveltyRequest, container=CONTAINER_DEP):
    started = perf_counter()
    result = await container.research_service.idea_novelty(request)
    return success_response(result, started)


@router.post("/trend")
async def trend(request: TrendRequest, container=CONTAINER_DEP):
    started = perf_counter()
    result = await container.research_service.trend(request)
    return success_response(result, started)


@router.post("/cross-domain")
async def cross_domain(request: CrossDomainRequest, container=CONTAINER_DEP):
    started = perf_counter()
    result = await container.research_service.cross_domain(request)
    return success_response(result, started)


@router.post("/study-plan")
async def study_plan(request: StudyPlanRequest, container=CONTAINER_DEP):
    started = perf_counter()
    result = await container.research_service.study_plan(request)
    return success_response(result, started)


@router.post("/paper-reading")
async def paper_reading(request: PaperReadingRequest, container=CONTAINER_DEP):
    started = perf_counter()
    result = await container.research_service.paper_reading(request)
    return success_response(result, started)


@router.post("/memory/transcript")
async def extract_transcript_memory(
    request: TranscriptMemoryExtractionRequest,
    container=CONTAINER_DEP,
):
    started = perf_counter()
    result = container.research_service.extract_transcript_memories(
        user_id=request.user_id,
        request_id=request.request_id,
        session_id=request.session_id,
        round_messages=[
            item.model_dump(mode="json") for item in request.round_messages
        ],
    )
    if request.wait_for_pending_extractions:
        wait = getattr(container.research_service, "wait_for_pending_extractions", None)
        if callable(wait):
            wait(timeout=300.0)
    return success_response(result, started)


@router.post("/ask/stream")
async def ask_stream(request: AskRequest, container=CONTAINER_DEP):
    return _stream_response(
        container,
        query=request.query,
        user_id=request.user_id,
        session_id=request.session_id,
        query_type=QueryType.QA,
        request_payload={
            "paper_ids": request.paper_ids,
            "rag_strategy": request.rag_strategy.value,
            "top_k": container.settings.final_citation_top_k,
            "conditional_memory_injection": _request_memory_injection_flag(
                container.settings, request
            ),
            **_memory_control_payload(request),
        },
    )


@router.post("/stream")
async def chat_stream(request: ChatRequest, container=CONTAINER_DEP):
    return _stream_response(
        container,
        query=request.query,
        user_id=request.user_id,
        session_id=request.session_id,
        query_type=None,
        request_payload={
            "paper_ids": request.paper_ids,
            "rag_strategy": request.rag_strategy.value,
            "top_k": container.settings.final_citation_top_k,
            "conditional_memory_injection": _request_memory_injection_flag(
                container.settings, request
            ),
            **_memory_control_payload(request),
        },
    )


@router.get("/ask/stream")
async def ask_stream_get(
    query: str = ASK_QUERY,
    user_id: str = USER_ID_QUERY,
    session_id: str | None = SESSION_QUERY,
    paper_ids: list[str] | None = PAPER_IDS_QUERY,
    rag_strategy: str = RAG_STRATEGY_QUERY,
    conditional_memory_injection: bool | None = CONDITIONAL_MEMORY_INJECTION_QUERY,
    container=CONTAINER_DEP,
):
    return _stream_response(
        container,
        query=query,
        user_id=user_id,
        session_id=session_id,
        query_type=QueryType.QA,
        request_payload={
            "paper_ids": paper_ids or [],
            "rag_strategy": rag_strategy,
            "top_k": container.settings.final_citation_top_k,
            "conditional_memory_injection": _memory_injection_flag(
                container.settings, conditional_memory_injection
            ),
        },
    )


@router.get("/stream")
async def chat_stream_get(
    query: str = ASK_QUERY,
    user_id: str = USER_ID_QUERY,
    session_id: str | None = SESSION_QUERY,
    paper_ids: list[str] | None = PAPER_IDS_QUERY,
    rag_strategy: str = RAG_STRATEGY_QUERY,
    conditional_memory_injection: bool | None = CONDITIONAL_MEMORY_INJECTION_QUERY,
    container=CONTAINER_DEP,
):
    return _stream_response(
        container,
        query=query,
        user_id=user_id,
        session_id=session_id,
        query_type=None,
        request_payload={
            "paper_ids": paper_ids or [],
            "rag_strategy": rag_strategy,
            "top_k": container.settings.final_citation_top_k,
            "conditional_memory_injection": _memory_injection_flag(
                container.settings, conditional_memory_injection
            ),
        },
    )


@router.post("/idea-novelty/stream")
async def idea_novelty_stream(request: IdeaNoveltyRequest, container=CONTAINER_DEP):
    return _stream_response(
        container,
        query=request.idea,
        user_id=request.user_id,
        session_id=request.session_id,
        query_type=QueryType.IDEA_NOVELTY,
        request_payload={
            "rag_strategy": request.rag_strategy.value,
            "max_papers": request.max_papers,
            "categories": request.categories,
            "date_from": request.time_range.start if request.time_range else None,
            "date_to": request.time_range.end if request.time_range else None,
            "conditional_memory_injection": _request_memory_injection_flag(
                container.settings, request
            ),
        },
    )


@router.get("/idea-novelty/stream")
async def idea_novelty_stream_get(
    idea: str = IDEA_QUERY,
    user_id: str = USER_ID_QUERY,
    session_id: str | None = SESSION_QUERY,
    rag_strategy: str = RAG_STRATEGY_QUERY,
    max_papers: int = MAX_PAPERS_QUERY,
    categories: list[str] | None = CATEGORIES_QUERY,
    conditional_memory_injection: bool | None = CONDITIONAL_MEMORY_INJECTION_QUERY,
    container=CONTAINER_DEP,
):
    return _stream_response(
        container,
        query=idea,
        user_id=user_id,
        session_id=session_id,
        query_type=QueryType.IDEA_NOVELTY,
        request_payload={
            "rag_strategy": rag_strategy,
            "max_papers": max_papers,
            "categories": categories or [],
            "date_from": None,
            "date_to": None,
            "conditional_memory_injection": _memory_injection_flag(
                container.settings, conditional_memory_injection
            ),
        },
    )


@router.post("/trend/stream")
async def trend_stream(request: TrendRequest, container=CONTAINER_DEP):
    return _stream_response(
        container,
        query=request.topic,
        user_id=request.user_id,
        session_id=request.session_id,
        query_type=QueryType.TREND,
        request_payload={
            "rag_strategy": container.settings.default_rag_strategy,
            "date_from": request.time_range.start if request.time_range else None,
            "date_to": request.time_range.end if request.time_range else None,
            "granularity": request.granularity,
            "conditional_memory_injection": _request_memory_injection_flag(
                container.settings, request
            ),
        },
    )


@router.get("/trend/stream")
async def trend_stream_get(
    topic: str = SHORT_TOPIC_QUERY,
    user_id: str = USER_ID_QUERY,
    session_id: str | None = SESSION_QUERY,
    granularity: str = GRANULARITY_QUERY,
    conditional_memory_injection: bool | None = CONDITIONAL_MEMORY_INJECTION_QUERY,
    container=CONTAINER_DEP,
):
    return _stream_response(
        container,
        query=topic,
        user_id=user_id,
        session_id=session_id,
        query_type=QueryType.TREND,
        request_payload={
            "rag_strategy": container.settings.default_rag_strategy,
            "date_from": None,
            "date_to": None,
            "granularity": granularity,
            "conditional_memory_injection": _memory_injection_flag(
                container.settings, conditional_memory_injection
            ),
        },
    )


@router.post("/cross-domain/stream")
async def cross_domain_stream(request: CrossDomainRequest, container=CONTAINER_DEP):
    return _stream_response(
        container,
        query=request.request,
        user_id=request.user_id,
        session_id=request.session_id,
        query_type=QueryType.CROSS_DOMAIN,
        request_payload={
            "rag_strategy": request.rag_strategy.value,
            "max_hypotheses": request.max_hypotheses,
            "conditional_memory_injection": _request_memory_injection_flag(
                container.settings, request
            ),
        },
    )


@router.get("/cross-domain/stream")
async def cross_domain_stream_get(
    request: str = CROSS_DOMAIN_REQUEST_QUERY,
    user_id: str = USER_ID_QUERY,
    session_id: str | None = SESSION_QUERY,
    rag_strategy: str = RAG_STRATEGY_QUERY,
    max_hypotheses: int = MAX_HYPOTHESES_QUERY,
    conditional_memory_injection: bool | None = CONDITIONAL_MEMORY_INJECTION_QUERY,
    container=CONTAINER_DEP,
):
    return _stream_response(
        container,
        query=request,
        user_id=user_id,
        session_id=session_id,
        query_type=QueryType.CROSS_DOMAIN,
        request_payload={
            "rag_strategy": rag_strategy,
            "max_hypotheses": max_hypotheses,
            "conditional_memory_injection": _memory_injection_flag(
                container.settings, conditional_memory_injection
            ),
        },
    )


@router.post("/study-plan/stream")
async def study_plan_stream(request: StudyPlanRequest, container=CONTAINER_DEP):
    request_payload = request.model_dump(mode="json", exclude={"user_id", "session_id"})
    request_payload["conditional_memory_injection"] = _request_memory_injection_flag(
        container.settings, request
    )
    return _stream_response(
        container,
        query=request.request or request.goal or "帮我制定一个学习计划",
        user_id=request.user_id,
        session_id=request.session_id,
        query_type=QueryType.STUDY_PLAN,
        request_payload=request_payload,
    )


@router.get("/study-plan/stream")
async def study_plan_stream_get(
    user_id: str = USER_ID_QUERY,
    session_id: str | None = SESSION_QUERY,
    request: str = STUDY_REQUEST_QUERY,
    goal: str | None = GOAL_QUERY,
    current_progress: str | None = CURRENT_PROGRESS_QUERY,
    read_papers: list[str] | None = PAPER_IDS_QUERY,
    known_topics: list[str] | None = CATEGORIES_QUERY,
    timeline_weeks: int | None = TIMELINE_WEEKS_QUERY,
    weekly_hours: int | None = WEEKLY_HOURS_QUERY,
    constraints: list[str] | None = CONSTRAINTS_QUERY,
    conditional_memory_injection: bool | None = CONDITIONAL_MEMORY_INJECTION_QUERY,
    container=CONTAINER_DEP,
):
    return _stream_response(
        container,
        query=request or goal or "帮我制定一个学习计划",
        user_id=user_id,
        session_id=session_id,
        query_type=QueryType.STUDY_PLAN,
        request_payload={
            "request": request,
            "goal": goal,
            "current_progress": current_progress,
            "read_papers": read_papers or [],
            "known_topics": known_topics or [],
            "timeline_weeks": timeline_weeks,
            "weekly_hours": weekly_hours,
            "constraints": constraints or [],
            "conditional_memory_injection": _memory_injection_flag(
                container.settings, conditional_memory_injection
            ),
        },
    )


@router.post("/paper-reading/stream")
async def paper_reading_stream(request: PaperReadingRequest, container=CONTAINER_DEP):
    request_payload = request.model_dump(mode="json", exclude={"user_id", "session_id"})
    request_payload["conditional_memory_injection"] = _request_memory_injection_flag(
        container.settings, request
    )
    return _stream_response(
        container,
        query=request.instruction,
        user_id=request.user_id,
        session_id=request.session_id,
        query_type=QueryType.PAPER_READING,
        request_payload=request_payload,
    )


@router.get("/paper-reading/stream")
async def paper_reading_stream_get(
    paper_id: str = PAPER_ID_QUERY,
    user_id: str = USER_ID_QUERY,
    session_id: str | None = SESSION_QUERY,
    instruction: str = INSTRUCTION_QUERY,
    section: str | None = SECTION_QUERY,
    paragraph_index: int | None = PARAGRAPH_INDEX_QUERY,
    depth: str = DEPTH_QUERY,
    conditional_memory_injection: bool | None = CONDITIONAL_MEMORY_INJECTION_QUERY,
    container=CONTAINER_DEP,
):
    return _stream_response(
        container,
        query=instruction,
        user_id=user_id,
        session_id=session_id,
        query_type=QueryType.PAPER_READING,
        request_payload={
            "paper_id": paper_id,
            "instruction": instruction,
            "section": section,
            "paragraph_index": paragraph_index,
            "depth": depth,
            "conditional_memory_injection": _memory_injection_flag(
                container.settings, conditional_memory_injection
            ),
        },
    )


def _stream_response(container, *, query, user_id, session_id, query_type, request_payload):
    async def event_stream():
        async for event, data in container.research_service.stream(
            query=query,
            user_id=user_id,
            session_id=session_id,
            query_type=query_type,
            request_payload=request_payload,
        ):
            yield format_sse(event, data)
        if request_payload.get("wait_for_pending_extractions"):
            wait = getattr(container.research_service, "wait_for_pending_extractions", None)
            if callable(wait):
                wait(timeout=300.0)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
