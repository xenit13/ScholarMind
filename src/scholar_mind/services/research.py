from __future__ import annotations

import logging
from time import perf_counter
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from scholar_mind.agents.common import merge_usage
from scholar_mind.agents.state import flatten_graph_state
from scholar_mind.eval.context import (
    finish_eval_context,
    init_eval_context,
)
from scholar_mind.models.domain import (
    AskRequest,
    CrossDomainRequest,
    IdeaNoveltyRequest,
    PaperReadingRequest,
    QueryType,
    ResearchAnswer,
    StudyPlanRequest,
    TrendRequest,
)
from scholar_mind.models.eval_models import AgentEvent, AnswerEvent
from scholar_mind.rag.top_k import FINAL_CITATION_TOP_K
from scholar_mind.utils.messages import serialize_messages

logger = logging.getLogger(__name__)
REQUEST_MEMORY_EXTRACTION_TASK = "scholar_mind.memory.extract_request"


def _enqueue_request_memory_extraction(
    *,
    user_id: str,
    request_id: str,
    round_messages: list[dict],
    explicit_memories: list[str],
) -> None:
    from scholar_mind.pipeline.tasks import celery_app

    kwargs = {
        "user_id": user_id,
        "request_id": request_id,
        "round_messages": round_messages,
        "explicit_memories": explicit_memories,
    }
    task = celery_app.tasks.get(REQUEST_MEMORY_EXTRACTION_TASK)
    if bool(celery_app.conf.task_always_eager) and task is not None:
        task.apply_async(kwargs=kwargs)
        return
    celery_app.send_task(REQUEST_MEMORY_EXTRACTION_TASK, kwargs=kwargs, retry=False)


def _resolve_citation_chunk_ids(
    citations: list[dict],
    retrieved_chunks: list[dict],
) -> list[str]:
    if not citations or not retrieved_chunks:
        return []

    matched_chunk_ids: list[str] = []
    used_chunk_ids: set[str] = set()
    for citation in citations:
        paper_id = citation.get("paper_id", "")
        section = citation.get("section", "")

        exact = next(
            (
                chunk for chunk in retrieved_chunks
                if chunk.get("paper_id") == paper_id
                and chunk.get("section") == section
                and chunk.get("chunk_id") not in used_chunk_ids
            ),
            None,
        )
        fallback = exact or next(
            (
                chunk for chunk in retrieved_chunks
                if chunk.get("paper_id") == paper_id
                and chunk.get("chunk_id") not in used_chunk_ids
            ),
            None,
        )
        if fallback and fallback.get("chunk_id"):
            chunk_id = fallback["chunk_id"]
            matched_chunk_ids.append(chunk_id)
            used_chunk_ids.add(chunk_id)
    return matched_chunk_ids


def _usage_metrics(state: dict) -> dict[str, int | float]:
    usage = state.get("llm_usage", {}) or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }


def _request_memory_injection_flag(settings, request) -> bool:
    if "conditional_memory_injection" in getattr(request, "model_fields_set", set()):
        return bool(request.conditional_memory_injection)
    return bool(getattr(settings, "conditional_memory_injection", False))


def _request_payload_with_memory_defaults(settings, request_payload: dict) -> dict:
    payload = dict(request_payload)
    payload.setdefault(
        "conditional_memory_injection",
        bool(getattr(settings, "conditional_memory_injection", False)),
    )
    return payload


def _memory_extraction_enabled(request_payload: dict | None) -> bool:
    if not isinstance(request_payload, dict):
        return True
    return bool(request_payload.get("memory_extraction_enabled", True))


def _request_memory_extraction_enabled(request_payload: dict | None) -> bool:
    if not isinstance(request_payload, dict):
        return True
    return bool(request_payload.get("request_memory_extraction_enabled", True))


def _runtime_metrics_snapshot(stored: dict) -> dict[str, int | float | list[str]]:
    usage = _usage_metrics(stored)
    return {
        "latency_ms": int(stored.get("latency_ms", 0) or 0),
        "retrieval_latency_ms": int(stored.get("rag_latency_ms", 0) or 0),
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
        "citations_count": len(stored.get("citations", [])),
        "retrieved_chunks_count": len(stored.get("retrieved_chunks", [])),
        "output_length": len(stored.get("final_answer", "")),
        "agent_path": [
            item.get("agent", "")
            for item in stored.get("agent_trace", [])
            if item.get("agent")
        ],
    }


def _serialize_agent_events(eval_ctx) -> list[dict]:
    return [
        {
            "event_id": event.event_id,
            "agent": event.agent,
            "duration_ms": event.duration_ms,
            "output_summary": event.output_summary,
            "started_at": event.started_at.isoformat() if event.started_at else None,
            "finished_at": event.finished_at.isoformat() if event.finished_at else None,
            "created_at": event.created_at.isoformat() if event.created_at else None,
        }
        for event in eval_ctx.agent_events
    ]


def _serialize_answer_event(eval_ctx) -> dict:
    if not eval_ctx.answer_events:
        return {}
    answer = eval_ctx.answer_events[-1]
    return {
        "event_id": answer.event_id,
        "draft": answer.draft,
        "final_answer": answer.final_answer,
        "citations": answer.citations,
        "citation_ids": answer.citation_ids,
        "created_at": answer.created_at.isoformat() if answer.created_at else None,
    }


class ResearchService:
    def __init__(
        self,
        settings,
        session_repository,
        metrics_repository,
        memory_manager,
        orchestrator,
        online_eval_repository=None,
        memory_eval_v2_repository=None,
        llm=None,
    ):
        self.settings = settings
        self.session_repository = session_repository
        self.metrics_repository = metrics_repository
        self.memory_manager = memory_manager
        self.orchestrator = orchestrator
        self.online_eval_repository = online_eval_repository
        self.memory_eval_v2_repository = memory_eval_v2_repository
        self.llm = llm

    async def ask(self, request: AskRequest) -> ResearchAnswer:
        state, request_id, latency = await self._execute(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id,
            query_type=QueryType.QA,
            request_payload={
                "paper_ids": request.paper_ids,
                "rag_strategy": request.rag_strategy.value,
                "top_k": getattr(self.settings, "final_citation_top_k", FINAL_CITATION_TOP_K),
                "conditional_memory_injection": _request_memory_injection_flag(
                    self.settings, request
                ),
            },
        )
        answer = ResearchAnswer(
            answer=state.get("final_answer", ""),
            citations=state.get("citations", []),
            related_papers=state.get("related_papers", []),
            rag_info={
                "strategy": state.get("rag_strategy", request.rag_strategy.value),
                "chunks_retrieved": len(state.get("retrieved_chunks", [])),
                "chunks_used": len(state.get("citations", [])),
                "retrieval_latency_ms": state.get("rag_latency_ms", 0),
            },
            agent_trace=state.get("agent_trace", []),
            session_id=state["session_id"],
        )
        self.metrics_repository.record_round(
            request_id=request_id,
            user_id=request.user_id,
            session_id=state["session_id"],
            query_type=QueryType.QA.value,
            success=True,
            retrieval_latency_ms=state.get("rag_latency_ms", 0),
            latency_ms=latency,
            citations_count=len(answer.citations),
            retrieved_chunks_count=len(state.get("retrieved_chunks", [])),
            output_length=len(answer.answer),
            agent_path=[item.agent for item in answer.agent_trace],
            **_usage_metrics(state),
        )
        return answer

    async def idea_novelty(self, request: IdeaNoveltyRequest) -> dict:
        state, request_id, latency = await self._execute(
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
                    self.settings, request
                ),
            },
        )
        report = state.get("report_payload", {})
        self.metrics_repository.record_round(
            request_id=request_id,
            user_id=request.user_id,
            session_id=state["session_id"],
            query_type=QueryType.IDEA_NOVELTY.value,
            success=True,
            retrieval_latency_ms=state.get("rag_latency_ms", 0),
            latency_ms=latency,
            citations_count=sum(
                len(item.get("evidence", [])) for item in report.get("overlapping_papers", [])
            ),
            retrieved_chunks_count=len(state.get("retrieved_chunks", [])),
            output_length=len(state.get("draft", "")),
            agent_path=[item["agent"] for item in state.get("agent_trace", [])],
            **_usage_metrics(state),
        )
        return {
            "idea_novelty": report,
            "papers_analyzed": len(
                {chunk["paper_id"] for chunk in state.get("retrieved_chunks", [])}
            ),
            "rag_info": {
                "strategy": state.get("rag_strategy"),
                "total_queries": len(state.get("sub_queries", [])) or 1,
            },
            "agent_trace": state.get("agent_trace", []),
            "session_id": state["session_id"],
        }

    async def trend(self, request: TrendRequest) -> dict:
        state, request_id, latency = await self._execute(
            query=request.topic,
            user_id=request.user_id,
            session_id=request.session_id,
            query_type=QueryType.TREND,
            request_payload={
                "categories": [],
                "date_from": request.time_range.start if request.time_range else None,
                "date_to": request.time_range.end if request.time_range else None,
                "granularity": request.granularity,
                "rag_strategy": self.settings.default_rag_strategy,
                "conditional_memory_injection": _request_memory_injection_flag(
                    self.settings, request
                ),
            },
        )
        self.metrics_repository.record_round(
            request_id=request_id,
            user_id=request.user_id,
            session_id=state["session_id"],
            query_type=QueryType.TREND.value,
            success=True,
            retrieval_latency_ms=state.get("rag_latency_ms", 0),
            latency_ms=latency,
            citations_count=0,
            retrieved_chunks_count=len(state.get("retrieved_chunks", [])),
            output_length=len(state.get("draft", "")),
            agent_path=[item["agent"] for item in state.get("agent_trace", [])],
            **_usage_metrics(state),
        )
        return {
            "trend": state.get("report_payload", state.get("trend_data", {})),
            "agent_trace": state.get("agent_trace", []),
            "session_id": state["session_id"],
        }

    async def cross_domain(self, request: CrossDomainRequest) -> dict:
        state, request_id, latency = await self._execute(
            query=request.request,
            user_id=request.user_id,
            session_id=request.session_id,
            query_type=QueryType.CROSS_DOMAIN,
            request_payload={
                "rag_strategy": request.rag_strategy.value,
                "max_hypotheses": request.max_hypotheses,
                "conditional_memory_injection": _request_memory_injection_flag(
                    self.settings, request
                ),
            },
        )
        self.metrics_repository.record_round(
            request_id=request_id,
            user_id=request.user_id,
            session_id=state["session_id"],
            query_type=QueryType.CROSS_DOMAIN.value,
            success=True,
            retrieval_latency_ms=state.get("rag_latency_ms", 0),
            latency_ms=latency,
            citations_count=0,
            retrieved_chunks_count=len(state.get("retrieved_chunks", [])),
            output_length=len(state.get("draft", "")),
            agent_path=[item["agent"] for item in state.get("agent_trace", [])],
            **_usage_metrics(state),
        )
        return {
            "cross_domain": state.get("report_payload", {}),
            "final_answer": state.get("final_answer", ""),
            "agent_trace": state.get("agent_trace", []),
            "session_id": state["session_id"],
        }

    async def study_plan(self, request: StudyPlanRequest) -> dict:
        query = request.request or request.goal or "帮我制定一个学习计划"
        state, request_id, latency = await self._execute(
            query=query,
            user_id=request.user_id,
            session_id=request.session_id,
            query_type=QueryType.STUDY_PLAN,
            request_payload={
                "request": request.request or "帮我制定一个学习计划",
                "goal": request.goal,
                "current_progress": request.current_progress,
                "read_papers": request.read_papers,
                "known_topics": request.known_topics,
                "timeline_weeks": request.timeline_weeks,
                "weekly_hours": request.weekly_hours,
                "constraints": request.constraints,
                "conditional_memory_injection": _request_memory_injection_flag(
                    self.settings, request
                ),
            },
        )
        self.metrics_repository.record_round(
            request_id=request_id,
            user_id=request.user_id,
            session_id=state["session_id"],
            query_type=QueryType.STUDY_PLAN.value,
            success=True,
            retrieval_latency_ms=state.get("rag_latency_ms", 0),
            latency_ms=latency,
            citations_count=0,
            retrieved_chunks_count=0,
            output_length=len(state.get("draft", "")),
            agent_path=[item["agent"] for item in state.get("agent_trace", [])],
            **_usage_metrics(state),
        )
        return {
            "study_plan": state.get("report_payload", state.get("study_plan", {})),
            "agent_trace": state.get("agent_trace", []),
            "session_id": state["session_id"],
        }

    async def paper_reading(self, request: PaperReadingRequest) -> dict:
        query = request.instruction or "开始精读"
        state, request_id, latency = await self._execute(
            query=query,
            user_id=request.user_id,
            session_id=request.session_id,
            query_type=QueryType.PAPER_READING,
            request_payload={
                "paper_id": request.paper_id,
                "instruction": request.instruction,
                "section": request.section,
                "paragraph_index": request.paragraph_index,
                "depth": request.depth,
                "conditional_memory_injection": _request_memory_injection_flag(
                    self.settings, request
                ),
            },
        )
        self.metrics_repository.record_round(
            request_id=request_id,
            user_id=request.user_id,
            session_id=state["session_id"],
            query_type=QueryType.PAPER_READING.value,
            success=True,
            retrieval_latency_ms=state.get("rag_latency_ms", 0),
            latency_ms=latency,
            citations_count=0,
            retrieved_chunks_count=0,
            output_length=len(state.get("draft", "")),
            agent_path=[item["agent"] for item in state.get("agent_trace", [])],
            **_usage_metrics(state),
        )
        return {
            "paper_reading": state.get("report_payload", {}),
            "agent_trace": state.get("agent_trace", []),
            "session_id": state["session_id"],
        }

    async def stream(
        self,
        query: str,
        user_id: str,
        session_id: str | None,
        query_type: QueryType | None,
        request_payload: dict,
    ):
        request_payload = _request_payload_with_memory_defaults(self.settings, request_payload)
        request_id = uuid4().hex
        session_id = session_id or str(uuid4())
        self.session_repository.create_or_get(user_id=user_id, session_id=session_id)

        # Initialize request audit context.
        eval_ctx = init_eval_context(
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
            query=query,
            query_type=query_type.value if query_type else "auto",
        )

        started = perf_counter()
        eval_finished = False
        compression_usage = {}
        try:
            previous = await self.orchestrator.get_state(session_id) or {}
            previous_messages, compression_usage = (
                self.memory_manager.compressor.compress_with_usage(
                    list(previous.get("messages", []))
                )
            )
            state = {
                "messages": previous_messages + [HumanMessage(content=query)],
                "request": {
                    "query": query,
                    "user_id": user_id,
                    "session_id": session_id,
                    "query_type_hint": query_type.value if query_type else None,
                    "payload": request_payload,
                },
                "telemetry": {"llm_usage": merge_usage(compression_usage)},
            }
            async for event in self.orchestrator.stream(state):
                yield event
            result = await self.orchestrator.get_state(session_id)
            if result is not None:
                latency = int((perf_counter() - started) * 1000)
                stored = self._persist_state(
                    user_id=user_id,
                    session_id=session_id,
                    request_id=request_id,
                    query=query,
                    request_payload=request_payload,
                    previous_state=previous,
                    result=result,
                )
                stored["latency_ms"] = latency
                if eval_ctx:
                    self._append_eval_trace_events(eval_ctx, stored, request_id)
                    finish_eval_context(eval_ctx, stored)
                    eval_finished = True
                if getattr(self.settings, "eval_enabled", True):
                    self._persist_request_audit(eval_ctx, stored, request_id)
        except Exception as exc:
            latency = int((perf_counter() - started) * 1000)
            stored = self._failed_audit_state(
                query_type=query_type,
                latency_ms=latency,
                error=exc,
                llm_usage=compression_usage,
            )
            if eval_ctx and not eval_finished:
                finish_eval_context(eval_ctx, stored)
            if getattr(self.settings, "eval_enabled", True):
                self._persist_request_audit(eval_ctx, stored, request_id, error=exc)
            raise

    async def _execute(
        self,
        query: str,
        user_id: str,
        session_id: str | None,
        query_type: QueryType | None,
        request_payload: dict,
    ):
        request_payload = _request_payload_with_memory_defaults(self.settings, request_payload)
        request_id = uuid4().hex
        session_id = session_id or str(uuid4())
        self.session_repository.create_or_get(user_id=user_id, session_id=session_id)

        # Initialize request audit context.
        eval_ctx = init_eval_context(
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
            query=query,
            query_type=query_type.value if query_type else "auto",
        )

        started = perf_counter()
        eval_finished = False
        compression_usage = {}
        try:
            previous = await self.orchestrator.get_state(session_id) or {}
            previous_messages, compression_usage = (
                self.memory_manager.compressor.compress_with_usage(
                    list(previous.get("messages", []))
                )
            )

            state = {
                "messages": previous_messages + [HumanMessage(content=query)],
                "request": {
                    "query": query,
                    "user_id": user_id,
                    "session_id": session_id,
                    "query_type_hint": query_type.value if query_type else None,
                    "payload": request_payload,
                },
                "telemetry": {"llm_usage": merge_usage(compression_usage)},
            }
            result = await self.orchestrator.run(state)
            latency = int((perf_counter() - started) * 1000)
            stored = self._persist_state(
                user_id=user_id,
                session_id=session_id,
                request_id=request_id,
                query=query,
                request_payload=request_payload,
                previous_state=previous,
                result=result,
            )
            stored["latency_ms"] = latency

            if eval_ctx:
                self._append_eval_trace_events(eval_ctx, stored, request_id)

            finish_eval_context(eval_ctx, stored)
            eval_finished = True
            if getattr(self.settings, "eval_enabled", True):
                self._persist_request_audit(eval_ctx, stored, request_id)

            return stored, request_id, latency
        except Exception as exc:
            latency = int((perf_counter() - started) * 1000)
            stored = self._failed_audit_state(
                query_type=query_type,
                latency_ms=latency,
                error=exc,
                llm_usage=compression_usage,
            )
            if eval_ctx and not eval_finished:
                finish_eval_context(eval_ctx, stored)
            if getattr(self.settings, "eval_enabled", True):
                self._persist_request_audit(eval_ctx, stored, request_id, error=exc)
            raise

    @staticmethod
    def _failed_audit_state(
        *,
        query_type: QueryType | None,
        latency_ms: int,
        error: Exception,
        llm_usage: dict | None = None,
    ) -> dict:
        return {
            "query_type": query_type.value if query_type else "auto",
            "final_answer": "",
            "citations": [],
            "retrieved_chunks": [],
            "rag_latency_ms": 0,
            "latency_ms": latency_ms,
            "llm_usage": llm_usage or {},
            "agent_trace": [],
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

    def _append_eval_trace_events(self, eval_ctx, stored: dict, request_id: str) -> None:
        citations = [
            c if isinstance(c, dict) else c.model_dump()
            for c in stored.get("citations", [])[:10]
        ]
        retrieved_chunks = stored.get("retrieved_chunks", [])
        eval_ctx.answer_events.append(
            AnswerEvent(
                request_id=request_id,
                draft=stored.get("draft", ""),
                final_answer=stored.get("final_answer", ""),
                citations=citations,
                citation_ids=_resolve_citation_chunk_ids(citations, retrieved_chunks),
            )
        )
        for item in stored.get("agent_trace", []):
            agent_name = item.get("agent", "")
            if not agent_name:
                continue
            eval_ctx.agent_events.append(
                AgentEvent(
                    request_id=request_id,
                    agent=agent_name,
                    duration_ms=int(item.get("duration_ms", 0)),
                    output_summary=stored.get("draft", "")[:280]
                    if agent_name == "researcher"
                    else stored.get("final_answer", "")[:280],
                )
            )

    def _persist_state(
        self,
        *,
        user_id: str,
        session_id: str,
        request_id: str,
        query: str,
        request_payload: dict,
        previous_state: dict,
        result: dict,
    ) -> dict:
        previous_messages = list(previous_state.get("messages", []))
        current_messages = list(result.get("messages", []))
        new_messages = current_messages[len(previous_messages) :]
        previous_tool_trace_messages = list(previous_state.get("tool_trace_messages", []))
        current_tool_trace_messages = list(result.get("tool_trace_messages", []))
        new_tool_trace_messages = current_tool_trace_messages[len(previous_tool_trace_messages) :]

        stored = flatten_graph_state(dict(result))
        stored["query"] = query
        stored["messages"] = serialize_messages(current_messages)
        stored.pop("tool_trace_messages", None)
        self.session_repository.update_from_state(
            user_id=user_id, session_id=session_id, state=stored
        )

        round_messages = list(new_messages)
        if new_tool_trace_messages:
            if round_messages and isinstance(round_messages[-1], AIMessage):
                round_messages = (
                    round_messages[:-1] + new_tool_trace_messages + round_messages[-1:]
                )
            else:
                round_messages.extend(new_tool_trace_messages)

        if round_messages and _memory_extraction_enabled(request_payload):
            prior_human_rounds = sum(
                1 for message in previous_messages if getattr(message, "type", "") == "human"
            )
            round_index = prior_human_rounds + 1
            pending_buffer = getattr(self.memory_manager, "pending_buffer", None)
            if pending_buffer is not None:
                pending_buffer.add_round(
                    user_id=user_id,
                    session_id=session_id,
                    request_id=request_id,
                    round_index=round_index,
                    messages=round_messages,
                )
            self.memory_manager.log_round(
                user_id=user_id,
                session_id=session_id,
                round_index=round_index,
                messages=round_messages,
                explicit_memories=stored.get("explicit_memory_candidates", []),
            )
            if _request_memory_extraction_enabled(request_payload):
                self._dispatch_request_memory_extraction(
                    user_id=user_id,
                    session_id=session_id,
                    request_id=request_id,
                    round_index=round_index,
                    round_messages=round_messages,
                    explicit_memories=stored.get("explicit_memory_candidates", []),
                )

        return stored


    def _dispatch_request_memory_extraction(
        self,
        *,
        user_id: str,
        request_id: str,
        round_messages: list,
        explicit_memories: list[str] | None,
        session_id: str | None = None,
        round_index: int | None = None,
    ) -> None:
        if self.memory_eval_v2_repository is None:
            return

        payload = []
        for item in serialize_messages(round_messages):
            entry = {"message": item}
            if session_id is not None:
                entry["thread_id"] = session_id
            if round_index is not None:
                entry["round_index"] = round_index
            payload.append(entry)
        started = perf_counter()
        dispatch_success = False

        try:
            _enqueue_request_memory_extraction(
                user_id=user_id,
                request_id=request_id,
                round_messages=payload,
                explicit_memories=explicit_memories or [],
            )
            dispatch_success = True
        except Exception:
            logger.exception(
                "Request-scoped memory extraction dispatch failed: request_id=%s",
                request_id,
            )

        self.memory_eval_v2_repository.save_memory_extraction_dispatch(
            request_id=request_id,
            user_id=user_id,
            dispatch_latency_ms=int((perf_counter() - started) * 1000),
            dispatch_success=dispatch_success,
        )

    def _persist_request_audit(
        self,
        eval_ctx,
        stored: dict,
        request_id: str,
        error: Exception | None = None,
    ) -> None:
        """Persist neutral request audit data and RAG retrieval events."""
        if eval_ctx is None or self.online_eval_repository is None:
            return

        try:
            has_error = error is not None or bool(stored.get("has_error", False))
            execution_health = {
                "total_latency_ms": int(stored.get("latency_ms", 0) or 0),
                "has_error": has_error,
                "has_retry": False,
                "has_fallback": False,
                "timeout": False,
            }
            if has_error:
                execution_health["error_type"] = (
                    type(error).__name__ if error is not None else stored.get("error_type", "")
                )
                execution_health["error_message"] = (
                    str(error) if error is not None else stored.get("error_message", "")
                )[:500]
            request_payload = {
                "request_id": request_id,
                "session_id": eval_ctx.session_id,
                "user_id": eval_ctx.user_id,
                "query": eval_ctx.query[:500],
                "query_type": stored.get("query_type", eval_ctx.query_type),
                "final_answer": stored.get("final_answer", "")[:2000],
                "memory_score": None,
                "execution_health_score": None if has_error else 1.0,
                "has_retry": False,
                "has_fallback": False,
                "runtime_metrics": _runtime_metrics_snapshot(stored),
                "execution_health": execution_health,
                "agent_trace": stored.get("agent_trace", []),
                "agent_events": _serialize_agent_events(eval_ctx),
                "answer_event": _serialize_answer_event(eval_ctx),
            }
            self.online_eval_repository.save_request_run(request_payload)
            for event in eval_ctx.rag_events:
                self.online_eval_repository.save_rag_retrieval_event(
                    event.model_dump(mode="json")
                )
            logger.debug("Request audit persisted: request_id=%s", request_id)
        except Exception:
            logger.exception("Request audit failed for request_id=%s", request_id)
