from __future__ import annotations

from operator import add
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

QueryTypeName = Literal[
    "qa",
    "idea_novelty",
    "trend",
    "cross_domain",
    "study_plan",
    "paper_reading",
]

AgentName = Literal[
    "researcher",
    "trend",
    "writer",
    "reviewer",
    "crossdomain",
    "hypothesis",
    "study_planner",
    "paper_reader",
]


def merge_state_dict(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class RequestState(TypedDict):
    query: NotRequired[str]
    user_id: NotRequired[str]
    session_id: NotRequired[str]
    query_type_hint: NotRequired[str | None]
    query_type: NotRequired[QueryTypeName]
    payload: NotRequired[dict[str, Any]]


class PlanningState(TypedDict):
    sub_queries: NotRequired[list[str]]
    active_agent: NotRequired[AgentName | None]


class MemoryState(TypedDict):
    explicit_candidates: NotRequired[list[str]]
    context: NotRequired[str]
    hit_count: NotRequired[int]


class RetrievalState(TypedDict):
    chunks: NotRequired[list[dict[str, Any]]]
    related_papers: NotRequired[list[dict[str, Any]]]
    rag_strategy: NotRequired[str]
    rag_latency_ms: NotRequired[int]


class CrossDomainState(TypedDict):
    intent: NotRequired[dict[str, Any]]
    resolved_source_papers: NotRequired[list[dict[str, Any]]]
    source_methodology: NotRequired[dict[str, Any]]
    candidates: NotRequired[list[dict[str, Any]]]
    transfer_analysis: NotRequired[list[dict[str, Any]]]
    hypotheses: NotRequired[list[dict[str, Any]]]
    novelty_checks: NotRequired[list[dict[str, Any]]]
    reference_metadata: NotRequired[list[dict[str, Any]]]


class PaperReadingState(TypedDict):
    active_paper_id: NotRequired[str]
    plan: NotRequired[list[dict[str, Any]]]
    completed_steps: NotRequired[list[dict[str, Any]]]
    action: NotRequired[dict[str, Any]]
    outline: NotRequired[list[dict[str, Any]]]
    cursor: NotRequired[dict[str, Any]]
    current_passage: NotRequired[dict[str, Any]]
    notes: NotRequired[dict[str, Any]]
    knowledge_links: NotRequired[list[dict[str, Any]]]


class OutputState(TypedDict):
    draft: NotRequired[str]
    final_answer: NotRequired[str]
    review_score: NotRequired[float]
    citations: NotRequired[list[dict[str, Any]]]
    report_payload: NotRequired[dict[str, Any]]
    trend_data: NotRequired[dict[str, Any]]
    study_plan: NotRequired[dict[str, Any]]


class TelemetryState(TypedDict):
    llm_usage: NotRequired[dict[str, float]]
    agent_trace: NotRequired[list[dict[str, Any]]]


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    tool_trace_messages: NotRequired[Annotated[list[BaseMessage], add_messages]]
    idea_chunk_batches: NotRequired[Annotated[list[list[dict[str, Any]]], add]]
    idea_latencies: NotRequired[Annotated[list[int], add]]
    request: Annotated[RequestState, merge_state_dict]
    planning: NotRequired[Annotated[PlanningState, merge_state_dict]]
    memory: NotRequired[Annotated[MemoryState, merge_state_dict]]
    retrieval: NotRequired[Annotated[RetrievalState, merge_state_dict]]
    cross_domain: NotRequired[Annotated[CrossDomainState, merge_state_dict]]
    reading: NotRequired[Annotated[PaperReadingState, merge_state_dict]]
    output: NotRequired[Annotated[OutputState, merge_state_dict]]
    telemetry: NotRequired[Annotated[TelemetryState, merge_state_dict]]


def request_value(state: dict[str, Any], key: str, default: Any = None) -> Any:
    aliases = {"payload": "request_payload"}
    return state.get("request", {}).get(key, state.get(aliases.get(key, key), default))


def planning_value(state: dict[str, Any], key: str, default: Any = None) -> Any:
    return state.get("planning", {}).get(key, state.get(key, default))


def memory_value(state: dict[str, Any], key: str, default: Any = None) -> Any:
    aliases = {"context": "memory_context", "hit_count": "memory_hit_count"}
    return state.get("memory", {}).get(key, state.get(aliases.get(key, key), default))


def retrieval_value(state: dict[str, Any], key: str, default: Any = None) -> Any:
    aliases = {
        "chunks": "retrieved_chunks",
        "rag_strategy": "rag_strategy",
        "rag_latency_ms": "rag_latency_ms",
    }
    return state.get("retrieval", {}).get(key, state.get(aliases.get(key, key), default))


def cross_domain_value(state: dict[str, Any], key: str, default: Any = None) -> Any:
    aliases = {
        "intent": "cross_domain_intent",
        "candidates": "cross_domain_candidates",
        "novelty_checks": "hypothesis_novelty_checks",
    }
    return state.get("cross_domain", {}).get(key, state.get(aliases.get(key, key), default))


def reading_value(state: dict[str, Any], key: str, default: Any = None) -> Any:
    aliases = {
        "active_paper_id": "active_paper_id",
        "plan": "reading_plan",
        "completed_steps": "completed_reading_steps",
        "action": "reading_action",
        "outline": "paper_outline",
        "cursor": "reading_cursor",
        "current_passage": "current_passage",
        "notes": "paper_notes",
        "knowledge_links": "knowledge_links",
    }
    return state.get("reading", {}).get(key, state.get(aliases.get(key, key), default))


def output_value(state: dict[str, Any], key: str, default: Any = None) -> Any:
    return state.get("output", {}).get(key, state.get(key, default))


def telemetry_value(state: dict[str, Any], key: str, default: Any = None) -> Any:
    return state.get("telemetry", {}).get(key, state.get(key, default))


def flatten_graph_state(state: dict[str, Any]) -> dict[str, Any]:
    flattened = dict(state)
    request = flattened.pop("request", {})
    planning = flattened.pop("planning", {})
    memory = flattened.pop("memory", {})
    retrieval = flattened.pop("retrieval", {})
    cross_domain = flattened.pop("cross_domain", {})
    reading = flattened.pop("reading", {})
    output = flattened.pop("output", {})
    telemetry = flattened.pop("telemetry", {})

    if request:
        flattened.update(
            {
                "query": request.get("query"),
                "user_id": request.get("user_id"),
                "session_id": request.get("session_id"),
                "query_type_hint": request.get("query_type_hint"),
                "query_type": request.get("query_type"),
                "request_payload": request.get("payload", {}),
            }
        )
    if planning:
        flattened.update(
            {
                "sub_queries": planning.get("sub_queries", []),
                "active_agent": planning.get("active_agent"),
            }
        )
    if memory:
        flattened.update(
            {
                "explicit_memory_candidates": memory.get("explicit_candidates", []),
                "memory_context": memory.get("context", ""),
                "memory_hit_count": memory.get("hit_count", 0),
            }
        )
    if retrieval:
        flattened.update(
            {
                "retrieved_chunks": retrieval.get("chunks", []),
                "related_papers": retrieval.get("related_papers", []),
                "rag_strategy": retrieval.get("rag_strategy"),
                "rag_latency_ms": retrieval.get("rag_latency_ms", 0),
            }
        )
    if cross_domain:
        flattened.update(
            {
                "cross_domain_intent": cross_domain.get("intent", {}),
                "resolved_source_papers": cross_domain.get("resolved_source_papers", []),
                "source_methodology": cross_domain.get("source_methodology", {}),
                "cross_domain_candidates": cross_domain.get("candidates", []),
                "transfer_analysis": cross_domain.get("transfer_analysis", []),
                "hypotheses": cross_domain.get("hypotheses", []),
                "hypothesis_novelty_checks": cross_domain.get("novelty_checks", []),
                "reference_metadata": cross_domain.get("reference_metadata", []),
            }
        )
    if reading:
        flattened.update(
            {
                "active_paper_id": reading.get("active_paper_id"),
                "reading_plan": reading.get("plan", []),
                "completed_reading_steps": reading.get("completed_steps", []),
                "reading_action": reading.get("action", {}),
                "paper_outline": reading.get("outline", []),
                "reading_cursor": reading.get("cursor", {}),
                "current_passage": reading.get("current_passage", {}),
                "paper_notes": reading.get("notes", {}),
                "knowledge_links": reading.get("knowledge_links", []),
            }
        )
    if output:
        flattened.update(output)
    if telemetry:
        flattened.update(telemetry)
    return flattened
