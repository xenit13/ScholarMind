from __future__ import annotations

from time import perf_counter
from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from scholar_mind.agents.common import (
    ainvoke_model,
    all_tool_messages,
    make_tool_trace_message,
    merge_usage,
    new_messages_since,
    route_tool_calls,
    usage_from_result,
)
from scholar_mind.agents.state import (
    cross_domain_value,
    request_value,
    retrieval_value,
    telemetry_value,
)
from scholar_mind.agents.tools.retrieval import retrieve_top10_similar_papers_payload


class _CrossDomainCandidateScore(BaseModel):
    paper_id: str
    methodology_similarity: float = 0.0
    transfer_rationale: str = ""


class _CrossDomainAssessment(BaseModel):
    source_method_summary: str = ""
    candidates: list[_CrossDomainCandidateScore] = Field(default_factory=list)


class _CrossDomainSubgraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    target_domains: list[str]
    source_papers: list[dict[str, Any]]
    source_methodology_summary: str
    rag_strategy: str
    cross_domain_candidates: NotRequired[list[dict[str, Any]]]
    rag_latency_ms: NotRequired[int]
    caller_agent: str
    llm_usage: NotRequired[dict[str, float]]


def _build_crossdomain_subgraph(llm, tools, cross_prompt):
    tool_llm = None
    if llm is not None and hasattr(llm, "bind_tools"):
        try:
            tool_llm = llm.bind_tools(tools)
        except Exception:
            tool_llm = None

    async def crossdomain_agent(state: _CrossDomainSubgraphState):
        if tool_llm is None:
            raise RuntimeError("CrossDomain primary requires a tool-capable LLM")

        agent_prompt = (
            f"{cross_prompt}\n\n"
            "## Runtime Tool Policy\n"
            "- Use `rag_top10_similar_papers` when candidate papers are missing or still insufficient.\n"
            "- Only score candidate paper ids that come from subgraph state or tool results.\n"
            "- Never invent candidate ids.\n\n"
            "## Prohibitions\n"
            "- Do not score papers outside the provided candidate set.\n"
            "- Do not substitute topical overlap for methodological similarity.\n\n"
            "## Runtime Output\n"
            "- When the available candidate papers are sufficient, stop calling tools.\n"
            "- Return JSON only.\n"
            "- Use exactly these top-level fields: `source_method_summary`, `candidates`.\n"
            "- Each item in `candidates` must contain: `paper_id`, `methodology_similarity`, `transfer_rationale`.\n\n"
            "## Example\n"
            "```json\n"
            "{\n"
            "  \"source_method_summary\": \"The source paper uses retrieval-guided planning before answer generation.\",\n"
            "  \"candidates\": [\n"
            "    {\n"
            "      \"paper_id\": \"2501.01234\",\n"
            "      \"methodology_similarity\": 0.78,\n"
            "      \"transfer_rationale\": \"Both works separate retrieval from downstream generation and rely on intermediate control decisions.\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "```"
        )
        agent_context = (
            f"User query: {state['query']}\n"
            f"Target domains: {state.get('target_domains', [])}\n"
            f"Source papers: {state.get('source_papers', [])}\n"
            f"Source methodology summary: {state.get('source_methodology_summary', '')}\n"
            f"Current candidate papers: {state.get('cross_domain_candidates', [])}\n"
        )
        subgraph_messages = state.get("messages", [])
        invoke_started = perf_counter()
        response = await ainvoke_model(
            tool_llm,
            [
                SystemMessage(content=agent_prompt),
                HumanMessage(content=agent_context),
                *subgraph_messages,
            ],
        )
        usage = usage_from_result(
            f"{agent_prompt}\n\n{agent_context}\n\n"
            + "\n".join(str(getattr(message, "content", "")) for message in subgraph_messages),
            str(getattr(response, "content", "")),
            perf_counter() - invoke_started,
            response,
        )
        result = {
            "messages": [response],
            "llm_usage": merge_usage(state.get("llm_usage"), usage),
        }
        if isinstance(response, AIMessage) and response.tool_calls:
            return result
        structured = _recover_crossdomain_assessment(response)
        if structured is None:
            raise RuntimeError("CrossDomain primary failed to produce structured similarity scoring")
        return result

    builder = StateGraph(_CrossDomainSubgraphState)
    builder.add_node("agent", crossdomain_agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        route_tool_calls,
        {
            "tools": "tools",
            "__end__": END,
        },
    )
    builder.add_edge("tools", "agent")
    return builder.compile()


def make_crossdomain_primary_node(paper_repository, llm, tools, prompt_catalog):
    cross_prompt = prompt_catalog.get("crossdomain")
    crossdomain_subgraph = _build_crossdomain_subgraph(llm, tools, cross_prompt)

    async def crossdomain_primary(state):
        started = perf_counter()
        request_payload = request_value(state, "payload", {})
        rag_strategy = request_payload.get("rag_strategy", "hybrid")
        parent_messages = list(state.get("messages", []))
        intent = cross_domain_value(state, "intent") or {
            "source_papers": [],
            "target_domains": [],
            "sources": [],
        }
        resolved_source_papers = paper_repository.resolve_paper_queries(
            intent.get("source_papers", [])
        )
        source_methodology = _build_source_methodology(
            resolved_source_papers=resolved_source_papers,
            planner_intent=intent,
        )
        source_papers_brief = [
            {
                "paper_id": item.get("paper_id"),
                "title": item.get("title"),
                "primary_category": item["categories"][0] if item.get("categories") else "",
            }
            for item in source_methodology.get("source_papers", [])
        ]
        subgraph_output = await crossdomain_subgraph.ainvoke(
            {
                "messages": parent_messages,
                "query": request_value(state, "query", ""),
                "target_domains": intent.get("target_domains", []),
                "source_papers": source_papers_brief,
                "source_methodology_summary": source_methodology.get("summary", ""),
                "cross_domain_candidates": cross_domain_value(state, "candidates", []),
                "rag_latency_ms": int(retrieval_value(state, "rag_latency_ms", 0)),
                "rag_strategy": rag_strategy,
                "caller_agent": "crossdomain",
            }
        )
        raw_candidates = subgraph_output.get("cross_domain_candidates", [])
        rag_latency_ms = int(subgraph_output.get("rag_latency_ms", 0))
        subgraph_messages = new_messages_since(parent_messages, subgraph_output.get("messages", []))
        final_response = subgraph_messages[-1] if subgraph_messages else None
        structured = _recover_crossdomain_assessment(final_response)
        if structured is None:
            raise RuntimeError("CrossDomain primary failed to produce structured similarity scoring")
        transfer_analysis = _build_transfer_analysis(
            raw_candidates=raw_candidates,
            structured=structured,
            source_methodology=source_methodology,
        )
        duration = int((perf_counter() - started) * 1000)
        result = {
            "cross_domain": {
                "intent": intent,
                "resolved_source_papers": resolved_source_papers,
                "source_methodology": source_methodology,
                "candidates": raw_candidates,
                "transfer_analysis": transfer_analysis,
            },
            "retrieval": {"rag_strategy": rag_strategy, "rag_latency_ms": rag_latency_ms},
            "planning": {"active_agent": None},
            "telemetry": {
                "llm_usage": merge_usage(
                    telemetry_value(state, "llm_usage"), subgraph_output.get("llm_usage")
                ),
                "agent_trace": telemetry_value(state, "agent_trace", [])
                + [{"agent": "crossdomain", "duration_ms": duration}],
            },
        }
        tool_trace_messages = all_tool_messages(subgraph_messages)
        if tool_trace_messages:
            result["tool_trace_messages"] = tool_trace_messages
        if final_response is not None:
            result["messages"] = [final_response]
        return result

    return crossdomain_primary


def make_crossdomain_fallback_node(paper_repository, rag_engine):
    async def crossdomain_fallback(state):
        started = perf_counter()
        request_payload = request_value(state, "payload", {})
        rag_strategy = request_payload.get("rag_strategy", "hybrid")
        intent = cross_domain_value(state, "intent") or {
            "source_papers": [],
            "target_domains": [],
            "sources": [],
        }
        resolved_source_papers = paper_repository.resolve_paper_queries(
            intent.get("source_papers", [])
        )
        source_methodology = _build_source_methodology(
            resolved_source_papers=resolved_source_papers,
            planner_intent=intent,
        )
        raw_candidates = cross_domain_value(state, "candidates", [])
        rag_latency_ms = int(retrieval_value(state, "rag_latency_ms", 0))
        tool_trace_messages = []
        if not raw_candidates:
            try:
                payload = retrieve_top10_similar_papers_payload(
                    rag_engine,
                    **_build_similarity_tool_args(
                        source_methodology=source_methodology,
                        target_domains=intent.get("target_domains", []),
                        strategy=rag_strategy,
                    ),
                )
                raw_candidates = list(payload.get("items", []))
                rag_latency_ms = int(payload.get("latency_ms", 0))
                tool_trace_messages.append(
                    make_tool_trace_message("rag_top10_similar_papers", payload)
                )
            except Exception:
                raw_candidates = []
                rag_latency_ms = 0
        transfer_analysis = _build_transfer_analysis(
            raw_candidates=raw_candidates,
            structured=None,
            source_methodology=source_methodology,
        )
        duration = int((perf_counter() - started) * 1000)
        result = {
            "messages": [AIMessage(content="Cross-domain candidate discovery complete")],
            "cross_domain": {
                "intent": intent,
                "resolved_source_papers": resolved_source_papers,
                "source_methodology": source_methodology,
                "candidates": raw_candidates,
                "transfer_analysis": transfer_analysis,
            },
            "retrieval": {"rag_strategy": rag_strategy, "rag_latency_ms": rag_latency_ms},
            "planning": {"active_agent": None},
            "telemetry": {
                "agent_trace": telemetry_value(state, "agent_trace", [])
                + [{"agent": "crossdomain", "duration_ms": duration}]
            },
        }
        if tool_trace_messages:
            result["tool_trace_messages"] = tool_trace_messages
        return result

    return crossdomain_fallback


def make_crossdomain_node(paper_repository, llm, tools, prompt_catalog):
    return make_crossdomain_primary_node(paper_repository, llm, tools, prompt_catalog)


def _build_source_methodology(
    *, resolved_source_papers: list[dict[str, Any]], planner_intent: dict[str, Any]
) -> dict[str, Any]:
    real_sources = [item for item in resolved_source_papers if item.get("resolved")]
    target_domains = [item for item in planner_intent.get("target_domains", []) if item]
    source_papers = real_sources or [
        {
            "requested_paper": item,
            "resolved": False,
            "paper_id": None,
            "title": item,
            "categories": [],
            "summary": item,
            "methodology_summary": item,
            "sources": [
                {
                    "kind": "planner_llm",
                    "label": item,
                    "metadata": {"resolution": "planner_only"},
                }
            ],
        }
        for item in planner_intent.get("source_papers", [])
        if item
    ]
    methodology_parts = [
        item.get("methodology_summary") or item.get("summary") or item.get("title", "")
        for item in source_papers
        if item.get("methodology_summary") or item.get("summary") or item.get("title")
    ]
    summary = "\n".join(methodology_parts).strip() or "No source methodology found."
    primary_categories = [
        item["categories"][0]
        for item in real_sources
        if item.get("categories")
    ]
    sources = []
    for item in source_papers:
        sources.extend(item.get("sources", []))
    if not sources:
        sources = planner_intent.get("sources", [])
    return {
        "summary": summary,
        "source_papers": source_papers,
        "source_primary_categories": primary_categories,
        "requested_target_domains": target_domains,
        "sources": sources,
    }


def _build_similarity_tool_args(
    *,
    source_methodology: dict[str, Any],
    target_domains: list[str],
    strategy: str,
) -> dict[str, Any]:
    resolved_paper_ids = [
        item["paper_id"]
        for item in source_methodology.get("source_papers", [])
        if item.get("paper_id")
    ]
    return {
        "source_summary": source_methodology.get("summary", ""),
        "target_domains": target_domains,
        "exclude_paper_ids": resolved_paper_ids,
        "exclude_primary_categories": source_methodology.get(
            "source_primary_categories", []
        ),
        "strategy": strategy,
    }


def _build_transfer_analysis(
    *,
    raw_candidates: list[dict[str, Any]],
    structured: _CrossDomainAssessment | None,
    source_methodology: dict[str, Any],
) -> list[dict[str, Any]]:
    score_map = {
        item.paper_id: item
        for item in (structured.candidates if structured else [])
    }
    max_retrieval_score = max(
        (float(item.get("retrieval_score", 0.0)) for item in raw_candidates),
        default=0.0,
    )
    transfer_analysis: list[dict[str, Any]] = []
    for candidate in raw_candidates:
        scored = score_map.get(candidate["paper_id"])
        if scored is not None:
            similarity = _clamp_similarity(scored.methodology_similarity)
            rationale = scored.transfer_rationale.strip()
        else:
            similarity = _normalized_retrieval_similarity(
                float(candidate.get("retrieval_score", 0.0)), max_retrieval_score
            )
            rationale = _fallback_transfer_rationale(source_methodology, candidate)
        transfer_analysis.append(
            {
                "paper_id": candidate["paper_id"],
                "title": candidate["title"],
                "categories": candidate.get("categories", []),
                "summary": candidate.get("summary", ""),
                "methodology_similarity": similarity,
                "transfer_rationale": rationale,
                "sources": candidate.get("sources", []),
            }
        )
    return transfer_analysis


def _fallback_transfer_rationale(
    source_methodology: dict[str, Any], candidate: dict[str, Any]
) -> str:
    source_summary = source_methodology.get("summary", "").strip()
    source_summary = source_summary[:120] if source_summary else "源方法"
    candidate_summary = candidate.get("summary", "").strip()
    candidate_summary = candidate_summary[:120] if candidate_summary else candidate["title"]
    return (
        "源方法与候选论文在问题结构上接近，"
        f"可尝试把“{source_summary}”迁移到“{candidate_summary}”。"
    )


def _normalized_retrieval_similarity(score: float, max_score: float) -> float:
    if max_score <= 0:
        return 0.0
    return round(max(0.0, min(score / max_score, 1.0)), 4)


def _clamp_similarity(score: float) -> float:
    return round(max(0.0, min(float(score), 1.0)), 4)


def _recover_crossdomain_assessment(raw) -> _CrossDomainAssessment | None:
    from scholar_mind.agents.common import extract_json_candidate, raw_output_text

    payload = extract_json_candidate(raw_output_text(raw).strip())
    if not isinstance(payload, dict):
        return None
    candidates = payload.get("candidates") or payload.get("papers") or []
    adapted = {
        "source_method_summary": payload.get("source_method_summary", ""),
        "candidates": candidates,
    }
    try:
        return _CrossDomainAssessment.model_validate(adapted)
    except Exception:
        return None
