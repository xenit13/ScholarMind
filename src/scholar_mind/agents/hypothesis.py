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
    merge_usage,
    new_messages_since,
    parse_tool_result,
    route_tool_calls,
    usage_from_result,
)
from scholar_mind.agents.state import cross_domain_value, request_value, telemetry_value


class _HypothesisDraft(BaseModel):
    hypothesis: str
    candidate_paper_ids: list[str] = Field(default_factory=list)
    novelty_is_novel: bool = True
    novelty_confidence: float = 0.0
    novelty_rationale: str = ""
    target_domain: str = ""
    core_intervention: str = ""
    datasets_or_tasks: list[str] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    ablations: list[str] = Field(default_factory=list)


class _HypothesisOutput(BaseModel):
    hypotheses: list[_HypothesisDraft] = Field(default_factory=list)


class _HypothesisSubgraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    source_methodology: dict[str, Any]
    transfer_analysis: list[dict[str, Any]]
    max_hypotheses: int
    llm_usage: NotRequired[dict[str, float]]


def _build_hypothesis_subgraph(llm, tools, prompt):
    tool_llm = None
    if llm is not None and hasattr(llm, "bind_tools"):
        try:
            tool_llm = llm.bind_tools(tools)
        except Exception:
            tool_llm = None

    async def hypothesis_agent(state: _HypothesisSubgraphState):
        if tool_llm is None:
            raise RuntimeError("Hypothesis primary requires a tool-capable LLM")

        system_prompt = (
            f"{prompt}\n\n"
            "## Runtime Tool Policy\n"
            "- Use `paper_methodology_lookup` only when candidate methodology details are needed.\n"
            "- Stop calling tools once the available candidate evidence is sufficient.\n\n"
            "## Runtime Generation Rules\n"
            "- Each hypothesis must combine the source method with multiple candidate "
            "papers when useful.\n"
            "- Judge novelty conservatively.\n"
            "- Propose a compact experiment design.\n\n"
            "## Prohibitions\n"
            "- Do not output unsupported hypotheses disconnected from the supplied evidence.\n"
            "- Do not overstate novelty when evidence is weak.\n\n"
            "## Runtime Output\n"
            "- Return JSON only.\n"
            "- Follow the schema and example defined in the base hypothesis prompt."
        )
        context = (
            f"User query: {state['query']}\n"
            f"Source methodology: {state['source_methodology']}\n"
            f"Filtered candidate papers: {state['transfer_analysis']}\n"
            f"Max hypotheses: {state['max_hypotheses']}\n"
        )
        subgraph_messages = state.get("messages", [])
        invoke_started = perf_counter()
        response = await ainvoke_model(
            tool_llm,
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context),
                *subgraph_messages,
            ],
        )
        usage = usage_from_result(
            f"{system_prompt}\n\n{context}\n\n"
            + "\n".join(str(getattr(message, "content", "")) for message in subgraph_messages),
            str(getattr(response, "content", "")),
            perf_counter() - invoke_started,
            response,
        )
        return {
            "messages": [response],
            "llm_usage": merge_usage(state.get("llm_usage"), usage),
        }

    builder = StateGraph(_HypothesisSubgraphState)
    builder.add_node("agent", hypothesis_agent)
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


def make_hypothesis_node(llm, tools, prompt_catalog):
    return make_hypothesis_primary_node(llm, tools, prompt_catalog)


def make_hypothesis_primary_node(llm, tools, prompt_catalog):
    hypothesis_subgraph = _build_hypothesis_subgraph(llm, tools, prompt_catalog.get("hypothesis"))

    async def hypothesis_node(state):
        started = perf_counter()
        request = request_value(state, "payload", {})
        parent_messages = list(state.get("messages", []))
        source_methodology = cross_domain_value(state, "source_methodology", {})
        transfer_analysis = sorted(
            [
                item
                for item in cross_domain_value(state, "transfer_analysis", [])
                if float(item.get("methodology_similarity", 0.0)) >= 0.5
            ],
            key=lambda item: float(item.get("methodology_similarity", 0.0)),
            reverse=True,
        )
        max_hypotheses = max(1, int(request.get("max_hypotheses", 3)))
        subgraph_output = await hypothesis_subgraph.ainvoke(
            {
                "messages": parent_messages,
                "query": request_value(state, "query", ""),
                "source_methodology": source_methodology,
                "transfer_analysis": transfer_analysis,
                "max_hypotheses": max_hypotheses,
            }
        )
        subgraph_messages = new_messages_since(parent_messages, subgraph_output.get("messages", []))
        final_response = subgraph_messages[-1] if subgraph_messages else None
        methodology_details = _collect_methodology_details_from_messages(subgraph_messages)
        structured = _recover_hypothesis_output(final_response)
        if structured is None:
            raise RuntimeError("Hypothesis primary failed to produce structured output")
        if not structured.hypotheses[:max_hypotheses]:
            raise RuntimeError("Hypothesis primary produced no hypothesis drafts")
        hypotheses = _build_hypotheses(
            source_methodology=source_methodology,
            transfer_analysis=transfer_analysis,
            methodology_details=methodology_details,
            structured=structured,
            max_hypotheses=max_hypotheses,
        )
        duration = int((perf_counter() - started) * 1000)
        result = {
            "cross_domain": {
                "hypotheses": hypotheses,
                "novelty_checks": [item["novelty_check"] for item in hypotheses],
            },
            "planning": {"active_agent": None},
            "telemetry": {
                "llm_usage": merge_usage(
                    telemetry_value(state, "llm_usage"), subgraph_output.get("llm_usage")
                ),
                "agent_trace": telemetry_value(state, "agent_trace", [])
                + [{"agent": "hypothesis", "duration_ms": duration}],
            },
        }
        tool_trace_messages = all_tool_messages(subgraph_messages)
        if tool_trace_messages:
            result["tool_trace_messages"] = tool_trace_messages
        if final_response is not None:
            result["messages"] = [final_response]
        return result

    return hypothesis_node


def make_hypothesis_fallback_node(prompt_catalog):
    async def hypothesis_fallback(state):
        started = perf_counter()
        request = request_value(state, "payload", {})
        source_methodology = cross_domain_value(state, "source_methodology", {})
        transfer_analysis = sorted(
            [
                item
                for item in cross_domain_value(state, "transfer_analysis", [])
                if float(item.get("methodology_similarity", 0.0)) >= 0.5
            ],
            key=lambda item: float(item.get("methodology_similarity", 0.0)),
            reverse=True,
        )
        methodology_details = _collect_methodology_details_from_messages(state.get("messages", []))
        max_hypotheses = max(1, int(request.get("max_hypotheses", 3)))
        detail_map = {item["paper_id"]: item for item in methodology_details if item.get("paper_id")}
        candidates = transfer_analysis[: max(2, max_hypotheses)]
        fallback_draft = (
            _fallback_hypothesis(source_methodology, candidates, detail_map)
            if candidates
            else None
        )
        hypotheses = _build_hypotheses(
            source_methodology=source_methodology,
            transfer_analysis=transfer_analysis,
            methodology_details=methodology_details,
            structured=(
                _HypothesisOutput(hypotheses=[fallback_draft])
                if fallback_draft is not None
                else None
            ),
            max_hypotheses=max_hypotheses,
        )
        duration = int((perf_counter() - started) * 1000)
        return {
            "messages": [AIMessage(content="Hypothesis generation complete")],
            "cross_domain": {
                "hypotheses": hypotheses,
                "novelty_checks": [item["novelty_check"] for item in hypotheses],
            },
            "planning": {"active_agent": None},
            "telemetry": {
                "agent_trace": telemetry_value(state, "agent_trace", [])
                + [{"agent": "hypothesis", "duration_ms": duration}]
            },
        }

    return hypothesis_fallback


def _collect_methodology_details_from_messages(messages) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for message in all_tool_messages(messages):
        if message.name != "paper_methodology_lookup":
            continue
        payload = parse_tool_result(message.content)
        if isinstance(payload, dict) and payload:
            payloads.append(payload)
    return payloads


def _build_hypotheses(
    *,
    source_methodology: dict[str, Any],
    transfer_analysis: list[dict[str, Any]],
    methodology_details: list[dict[str, Any]],
    structured: _HypothesisOutput | None,
    max_hypotheses: int,
) -> list[dict[str, Any]]:
    detail_map = {item["paper_id"]: item for item in methodology_details if item.get("paper_id")}
    candidates = transfer_analysis[: max(2, max_hypotheses)]
    if not candidates:
        return []
    drafts = structured.hypotheses[:max_hypotheses] if structured else []
    if not drafts:
        return []

    hypotheses: list[dict[str, Any]] = []
    for draft in drafts[:max_hypotheses]:
        selected_ids = draft.candidate_paper_ids or [item["paper_id"] for item in candidates[:2]]
        selected_candidates = [
            item for item in candidates if item["paper_id"] in set(selected_ids)
        ] or candidates[:2]
        evidence = [
            {
                "paper_id": source.get("paper_id") or "planner-source",
                "title": source.get("title", source.get("requested_paper", "source")),
                "claim": source.get("methodology_summary") or source.get("summary", ""),
                "role": "source",
                "sources": source.get("sources", []),
            }
            for source in source_methodology.get("source_papers", [])
        ]
        for candidate in selected_candidates:
            evidence.append(
                {
                    "paper_id": candidate["paper_id"],
                    "title": candidate["title"],
                    "claim": candidate.get("summary", ""),
                    "role": "candidate",
                    "sources": candidate.get("sources", []),
                }
            )
            detail = detail_map.get(candidate["paper_id"])
            if detail:
                evidence.append(
                    {
                        "paper_id": detail["paper_id"],
                        "title": detail["title"],
                        "claim": detail.get("methodology_summary", ""),
                        "role": "methodology_lookup",
                        "sources": detail.get("sources", []),
                    }
                )
        hypothesis_sources = []
        for item in evidence:
            hypothesis_sources.extend(item.get("sources", []))
        hypothesis_sources.append(
            {
                "kind": "llm_hypothesis",
                "label": draft.hypothesis,
                "metadata": {"candidate_paper_ids": selected_ids},
            }
        )
        hypotheses.append(
            {
                "hypothesis": draft.hypothesis,
                "candidate_paper_ids": selected_ids,
                "supporting_evidence": evidence,
                "novelty_check": {
                    "is_novel": bool(draft.novelty_is_novel),
                    "confidence": _clamp_confidence(draft.novelty_confidence),
                    "rationale": draft.novelty_rationale,
                    "sources": hypothesis_sources,
                },
                "experiment_design": {
                    "target_domain": draft.target_domain
                    or _infer_target_domain(selected_candidates, source_methodology),
                    "core_intervention": draft.core_intervention,
                    "datasets_or_tasks": draft.datasets_or_tasks,
                    "baselines": draft.baselines,
                    "metrics": draft.metrics,
                    "ablations": draft.ablations,
                },
                "sources": hypothesis_sources,
            }
        )
    return hypotheses


def _fallback_hypothesis(
    source_methodology: dict[str, Any],
    candidates: list[dict[str, Any]],
    detail_map: dict[str, dict[str, Any]],
) -> _HypothesisDraft:
    primary = candidates[0]
    secondary = candidates[1] if len(candidates) > 1 else primary
    primary_detail = detail_map.get(primary["paper_id"], {})
    source_summary = source_methodology.get("summary", "源方法")
    target_domain = _infer_target_domain(candidates[:2], source_methodology)
    return _HypothesisDraft(
        hypothesis=(
            f"将源方法中的关键规划/迁移机制，与《{primary['title']}》和《{secondary['title']}》"
            f"体现的问题结构结合，可能在 {target_domain or '新领域'} 中形成更稳定的跨领域方法。"
        ),
        candidate_paper_ids=[primary["paper_id"], secondary["paper_id"]],
        novelty_is_novel=True,
        novelty_confidence=0.62,
        novelty_rationale=(
            f"当前候选论文与源方法存在结构相似性，但尚未看到完全相同的组合式迁移方案。"
            f"源依据：{source_summary[:100]}。"
            "候选方法依据："
            f"{primary_detail.get('methodology_summary', primary.get('summary', ''))[:100]}。"
        ),
        target_domain=target_domain,
        core_intervention="将源论文中的规划/迁移机制嵌入目标领域流程",
        datasets_or_tasks=[target_domain] if target_domain else [],
        baselines=[primary["title"], secondary["title"], "source-only adaptation"],
        metrics=["task success", "grounding quality", "stability"],
        ablations=["remove planning component", "remove transfer constraint"],
    )


def _infer_target_domain(
    selected_candidates: list[dict[str, Any]],
    source_methodology: dict[str, Any],
) -> str:
    requested = source_methodology.get("requested_target_domains", [])
    if requested:
        return requested[0]
    for candidate in selected_candidates:
        categories = candidate.get("categories", [])
        if categories:
            return categories[0]
    return ""


def _clamp_confidence(value: float) -> float:
    return round(max(0.0, min(float(value), 1.0)), 4)


def _recover_hypothesis_output(raw) -> _HypothesisOutput | None:
    from scholar_mind.agents.common import extract_json_candidate, raw_output_text

    payload = extract_json_candidate(raw_output_text(raw).strip())
    if not isinstance(payload, dict):
        return None
    hypotheses = payload.get("hypotheses") or payload.get("items") or []
    try:
        return _HypothesisOutput.model_validate({"hypotheses": hypotheses})
    except Exception:
        return None
