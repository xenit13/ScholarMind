from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from scholar_mind.agents.common import (
    ainvoke_model,
    all_tool_messages,
    make_tool_trace_message,
    merge_usage,
    new_messages_since,
    parse_tool_result,
    route_tool_calls,
    usage_from_result,
)
from scholar_mind.agents.novelty import build_evidence_cards, build_novelty_payload
from scholar_mind.agents.state import (
    cross_domain_value,
    output_value,
    request_value,
    retrieval_value,
    telemetry_value,
)
from scholar_mind.models.domain import QueryType
from scholar_mind.rag.top_k import IDEA_EVIDENCE_TOP_K


class _WriterSubgraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    system_prompt: str
    context: str
    llm_usage: NotRequired[dict[str, float]]


def _build_writer_subgraph(llm, tools):
    tool_llm = None
    if llm is not None and hasattr(llm, "bind_tools"):
        try:
            tool_llm = llm.bind_tools(tools)
        except Exception:
            tool_llm = None

    async def writer_agent(state: _WriterSubgraphState):
        if tool_llm is None:
            raise RuntimeError("Writer primary requires a tool-capable LLM")

        subgraph_messages = state.get("messages", [])
        invoke_started = perf_counter()
        response = await ainvoke_model(
            tool_llm,
            [
                SystemMessage(content=state["system_prompt"]),
                HumanMessage(content=state["context"]),
                *subgraph_messages,
            ],
        )
        usage = usage_from_result(
            f"{state['system_prompt']}\n\n{state['context']}\n\n"
            + "\n".join(str(getattr(message, "content", "")) for message in subgraph_messages),
            str(getattr(response, "content", "")),
            perf_counter() - invoke_started,
            response,
        )
        return {
            "messages": [response],
            "llm_usage": merge_usage(state.get("llm_usage"), usage),
        }

    builder = StateGraph(_WriterSubgraphState)
    builder.add_node("agent", writer_agent)
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


def make_writer_node(llm, tools, prompt_catalog):
    return make_writer_primary_node(llm, tools, prompt_catalog)


def make_writer_primary_node(llm, tools, prompt_catalog):
    writer_prompt = prompt_catalog.get("writer")
    writer_subgraph = _build_writer_subgraph(llm, tools)

    async def writer_node(state):
        started = perf_counter()
        query = request_value(state, "query", "")
        query_type = QueryType(request_value(state, "query_type"))
        parent_messages = list(state.get("messages", []))

        if query_type in {QueryType.IDEA_NOVELTY, QueryType.CROSS_DOMAIN}:
            if query_type == QueryType.IDEA_NOVELTY:
                evidence_cards = build_evidence_cards(query, retrieval_value(state, "chunks", []))
                paper_ids = [
                    item["paper_id"]
                    for item in evidence_cards[:IDEA_EVIDENCE_TOP_K]
                    if item.get("paper_id")
                ]
                system_prompt = (
                    f"{writer_prompt}\n"
                    "\n## Runtime Role\n"
                    "You are drafting an idea novelty report.\n\n"
                    "## Runtime Tool Policy\n"
                    "- Use `citation_lookup` only for paper ids already present in the evidence cards.\n"
                    "- Do not expand the reference set speculatively.\n\n"
                    "## Prohibitions\n"
                    "- Do not introduce unsupported claims or references.\n"
                    "- Do not add generic filler.\n\n"
                    "## Runtime Output\n"
                    "- When ToolMessages are sufficient, stop calling tools.\n"
                    "- Return only the novelty summary in plain prose."
                )
                context = (
                    f"Idea: {query}\n"
                    f"Evidence cards: {evidence_cards[:IDEA_EVIDENCE_TOP_K]}\n"
                    f"Candidate paper ids: {paper_ids}"
                )
            else:
                paper_ids = _crossdomain_reference_ids(state)
                system_prompt = (
                    f"{writer_prompt}\n"
                    "\n## Runtime Role\n"
                    "Write a concise cross-domain transfer report in Chinese.\n\n"
                    "## Runtime Tool Policy\n"
                    "- Use `citation_lookup` only for paper ids already present in the report payload.\n"
                    "- Do not introduce new references outside the payload.\n\n"
                    "## Prohibitions\n"
                    "- Do not introduce unsupported claims or references.\n"
                    "- Do not drift away from concise Chinese report prose.\n\n"
                    "## Runtime Output\n"
                    "- When ToolMessages are sufficient, stop calling tools.\n"
                    "- Return only the final report prose."
                )
                context = (
                    f"User query: {query}\n"
                    f"Planner intent: {cross_domain_value(state, 'intent', {})}\n"
                    f"Source methodology: {cross_domain_value(state, 'source_methodology', {})}\n"
                    f"Candidate papers: {cross_domain_value(state, 'transfer_analysis', [])}\n"
                    f"Hypotheses: {cross_domain_value(state, 'hypotheses', [])}\n"
                    f"Reference paper ids: {paper_ids}"
                )

            subgraph_output = await writer_subgraph.ainvoke(
                {
                    "messages": parent_messages,
                    "system_prompt": system_prompt,
                    "context": context,
                }
            )
            subgraph_messages = new_messages_since(parent_messages, subgraph_output.get("messages", []))
            final_response = subgraph_messages[-1] if subgraph_messages else None
            citation_payload = _citation_payload_from_messages(subgraph_messages)
            evidence_cards = build_evidence_cards(query, retrieval_value(state, "chunks", []))
            duration = int((perf_counter() - started) * 1000)
            if query_type == QueryType.IDEA_NOVELTY:
                citation_meta = {item["paper_id"]: item for item in citation_payload}
                report = build_novelty_payload(query, evidence_cards)
                report["references"] = [
                    {
                        "paper_id": item["paper_id"],
                        "title": citation_meta.get(item["paper_id"], {}).get(
                            "title", item["title"]
                        ),
                        "year": citation_meta.get(item["paper_id"], {}).get(
                            "year", item.get("year")
                        ),
                    }
                    for item in evidence_cards[:IDEA_EVIDENCE_TOP_K]
                ]
                result = {
                    "output": {
                        "draft": str(getattr(final_response, "content", "")).strip()
                        or report["novelty_report"]["summary"],
                        "report_payload": report,
                    },
                    "planning": {"active_agent": None},
                    "telemetry": {
                        "llm_usage": merge_usage(
                            telemetry_value(state, "llm_usage"), subgraph_output.get("llm_usage")
                        ),
                        "agent_trace": telemetry_value(state, "agent_trace", [])
                        + [{"agent": "writer", "duration_ms": duration}],
                    },
                }
            else:
                report = {
                    "planner_intent": cross_domain_value(state, "intent", {}),
                    "source_methodology": cross_domain_value(state, "source_methodology", {}),
                    "candidate_papers": cross_domain_value(state, "transfer_analysis", []),
                    "hypotheses": cross_domain_value(state, "hypotheses", []),
                    "references": citation_payload,
                }
                result = {
                    "output": {
                        "draft": str(getattr(final_response, "content", "")).strip()
                        or _fallback_crossdomain_draft(report),
                        "report_payload": report,
                    },
                    "cross_domain": {"reference_metadata": citation_payload},
                    "planning": {"active_agent": None},
                    "telemetry": {
                        "llm_usage": merge_usage(
                            telemetry_value(state, "llm_usage"), subgraph_output.get("llm_usage")
                        ),
                        "agent_trace": telemetry_value(state, "agent_trace", [])
                        + [{"agent": "writer", "duration_ms": duration}],
                    },
                }
            tool_trace_messages = all_tool_messages(subgraph_messages)
            if tool_trace_messages:
                result["tool_trace_messages"] = tool_trace_messages
            if final_response is not None:
                result["messages"] = [final_response]
            return result

        if query_type == QueryType.TREND:
            report = output_value(state, "trend_data", {})
            draft = report.get("summary") or output_value(state, "draft", "") or "Trend analysis completed."
        elif query_type == QueryType.STUDY_PLAN:
            report = output_value(state, "study_plan", output_value(state, "report_payload", {}))
            draft = output_value(state, "draft", "") or report.get(
                "goal_summary", "Study plan generated."
            )
        elif query_type == QueryType.PAPER_READING:
            report = output_value(state, "report_payload", {})
            draft = output_value(state, "draft", "") or report.get("explanation", {}).get(
                "plain_language", "Paper reading step generated."
            )
        else:
            report = output_value(state, "report_payload", {})
            draft = output_value(state, "draft", "")

        duration = int((perf_counter() - started) * 1000)
        return {
            "output": {"draft": draft, "report_payload": report},
            "telemetry": {
                "llm_usage": merge_usage(telemetry_value(state, "llm_usage")),
                "agent_trace": telemetry_value(state, "agent_trace", [])
                + [{"agent": "writer", "duration_ms": duration}],
            },
        }

    return writer_node


def make_writer_fallback_node(paper_repository, prompt_catalog):
    async def writer_fallback(state):
        started = perf_counter()
        query = request_value(state, "query", "")
        query_type = QueryType(request_value(state, "query_type"))
        if query_type == QueryType.IDEA_NOVELTY:
            evidence_cards = build_evidence_cards(query, retrieval_value(state, "chunks", []))
            citation_payload = await asyncio.to_thread(
                _citation_payload_for_ids,
                paper_repository,
                [
                    item["paper_id"]
                    for item in evidence_cards[:IDEA_EVIDENCE_TOP_K]
                    if item.get("paper_id")
                ],
            )
            citation_meta = {item["paper_id"]: item for item in citation_payload}
            report = build_novelty_payload(query, evidence_cards)
            report["references"] = [
                {
                    "paper_id": item["paper_id"],
                    "title": citation_meta.get(item["paper_id"], {}).get("title", item["title"]),
                    "year": citation_meta.get(item["paper_id"], {}).get("year", item.get("year")),
                }
                for item in evidence_cards[:IDEA_EVIDENCE_TOP_K]
            ]
            duration = int((perf_counter() - started) * 1000)
            return {
                "output": {
                    "draft": report["novelty_report"]["summary"],
                    "report_payload": report,
                },
                "planning": {"active_agent": None},
                "tool_trace_messages": [
                    make_tool_trace_message("citation_lookup", citation_payload)
                ],
                "telemetry": {
                    "agent_trace": telemetry_value(state, "agent_trace", [])
                    + [{"agent": "writer", "duration_ms": duration}]
                },
            }

        if query_type == QueryType.CROSS_DOMAIN:
            citation_payload = await asyncio.to_thread(
                _citation_payload_for_ids,
                paper_repository,
                _crossdomain_reference_ids(state),
            )
            report = {
                "planner_intent": cross_domain_value(state, "intent", {}),
                "source_methodology": cross_domain_value(state, "source_methodology", {}),
                "candidate_papers": cross_domain_value(state, "transfer_analysis", []),
                "hypotheses": cross_domain_value(state, "hypotheses", []),
                "references": citation_payload,
            }
            duration = int((perf_counter() - started) * 1000)
            return {
                "output": {
                    "draft": _fallback_crossdomain_draft(report),
                    "report_payload": report,
                },
                "cross_domain": {"reference_metadata": citation_payload},
                "planning": {"active_agent": None},
                "tool_trace_messages": [
                    make_tool_trace_message("citation_lookup", citation_payload)
                ],
                "telemetry": {
                    "agent_trace": telemetry_value(state, "agent_trace", [])
                    + [{"agent": "writer", "duration_ms": duration}]
                },
            }

        report = (
            output_value(state, "trend_data")
            or output_value(state, "study_plan")
            or output_value(state, "report_payload", {})
        )
        draft = output_value(state, "draft", "") or report.get("summary") or report.get("goal_summary", "")
        duration = int((perf_counter() - started) * 1000)
        return {
            "output": {"draft": draft, "report_payload": report},
            "planning": {"active_agent": None},
            "telemetry": {
                "agent_trace": telemetry_value(state, "agent_trace", [])
                + [{"agent": "writer", "duration_ms": duration}]
            },
        }

    return writer_fallback


def _crossdomain_reference_ids(state: dict) -> list[str]:
    paper_ids: list[str] = []
    seen: set[str] = set()
    for source in cross_domain_value(state, "source_methodology", {}).get("source_papers", []):
        paper_id = source.get("paper_id")
        if paper_id and paper_id not in seen:
            paper_ids.append(paper_id)
            seen.add(paper_id)
    for candidate in cross_domain_value(state, "transfer_analysis", []):
        paper_id = candidate.get("paper_id")
        if paper_id and paper_id not in seen:
            paper_ids.append(paper_id)
            seen.add(paper_id)
    for hypothesis in cross_domain_value(state, "hypotheses", []):
        for paper_id in hypothesis.get("candidate_paper_ids", []):
            if paper_id and paper_id not in seen:
                paper_ids.append(paper_id)
                seen.add(paper_id)
    return paper_ids


def _citation_payload_from_messages(messages: list) -> list[dict]:
    payload: list[dict] = []
    for message in all_tool_messages(messages):
        if message.name != "citation_lookup":
            continue
        parsed = parse_tool_result(message.content)
        if isinstance(parsed, list):
            payload = parsed
    return payload


def _citation_payload_for_ids(paper_repository, paper_ids: list[str]) -> list[dict]:
    payload = []
    seen: set[str] = set()
    for paper_id in paper_ids:
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        paper = paper_repository.get_paper(paper_id)
        if paper is None:
            continue
        author_text = ", ".join(paper.authors[:4])
        if len(paper.authors) > 4:
            author_text = f"{author_text}, et al."
        payload.append(
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "authors": paper.authors,
                "year": paper.publish_date.year,
                "publish_date": paper.publish_date.isoformat(),
                "categories": list(paper.categories),
                "citation_count": paper.citation_count,
                "formatted_reference": (
                    f"{author_text} ({paper.publish_date.year}). "
                    f"{paper.title}. {paper.paper_id}."
                ),
                "sources": [
                    {
                        "kind": "citation_lookup",
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                    }
                ],
            }
        )
    return payload


def _fallback_crossdomain_draft(report: dict) -> str:
    source_titles = [
        item.get("title", item.get("requested_paper", "source"))
        for item in report.get("source_methodology", {}).get("source_papers", [])
    ]
    candidate_count = len(report.get("candidate_papers", []))
    hypothesis_count = len(report.get("hypotheses", []))
    source_text = "、".join(source_titles) if source_titles else "源论文"
    return (
        f"本报告基于 {source_text} 的方法论线索，筛出了 {candidate_count} 篇跨领域候选论文，"
        f"并进一步生成 {hypothesis_count} 条研究假设与实验建议。"
    )
