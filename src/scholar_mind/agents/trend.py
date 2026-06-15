from __future__ import annotations

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
from scholar_mind.agents.state import request_value, telemetry_value
from scholar_mind.utils.text import top_keywords


class _TrendSubgraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    keyword_names: list[str]
    categories: list[str]
    date_from: str | None
    date_to: str | None
    granularity: str
    llm_usage: NotRequired[dict[str, float]]


def _build_trend_subgraph(llm, tools, trend_prompt):
    tool_llm = None
    if llm is not None and hasattr(llm, "bind_tools"):
        try:
            tool_llm = llm.bind_tools(tools)
        except Exception:
            tool_llm = None

    async def trend_agent(state: _TrendSubgraphState):
        if tool_llm is None:
            raise RuntimeError("Trend primary requires a tool-capable LLM")

        system_prompt = (
            f"{trend_prompt}\n\n"
            "## Runtime Tool Policy\n"
            "- Use analytics tools to gather publication counts, keyword trends, and representative papers.\n"
            "- Prefer statistics before narrative interpretation.\n"
            "- Stop as soon as the evidence is sufficient.\n\n"
            "## Prohibitions\n"
            "- Do not overstate causality or certainty.\n"
            "- Do not describe unsupported trend shifts.\n\n"
            "## Runtime Output\n"
            "- Reply with a concise trend summary grounded in the available statistics.\n"
            "- Keep the summary evidence-grounded and concise."
        )
        context = (
            f"Topic: {state['query']}\n"
            f"Suggested keywords: {state['keyword_names']}\n"
            f"Categories: {state.get('categories', [])}\n"
            f"Date from: {state.get('date_from') or '(none)'}\n"
            f"Date to: {state.get('date_to') or '(none)'}\n"
            f"Granularity: {state['granularity']}"
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

    builder = StateGraph(_TrendSubgraphState)
    builder.add_node("agent", trend_agent)
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


def make_trend_node(paper_repository, llm, tools, prompt_catalog):
    return make_trend_primary_node(paper_repository, llm, tools, prompt_catalog)


def make_trend_primary_node(paper_repository, llm, tools, prompt_catalog):
    trend_subgraph = _build_trend_subgraph(llm, tools, prompt_catalog.get("trend"))

    async def trend_node(state):
        started = perf_counter()
        payload = request_value(state, "payload", {})
        parent_messages = list(state.get("messages", []))
        query = request_value(state, "query", "")
        keyword_names = paper_repository.top_keywords_for_topic(
            query, limit=4
        ) or top_keywords(query, limit=4)

        subgraph_output = await trend_subgraph.ainvoke(
            {
                "messages": parent_messages,
                "query": query,
                "keyword_names": keyword_names,
                "categories": payload.get("categories", []),
                "date_from": payload.get("date_from").isoformat() if payload.get("date_from") else None,
                "date_to": payload.get("date_to").isoformat() if payload.get("date_to") else None,
                "granularity": payload.get("granularity", "quarterly"),
            }
        )

        counts = []
        keyword_stats = []
        representative = []
        subgraph_messages = new_messages_since(parent_messages, subgraph_output.get("messages", []))
        for message in all_tool_messages(subgraph_messages):
            parsed = parse_tool_result(message.content)
            if message.name == "paper_count_stats":
                counts = parsed if isinstance(parsed, list) else []
            elif message.name == "keyword_trend_stats":
                keyword_stats = parsed if isinstance(parsed, list) else []
            elif message.name == "paper_search":
                representative = parsed.get("papers", []) if isinstance(parsed, dict) else []

        latest_by_keyword = {item["keyword"]: item for item in keyword_stats}
        final_response = subgraph_messages[-1] if subgraph_messages else None
        summary = str(getattr(final_response, "content", "")).strip() or (
            f"Trend analysis covers {len(counts)} time buckets and "
            f"{len(representative)} representative papers."
        )
        trend_data = {
            "paper_count_by_period": counts,
            "emerging_keywords": [
                {"keyword": keyword, "growth_rate": round(data["growth_rate"], 4)}
                for keyword, data in latest_by_keyword.items()
            ],
            "hot_subtopics": [
                {
                    "topic": keyword,
                    "paper_count": sum(
                        1 for item in keyword_stats if item["keyword"] == keyword and item["count"] > 0
                    ),
                    "key_papers": [paper["paper_id"] for paper in representative[i * 2:(i + 1) * 2]],
                }
                for i, keyword in enumerate(keyword_names[:3])
            ],
            "representative_papers": representative,
            "summary": summary,
        }
        duration = int((perf_counter() - started) * 1000)
        result = {
            "output": {"trend_data": trend_data},
            "planning": {"active_agent": None},
            "telemetry": {
                "llm_usage": merge_usage(
                    telemetry_value(state, "llm_usage"), subgraph_output.get("llm_usage")
                ),
                "agent_trace": telemetry_value(state, "agent_trace", [])
                + [{"agent": "trend", "duration_ms": duration}],
            },
        }
        tool_trace_messages = all_tool_messages(subgraph_messages)
        if tool_trace_messages:
            result["tool_trace_messages"] = tool_trace_messages
        if final_response is not None:
            result["messages"] = [final_response]
        return result

    return trend_node


def make_trend_fallback_node(paper_repository):
    async def trend_fallback(state):
        started = perf_counter()
        payload = request_value(state, "payload", {})
        query = request_value(state, "query", "")
        keyword_names = paper_repository.top_keywords_for_topic(
            query, limit=4
        ) or top_keywords(query, limit=4)
        counts = paper_repository.paper_count_stats(
            topic=query,
            categories=payload.get("categories", []),
            date_from=payload.get("date_from"),
            date_to=payload.get("date_to"),
            granularity=payload.get("granularity", "quarterly"),
        )
        keyword_stats = paper_repository.keyword_trend_stats(
            keywords=keyword_names,
            date_from=payload.get("date_from"),
            date_to=payload.get("date_to"),
            granularity=payload.get("granularity", "quarterly"),
        )
        representative, _ = paper_repository.search_papers(
            query,
            categories=payload.get("categories", []),
            date_from=payload.get("date_from"),
            date_to=payload.get("date_to"),
            sort_by="citations",
            page=1,
            page_size=5,
        )
        latest_by_keyword = {item["keyword"]: item for item in keyword_stats}
        summary = (
            f"Trend analysis covers {len(counts)} time buckets and "
            f"{len(representative)} representative papers."
        )
        trend_data = {
            "paper_count_by_period": counts,
            "emerging_keywords": [
                {"keyword": keyword, "growth_rate": round(data["growth_rate"], 4)}
                for keyword, data in latest_by_keyword.items()
            ],
            "hot_subtopics": [
                {
                    "topic": keyword,
                    "paper_count": sum(
                        1
                        for item in keyword_stats
                        if item["keyword"] == keyword and item["count"] > 0
                    ),
                    "key_papers": [paper["paper_id"] for paper in representative[i * 2:(i + 1) * 2]],
                }
                for i, keyword in enumerate(keyword_names[:3])
            ],
            "representative_papers": representative,
            "summary": summary,
        }
        duration = int((perf_counter() - started) * 1000)
        return {
            "output": {"trend_data": trend_data},
            "planning": {"active_agent": None},
            "tool_trace_messages": [
                make_tool_trace_message("paper_count_stats", counts),
                make_tool_trace_message("keyword_trend_stats", keyword_stats),
                make_tool_trace_message(
                    "paper_search",
                    {"papers": representative, "total": len(representative)},
                ),
            ],
            "telemetry": {
                "agent_trace": telemetry_value(state, "agent_trace", [])
                + [{"agent": "trend", "duration_ms": duration}]
            },
        }

    return trend_fallback
