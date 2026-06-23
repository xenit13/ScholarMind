from __future__ import annotations

from time import perf_counter
from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from scholar_mind.agents.common import (
    ainvoke_model,
    all_tool_messages,
    make_tool_trace_message,
    merge_usage,
    new_messages_since,
    parse_tool_result,
    rerank_retrieved_chunks,
    route_tool_calls,
    usage_from_result,
)
from scholar_mind.agents.state import memory_value, request_value, telemetry_value
from scholar_mind.eval.context import get_eval_context
from scholar_mind.models.domain import QueryType
from scholar_mind.models.eval_models import RagRetrievalEventV2
from scholar_mind.rag.top_k import FINAL_CITATION_TOP_K
from scholar_mind.utils.text import truncate


class _ResearchSubgraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    query_type: str
    retrieval_query: str
    rag_strategy: str
    top_k: int
    paper_ids: list[str]
    categories: list[str]
    date_from: str | None
    date_to: str | None
    source_paper_id: str | None
    source_paper_abstract: str
    memory_context: str
    caller_agent: str
    llm_usage: NotRequired[dict[str, float]]


def _retrieve_chunks(
    rag_engine,
    payload,
    query: str,
    *,
    caller_agent: str | None = None,
) -> tuple[list[dict], int]:
    strategy = payload.get("rag_strategy", "hybrid")
    top_k = payload.get("max_papers", payload.get("top_k", FINAL_CITATION_TOP_K))
    filters = {
        "paper_ids": payload.get("paper_ids", []),
        "categories": payload.get("categories", []),
        "date_from": payload.get("date_from"),
        "date_to": payload.get("date_to"),
    }
    chunks, latency = rag_engine.retrieve_sync(
        query=query,
        strategy=strategy,
        top_k=top_k,
        filters=filters,
    )
    chunk_dicts = [chunk.model_dump(mode="json") for chunk in chunks]
    eval_ctx = get_eval_context()
    if eval_ctx is not None:
        eval_ctx.rag_events.append(
            RagRetrievalEventV2(
                request_id=eval_ctx.request_id,
                query=query,
                strategy=strategy,
                top_k=top_k,
                filters=filters,
                latency_ms=latency,
                returned_contexts=[chunk.get("content", "")[:1200] for chunk in chunk_dicts[:10]],
                returned_chunk_ids=[chunk.get("chunk_id", "") for chunk in chunk_dicts],
                returned_paper_ids=list(
                    {chunk.get("paper_id", "") for chunk in chunk_dicts if chunk.get("paper_id")}
                ),
                caller_agent=caller_agent,
                tool_name="rag_retrieve",
            )
        )
    return chunk_dicts, latency


def _build_research_subgraph(llm, tools, researcher_prompt):
    tool_llm = None
    if llm is not None and hasattr(llm, "bind_tools"):
        try:
            tool_llm = llm.bind_tools(tools)
        except Exception:
            tool_llm = None

    async def researcher_agent(state: _ResearchSubgraphState):
        if tool_llm is None:
            raise RuntimeError("Researcher primary requires a tool-capable LLM")

        system_prompt = (
            f"{researcher_prompt}\n\n"
            "## Runtime Tool Policy\n"
            "- Use the available retrieval tools only when they materially improve "
            "answer quality.\n"
            "- Prefer `rag_retrieve` for evidence collection.\n"
            "- Use `related_papers` only when a source paper is provided and the "
            "related set is useful.\n"
            "- Never fabricate tool outputs or claim retrieval happened when it did not.\n"
            "\n## Prohibitions\n"
            "- Do not keep calling tools after evidence is sufficient.\n"
            "- Do not answer beyond the support of retrieved evidence.\n"
        )
        if state["query_type"] == QueryType.QA.value:
            system_prompt += (
                "\n## Runtime Stop Condition\n"
                "- Treat Memory context as valid user-specific evidence.\n"
                "- If Memory context directly answers the query, answer from it without "
                "calling tools.\n"
                "- When the available ToolMessages are sufficient, stop calling tools.\n"
                "- Answer the user query directly in concise prose grounded in "
                "retrieved evidence or Memory context."
            )
        else:
            system_prompt += (
                "\n## Runtime Stop Condition\n"
                "- When the available ToolMessages are sufficient, stop calling tools.\n"
                "- Reply with a short evidence-status note only."
            )
        context = (
            f"Query type: {state['query_type']}\n"
            f"User query: {state['query']}\n"
            f"Retrieval query: {state['retrieval_query']}\n"
            f"RAG strategy: {state['rag_strategy']}\n"
            f"Top K: {state['top_k']}\n"
            f"Paper IDs: {state.get('paper_ids', [])}\n"
            f"Categories: {state.get('categories', [])}\n"
            f"Date from: {state.get('date_from') or '(none)'}\n"
            f"Date to: {state.get('date_to') or '(none)'}\n"
            f"Source paper id: {state.get('source_paper_id') or '(none)'}\n"
            f"Source paper abstract: {state.get('source_paper_abstract') or '(none)'}\n"
            f"Memory context:\n{state.get('memory_context') or '(none)'}"
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

    builder = StateGraph(_ResearchSubgraphState)
    builder.add_node("agent", researcher_agent)
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


def make_research_node(paper_repository, llm, tools, prompt_catalog):
    return make_research_primary_node(paper_repository, llm, tools, prompt_catalog)


def make_research_primary_node(paper_repository, llm, tools, prompt_catalog):
    research_subgraph = _build_research_subgraph(llm, tools, prompt_catalog.get("researcher"))

    async def researcher_node(state):
        started = perf_counter()
        payload = request_value(state, "payload", {})
        parent_messages = list(state.get("messages", []))
        source_paper = None
        query = request_value(state, "query", "")
        query_type = request_value(state, "query_type")
        retrieval_query = query
        if query_type == QueryType.CROSS_DOMAIN.value and payload.get("paper_id"):
            source_paper = paper_repository.get_paper(payload["paper_id"])
            if source_paper:
                retrieval_query = f"{source_paper.title} {source_paper.abstract}"

        subgraph_output = await research_subgraph.ainvoke(
            {
                "messages": parent_messages,
                "query": query,
                "query_type": query_type,
                "retrieval_query": retrieval_query,
                "rag_strategy": payload.get("rag_strategy", "hybrid"),
                "top_k": payload.get("max_papers", payload.get("top_k", FINAL_CITATION_TOP_K)),
                "paper_ids": payload.get("paper_ids", []),
                "categories": payload.get("categories", []),
                "date_from": (
                    payload.get("date_from").isoformat()
                    if payload.get("date_from")
                    else None
                ),
                "date_to": payload.get("date_to").isoformat() if payload.get("date_to") else None,
                "source_paper_id": payload.get("paper_id"),
                "source_paper_abstract": source_paper.abstract if source_paper else "",
                "memory_context": memory_value(state, "context", ""),
                "caller_agent": "researcher",
            }
        )

        dedup: dict[str, dict] = {}
        total_latency = 0
        related = []
        subgraph_messages = new_messages_since(parent_messages, subgraph_output.get("messages", []))
        for message in all_tool_messages(subgraph_messages):
            parsed = parse_tool_result(message.content)
            if message.name == "rag_retrieve":
                total_latency += int(parsed.get("latency_ms", 0))
                for chunk in parsed.get("chunks", []):
                    current = dedup.get(chunk["chunk_id"])
                    if current is None or float(chunk.get("score", 0.0)) > float(
                        current.get("score", 0.0)
                    ):
                        dedup[chunk["chunk_id"]] = chunk
            elif message.name == "related_papers":
                related = parsed if isinstance(parsed, list) else []

        final_response = subgraph_messages[-1] if subgraph_messages else None
        duration = int((perf_counter() - started) * 1000)
        draft = None
        if query_type == QueryType.QA.value:
            draft = _select_qa_draft(
                str(getattr(final_response, "content", "")).strip(),
                query,
                list(dedup.values()),
                memory_value(state, "context", ""),
            )
        result = {
            "retrieval": {
                "chunks": list(dedup.values()),
                "related_papers": related,
                "rag_strategy": payload.get("rag_strategy", "hybrid"),
                "rag_latency_ms": total_latency,
            },
            "output": {"draft": draft},
            "planning": {"active_agent": None},
            "telemetry": {
                "llm_usage": merge_usage(
                    telemetry_value(state, "llm_usage"), subgraph_output.get("llm_usage")
                ),
                "agent_trace": telemetry_value(state, "agent_trace", [])
                + [{"agent": "researcher", "duration_ms": duration}],
            },
        }
        tool_trace_messages = all_tool_messages(subgraph_messages)
        if tool_trace_messages:
            result["tool_trace_messages"] = tool_trace_messages
        if final_response is not None:
            result["messages"] = [final_response]
        return result

    return researcher_node


def make_research_fallback_node(paper_repository, rag_engine):
    async def researcher_fallback(state):
        started = perf_counter()
        payload = request_value(state, "payload", {})
        query = request_value(state, "query", "")
        query_type = request_value(state, "query_type")
        retrieval_query = query
        if query_type == QueryType.CROSS_DOMAIN.value and payload.get("paper_id"):
            source_paper = paper_repository.get_paper(payload["paper_id"])
            if source_paper:
                retrieval_query = f"{source_paper.title} {source_paper.abstract}"
        chunks, latency = _retrieve_chunks(
            rag_engine,
            payload,
            retrieval_query,
            caller_agent="researcher",
        )
        related = (
            paper_repository.related_papers(payload["paper_id"], limit=5)
            if payload.get("paper_id")
            else []
        )
        duration = int((perf_counter() - started) * 1000)
        result = {
            "retrieval": {
                "chunks": chunks,
                "related_papers": related,
                "rag_strategy": payload.get("rag_strategy", "hybrid"),
                "rag_latency_ms": latency,
            },
            "output": {
                "draft": _qa_draft_from_chunks_or_memory(
                    query,
                    chunks,
                    memory_value(state, "context", ""),
                )
                if query_type == QueryType.QA.value
                else None
            },
            "planning": {"active_agent": None},
            "telemetry": {
                "agent_trace": telemetry_value(state, "agent_trace", [])
                + [{"agent": "researcher", "duration_ms": duration}]
            },
        }
        tool_trace_messages = [
            make_tool_trace_message(
                "rag_retrieve",
                {"chunks": chunks, "latency_ms": latency},
            )
        ]
        if related:
            tool_trace_messages.append(
                make_tool_trace_message("related_papers", related)
            )
        result["tool_trace_messages"] = tool_trace_messages
        return result

    return researcher_fallback


def make_idea_research_node(rag_engine):
    async def idea_research_node(state):
        query = request_value(state, "query", "")
        chunks, latency = _retrieve_chunks(
            rag_engine,
            request_value(state, "payload", {}),
            query,
            caller_agent="idea_research",
        )
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"Researcher gathered {len(chunks)} chunks for idea sub-query: "
                        f"{query}"
                    )
                )
            ],
            "idea_chunk_batches": [chunks],
            "idea_latencies": [latency],
        }

    return idea_research_node


def _select_qa_draft(
    model_text: str,
    query: str,
    chunks: list[dict],
    memory_context: str = "",
) -> str:
    if _is_no_information_answer(model_text) and memory_context.strip():
        return memory_context.strip()
    if _is_grounded_qa_answer(model_text):
        return model_text
    return _qa_draft_from_chunks_or_memory(query, chunks, memory_context)


def _qa_draft_from_chunks_or_memory(
    query: str,
    chunks: list[dict],
    memory_context: str,
) -> str:
    chunk_draft = _qa_draft_from_chunks(query, chunks)
    if chunk_draft:
        return chunk_draft
    return memory_context.strip()


def _is_no_information_answer(text: str) -> bool:
    lowered = text.strip().lower()
    return any(
        marker in lowered
        for marker in (
            "no information available",
            "not enough information",
            "cannot determine",
            "unable to determine",
            "无法确定",
            "未提及",
            "没有足够",
        )
    )


def _is_grounded_qa_answer(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return False
    lowered = cleaned.lower()
    generic_markers = [
        "collected enough evidence",
        "gathered enough evidence",
        "researcher collected",
        "requesting evidence",
    ]
    return not any(marker in lowered for marker in generic_markers)


def _qa_draft_from_chunks(query: str, chunks: list[dict]) -> str:
    ranked = rerank_retrieved_chunks(query, chunks, limit=3)
    snippets = [
        truncate(str(chunk.get("content", "")).strip(), 180)
        for chunk in ranked
        if str(chunk.get("content", "")).strip()
    ]
    if not snippets:
        return ""
    if len(snippets) == 1:
        return snippets[0]
    return " ".join(snippets[:2])
