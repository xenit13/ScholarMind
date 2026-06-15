from __future__ import annotations

from time import perf_counter

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from scholar_mind.agents.common import (
    ainvoke_structured_output_with_raw,
    extract_json_candidate,
    merge_usage,
    raw_output_text,
    rerank_retrieved_chunks,
)
from scholar_mind.agents.state import (
    memory_value,
    output_value,
    request_value,
    retrieval_value,
    telemetry_value,
)
from scholar_mind.models.domain import QueryType, ReviewerOutput
from scholar_mind.rag.top_k import FINAL_CITATION_TOP_K
from scholar_mind.utils.text import truncate


def _normalize_review_output(candidate: str | None, fallback: str) -> str:
    if not candidate:
        return fallback
    text = candidate.strip()
    lowered = text.lower()
    if "revised draft:" in lowered:
        marker_index = lowered.index("revised draft:")
        cleaned = text[marker_index + len("revised draft:") :].strip(" *\n\t:")
        return cleaned or fallback
    if "unsupported claims identified" in lowered:
        return fallback
    if lowered.startswith("revised answer:"):
        _, _, remainder = text.partition(":")
        cleaned = remainder.strip()
        return cleaned or fallback
    return text


def _review_context_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    return [
        message
        for message in messages
        if not isinstance(message, ToolMessage)
        and not (isinstance(message, AIMessage) and message.tool_calls)
    ]


def make_reviewer_node(llm, prompt_catalog):
    return make_reviewer_primary_node(llm, prompt_catalog)


def make_reviewer_primary_node(llm, prompt_catalog):
    reviewer_prompt = prompt_catalog.get("reviewer")

    async def reviewer_node(state):
        started = perf_counter()
        query = request_value(state, "query", "")
        query_type = QueryType(request_value(state, "query_type"))
        chunks = retrieval_value(state, "chunks", [])
        if query_type == QueryType.PAPER_READING:
            final_answer = (
                output_value(state, "report_payload", {})
                .get("explanation", {})
                .get("plain_language", output_value(state, "draft", ""))
            )
            review_score = 1.0 if final_answer else 0.0
            duration = int((perf_counter() - started) * 1000)
            return {
                "output": {
                    "final_answer": final_answer,
                    "review_score": review_score,
                    "citations": [],
                },
                "telemetry": {
                    "llm_usage": merge_usage(telemetry_value(state, "llm_usage")),
                    "agent_trace": telemetry_value(state, "agent_trace", [])
                    + [{"agent": "reviewer", "duration_ms": duration}],
                },
            }
        citations = [
            {
                "paper_id": chunk["paper_id"],
                "title": chunk["title"],
                "section": chunk["section"],
                "quote": truncate(chunk["content"], 140),
                "relevance_score": round(float(chunk["score"]), 4),
            }
            for chunk in (
                rerank_retrieved_chunks(query, chunks, limit=FINAL_CITATION_TOP_K)
                if query_type == QueryType.QA
                else chunks[:FINAL_CITATION_TOP_K]
            )
        ]
        base_answer = output_value(state, "draft", "")
        if llm is None:
            raise RuntimeError("Reviewer primary requires an LLM")
        memory_context = memory_value(state, "context", "").strip() or "(none)"
        review_prompt = [
            SystemMessage(content=reviewer_prompt),
            *_review_context_messages(list(state.get("messages", []))),
            HumanMessage(
                content=(
                    f"Query type: {query_type.value}\n"
                    f"User query: {query}\n"
                    f"Memory context:\n{memory_context}\n\n"
                    f"Draft:\n{base_answer}\n"
                    f"Citations: {citations}"
                )
            ),
        ]
        structured, usage, response = await ainvoke_structured_output_with_raw(
            llm,
            review_prompt,
            ReviewerOutput,
            recover=_recover_reviewer_output,
        )
        final_answer = _normalize_review_output(
            structured.final_answer if structured else None,
            base_answer,
        )
        if structured:
            review_score = round(max(0.0, min(float(structured.review_score), 1.0)), 4)
        else:
            citation_coverage = min(len(citations), FINAL_CITATION_TOP_K) / FINAL_CITATION_TOP_K
            answer_signal = min(len(final_answer.split()), 80) / 80 if final_answer else 0.0
            review_score = round((citation_coverage * 0.6) + (answer_signal * 0.4), 4)
        duration = int((perf_counter() - started) * 1000)
        result = {
            "output": {
                "final_answer": final_answer,
                "review_score": review_score,
                "citations": citations,
            },
            "telemetry": {
                "llm_usage": merge_usage(telemetry_value(state, "llm_usage"), usage),
                "agent_trace": telemetry_value(state, "agent_trace", [])
                + [{"agent": "reviewer", "duration_ms": duration}],
            },
        }
        if response is not None:
            result["messages"] = [response]
        return result

    return reviewer_node


def make_reviewer_fallback_node(prompt_catalog):
    async def reviewer_fallback(state):
        started = perf_counter()
        query = request_value(state, "query", "")
        query_type = QueryType(request_value(state, "query_type"))
        chunks = retrieval_value(state, "chunks", [])
        if query_type == QueryType.PAPER_READING:
            final_answer = (
                output_value(state, "report_payload", {})
                .get("explanation", {})
                .get("plain_language", output_value(state, "draft", ""))
            )
            review_score = 1.0 if final_answer else 0.0
            duration = int((perf_counter() - started) * 1000)
            return {
                "messages": [AIMessage(content="Reviewer finalized the response")],
                "output": {
                    "final_answer": final_answer,
                    "review_score": review_score,
                    "citations": [],
                },
                "telemetry": {
                    "llm_usage": merge_usage(telemetry_value(state, "llm_usage")),
                    "agent_trace": telemetry_value(state, "agent_trace", [])
                    + [{"agent": "reviewer", "duration_ms": duration}],
                },
            }
        citations = [
            {
                "paper_id": chunk["paper_id"],
                "title": chunk["title"],
                "section": chunk["section"],
                "quote": truncate(chunk["content"], 140),
                "relevance_score": round(float(chunk["score"]), 4),
            }
            for chunk in (
                rerank_retrieved_chunks(query, chunks, limit=FINAL_CITATION_TOP_K)
                if query_type == QueryType.QA
                else chunks[:FINAL_CITATION_TOP_K]
            )
        ]
        base_answer = output_value(state, "draft", "")
        citation_coverage = min(len(citations), FINAL_CITATION_TOP_K) / FINAL_CITATION_TOP_K
        answer_signal = min(len(base_answer.split()), 80) / 80 if base_answer else 0.0
        review_score = round((citation_coverage * 0.6) + (answer_signal * 0.4), 4)
        duration = int((perf_counter() - started) * 1000)
        return {
            "messages": [AIMessage(content="Reviewer finalized the response")],
            "output": {
                "final_answer": base_answer,
                "review_score": review_score,
                "citations": citations,
            },
            "telemetry": {
                "llm_usage": merge_usage(telemetry_value(state, "llm_usage")),
                "agent_trace": telemetry_value(state, "agent_trace", [])
                + [{"agent": "reviewer", "duration_ms": duration}],
            },
        }

    return reviewer_fallback


def _recover_reviewer_output(raw) -> ReviewerOutput | None:
    text = raw_output_text(raw).strip()
    payload = extract_json_candidate(text)
    if isinstance(payload, dict):
        try:
            return ReviewerOutput.model_validate(payload)
        except Exception:
            return None
    if not text:
        return None
    return ReviewerOutput(
        final_answer=text,
        review_score=0.5,
        notes="recovered from plain text",
    )
