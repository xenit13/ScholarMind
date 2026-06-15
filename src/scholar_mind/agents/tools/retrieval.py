from __future__ import annotations

import json
from datetime import date
from typing import Any

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from scholar_mind.agents.state import planning_value
from scholar_mind.eval.context import get_eval_context
from scholar_mind.models.eval_models import RagRetrievalEventV2
from scholar_mind.rag.top_k import (
    CROSS_DOMAIN_CANDIDATE_TOP_K,
    FINAL_CITATION_TOP_K,
)
from scholar_mind.utils.text import overlap_score


def _tool_command(name: str, payload: Any, runtime: ToolRuntime, **extra_update) -> Command:
    return Command(
        update={
            **extra_update,
            "messages": [
                ToolMessage(
                    content=json.dumps(payload, ensure_ascii=False),
                    tool_call_id=runtime.tool_call_id,
                    name=name,
                )
            ],
        }
    )


def _caller_agent(runtime: ToolRuntime, default: str) -> str:
    state = getattr(runtime, "state", None)
    if isinstance(state, dict):
        value = state.get("caller_agent") or planning_value(state, "active_agent")
    else:
        value = getattr(state, "caller_agent", None) or getattr(state, "active_agent", None)
    return str(value or default)


def retrieve_top10_similar_papers_payload(
    rag_engine,
    *,
    source_summary: str,
    target_domains: list[str] | None = None,
    exclude_paper_ids: list[str] | None = None,
    exclude_primary_categories: list[str] | None = None,
    strategy: str = "hybrid",
    candidate_top_k: int = CROSS_DOMAIN_CANDIDATE_TOP_K,
) -> dict[str, Any]:
    domain_terms = [item.strip() for item in (target_domains or []) if item and item.strip()]
    query = source_summary.strip()
    if domain_terms:
        query = f"{query}\n\nTarget domains: {', '.join(domain_terms)}"
    chunks, latency = rag_engine.retrieve_sync(
        query=query,
        strategy=strategy,
        top_k=candidate_top_k,
        filters={},
    )
    return _build_top10_payload(
        rag_engine,
        chunks=chunks,
        latency=latency,
        query=query,
        target_domains=domain_terms,
        exclude_paper_ids=exclude_paper_ids,
        exclude_primary_categories=exclude_primary_categories,
        candidate_top_k=candidate_top_k,
    )


def _build_top10_payload(
    rag_engine,
    chunks,
    latency: int,
    query: str,
    target_domains: list[str],
    exclude_paper_ids: list[str] | None,
    exclude_primary_categories: list[str] | None,
    candidate_top_k: int,
) -> dict[str, Any]:
    exclude_ids = set(exclude_paper_ids or [])
    exclude_categories = set(exclude_primary_categories or [])
    paper_repository = rag_engine.paper_repository
    grouped: dict[str, dict] = {}
    grouped_domain_filtered: dict[str, dict] = {}
    for chunk in chunks:
        if chunk.paper_id in exclude_ids:
            continue
        paper = paper_repository.get_paper(chunk.paper_id)
        if paper is None:
            continue
        primary_category = paper.categories[0] if paper.categories else ""
        if not target_domains and primary_category in exclude_categories:
            continue
        domain_corpus = " ".join([paper.title, paper.abstract, *paper.categories]).strip()
        payload = grouped.setdefault(
            chunk.paper_id,
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "categories": list(paper.categories),
                "summary": paper.abstract.strip()[:320],
                "retrieval_score": 0.0,
                "supporting_chunks": [],
                "sources": [],
            },
        )
        payload["retrieval_score"] = max(payload["retrieval_score"], float(chunk.score))
        if len(payload["supporting_chunks"]) < 3:
            payload["supporting_chunks"].append(
                {
                    "chunk_id": chunk.chunk_id,
                    "section": chunk.section,
                    "content": chunk.content[:280],
                    "score": round(float(chunk.score), 4),
                }
            )
        payload["sources"].append(
            {
                "kind": "rag_top10_similar_papers",
                "paper_id": paper.paper_id,
                "title": paper.title,
                "section": chunk.section,
                "chunk_id": chunk.chunk_id,
                "score": round(float(chunk.score), 4),
            }
        )
        if target_domains and overlap_score(" ".join(target_domains), domain_corpus) > 0:
            grouped_domain_filtered[chunk.paper_id] = payload
    ranked_source = grouped_domain_filtered if grouped_domain_filtered else grouped
    ranked = sorted(
        ranked_source.values(),
        key=lambda item: float(item["retrieval_score"]),
        reverse=True,
    )[:candidate_top_k]
    return {
        "query": query,
        "target_domains": target_domains,
        "items": ranked,
        "latency_ms": latency,
    }


def build_retrieval_tools(rag_engine):
    @tool
    async def rag_retrieve(
        query: str,
        runtime: ToolRuntime,
        strategy: str = "hybrid",
        top_k: int = FINAL_CITATION_TOP_K,
        paper_ids: list[str] | None = None,
        categories: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> Command:
        """Retrieve relevant paper chunks from the indexed corpus."""
        filters = {
            "paper_ids": paper_ids or [],
            "categories": categories or [],
            "date_from": date.fromisoformat(date_from) if date_from else None,
            "date_to": date.fromisoformat(date_to) if date_to else None,
        }
        chunks, latency = rag_engine.retrieve_sync(
            query=query,
            strategy=strategy,
            top_k=top_k,
            filters=filters,
        )
        # Record RAG retrieval event for request audit.
        eval_ctx = get_eval_context()
        if eval_ctx is not None:
            chunk_ids = [c.chunk_id for c in chunks]
            paper_ids_found = list({c.paper_id for c in chunks})
            eval_ctx.rag_events.append(
                RagRetrievalEventV2(
                    request_id=eval_ctx.request_id,
                    query=query,
                    strategy=strategy,
                    top_k=top_k,
                    filters=filters,
                    latency_ms=latency,
                    returned_contexts=[c.content[:1200] for c in chunks[:10]],
                    returned_chunk_ids=chunk_ids,
                    returned_paper_ids=paper_ids_found,
                    caller_agent=_caller_agent(runtime, "researcher"),
                    tool_name="rag_retrieve",
                )
            )
        return _tool_command(
            "rag_retrieve",
            {
                "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
                "latency_ms": latency,
            },
            runtime,
        )

    @tool
    async def rag_top10_similar_papers(
        source_summary: str,
        runtime: ToolRuntime,
        target_domains: list[str] | None = None,
        exclude_paper_ids: list[str] | None = None,
        exclude_primary_categories: list[str] | None = None,
        strategy: str = "hybrid",
    ) -> Command:
        """Retrieve the top 10 candidate papers for cross-domain transfer from RAG."""
        payload = retrieve_top10_similar_papers_payload(
            rag_engine,
            source_summary=source_summary,
            target_domains=target_domains,
            exclude_paper_ids=exclude_paper_ids,
            exclude_primary_categories=exclude_primary_categories,
            strategy=strategy,
        )
        # Record RAG retrieval event for request audit.
        eval_ctx = get_eval_context()
        if eval_ctx is not None:
            items = payload.get("items", [])
            eval_ctx.rag_events.append(
                RagRetrievalEventV2(
                    request_id=eval_ctx.request_id,
                    query=source_summary,
                    strategy=strategy,
                    top_k=CROSS_DOMAIN_CANDIDATE_TOP_K,
                    filters={"target_domains": target_domains or []},
                    latency_ms=payload.get("latency_ms", 0),
                    returned_contexts=[item.get("title", "") for item in items[:10]],
                    returned_chunk_ids=[],
                    returned_paper_ids=[
                        item.get("paper_id", "") for item in items
                    ],
                    caller_agent=_caller_agent(runtime, "crossdomain"),
                    tool_name="rag_top10_similar_papers",
                )
            )
        return Command(
            update={
                "cross_domain_candidates": payload["items"],
                "rag_latency_ms": payload["latency_ms"],
                "rag_strategy": strategy,
                "messages": [
                    ToolMessage(
                        content=json.dumps(payload, ensure_ascii=False),
                        tool_call_id=runtime.tool_call_id,
                        name="rag_top10_similar_papers",
                    )
                ],
            }
        )

    return {
        "rag_retrieve": rag_retrieve,
        "rag_top10_similar_papers": rag_top10_similar_papers,
    }
