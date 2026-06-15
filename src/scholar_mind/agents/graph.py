from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from time import perf_counter
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy, Send

from scholar_mind.agents.crossdomain import (
    make_crossdomain_fallback_node,
    make_crossdomain_primary_node,
)
from scholar_mind.agents.hypothesis import (
    make_hypothesis_fallback_node,
    make_hypothesis_primary_node,
)
from scholar_mind.agents.paper_reader import (
    make_paper_reader_fallback_node,
    make_paper_reader_primary_node,
)
from scholar_mind.agents.planner import make_planner_fallback_node, make_planner_primary_node
from scholar_mind.agents.prompts import PromptCatalog
from scholar_mind.agents.researcher import (
    make_idea_research_node,
    make_research_fallback_node,
    make_research_primary_node,
)
from scholar_mind.agents.reviewer import make_reviewer_fallback_node, make_reviewer_primary_node
from scholar_mind.agents.state import (
    GraphState,
    output_value,
    planning_value,
    request_value,
    retrieval_value,
    telemetry_value,
)
from scholar_mind.agents.study_planner import (
    make_study_planner_fallback_node,
    make_study_planner_primary_node,
)
from scholar_mind.agents.tools import build_tool_registry, get_tools_for
from scholar_mind.agents.trend import make_trend_fallback_node, make_trend_primary_node
from scholar_mind.agents.writer import make_writer_fallback_node, make_writer_primary_node
from scholar_mind.models.domain import QueryType

GLOBAL_RETRY_POLICY = RetryPolicy(max_attempts=2, initial_interval=1.0, backoff_factor=2.0)
STREAM_END = object()


async def _resolve_async(value):
    if inspect.isawaitable(value):
        return await value
    return value


def wrap_node(primary, fallback):
    async def node(state, runtime: Runtime):
        try:
            return await _resolve_async(primary(state))
        except Exception:
            if runtime.execution_info.node_attempt == 1:
                raise
            return await _resolve_async(fallback(state))

    return node


class SafeStateGraph(StateGraph):
    def add_safe_node(self, name, primary, fallback, **kwargs):
        return super().add_node(
            name,
            wrap_node(primary, fallback),
            retry_policy=GLOBAL_RETRY_POLICY,
            **kwargs,
        )


class AgentOrchestrator:
    def __init__(
        self,
        paper_repository,
        rag_engine,
        memory_manager,
        checkpointer,
        *,
        chat_models: dict[str, Any] | None = None,
        prompt_root: Path | None = None,
    ):
        self.paper_repository = paper_repository
        self.rag_engine = rag_engine
        self.memory_manager = memory_manager
        self.checkpointer = checkpointer
        self.chat_models = chat_models or {"reasoning": None, "light": None}
        self.prompt_catalog = PromptCatalog(prompt_root or Path("config/prompts"))
        self.tool_registry = build_tool_registry(
            paper_repository=paper_repository, rag_engine=rag_engine
        )
        self._checkpointer_factory = None
        if not self._looks_like_checkpointer(checkpointer):
            self._checkpointer_factory = checkpointer
            self.checkpointer = None
        self.graph = None
        self._graph_lock = None
        self._graph_loop = None

    def _build_graph(self):
        graph = SafeStateGraph(GraphState)
        researcher_tools = [
            self.tool_registry["rag_retrieve"],
            self.tool_registry["related_papers"],
        ]
        crossdomain_tools = [self.tool_registry["rag_top10_similar_papers"]]
        hypothesis_tools = [self.tool_registry["paper_methodology_lookup"]]
        trend_tools = get_tools_for("trend", self.tool_registry)
        writer_tools = [self.tool_registry["citation_lookup"]]

        graph.add_safe_node(
            "planner",
            make_planner_primary_node(
                self.chat_models.get("light"),
                self.memory_manager,
                self.prompt_catalog,
                self.paper_repository,
            ),
            make_planner_fallback_node(
                self.memory_manager,
                self.paper_repository,
            ),
        )
        graph.add_safe_node(
            "researcher",
            make_research_primary_node(
                self.paper_repository,
                self.chat_models.get("light"),
                researcher_tools,
                self.prompt_catalog,
            ),
            make_research_fallback_node(self.paper_repository, self.rag_engine),
        )
        graph.add_safe_node(
            "idea_research_dispatch",
            self._dispatch_idea_research,
            self._dispatch_idea_research,
        )
        idea_research = make_idea_research_node(self.rag_engine)
        graph.add_safe_node(
            "idea_research",
            idea_research,
            idea_research,
        )
        graph.add_safe_node(
            "research_gather",
            self._gather_research_results,
            self._gather_research_results,
        )
        graph.add_safe_node(
            "trend",
            make_trend_primary_node(
                self.paper_repository,
                self.chat_models.get("light"),
                trend_tools,
                self.prompt_catalog,
            ),
            make_trend_fallback_node(self.paper_repository),
        )
        graph.add_safe_node(
            "crossdomain",
            make_crossdomain_primary_node(
                self.paper_repository,
                self.chat_models.get("reasoning"),
                crossdomain_tools,
                self.prompt_catalog,
            ),
            make_crossdomain_fallback_node(
                self.paper_repository,
                self.rag_engine,
            ),
        )
        graph.add_safe_node(
            "hypothesis",
            make_hypothesis_primary_node(
                self.chat_models.get("reasoning"),
                hypothesis_tools,
                self.prompt_catalog,
            ),
            make_hypothesis_fallback_node(self.prompt_catalog),
        )
        graph.add_safe_node(
            "study_planner",
            make_study_planner_primary_node(
                self.paper_repository, self.chat_models.get("reasoning"), self.prompt_catalog
            ),
            make_study_planner_fallback_node(self.paper_repository),
        )
        graph.add_safe_node(
            "paper_reader",
            make_paper_reader_primary_node(
                self.paper_repository,
                self.chat_models.get("light") or self.chat_models.get("reasoning"),
                self.chat_models.get("reasoning"),
                self.prompt_catalog,
            ),
            make_paper_reader_fallback_node(self.paper_repository),
        )
        graph.add_safe_node(
            "writer",
            make_writer_primary_node(
                self.chat_models.get("reasoning"),
                writer_tools,
                self.prompt_catalog,
            ),
            make_writer_fallback_node(self.paper_repository, self.prompt_catalog),
        )
        graph.add_safe_node(
            "reviewer",
            make_reviewer_primary_node(
                self.chat_models.get("light") or self.chat_models.get("reasoning"),
                self.prompt_catalog,
            ),
            make_reviewer_fallback_node(self.prompt_catalog),
        )
        graph.set_entry_point("planner")
        graph.add_conditional_edges(
            "planner",
            self._aroute_after_planner,
            {
                "researcher": "researcher",
                "crossdomain": "crossdomain",
                "idea_research_dispatch": "idea_research_dispatch",
                "study_planner": "study_planner",
                "paper_reader": "paper_reader",
            },
        )
        graph.add_conditional_edges("idea_research_dispatch", self._adispatch_sub_queries)
        graph.add_edge("idea_research", "research_gather")
        graph.add_edge("research_gather", "writer")
        graph.add_edge("study_planner", "reviewer")
        graph.add_edge("paper_reader", "reviewer")

        graph.add_conditional_edges(
            "researcher",
            self._ashould_continue_after_research,
            {
                "reviewer": "reviewer",
                "trend": "trend",
                "crossdomain": "crossdomain",
            },
        )
        graph.add_edge("trend", "writer")
        graph.add_conditional_edges(
            "crossdomain",
            self._ashould_continue_after_crossdomain,
            {
                "hypothesis": "hypothesis",
            },
        )
        graph.add_edge("hypothesis", "writer")
        graph.add_edge("writer", "reviewer")
        graph.add_edge("reviewer", END)
        return graph.compile(checkpointer=self.checkpointer)

    async def run(self, state: GraphState | None, session_id: str | None = None) -> dict[str, Any]:
        thread_id = session_id or (request_value(state, "session_id") if state else None)
        if not thread_id:
            raise ValueError("session_id is required to invoke the LangGraph runtime")
        config = self._config(thread_id)
        graph = await self._ensure_graph()
        return await graph.ainvoke(state, config=config)

    async def stream(self, state: GraphState):
        session_id = request_value(state, "session_id")
        config = self._config(session_id)
        graph = await self._ensure_graph()
        async for update in graph.astream(state, config=config, stream_mode="updates"):
            for event in self._events_from_update(update, session_id):
                yield event
        yield ("done", {"session_id": session_id})

    async def get_state(self, session_id: str) -> dict[str, Any] | None:
        config = self._config(session_id)
        graph = await self._ensure_graph()
        if self.checkpointer is not None:
            checkpoint = await self._get_checkpoint(config)
            if checkpoint is None:
                return None
        snapshot = await graph.aget_state(config)
        return dict(snapshot.values)

    async def resume(self, session_id: str) -> dict[str, Any]:
        return await self.run(None, session_id=session_id)

    async def aclose(self) -> None:
        checkpointer = getattr(self, "checkpointer", None)
        if checkpointer is None:
            return None
        await self._close_checkpointer(checkpointer)
        self.graph = None
        self.checkpointer = None
        self._graph_loop = None
        return None

    @staticmethod
    def execution_path(query_type: QueryType) -> list[str]:
        if query_type == QueryType.QA:
            return ["planner", "researcher", "reviewer"]
        if query_type == QueryType.IDEA_NOVELTY:
            return ["planner", "idea_research", "research_gather", "writer", "reviewer"]
        if query_type == QueryType.TREND:
            return ["planner", "researcher", "trend", "writer", "reviewer"]
        if query_type == QueryType.CROSS_DOMAIN:
            return ["planner", "crossdomain", "hypothesis", "writer", "reviewer"]
        if query_type == QueryType.STUDY_PLAN:
            return ["planner", "study_planner", "reviewer"]
        if query_type == QueryType.PAPER_READING:
            return ["planner", "paper_reader", "reviewer"]
        return ["planner", "researcher", "reviewer"]

    @staticmethod
    def _route_after_planner(state: GraphState) -> str:
        query_type = request_value(state, "query_type")
        if query_type == QueryType.IDEA_NOVELTY.value:
            return "idea_research_dispatch"
        if query_type == QueryType.CROSS_DOMAIN.value:
            return "crossdomain"
        if query_type == QueryType.STUDY_PLAN.value:
            return "study_planner"
        if query_type == QueryType.PAPER_READING.value:
            return "paper_reader"
        return "researcher"

    @staticmethod
    def _dispatch_idea_research(state: GraphState) -> dict[str, Any]:
        return {
            "planning": {
                "sub_queries": planning_value(state, "sub_queries", [])
                or [request_value(state, "query")]
            }
        }

    @staticmethod
    def _dispatch_sub_queries(state: GraphState) -> list[Send]:
        request = {
            "query": request_value(state, "query"),
            "user_id": request_value(state, "user_id"),
            "session_id": request_value(state, "session_id"),
            "query_type_hint": request_value(state, "query_type_hint"),
            "query_type": request_value(state, "query_type"),
            "payload": request_value(state, "payload", {}),
            **state.get("request", {}),
        }
        return [
            Send("idea_research", {**state, "request": {**request, "query": sub_query}})
            for sub_query in planning_value(state, "sub_queries", [])
        ]

    @staticmethod
    def _gather_research_results(state: GraphState) -> dict[str, Any]:
        started = perf_counter()
        deduped: dict[str, dict[str, Any]] = {}
        for batch in state.get("idea_chunk_batches", []):
            for chunk in batch:
                current = deduped.get(chunk["chunk_id"])
                if current is None or float(chunk.get("score", 0.0)) > float(
                    current.get("score", 0.0)
                ):
                    deduped[chunk["chunk_id"]] = chunk
        duration = int((perf_counter() - started) * 1000)
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"Researcher gathered {len(deduped)} unique chunks across "
                        f"{len(planning_value(state, 'sub_queries', [])) or 1} idea sub-queries"
                    )
                )
            ],
            "retrieval": {
                "chunks": list(deduped.values()),
                "rag_strategy": request_value(state, "payload", {}).get("rag_strategy", "hybrid"),
                "rag_latency_ms": sum(state.get("idea_latencies", [])),
            },
            "telemetry": {
                "agent_trace": telemetry_value(state, "agent_trace", [])
                + [{"agent": "researcher", "duration_ms": duration}]
            },
        }

    def _should_continue_after_research(self, state: GraphState) -> str:
        query_type = request_value(state, "query_type")
        if query_type == QueryType.TREND.value:
            return "trend"
        if query_type == QueryType.QA.value:
            return "reviewer"
        return "crossdomain"

    def _should_continue_after_crossdomain(self, state: GraphState) -> str:
        return "hypothesis"

    async def _aroute_after_planner(self, state: GraphState) -> str:
        return self._route_after_planner(state)

    async def _adispatch_sub_queries(self, state: GraphState) -> list[Send]:
        return self._dispatch_sub_queries(state)

    async def _ashould_continue_after_research(self, state: GraphState) -> str:
        return self._should_continue_after_research(state)

    async def _ashould_continue_after_crossdomain(self, state: GraphState) -> str:
        return self._should_continue_after_crossdomain(state)

    @staticmethod
    def _config(session_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": session_id}}

    @staticmethod
    def _looks_like_checkpointer(candidate: Any) -> bool:
        return any(
            hasattr(candidate, attr)
            for attr in ("get", "aget", "get_tuple", "aget_tuple", "put", "aput")
        )

    async def _ensure_graph(self):
        checkpointer_factory = getattr(self, "_checkpointer_factory", None)
        graph_loop = getattr(self, "_graph_loop", None)
        current_loop = asyncio.get_running_loop()
        if self.graph is not None and (
            checkpointer_factory is None or graph_loop in {None, current_loop}
        ):
            return self.graph
        if getattr(self, "_graph_lock", None) is None:
            self._graph_lock = asyncio.Lock()
        async with self._graph_lock:
            checkpointer_factory = getattr(self, "_checkpointer_factory", None)
            graph_loop = getattr(self, "_graph_loop", None)
            current_loop = asyncio.get_running_loop()
            if self.graph is not None and (
                checkpointer_factory is None or graph_loop in {None, current_loop}
            ):
                return self.graph
            if checkpointer_factory is not None and graph_loop not in {None, current_loop}:
                if self.checkpointer is not None:
                    await self._close_checkpointer(self.checkpointer)
                self.graph = None
                self.checkpointer = None
            if self.checkpointer is None and checkpointer_factory is not None:
                factory_result = checkpointer_factory()
                self.checkpointer = await _resolve_async(factory_result)
            self.graph = self._build_graph()
            self._graph_loop = current_loop
            return self.graph

    async def _get_checkpoint(self, config: dict[str, dict[str, str]]):
        aget = getattr(self.checkpointer, "aget", None)
        if callable(aget):
            return await _resolve_async(aget(config))
        return await asyncio.to_thread(self.checkpointer.get, config)

    async def _close_checkpointer(self, checkpointer) -> None:
        context_manager = getattr(checkpointer, "_context_manager", None)
        if context_manager is not None and hasattr(context_manager, "__aexit__"):
            await _resolve_async(context_manager.__aexit__(None, None, None))
            return
        aclose = getattr(checkpointer, "aclose", None)
        if callable(aclose):
            await _resolve_async(aclose())
            return
        close = getattr(checkpointer, "close", None)
        if callable(close):
            await _resolve_async(close())
            return
        conn = getattr(checkpointer, "conn", None)
        if conn is not None and hasattr(conn, "close"):
            await _resolve_async(conn.close())

    def _events_from_update(
        self, update: dict[str, Any], session_id: str
    ) -> list[tuple[str, dict[str, Any]]]:
        events: list[tuple[str, dict[str, Any]]] = []
        for node_name, delta in update.items():
            if node_name == "planner":
                events.append(("plan", {"status": "planning", "message": "Analyzing request..."}))
            elif node_name == "researcher" and retrieval_value(delta, "chunks") is not None:
                events.append(
                    (
                        "progress",
                        {
                            "status": "researching",
                            "agent": "researcher",
                            "chunks_found": len(retrieval_value(delta, "chunks", [])),
                        },
                    )
                )
            elif node_name == "idea_research" and delta.get("idea_chunk_batches") is not None:
                events.append(
                    (
                        "progress",
                        {
                            "status": "researching",
                            "agent": "researcher",
                            "chunks_found": sum(
                                len(batch) for batch in delta.get("idea_chunk_batches", [])
                            ),
                        },
                    )
                )
            elif node_name == "research_gather" and retrieval_value(delta, "chunks") is not None:
                events.append(
                    (
                        "progress",
                        {
                            "status": "researching",
                            "agent": "researcher",
                            "chunks_found": len(retrieval_value(delta, "chunks", [])),
                        },
                    )
                )
            elif node_name == "trend":
                events.append(("progress", {"status": "analyzing", "agent": "trend"}))
            elif node_name == "crossdomain":
                events.append(("progress", {"status": "mapping", "agent": "crossdomain"}))
            elif node_name == "hypothesis":
                events.append(("progress", {"status": "hypothesizing", "agent": "hypothesis"}))
            elif node_name == "study_planner":
                events.append(("progress", {"status": "planning", "agent": "study_planner"}))
            elif node_name == "paper_reader":
                events.append(("progress", {"status": "reading", "agent": "paper_reader"}))
            elif node_name == "writer" and output_value(delta, "draft"):
                continue
            elif node_name == "reviewer":
                events.append(
                    (
                        "progress",
                        {"status": "reviewing", "agent": "reviewer", "session_id": session_id},
                    )
                )
                if output_value(delta, "final_answer"):
                    events.append(
                        (
                            "answer",
                            {
                                "answer": output_value(delta, "final_answer"),
                                "citations": output_value(delta, "citations", []),
                                "review_score": output_value(delta, "review_score"),
                            },
                        )
                    )
        return events
