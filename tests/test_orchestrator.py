from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langgraph.runtime import ExecutionInfo, Runtime
from langgraph.types import Send

from scholar_mind.agents.graph import AgentOrchestrator, wrap_node
from scholar_mind.agents.state import flatten_graph_state


class DummyGraph:
    async def ainvoke(self, state, *, config):
        await asyncio.sleep(0.05)
        return {"state": state, "config": config}

    async def astream(self, _state, *, config, stream_mode):
        assert config == {"configurable": {"thread_id": "sess-2"}}
        assert stream_mode == "updates"
        yield {"planner": {}}
        await asyncio.sleep(0.05)
        yield {"reviewer": {"draft": "ok", "final_answer": "ok"}}

    async def aget_state(self, *_args, **_kwargs):
        await asyncio.sleep(0.05)
        return SimpleNamespace(values={"messages": [], "final_answer": "ok"})


class DummyCheckpointer:
    async def aget(self, _config):
        await asyncio.sleep(0.05)
        return {"checkpoint": "present"}


@pytest.mark.asyncio
async def test_orchestrator_run_does_not_block_event_loop():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.graph = DummyGraph()
    orchestrator.checkpointer = DummyCheckpointer()

    task = asyncio.create_task(orchestrator.run({"session_id": "sess-1"}, session_id=None))

    await asyncio.sleep(0.01)

    assert task.done() is False
    result = await task
    assert result["config"] == {"configurable": {"thread_id": "sess-1"}}


@pytest.mark.asyncio
async def test_orchestrator_stream_does_not_block_event_loop():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.graph = DummyGraph()
    orchestrator.checkpointer = DummyCheckpointer()
    events = []

    async def consume():
        async for event in orchestrator.stream({"session_id": "sess-2"}):
            events.append(event)

    task = asyncio.create_task(consume())

    await asyncio.sleep(0.01)

    assert task.done() is False
    await task
    assert events[0][0] == "plan"
    assert all(event[0] != "chunk" for event in events)
    assert events[-1] == ("done", {"session_id": "sess-2"})


@pytest.mark.asyncio
async def test_orchestrator_get_state_does_not_block_event_loop():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator.graph = DummyGraph()
    orchestrator.checkpointer = DummyCheckpointer()

    task = asyncio.create_task(orchestrator.get_state("sess-3"))

    await asyncio.sleep(0.01)

    assert task.done() is False
    state = await task
    assert state == {"messages": [], "final_answer": "ok"}


@pytest.mark.asyncio
async def test_wrap_node_retries_once_then_uses_fallback():
    def primary(_state):
        raise RuntimeError("boom")

    def fallback(_state):
        return {"final_answer": "fallback"}

    node = wrap_node(primary, fallback)

    with pytest.raises(RuntimeError):
        await node(
            {},
            Runtime(
                execution_info=ExecutionInfo(
                    checkpoint_id="c1",
                    checkpoint_ns="root",
                    task_id="t1",
                    node_attempt=1,
                )
            ),
        )

    result = await node(
        {},
        Runtime(
            execution_info=ExecutionInfo(
                checkpoint_id="c1",
                checkpoint_ns="root",
                task_id="t1",
                node_attempt=2,
            )
        ),
    )

    assert result == {"final_answer": "fallback"}


def test_route_after_research_routes_directly_without_global_tools():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)

    assert (
        orchestrator._should_continue_after_research(
            {"query_type": "qa", "messages": [AIMessage(content="grounded answer")]}
        )
        == "reviewer"
    )
    assert (
        orchestrator._should_continue_after_research(
            {"query_type": "trend", "messages": [AIMessage(content="done")]}
        )
        == "trend"
    )
    assert (
        orchestrator._should_continue_after_research(
            {"query_type": "cross_domain", "messages": [AIMessage(content="done")]}
        )
        == "crossdomain"
    )


def test_planner_routes_idea_novelty_to_dispatch():
    assert (
        AgentOrchestrator._route_after_planner({"query_type": "idea_novelty"})
        == "idea_research_dispatch"
    )


def test_planner_routes_cross_domain_directly():
    assert AgentOrchestrator._route_after_planner({"query_type": "cross_domain"}) == "crossdomain"


def test_crossdomain_always_continues_to_hypothesis():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)

    assert (
        orchestrator._should_continue_after_crossdomain(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "rag_top10_similar_papers",
                                "args": {"source_summary": "planning"},
                                "id": "call-1",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            }
        )
        == "hypothesis"
    )
    assert orchestrator._should_continue_after_crossdomain(
        {"messages": [AIMessage(content="done")]}
    ) == "hypothesis"


def test_dispatch_sub_queries_returns_send_workers():
    sends = AgentOrchestrator._dispatch_sub_queries(
        {"query": "topic", "sub_queries": ["q1", "q2"], "messages": []}
    )

    assert all(isinstance(item, Send) for item in sends)
    assert [item.node for item in sends] == ["idea_research", "idea_research"]
    assert [flatten_graph_state(item.arg)["query"] for item in sends] == ["q1", "q2"]


def test_gather_research_results_dedupes_parallel_chunks():
    result = AgentOrchestrator._gather_research_results(
        {
            "request_payload": {"rag_strategy": "hybrid"},
            "sub_queries": ["q1", "q2"],
            "agent_trace": [{"agent": "planner", "duration_ms": 1}],
            "idea_latencies": [5, 7],
            "idea_chunk_batches": [
                [{"chunk_id": "c1", "score": 0.3}, {"chunk_id": "c2", "score": 0.4}],
                [{"chunk_id": "c1", "score": 0.6}],
            ],
        }
    )

    result = flatten_graph_state(result)

    assert result["rag_latency_ms"] == 12
    assert result["rag_strategy"] == "hybrid"
    assert {chunk["chunk_id"] for chunk in result["retrieved_chunks"]} == {"c1", "c2"}
    assert next(chunk for chunk in result["retrieved_chunks"] if chunk["chunk_id"] == "c1")[
        "score"
    ] == 0.6
