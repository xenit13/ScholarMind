from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from scholar_mind.agents.state import GraphState, merge_state_dict


def test_graph_state_groups_non_reducer_fields_under_structured_names():
    keys = set(GraphState.__annotations__)

    assert {
        "messages",
        "tool_trace_messages",
        "idea_chunk_batches",
        "idea_latencies",
        "request",
        "planning",
        "memory",
        "retrieval",
        "cross_domain",
        "reading",
        "output",
        "telemetry",
    } <= keys
    assert {
        "query",
        "user_id",
        "session_id",
        "query_type_hint",
        "query_type",
        "request_payload",
        "memory_context",
        "retrieved_chunks",
        "report_payload",
        "llm_usage",
        "agent_trace",
    }.isdisjoint(keys)


def test_nested_state_reducer_preserves_existing_subfields():
    merged = merge_state_dict(
        {"draft": "draft text", "report_payload": {"kind": "trend"}},
        {"final_answer": "final", "review_score": 0.9},
    )

    assert merged == {
        "draft": "draft text",
        "report_payload": {"kind": "trend"},
        "final_answer": "final",
        "review_score": 0.9,
    }


def test_langgraph_uses_nested_state_reducer_between_nodes():
    builder = StateGraph(GraphState)

    def writer(_state):
        return {"output": {"draft": "draft text", "report_payload": {"kind": "trend"}}}

    def reviewer(_state):
        return {"output": {"final_answer": "final", "review_score": 0.9}}

    builder.add_node("writer", writer)
    builder.add_node("reviewer", reviewer)
    builder.add_edge(START, "writer")
    builder.add_edge("writer", "reviewer")
    builder.add_edge("reviewer", END)

    graph = builder.compile()
    result = graph.invoke({"messages": [], "request": {"session_id": "s1"}})

    assert result["output"] == {
        "draft": "draft text",
        "report_payload": {"kind": "trend"},
        "final_answer": "final",
        "review_score": 0.9,
    }
