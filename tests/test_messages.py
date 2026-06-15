from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from scholar_mind.agents.common import recent_tool_messages
from scholar_mind.utils.messages import deserialize_messages, serialize_messages


def test_message_serialization_uses_native_langchain_shape():
    messages = [
        HumanMessage(content="hi", id="h1"),
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {"name": "rag_retrieve", "args": {"query": "What is RAG?"}, "id": "call_1", "type": "tool_call"}
            ],
        ),
        ToolMessage(
            content='{"chunks": [], "latency_ms": 5}',
            tool_call_id="call_1",
            name="rag_retrieve",
            id="t1",
            status="success",
        ),
    ]

    payload = serialize_messages(messages)

    assert payload[0]["type"] == "human"
    assert payload[0]["data"]["id"] == "h1"
    assert payload[1]["type"] == "ai"
    assert payload[1]["data"]["tool_calls"][0]["name"] == "rag_retrieve"
    assert payload[2]["type"] == "tool"
    assert payload[2]["data"]["tool_call_id"] == "call_1"
    assert payload[2]["data"]["name"] == "rag_retrieve"


def test_message_serialization_round_trip_restores_native_messages():
    payload = [
        {
            "type": "human",
            "data": {
                "content": "hello",
                "additional_kwargs": {},
                "response_metadata": {},
                "type": "human",
                "name": None,
                "id": "h1",
            },
        },
        {
            "type": "tool",
            "data": {
                "content": '{"ok": true}',
                "additional_kwargs": {},
                "response_metadata": {},
                "type": "tool",
                "name": "paper_search",
                "id": "t1",
                "tool_call_id": "call_1",
                "artifact": None,
                "status": "success",
            },
        },
    ]

    messages = deserialize_messages(payload)

    assert messages[0].type == "human"
    assert messages[0].content == "hello"
    assert messages[1].type == "tool"
    assert messages[1].name == "paper_search"
    assert messages[1].tool_call_id == "call_1"


def test_recent_tool_messages_keeps_current_tool_loop_window():
    messages = [
        HumanMessage(content="What does hybrid retrieval improve?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "rag_retrieve",
                    "args": {"query": "hybrid retrieval"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content='{"chunks": [{"chunk_id": "c1"}], "latency_ms": 3}',
            tool_call_id="call_1",
            name="rag_retrieve",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "related_papers",
                    "args": {"paper_id": "p1", "limit": 5},
                    "id": "call_2",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content='[{"paper_id": "p2"}]',
            tool_call_id="call_2",
            name="related_papers",
        ),
    ]

    tool_messages = recent_tool_messages(messages)

    assert [message.name for message in tool_messages] == ["rag_retrieve", "related_papers"]
