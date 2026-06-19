from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from scholar_mind.utils.messages import deserialize_messages, serialize_messages


def test_message_serialization_uses_native_langchain_shape():
    messages = [
        HumanMessage(content="hi", id="h1"),
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "name": "memory_lookup",
                    "args": {"query": "What does the user prefer?"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content='{"memories": [], "latency_ms": 5}',
            tool_call_id="call_1",
            name="memory_lookup",
            id="t1",
            status="success",
        ),
    ]

    payload = serialize_messages(messages)

    assert payload[0]["type"] == "human"
    assert payload[0]["data"]["id"] == "h1"
    assert payload[1]["type"] == "ai"
    assert payload[1]["data"]["tool_calls"][0]["name"] == "memory_lookup"
    assert payload[2]["type"] == "tool"
    assert payload[2]["data"]["tool_call_id"] == "call_1"
    assert payload[2]["data"]["name"] == "memory_lookup"


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
                "name": "memory_lookup",
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
    assert messages[1].name == "memory_lookup"
    assert messages[1].tool_call_id == "call_1"
