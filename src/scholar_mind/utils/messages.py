from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict


def serialize_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    return messages_to_dict(messages)


def deserialize_messages(payload: list[dict[str, Any]]) -> list[BaseMessage]:
    return messages_from_dict(payload)
