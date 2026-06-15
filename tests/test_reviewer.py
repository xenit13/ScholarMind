from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from scholar_mind.agents.reviewer import (
    _normalize_review_output,
)
from scholar_mind.agents.reviewer import (
    make_reviewer_node as _make_reviewer_node,
)
from scholar_mind.agents.state import flatten_graph_state


def make_reviewer_node(*args, **kwargs):
    node = _make_reviewer_node(*args, **kwargs)

    async def _node(state):
        return flatten_graph_state(await node(state))

    return _node


class _RawResult:
    def __init__(self, content: str, usage_metadata: dict[str, int]):
        self.content = content
        self.usage_metadata = usage_metadata


class _Runnable:
    def __init__(self, llm, payload):
        self.llm = llm
        self.payload = payload

    def invoke(self, prompt):
        self.llm.prompts.append(prompt)
        return self.payload

    async def ainvoke(self, prompt):
        return self.invoke(prompt)


class _StructuredOutputLLM:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.prompts: list[str] = []

    def with_structured_output(self, _schema, include_raw: bool = False):
        assert include_raw is True
        return _Runnable(self, self.payloads.pop(0))


class _PromptCatalog:
    def get(self, name: str) -> str:
        return f"{name} prompt"


def test_normalize_review_output_strips_audit_scaffold():
    candidate = (
        "**Unsupported Claims Identified:**\n"
        "- claim one\n\n"
        "**Revised Draft:**\n"
        "Hybrid retrieval improves recall by combining semantic matches with exact keyword hits."
    )

    normalized = _normalize_review_output(candidate, "fallback answer")

    assert normalized == (
        "Hybrid retrieval improves recall by combining semantic matches with exact keyword hits."
    )


def test_normalize_review_output_falls_back_when_only_audit_notes_exist():
    normalized = _normalize_review_output(
        "**Unsupported Claims Identified:**\n- claim one",
        "fallback answer",
    )

    assert normalized == "fallback answer"


@pytest.mark.asyncio
async def test_reviewer_short_circuits_for_paper_reading():
    node = make_reviewer_node(llm=None, prompt_catalog=_PromptCatalog())

    result = await node(
        {
            "query": "开始精读",
            "query_type": "paper_reading",
            "draft": "论文在强调稳定证据的重要性。",
            "report_payload": {
                "explanation": {"plain_language": "论文在强调稳定证据的重要性。"}
            },
            "agent_trace": [],
        }
    )

    assert result["final_answer"] == "论文在强调稳定证据的重要性。"
    assert result["review_score"] == 1.0
    assert result["citations"] == []


@pytest.mark.asyncio
async def test_reviewer_recovers_plain_text_structured_output():
    llm = _StructuredOutputLLM(
        [
            {
                "parsed": None,
                "raw": _RawResult(
                    "The answer stays grounded in the retrieved evidence.",
                    {"input_tokens": 6, "output_tokens": 4, "total_tokens": 10},
                ),
                "parsing_error": ValueError("invalid json"),
            }
        ]
    )
    node = make_reviewer_node(llm=llm, prompt_catalog=_PromptCatalog())

    result = await node(
        {
            "query": "What is retrieval grounding?",
            "query_type": "qa",
            "draft": "fallback answer",
            "retrieved_chunks": [],
            "agent_trace": [],
        }
    )

    assert result["final_answer"] == "The answer stays grounded in the retrieved evidence."
    assert result["review_score"] == 0.5
    assert result["llm_usage"]["total_tokens"] == 10


@pytest.mark.asyncio
async def test_reviewer_prompt_includes_memory_and_recent_messages():
    llm = _StructuredOutputLLM(
        [
            {
                "parsed": None,
                "raw": _RawResult(
                    "Final answer for the user.",
                    {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12},
                ),
                "parsing_error": ValueError("invalid json"),
            }
        ]
    )
    node = make_reviewer_node(llm=llm, prompt_catalog=_PromptCatalog())

    await node(
        {
            "query": "你是谁",
            "query_type": "qa",
            "draft": "我是你的 AI 助手，老板。",
            "memory_context": "- 用户希望被称呼为老板",
            "messages": [
                HumanMessage(content="在后续的对话中，称呼我为老板"),
                AIMessage(content="好的，老板。"),
                HumanMessage(content="你是谁"),
            ],
            "retrieved_chunks": [],
            "agent_trace": [],
        }
    )

    prompt = llm.prompts[0]
    assert isinstance(prompt[0], SystemMessage)
    assert prompt[1].content == "在后续的对话中，称呼我为老板"
    assert prompt[2].content == "好的，老板。"
    assert prompt[3].content == "你是谁"
    assert "Memory context:\n- 用户希望被称呼为老板" in prompt[4].content
    assert "Draft:\n我是你的 AI 助手，老板。" in prompt[4].content


@pytest.mark.asyncio
async def test_reviewer_prompt_omits_tool_call_messages():
    llm = _StructuredOutputLLM(
        [
            {
                "parsed": None,
                "raw": _RawResult(
                    "Final answer for the user.",
                    {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12},
                ),
                "parsing_error": ValueError("invalid json"),
            }
        ]
    )
    node = make_reviewer_node(llm=llm, prompt_catalog=_PromptCatalog())

    await node(
        {
            "query": "summarize",
            "query_type": "qa",
            "draft": "fallback answer",
            "messages": [
                HumanMessage(content="summarize"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "search", "args": {"query": "summarize"}, "id": "call_1"}
                    ],
                ),
                ToolMessage(content="tool result", tool_call_id="call_1"),
                AIMessage(content="ordinary assistant context"),
            ],
            "retrieved_chunks": [],
            "agent_trace": [],
        }
    )

    prompt = llm.prompts[0]
    assert [message.content for message in prompt[1:3]] == [
        "summarize",
        "ordinary assistant context",
    ]
    assert all(not getattr(message, "tool_calls", None) for message in prompt)
    assert all(not isinstance(message, ToolMessage) for message in prompt)
