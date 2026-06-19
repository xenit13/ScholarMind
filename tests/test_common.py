from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel

from scholar_mind.models.structured_output import (
    ainvoke_structured_output,
    empty_usage,
    invoke_structured_output,
)


class _RetrySchema(BaseModel):
    answer: str


class _RawResult:
    def __init__(self, content: str, usage_metadata: dict[str, int]):
        self.content = content
        self.usage_metadata = usage_metadata


class _Runnable:
    def __init__(self, llm, payload):
        self.llm = llm
        self.payload = payload

    def invoke(self, prompt: str):
        self.llm.prompts.append(prompt)
        return self.payload


class _StructuredOutputLLM:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.prompts: list[str] = []
        self.include_raw_flags: list[bool] = []

    def with_structured_output(self, _schema, include_raw: bool = False):
        self.include_raw_flags.append(include_raw)
        return _Runnable(self, self.payloads.pop(0))


class _MethodAwareStructuredOutputLLM:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.prompts: list[str] = []
        self.include_raw_flags: list[bool] = []
        self.methods: list[str | None] = []

    def with_structured_output(
        self,
        _schema,
        include_raw: bool = False,
        method: str | None = None,
    ):
        self.include_raw_flags.append(include_raw)
        self.methods.append(method)
        return _Runnable(self, self.payloads.pop(0))


class _AsyncRunnable(_Runnable):
    async def ainvoke(self, prompt: str):
        return self.invoke(prompt)


class _AsyncMethodAwareStructuredOutputLLM(_MethodAwareStructuredOutputLLM):
    def with_structured_output(
        self,
        _schema,
        include_raw: bool = False,
        method: str | None = None,
    ):
        self.include_raw_flags.append(include_raw)
        self.methods.append(method)
        return _AsyncRunnable(self, self.payloads.pop(0))


def test_invoke_structured_output_retries_once_and_merges_usage(caplog):
    llm = _StructuredOutputLLM(
        [
            {
                "parsed": None,
                "raw": _RawResult(
                    '{"answer": 3}',
                    {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                ),
                "parsing_error": ValueError("Field 'answer' should be a string"),
            },
            {
                "parsed": _RetrySchema(answer="ok"),
                "raw": _RawResult(
                    '{"answer": "ok"}',
                    {"input_tokens": 6, "output_tokens": 3, "total_tokens": 9},
                ),
                "parsing_error": None,
            },
        ]
    )

    with caplog.at_level(logging.ERROR):
        parsed, usage = invoke_structured_output(llm, "Explain retrieval.", _RetrySchema)

    assert parsed == _RetrySchema(answer="ok")
    assert usage["prompt_tokens"] == 16
    assert usage["completion_tokens"] == 5
    assert usage["total_tokens"] == 21
    assert usage["latency_ms"] >= 0
    assert len(llm.prompts) == 2
    assert llm.include_raw_flags == [True, True]
    assert llm.prompts[1].startswith("You must output valid JSON matching this schema.")
    assert "Error: Field 'answer' should be a string" in llm.prompts[1]
    assert 'Previous output: {"answer": 3}' in llm.prompts[1]
    assert "correct output schema:" in llm.prompts[1]
    assert "Structured output parse failed" in caplog.text
    assert "Field 'answer' should be a string" in caplog.text
    assert '{"answer": "int"}' in caplog.text


def test_invoke_structured_output_uses_function_calling_when_supported():
    llm = _MethodAwareStructuredOutputLLM(
        [
            {
                "parsed": _RetrySchema(answer="ok"),
                "raw": _RawResult(
                    '{"answer": "ok"}',
                    {"input_tokens": 6, "output_tokens": 3, "total_tokens": 9},
                ),
                "parsing_error": None,
            },
        ]
    )

    parsed, usage = invoke_structured_output(llm, "Explain retrieval.", _RetrySchema)

    assert parsed == _RetrySchema(answer="ok")
    assert usage["total_tokens"] == 9
    assert llm.include_raw_flags == [True]
    assert llm.methods == ["function_calling"]


@pytest.mark.asyncio
async def test_ainvoke_structured_output_uses_function_calling_when_supported():
    llm = _AsyncMethodAwareStructuredOutputLLM(
        [
            {
                "parsed": _RetrySchema(answer="ok"),
                "raw": _RawResult(
                    '{"answer": "ok"}',
                    {"input_tokens": 6, "output_tokens": 3, "total_tokens": 9},
                ),
                "parsing_error": None,
            },
        ]
    )

    parsed, usage = await ainvoke_structured_output(
        llm,
        "Explain retrieval.",
        _RetrySchema,
    )

    assert parsed == _RetrySchema(answer="ok")
    assert usage["total_tokens"] == 9
    assert llm.include_raw_flags == [True]
    assert llm.methods == ["function_calling"]


def test_invoke_structured_output_returns_empty_usage_when_retry_still_fails():
    llm = _StructuredOutputLLM(
        [
            {
                "parsed": None,
                "raw": _RawResult(
                    '{"answer": 3}',
                    {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                ),
                "parsing_error": ValueError("Field 'answer' should be a string"),
            },
            {
                "parsed": None,
                "raw": _RawResult(
                    '{"answer": false}',
                    {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5},
                ),
                "parsing_error": ValueError("Field 'answer' should be a string"),
            },
        ]
    )

    parsed, usage = invoke_structured_output(llm, "Explain retrieval.", _RetrySchema)

    assert parsed is None
    assert usage == empty_usage()
    assert len(llm.prompts) == 2
    assert llm.include_raw_flags == [True, True]


def test_invoke_structured_output_skips_retry_when_failure_is_not_retryable():
    llm = _StructuredOutputLLM(
        [
            {
                "parsed": None,
                "raw": _RawResult("", {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}),
                "parsing_error": ValueError("invalid json"),
            }
        ]
    )

    parsed, usage = invoke_structured_output(llm, "Explain retrieval.", _RetrySchema)

    assert parsed is None
    assert usage == empty_usage()
    assert len(llm.prompts) == 1


def test_invoke_structured_output_recovers_generic_fenced_json():
    llm = _StructuredOutputLLM(
        [
            {
                "parsed": None,
                "raw": _RawResult(
                    '```json\n{"answer":"ok"}\n```',
                    {"input_tokens": 8, "output_tokens": 5, "total_tokens": 13},
                ),
                "parsing_error": ValueError("invalid json"),
            }
        ]
    )

    parsed, usage = invoke_structured_output(llm, "Explain retrieval.", _RetrySchema)

    assert parsed == _RetrySchema(answer="ok")
    assert usage["total_tokens"] == 13
    assert len(llm.prompts) == 1
