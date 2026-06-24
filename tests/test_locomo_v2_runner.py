"""Tests for the LOCOMO v2 evaluation runner.

Uses a fake ResearchService to verify the runner:
- Replays complete transcript rounds directly into memory extraction
- Asks each QA and captures the prediction
- Populates prediction_key and {prediction_key}_context on each QA
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from scholar_mind.eval.locomo_v2_runner import (
    HttpResearchServiceClient,
    ask_question,
    replay_memory_turns,
    run_locomo_v2_eval,
)


def _turn(
    speaker: str,
    text: str,
    *,
    seed_id: str | None = None,
    is_distractor: bool | None = None,
    dia_id: str = "s1:1",
) -> dict[str, Any]:
    if is_distractor is None:
        is_distractor = seed_id is None
    return {
        "speaker": speaker,
        "dia_id": dia_id,
        "text": text,
        "metadata": {
            "seed_id": seed_id,
            "memory_type": None,
            "is_distractor": is_distractor,
        },
    }


def _make_conversation(turns_by_session: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    conv: dict[str, Any] = {"speaker_a": "user", "speaker_b": "assistant"}
    for session_idx, turns in turns_by_session.items():
        conv[f"session_{session_idx}_date_time"] = f"2026-05-{(session_idx - 1) * 3 + 3:02d}"
        conv[f"session_{session_idx}"] = turns
    return conv


class _FakeResearchService:
    """Captures stream() calls; returns canned answer events."""

    def __init__(self, *, answer: str = "fake answer", citations: list[dict] | None = None):
        self.settings = type("S", (), {"final_citation_top_k": 4})()
        self.answer = answer
        self.citations = citations or []
        self.stream_calls: list[dict[str, Any]] = []
        self.transcript_calls: list[dict[str, Any]] = []
        self.memory_manager = type("M", (), {"extract_pending_memories": self._noop})()

    @staticmethod
    def _noop(*_, **__): ...

    async def stream(
        self,
        *,
        query: str,
        user_id: str,
        session_id: str | None,
        query_type: Any,
        request_payload: dict,
    ) -> AsyncIterator[tuple[str, Any]]:
        self.stream_calls.append(
            {
                "query": query,
                "user_id": user_id,
                "session_id": session_id,
                "query_type": query_type,
                "request_payload": request_payload,
            }
        )
        # Extraction-only stream calls should not emit answer events.
        if request_payload.get("memory_extraction_enabled"):
            return
        yield "answer", {"answer": self.answer, "citations": self.citations}

    async def extract_transcript_memories(
        self,
        *,
        user_id: str,
        request_id: str,
        session_id: str,
        round_messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.transcript_calls.append(
            {
                "user_id": user_id,
                "request_id": request_id,
                "session_id": session_id,
                "round_messages": round_messages,
            }
        )
        return {"request_id": request_id, "dispatched": True}


class _FakeHttpStream:
    def __init__(self, lines: list[str]):
        self.lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        for line in self.lines:
            yield line


class _FakeHttpClient:
    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    def stream(self, method: str, url: str, *, json: dict[str, Any]):
        self.calls.append({"method": method, "url": url, "json": json})
        return _FakeHttpStream([
            "event: answer",
            (
                'data: {"answer": "http answer", '
                '"citations": [{"paper_id": "p1", "section": "abstract"}]}'
            ),
            "",
        ])

    async def post(self, url: str, *, json: dict[str, Any]):
        self.calls.append({"method": "POST", "url": url, "json": json})
        return type(
            "Resp",
            (),
            {
                "raise_for_status": lambda self: None,
                "json": lambda self: {
                    "success": True,
                    "data": {"request_id": json["request_id"], "dispatched": True},
                },
            },
        )()


# ---------- replay_memory_turns ----------


@pytest.mark.asyncio
async def test_replay_extracts_complete_transcript_rounds_without_streaming():
    conv = _make_conversation({
        1: [
            _turn("assistant", "assistant-only preface", seed_id="seed_skip", dia_id="s1:1"),
            _turn("user", "memory one", seed_id="seed_a", dia_id="s1:2"),
            _turn("assistant", "assistant response", seed_id="seed_a", dia_id="s1:3"),
            _turn("user", "distractor", is_distractor=True, dia_id="s1:4"),
            _turn("assistant", "assistant response to distractor", dia_id="s1:5"),
        ]
    })
    fake = _FakeResearchService()
    count = await replay_memory_turns(
        research_service=fake,
        conversation=conv,
        user_id="u1",
        top_k=4,
    )
    assert count == 2
    assert fake.stream_calls == []
    assert len(fake.transcript_calls) == 2

    first_round = fake.transcript_calls[0]["round_messages"]
    assert [item["message_id"] for item in first_round] == ["s1:2", "s1:3"]
    assert [item["message"]["type"] for item in first_round] == ["human", "ai"]
    assert first_round[0]["metadata"]["seed_id"] == "seed_a"
    assert first_round[1]["metadata"]["seed_id"] == "seed_a"

    second_round = fake.transcript_calls[1]["round_messages"]
    assert [item["message_id"] for item in second_round] == ["s1:4", "s1:5"]
    assert second_round[0]["metadata"]["is_distractor"] is True


@pytest.mark.asyncio
async def test_replay_calls_wait_for_pending_extractions():
    """Replay should wait for queued Celery tasks before returning."""
    conv = _make_conversation({1: [_turn("user", "memory", seed_id="seed_a")]})
    wait_calls: list[dict] = []

    fake = _FakeResearchService()

    def wait_fn(*, timeout: float = 300.0):
        wait_calls.append({"timeout": timeout})
        return {"total": 0, "succeeded": 0, "failed": 0}

    fake.wait_for_pending_extractions = wait_fn
    await replay_memory_turns(
        research_service=fake,
        conversation=conv,
        user_id="u1",
        top_k=4,
    )
    assert wait_calls == [{"timeout": 300.0}]


@pytest.mark.asyncio
async def test_replay_starts_new_round_at_each_user_turn():
    conv = _make_conversation({
        1: [
            _turn("user", "first user message", seed_id="seed_a", dia_id="s1:1"),
            _turn("user", "second user message", seed_id="seed_b", dia_id="s1:2"),
            _turn("assistant", "assistant follows second", seed_id="seed_b", dia_id="s1:3"),
        ]
    })
    fake = _FakeResearchService()

    count = await replay_memory_turns(
        research_service=fake,
        conversation=conv,
        user_id="u1",
        top_k=4,
    )

    assert count == 2
    assert [item["message_id"] for item in fake.transcript_calls[0]["round_messages"]] == [
        "s1:1"
    ]
    assert fake.transcript_calls[0]["round_messages"][0]["round_index"] == 1
    assert [item["message_id"] for item in fake.transcript_calls[1]["round_messages"]] == [
        "s1:2",
        "s1:3",
    ]
    assert fake.transcript_calls[1]["round_messages"][0]["round_index"] == 2


@pytest.mark.asyncio
async def test_http_research_service_client_posts_stream_payload_and_parses_sse():
    _FakeHttpClient.calls = []
    client = HttpResearchServiceClient(
        "http://api.test",
        http_client_factory=_FakeHttpClient,
    )

    events = [
        item
        async for item in client.stream(
            query="q",
            user_id="u1",
            session_id="s1",
            query_type=None,
            request_payload={
                "paper_ids": [],
                "rag_strategy": "hybrid",
                "top_k": 4,
                "conditional_memory_injection": False,
                "memory_extraction_enabled": True,
                "request_memory_extraction_enabled": True,
                "wait_for_pending_extractions": True,
            },
        )
    ]

    assert events == [
        (
            "answer",
            {
                "answer": "http answer",
                "citations": [{"paper_id": "p1", "section": "abstract"}],
            },
        )
    ]
    assert _FakeHttpClient.calls == [
        {
            "method": "POST",
            "url": "http://api.test/api/v1/research/stream",
            "json": {
                "query": "q",
                "user_id": "u1",
                "session_id": "s1",
                "paper_ids": [],
                "rag_strategy": "hybrid",
                "conditional_memory_injection": False,
                "memory_extraction_enabled": True,
                "request_memory_extraction_enabled": True,
                "wait_for_pending_extractions": True,
            },
        }
    ]


@pytest.mark.asyncio
async def test_http_research_service_client_posts_transcript_extraction_payload():
    _FakeHttpClient.calls = []
    client = HttpResearchServiceClient(
        "http://api.test",
        http_client_factory=_FakeHttpClient,
    )

    result = await client.extract_transcript_memories(
        user_id="u1",
        request_id="req1",
        session_id="s1",
        round_messages=[{"message_id": "s1:1", "message": {"type": "human", "data": {}}}],
    )

    assert result == {"request_id": "req1", "dispatched": True}
    assert _FakeHttpClient.calls == [
        {
            "method": "POST",
            "url": "http://api.test/api/v1/research/memory/transcript",
            "json": {
                "user_id": "u1",
                "request_id": "req1",
                "session_id": "s1",
                "round_messages": [
                    {"message_id": "s1:1", "message": {"type": "human", "data": {}}}
                ],
                "wait_for_pending_extractions": True,
            },
        }
    ]


# ---------- ask_question ----------


@pytest.mark.asyncio
async def test_ask_question_captures_answer_and_citations():
    fake = _FakeResearchService(
        answer="anchor paper",
        citations=[{"paper_id": "2604.20779", "section": "abstract"}],
    )
    answer, evidence = await ask_question(
        research_service=fake,
        question="这个项目的 paper role 是什么?",
        user_id="u1",
        session_id="u1-q0001",
        top_k=4,
    )
    assert answer == "anchor paper"
    assert evidence == ["2604.20779::abstract"]
    # The query sent to the model should include the answer instruction
    sent_query = fake.stream_calls[0]["query"]
    assert "请只输出最终短答案" in sent_query
    assert "memory_extraction_enabled" in fake.stream_calls[0]["request_payload"]
    assert fake.stream_calls[0]["request_payload"]["memory_extraction_enabled"] is False


@pytest.mark.asyncio
async def test_ask_question_handles_missing_citations():
    fake = _FakeResearchService(answer="some answer", citations=[])
    answer, evidence = await ask_question(
        research_service=fake,
        question="q",
        user_id="u1",
        session_id="s1",
        top_k=4,
    )
    assert answer == "some answer"
    assert evidence == []


# ---------- run_locomo_v2_eval ----------


def _make_sample(qas: list[dict], *, conversation_turns: list[dict] | None = None) -> dict:
    return {
        "sample_id": "test_p01",
        "persona": {"persona_id": "p01", "user_id": "u1", "background": "b"},
        "conversation": _make_conversation({
            1: conversation_turns or [_turn("user", "memory", seed_id="seed_a")],
        }),
        "qa": qas,
    }


@pytest.mark.asyncio
async def test_run_eval_populates_prediction_on_each_qa():
    sample = _make_sample([
        {
            "question": f"q{i}",
            "answer": f"gold{i}",
            "category": 1,
            "evidence": ["s1:1"],
            "metadata": {},
        }
        for i in range(3)
    ])
    fake = _FakeResearchService(answer="model prediction")
    out = await run_locomo_v2_eval(
        research_service=fake,
        samples=[sample],
        prediction_key="test_model",
    )
    for qa in out[0]["qa"]:
        assert qa["test_model"] == "model prediction"
        assert qa["test_model_context"] == []


@pytest.mark.asyncio
async def test_run_eval_replays_memory_before_asking(tmp_path):
    sample = _make_sample([
        {"question": "q1", "answer": "a1", "category": 1, "evidence": [], "metadata": {}},
    ])
    fake = _FakeResearchService(answer="answer")
    progress = tmp_path / "progress.json"
    await run_locomo_v2_eval(
        research_service=fake,
        samples=[sample],
        prediction_key="m",
        progress_file=progress,
    )
    # Replay uses transcript extraction; only the QA phase uses stream().
    assert len(fake.transcript_calls) == 1
    assert len(fake.stream_calls) == 1
    assert fake.stream_calls[0]["request_payload"]["memory_extraction_enabled"] is False
    # Progress file should exist and contain the prediction
    assert progress.exists()
    payload = json.loads(progress.read_text(encoding="utf-8"))
    assert payload[0]["qa"][0]["m"] == "answer"


@pytest.mark.asyncio
async def test_run_eval_limit_caps_total_qas():
    samples = [
        _make_sample([
            {"question": f"q{i}", "answer": f"a{i}", "category": 1, "evidence": [], "metadata": {}}
            for i in range(5)
        ])
        for _ in range(3)
    ]
    fake = _FakeResearchService(answer="ans")
    out = await run_locomo_v2_eval(
        research_service=fake,
        samples=samples,
        prediction_key="m",
        limit=7,
    )
    # Only 7 QAs should be populated; the rest should have no prediction
    populated = sum(1 for s in out for q in s["qa"] if "m" in q)
    assert populated == 7


@pytest.mark.asyncio
async def test_run_eval_uses_persona_user_id():
    sample = _make_sample(
        [{"question": "q", "answer": "a", "category": 1, "evidence": [], "metadata": {}}],
    )
    sample["persona"]["user_id"] = "custom_user_42"
    fake = _FakeResearchService(answer="ans")
    await run_locomo_v2_eval(
        research_service=fake,
        samples=[sample],
        prediction_key="m",
    )
    user_ids = {c["user_id"] for c in fake.stream_calls}
    assert user_ids == {"custom_user_42"}
