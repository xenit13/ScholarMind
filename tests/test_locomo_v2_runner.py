"""Tests for the LOCOMO v2 evaluation runner.

Uses a fake ResearchService to verify the runner:
- Replays only memory-bearing user turns (skips distractors and assistant turns)
- Asks each QA and captures the prediction
- Populates prediction_key and {prediction_key}_context on each QA
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from scholar_mind.eval.locomo_v2_runner import (
    _iter_memory_bearing_turns,
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
) -> dict[str, Any]:
    if is_distractor is None:
        is_distractor = seed_id is None
    return {
        "speaker": speaker,
        "dia_id": "s1:1",
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
        # memory_extraction_enabled=True calls are replays; don't emit an answer
        if request_payload.get("memory_extraction_enabled"):
            return
        yield "answer", {"answer": self.answer, "citations": self.citations}


# ---------- _iter_memory_bearing_turns ----------


def test_iter_skips_distractor_turns():
    conv = _make_conversation({
        1: [
            _turn("user", "memory one", seed_id="seed_a"),
            _turn("assistant", "ok"),
            _turn("user", "distractor chat", is_distractor=True),
        ]
    })
    turns = _iter_memory_bearing_turns(conv)
    assert len(turns) == 1
    assert turns[0][1]["metadata"]["seed_id"] == "seed_a"


def test_iter_skips_assistant_turns_even_if_seed_set():
    conv = _make_conversation({
        1: [
            _turn("assistant", "assistant repeating memory", seed_id="seed_a"),
            _turn("user", "real memory", seed_id="seed_a"),
        ]
    })
    turns = _iter_memory_bearing_turns(conv)
    assert len(turns) == 1
    assert turns[0][1]["speaker"] == "user"


def test_iter_skips_empty_text():
    conv = _make_conversation({
        1: [
            _turn("user", "", seed_id="seed_a"),
            _turn("user", "real", seed_id="seed_b"),
        ]
    })
    turns = _iter_memory_bearing_turns(conv)
    assert len(turns) == 1
    assert turns[0][1]["metadata"]["seed_id"] == "seed_b"


def test_iter_returns_in_session_order():
    conv = _make_conversation({
        1: [_turn("user", "first", seed_id="s1")],
        2: [_turn("user", "second", seed_id="s2")],
        3: [_turn("user", "third", seed_id="s3")],
    })
    turns = _iter_memory_bearing_turns(conv)
    sessions = [t[0] for t in turns]
    assert sessions == [1, 2, 3]


# ---------- replay_memory_turns ----------


@pytest.mark.asyncio
async def test_replay_calls_stream_for_each_memory_turn():
    conv = _make_conversation({
        1: [
            _turn("user", "memory one", seed_id="seed_a"),
            _turn("user", "memory two", seed_id="seed_b"),
            _turn("user", "distractor", is_distractor=True),
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
    # Each replay should set memory_extraction_enabled=True
    assert len(fake.stream_calls) == 2
    for call in fake.stream_calls:
        assert call["request_payload"]["memory_extraction_enabled"] is True
        assert call["request_payload"]["memory_extraction_enabled"] is True
    # Each call should use a distinct session_id (so memory extracts don't collide)
    session_ids = [c["session_id"] for c in fake.stream_calls]
    assert len(set(session_ids)) == 2


@pytest.mark.asyncio
async def test_replay_calls_extract_pending_at_end():
    conv = _make_conversation({1: [_turn("user", "memory", seed_id="seed_a")]})
    extract_calls: list[dict] = []

    class FakeMM:
        def extract_pending_memories(self, *, user_id):
            extract_calls.append({"user_id": user_id})

    fake = _FakeResearchService()
    fake.memory_manager = FakeMM()
    await replay_memory_turns(
        research_service=fake,
        conversation=conv,
        user_id="u1",
        top_k=4,
    )
    assert extract_calls == [{"user_id": "u1"}]


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
    # Should be: 1 replay (memory) + 1 ask = 2 stream calls
    assert len(fake.stream_calls) == 2
    assert fake.stream_calls[0]["request_payload"]["memory_extraction_enabled"] is True
    assert fake.stream_calls[1]["request_payload"]["memory_extraction_enabled"] is False
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
