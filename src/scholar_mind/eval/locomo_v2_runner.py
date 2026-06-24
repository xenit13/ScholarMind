"""End-to-end LOCOMO v2 evaluation runner.

Imports a persona's transcript rounds into memory, then asks each QA and
captures the prediction. Designed for the schema produced by
scholar_mind.eval.locomo_build.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from inspect import isawaitable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol
from uuid import uuid4

import httpx
from langchain_core.messages import AIMessage, HumanMessage

from scholar_mind.models.domain import QueryType
from scholar_mind.rag.top_k import FINAL_CITATION_TOP_K
from scholar_mind.utils.messages import serialize_messages

logger = logging.getLogger(__name__)


_ANSWER_INSTRUCTION = (
    "请只输出最终短答案，不要解释；如果答案包含多项，用英文逗号和空格分隔；"
    "如果没有足够记忆支持答案，回答 No information available."
)


class ResearchServiceLike(Protocol):
    """Minimal ResearchService surface used by the runner."""

    settings: Any

    def stream(
        self,
        *,
        query: str,
        user_id: str,
        session_id: str | None,
        query_type: QueryType | None,
        request_payload: dict,
    ) -> AsyncIterator[tuple[str, Any]]: ...

    def extract_transcript_memories(
        self,
        *,
        user_id: str,
        request_id: str,
        session_id: str,
        round_messages: list[dict[str, Any]],
    ) -> Any: ...


class HttpResearchServiceClient:
    """Small adapter exposing the ResearchService stream surface over HTTP."""

    def __init__(
        self,
        api_url: str,
        *,
        http_client_factory=httpx.AsyncClient,
        timeout: float | None = None,
    ):
        self.api_url = api_url.rstrip("/")
        self._http_client_factory = http_client_factory
        self._timeout = timeout
        self.settings = SimpleNamespace(final_citation_top_k=FINAL_CITATION_TOP_K)

    async def stream(
        self,
        *,
        query: str,
        user_id: str,
        session_id: str | None,
        query_type: QueryType | None,
        request_payload: dict,
    ) -> AsyncIterator[tuple[str, Any]]:
        endpoint = "ask/stream" if query_type == QueryType.QA else "stream"
        body = {
            "query": query,
            "user_id": user_id,
            "session_id": session_id,
            "paper_ids": request_payload.get("paper_ids", []),
            "rag_strategy": request_payload.get("rag_strategy", "hybrid"),
            "conditional_memory_injection": bool(
                request_payload.get("conditional_memory_injection", False)
            ),
        }
        for key in (
            "memory_extraction_enabled",
            "request_memory_extraction_enabled",
            "wait_for_pending_extractions",
        ):
            if key in request_payload:
                body[key] = request_payload[key]

        async with self._http_client_factory(timeout=self._timeout) as client:
            async with client.stream(
                "POST",
                f"{self.api_url}/api/v1/research/{endpoint}",
                json=body,
            ) as response:
                response.raise_for_status()
                async for event in _iter_sse_events(response.aiter_lines()):
                    yield event

    async def extract_transcript_memories(
        self,
        *,
        user_id: str,
        request_id: str,
        session_id: str,
        round_messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        body = {
            "user_id": user_id,
            "request_id": request_id,
            "session_id": session_id,
            "round_messages": round_messages,
            "wait_for_pending_extractions": True,
        }
        async with self._http_client_factory(timeout=self._timeout) as client:
            response = await client.post(
                f"{self.api_url}/api/v1/research/memory/transcript",
                json=body,
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and "data" in payload:
                return payload["data"]
            return payload


async def _iter_sse_events(lines: AsyncIterator[str]) -> AsyncIterator[tuple[str, Any]]:
    event_name = "message"
    data_lines: list[str] = []
    async for line in lines:
        if not line:
            if data_lines:
                yield event_name, _parse_sse_data("\n".join(data_lines))
            event_name = "message"
            data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())
    if data_lines:
        yield event_name, _parse_sse_data("\n".join(data_lines))


def _parse_sse_data(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def _iter_transcript_rounds(
    conversation: dict[str, Any],
) -> list[tuple[int, str, list[dict[str, Any]]]]:
    """Yield transcript rounds as one user turn plus following assistant turns."""
    rounds: list[tuple[int, str, list[dict[str, Any]]]] = []
    session_numbers = sorted(
        int(match.group(1))
        for key in conversation
        if (match := re.fullmatch(r"session_(\d+)", key))
        and isinstance(conversation.get(key), list)
    )
    for session_index in session_numbers:
        session_date = str(conversation.get(f"session_{session_index}_date_time", ""))
        current_round: list[dict[str, Any]] = []
        for turn in conversation.get(f"session_{session_index}", []):
            if not isinstance(turn, dict):
                continue
            speaker = str(turn.get("speaker", "")).lower()
            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            if "user" in speaker:
                if current_round:
                    rounds.append((session_index, session_date, current_round))
                current_round = [turn]
            elif "assistant" in speaker and current_round:
                current_round.append(turn)
        if current_round:
            rounds.append((session_index, session_date, current_round))
    return rounds


def _round_messages_payload(
    *,
    turns: list[dict[str, Any]],
    session_id: str,
    session_index: int,
    session_date: str,
    round_index: int,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for idx, turn in enumerate(turns):
        speaker = str(turn.get("speaker", "")).lower()
        content = str(turn.get("text", "")).strip()
        if not content:
            continue
        message = (
            HumanMessage(content=content)
            if "user" in speaker
            else AIMessage(content=content)
        )
        metadata = dict(turn.get("metadata", {}) or {})
        metadata.update(
            {
                "speaker": turn.get("speaker", ""),
                "session_index": session_index,
                "session_date": session_date,
            }
        )
        payload.append(
            {
                "message": serialize_messages([message])[0],
                "message_id": str(turn.get("dia_id") or f"s{session_index}:{idx + 1}"),
                "thread_id": session_id,
                "round_index": round_index,
                "metadata": metadata,
            }
        )
    return payload


async def _resolve(value: Any) -> Any:
    if isawaitable(value):
        return await value
    return value


async def _drain_stream(stream: AsyncIterator[tuple[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Consume a research_service.stream() generator, returning (answer_text, citations)."""
    answer_text = ""
    citations: list[dict[str, Any]] = []
    async for event, data in stream:
        if event == "answer":
            answer_text = str(_value(data, "answer")) or answer_text
            cited = _value(data, "citations") or []
            if cited:
                citations = list(cited)
    return answer_text, citations


def _value(obj: Any, key: str, default: Any = None) -> Any:
    """Safe .get(key) for dict or .key for object; returns default if missing."""
    if isinstance(obj, dict):
        result = obj.get(key, default)
    else:
        result = getattr(obj, key, default)
    return result if result is not None else default


def _citation_evidence_id(citation: Any) -> str:
    paper_id = str(_value(citation, "paper_id", "")).strip()
    section = str(_value(citation, "section", "")).strip()
    if not paper_id:
        return ""
    return f"{paper_id}::{section}" if section else f"{paper_id}::metadata"


async def replay_memory_turns(
    *,
    research_service: ResearchServiceLike,
    conversation: dict[str, Any],
    user_id: str,
    top_k: int,
    extraction_timeout: float = 300.0,
) -> int:
    """Import transcript rounds into the memory system, then wait for extraction.

    Returns the count of transcript rounds replayed (for logging/progress reporting).
    """
    _ = top_k
    replays = 0
    replay_session_id = f"{user_id}-locomo-replay"
    for session_index, session_date, turns in _iter_transcript_rounds(conversation):
        round_messages = _round_messages_payload(
            turns=turns,
            session_id=replay_session_id,
            session_index=session_index,
            session_date=session_date,
            round_index=replays + 1,
        )
        if not round_messages:
            continue
        await _resolve(
            research_service.extract_transcript_memories(
                request_id=f"locomo-replay-{uuid4().hex}",
                session_id=replay_session_id,
                round_messages=round_messages,
                user_id=user_id,
            )
        )
        replays += 1

    # Wait for all queued extraction tasks to finish before returning, so the
    # QA phase can rely on memory being populated. Production never does this —
    # only specialized runners (LOCOMO v2) opt in via this explicit call.
    wait = getattr(research_service, "wait_for_pending_extractions", None)
    if callable(wait):
        wait(timeout=extraction_timeout)
    return replays


async def ask_question(
    *,
    research_service: ResearchServiceLike,
    question: str,
    user_id: str,
    session_id: str,
    top_k: int,
) -> tuple[str, list[str]]:
    """Ask one QA, returning (prediction_text, citation_evidence_ids).

    Sets request_memory_extraction_enabled=False so the QA itself doesn't
    pollute memory (replay phase is the only memory source in LOCOMO mode).
    """
    query = f"{question.strip()}\n\n{_ANSWER_INSTRUCTION}"
    answer_text, citations = await _drain_stream(
        research_service.stream(
            query=query,
            user_id=user_id,
            session_id=session_id,
            query_type=QueryType.QA,
            request_payload={
                "paper_ids": [],
                "rag_strategy": "hybrid",
                "top_k": top_k,
                "conditional_memory_injection": False,
                "memory_extraction_enabled": False,
                "request_memory_extraction_enabled": False,
            },
        )
    )
    evidence_ids = [
        eid for citation in citations if (eid := _citation_evidence_id(citation))
    ]
    return answer_text, evidence_ids


async def run_locomo_v2_eval(
    *,
    research_service: ResearchServiceLike,
    samples: list[dict[str, Any]],
    prediction_key: str = "scholarmind",
    limit: int | None = None,
    progress_file: Path | None = None,
) -> list[dict[str, Any]]:
    """Run end-to-end evaluation on samples, populating prediction_key on each QA.

    - Per sample: replay memory-bearing turns, then ask each QA serially.
    - Progress is written to progress_file (if given) after each QA for observability.
      Resume is NOT supported — the user explicitly opted out.
    - `limit` caps total QAs across all samples for smoke testing.
    """
    settings = getattr(research_service, "settings", None)
    top_k = getattr(settings, "final_citation_top_k", FINAL_CITATION_TOP_K)
    remaining = limit if limit is not None else float("inf")
    question_index = 0
    for sample_idx, sample in enumerate(samples):
        persona = sample.get("persona", {}) or {}
        user_id = persona.get("user_id") or f"locomo_v2_p{sample_idx + 1:02d}"
        conversation = sample.get("conversation", {}) or {}
        replays = await replay_memory_turns(
            research_service=research_service,
            conversation=conversation,
            user_id=user_id,
            top_k=top_k,
        )
        logger.info(
            "sample[%s] persona=%s replayed %d memory turns",
            sample_idx,
            user_id,
            replays,
        )
        for qa_idx, qa in enumerate(sample.get("qa", [])):
            if remaining <= 0:
                return samples
            question_index += 1
            session_id = f"{user_id}-q{question_index:04d}"
            answer_text, evidence_ids = await ask_question(
                research_service=research_service,
                question=str(qa["question"]),
                user_id=user_id,
                session_id=session_id,
                top_k=top_k,
            )
            qa[prediction_key] = answer_text
            qa[f"{prediction_key}_context"] = evidence_ids
            logger.info(
                "sample[%s] qa[%d/%d] cat=%s -> %r",
                sample_idx,
                qa_idx + 1,
                len(sample.get("qa", [])),
                qa.get("category"),
                answer_text[:80],
            )
            if progress_file is not None:
                progress_file.parent.mkdir(parents=True, exist_ok=True)
                tmp = progress_file.with_name(f"{progress_file.name}.tmp")
                tmp.write_text(
                    json.dumps(samples, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                tmp.replace(progress_file)
            remaining -= 1
    return samples
