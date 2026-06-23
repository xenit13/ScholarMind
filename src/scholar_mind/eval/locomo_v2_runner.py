"""End-to-end LOCOMO v2 evaluation runner.

Replays a persona's memory-bearing conversation turns through ResearchService
to populate memory, then asks each QA and captures the prediction. Designed
for the schema produced by scholar_mind.eval.locomo_build (Turn.metadata.seed_id
identifies memory-bearing turns; Turn.metadata.is_distractor flags distractors).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol

from scholar_mind.models.domain import QueryType
from scholar_mind.rag.top_k import FINAL_CITATION_TOP_K

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


def _iter_memory_bearing_turns(
    conversation: dict[str, Any],
) -> list[tuple[int, dict[str, Any]]]:
    """Yield (session_index, turn) for user-side memory-bearing turns.

    Memory-bearing means metadata.seed_id is not null AND metadata.is_distractor
    is False. Distractor turns and assistant turns are skipped — only user input
    seeds memory extraction.
    """
    turns: list[tuple[int, dict[str, Any]]] = []
    session_numbers = sorted(
        int(match.group(1))
        for key in conversation
        if (match := re.fullmatch(r"session_(\d+)", key))
        and isinstance(conversation.get(key), list)
    )
    for session_index in session_numbers:
        for turn in conversation.get(f"session_{session_index}", []):
            if not isinstance(turn, dict):
                continue
            metadata = turn.get("metadata", {}) or {}
            if metadata.get("is_distractor"):
                continue
            if metadata.get("seed_id") is None:
                continue
            if "user" not in str(turn.get("speaker", "")).lower():
                continue
            text = str(turn.get("text", "")).strip()
            if text:
                turns.append((session_index, turn))
    return turns


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
    """Feed memory-bearing user turns into the memory system, then wait for
    all in-flight extraction Celery tasks to complete.

    Returns the count of turns replayed (for logging/progress reporting).
    """
    replays = 0
    for session_index, turn in _iter_memory_bearing_turns(conversation):
        session_id = f"{user_id}-seed-s{session_index:03d}-{replays:04d}"
        async for _event, _data in research_service.stream(
            query=str(turn["text"]),
            user_id=user_id,
            session_id=session_id,
            query_type=QueryType.QA,
            request_payload={
                "paper_ids": [],
                "rag_strategy": "hybrid",
                "top_k": top_k,
                "conditional_memory_injection": False,
                "memory_extraction_enabled": True,
                "request_memory_extraction_enabled": True,
            },
        ):
            continue
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
