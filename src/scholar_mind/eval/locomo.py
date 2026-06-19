from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


@dataclass(frozen=True)
class OfficialLocomoTurn:
    session_number: int
    session_id: str
    timestamp: str
    dialog_id: str
    speaker: str
    text: str
    image_caption: str = ""


@dataclass(frozen=True)
class OfficialLocomoQuestion:
    question_id: str
    question: str
    answer: Any
    evidence: list[str]
    category: Any
    raw: dict[str, Any]


@dataclass(frozen=True)
class OfficialLocomoSample:
    sample_id: str
    speaker_a: str
    speaker_b: str
    turns: list[OfficialLocomoTurn]
    questions: list[OfficialLocomoQuestion]


_SESSION_RE = re.compile(r"^session_(?P<number>\d+)$")


def load_official_locomo(data_file: str | Path, *, limit: int | None = None):
    path = Path(data_file)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("LOCOMO_DATA_MUST_BE_A_LIST")
    rows = payload[:limit] if limit is not None else payload
    return [_parse_sample(row, index) for index, row in enumerate(rows)]


def run_official_locomo(
    *,
    data_file: str | Path,
    out_file: str | Path,
    memory_manager,
    model_key: str = "scholarmind_memory",
    limit: int | None = None,
    ingest: bool = True,
) -> dict[str, Any]:
    samples = load_official_locomo(data_file, limit=limit)
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prediction_key = f"{model_key}_prediction"
    output_rows: list[dict[str, Any]] = []
    question_count = 0
    for sample in samples:
        user_id = f"locomo:{sample.sample_id}"
        if ingest:
            _ingest_sample(memory_manager, sample, user_id=user_id)
        qa_rows: list[dict[str, Any]] = []
        for question in sample.questions:
            context, hit_count = memory_manager.get_context_sync(
                user_id=user_id,
                current_query=question.question,
            )
            answer = _answer_question(
                llm=getattr(memory_manager, "llm", None),
                question=question.question,
                memory_context=context,
            )
            qa_row = dict(question.raw)
            qa_row[prediction_key] = answer
            qa_row[f"{model_key}_memory_context"] = context
            qa_row[f"{model_key}_memory_hit_count"] = hit_count
            qa_rows.append(qa_row)
            question_count += 1
        output_rows.append({"sample_id": sample.sample_id, "qa": qa_rows})

    out_path.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "sample_count": len(samples),
        "question_count": question_count,
        "out_file": str(out_path),
        "model_key": model_key,
    }


def _parse_sample(row: dict[str, Any], index: int) -> OfficialLocomoSample:
    sample_id = str(row.get("sample_id") or f"sample_{index}")
    conversation = row.get("conversation")
    if not isinstance(conversation, dict):
        raise ValueError(f"LOCOMO_SAMPLE_MISSING_CONVERSATION: {sample_id}")
    qa_rows = row.get("qa")
    if not isinstance(qa_rows, list):
        raise ValueError(f"LOCOMO_SAMPLE_MISSING_QA: {sample_id}")
    return OfficialLocomoSample(
        sample_id=sample_id,
        speaker_a=str(conversation.get("speaker_a") or ""),
        speaker_b=str(conversation.get("speaker_b") or ""),
        turns=_parse_turns(conversation),
        questions=_parse_questions(sample_id, qa_rows),
    )


def _parse_turns(conversation: dict[str, Any]) -> list[OfficialLocomoTurn]:
    turns: list[OfficialLocomoTurn] = []
    for key in sorted(conversation, key=_session_sort_key):
        match = _SESSION_RE.match(key)
        if match is None:
            continue
        session_number = int(match.group("number"))
        session_rows = conversation.get(key)
        if not isinstance(session_rows, list):
            continue
        timestamp = str(conversation.get(f"{key}_date_time") or "")
        for turn_index, turn in enumerate(session_rows):
            if not isinstance(turn, dict):
                continue
            turns.append(
                OfficialLocomoTurn(
                    session_number=session_number,
                    session_id=key,
                    timestamp=timestamp,
                    dialog_id=str(turn.get("dia_id") or f"{key}:{turn_index}"),
                    speaker=str(turn.get("speaker") or ""),
                    text=str(turn.get("text") or ""),
                    image_caption=str(turn.get("blip_caption") or ""),
                )
            )
    return turns


def _parse_questions(
    sample_id: str, qa_rows: list[dict[str, Any]]
) -> list[OfficialLocomoQuestion]:
    questions: list[OfficialLocomoQuestion] = []
    for index, row in enumerate(qa_rows):
        if not isinstance(row, dict):
            continue
        questions.append(
            OfficialLocomoQuestion(
                question_id=f"{sample_id}:qa:{index}",
                question=str(row.get("question") or ""),
                answer=row.get("answer"),
                evidence=list(row.get("evidence") or []),
                category=row.get("category"),
                raw=dict(row),
            )
        )
    return questions


def _ingest_sample(memory_manager, sample: OfficialLocomoSample, *, user_id: str) -> None:
    by_session: dict[int, list[OfficialLocomoTurn]] = {}
    for turn in sample.turns:
        by_session.setdefault(turn.session_number, []).append(turn)
    for session_number in sorted(by_session):
        transcript = _session_transcript(by_session[session_number])
        memory_manager.log_round(
            user_id=user_id,
            session_id=f"locomo:{sample.sample_id}:session_{session_number}",
            round_index=session_number,
            messages=[HumanMessage(content=transcript)],
            explicit_memories=None,
        )
    memory_manager.extract_pending_memories(user_id=user_id)


def _session_transcript(turns: list[OfficialLocomoTurn]) -> str:
    lines: list[str] = []
    for turn in turns:
        prefix = f"[{turn.timestamp}] " if turn.timestamp else ""
        line = f"{prefix}{turn.speaker} ({turn.dialog_id}): {turn.text}".strip()
        if turn.image_caption:
            line += f"\nImage caption: {turn.image_caption}"
        lines.append(line)
    return "\n".join(lines)


def _answer_question(*, llm, question: str, memory_context: str) -> str:
    if llm is None or not hasattr(llm, "invoke"):
        return ""
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Answer the official LoCoMo question using the supplied memory "
                    "context. Preserve dates and names exactly when known. If the "
                    "context is insufficient, answer with the best concise response."
                )
            ),
            HumanMessage(
                content=(
                    f"Memory context:\n{memory_context or '(none)'}\n\n"
                    f"Question:\n{question}\n\nAnswer:"
                )
            ),
        ]
    )
    return str(getattr(response, "content", response)).strip()


def _session_sort_key(value: str) -> tuple[int, str]:
    match = _SESSION_RE.match(value)
    if match is None:
        return 10**9, value
    return int(match.group("number")), value
