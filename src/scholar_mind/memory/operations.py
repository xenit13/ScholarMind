from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from math import sqrt
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from scholar_mind.memory.decay import default_decay_parameters
from scholar_mind.memory.discrete import (
    DISCRETE_CONFLICT_MIN_CONFIDENCE,
    discrete_value_token,
    normalize_discrete_structured,
    parse_discrete_fact,
    skips_temporal_conflict,
)
from scholar_mind.memory.repository import MemoryRepository
from scholar_mind.models.domain import (
    MemoryCandidate,
    MemoryOperationEvent,
    MemoryOperationName,
    MemoryOperationResult,
    MemoryStatus,
    StructuredMemoryRecord,
)
from scholar_mind.models.structured_output import invoke_structured_output_once

LOW_CONFIDENCE_THRESHOLD = 0.5
EMBEDDING_AUTO_MATCH_THRESHOLD = 0.82
EMBEDDING_JUDGE_MIN_THRESHOLD = 0.65
LEXICAL_DUPLICATE_THRESHOLD = 0.90
LEXICAL_UPDATE_THRESHOLD = 0.45
JUDGE_MIN_CONFIDENCE = 0.6
GENERIC_MEMORY_TYPE = "interaction_summary"
CONCRETE_MEMORY_TYPES = {
    "preference",
    "research_interest",
    "knowledge_level",
    "goal",
    "workflow",
    "project_constraint",
    "paper_read",
    "feedback",
}
logger = logging.getLogger(__name__)


class MemoryMatchJudgeOutput(BaseModel):
    relation: Literal["duplicate", "update", "distinct"]
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


@dataclass(frozen=True)
class MemoryMatch:
    record: StructuredMemoryRecord
    relation: Literal["duplicate", "update"]
    source: Literal["exact", "lexical", "embedding", "judge"]
    score: float | None = None
    reason: str = ""


class MemoryOperationApplier:
    def __init__(self, repository: MemoryRepository, index, embedder, llm=None):
        self.repository = repository
        self.index = index
        self.embedder = embedder
        self.llm = llm

    def apply_candidate(
        self,
        *,
        user_id: str,
        candidate: MemoryCandidate,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> MemoryOperationResult:
        candidate = _normalize_candidate_structured(candidate)
        operation = _requested_operation(candidate)
        if operation == MemoryOperationName.DELETE:
            match = self._find_matching_record(
                user_id=user_id,
                candidate=candidate,
                status=MemoryStatus.ACTIVE,
            )
            return self._delete(
                user_id=user_id,
                candidate=candidate,
                existing=match.record if match is not None else None,
                request_id=request_id,
                session_id=session_id,
            )
        if operation == MemoryOperationName.ARCHIVE:
            match = self._find_matching_record(
                user_id=user_id,
                candidate=candidate,
                status=MemoryStatus.ACTIVE,
            )
            return self._archive(
                user_id=user_id,
                candidate=candidate,
                existing=match.record if match is not None else None,
                request_id=request_id,
                session_id=session_id,
            )
        if operation == MemoryOperationName.RESTORE:
            match = self._find_matching_record(
                user_id=user_id,
                candidate=candidate,
                status=MemoryStatus.ARCHIVED,
            )
            return self._restore(
                user_id=user_id,
                candidate=candidate,
                archived=match.record if match is not None else None,
                request_id=request_id,
                session_id=session_id,
            )
        discrete_result = self._apply_discrete_candidate(
            user_id=user_id,
            candidate=candidate,
            request_id=request_id,
            session_id=session_id,
        )
        if discrete_result is not None:
            return discrete_result
        match = self._find_matching_record(
            user_id=user_id,
            candidate=candidate,
            status=MemoryStatus.ACTIVE,
        )
        if candidate.confidence < LOW_CONFIDENCE_THRESHOLD:
            return self._none(
                user_id=user_id,
                candidate=candidate,
                existing=match.record if match is not None else None,
                request_id=request_id,
                session_id=session_id,
                reason="candidate confidence is below threshold",
            )
        if match is None:
            return self._add(
                user_id=user_id,
                candidate=candidate,
                request_id=request_id,
                session_id=session_id,
            )
        if match.relation == "duplicate":
            if _candidate_adds_discrete_structure(match.record, candidate):
                return self._update(
                    user_id=user_id,
                    candidate=candidate,
                    existing=match.record,
                    request_id=request_id,
                    session_id=session_id,
                )
            return self._none(
                user_id=user_id,
                candidate=candidate,
                existing=match.record,
                request_id=request_id,
                session_id=session_id,
                reason=match.reason or "candidate duplicates existing memory",
            )
        return self._update(
            user_id=user_id,
            candidate=candidate,
            existing=match.record,
            request_id=request_id,
            session_id=session_id,
        )

    def reject_candidate(
        self,
        *,
        user_id: str,
        candidate: MemoryCandidate,
        reason: str,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> MemoryOperationResult:
        candidate = _normalize_candidate_structured(candidate)
        return self._none(
            user_id=user_id,
            candidate=candidate,
            existing=None,
            request_id=request_id,
            session_id=session_id,
            reason=reason,
        )

    def _add(
        self,
        *,
        user_id: str,
        candidate: MemoryCandidate,
        request_id: str | None,
        session_id: str | None,
    ) -> MemoryOperationResult:
        record = self._build_new_record(
            user_id=user_id,
            candidate=candidate,
            request_id=request_id,
            session_id=session_id,
        )
        self.repository.upsert(record)
        self._upsert_index(record)
        event = self._record_event(
            user_id=user_id,
            operation=MemoryOperationName.ADD,
            candidate=candidate,
            new_record=record,
            request_id=request_id,
            session_id=session_id,
            reason="new durable memory",
        )
        return MemoryOperationResult(
            operation=MemoryOperationName.ADD,
            memory_id=record.memory_id,
            record=record,
            event_id=event.event_id,
            reason=event.reason,
        )

    def _build_new_record(
        self,
        *,
        user_id: str,
        candidate: MemoryCandidate,
        request_id: str | None,
        session_id: str | None,
        supersedes: list[str] | None = None,
    ) -> StructuredMemoryRecord:
        now = datetime.now(UTC)
        decay_rate, decay_floor = default_decay_parameters(
            candidate.memory_type,
            candidate.source,
        )
        return StructuredMemoryRecord(
            memory_id=f"mem_{uuid4().hex}",
            user_id=user_id,
            scope="user",
            session_id=session_id,
            request_id=request_id,
            memory_type=candidate.memory_type,
            content=candidate.content,
            structured=candidate.structured,
            keywords=candidate.keywords,
            source=candidate.source,
            evidence=candidate.evidence,
            importance=candidate.importance,
            confidence=candidate.confidence,
            status=MemoryStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            decay_rate=decay_rate,
            decay_floor=decay_floor,
            supersedes=supersedes or [],
        )

    def _apply_discrete_candidate(
        self,
        *,
        user_id: str,
        candidate: MemoryCandidate,
        request_id: str | None,
        session_id: str | None,
    ) -> MemoryOperationResult | None:
        candidate_fact = parse_discrete_fact(candidate.structured)
        if candidate_fact is None:
            return None
        same_key_records: list[tuple[StructuredMemoryRecord, object]] = []
        for record in self.repository.list_by_status(user_id, MemoryStatus.ACTIVE):
            record_fact = parse_discrete_fact(record.structured)
            if record_fact is None:
                continue
            if record_fact.conflict_key == candidate_fact.conflict_key:
                same_key_records.append((record, record_fact))
        if not same_key_records:
            return None

        comparable_records: list[tuple[StructuredMemoryRecord, object]] = []
        for record, record_fact in same_key_records:
            if skips_temporal_conflict(candidate_fact, record_fact):
                continue
            comparable_records.append((record, record_fact))
        if not comparable_records:
            return self._add(
                user_id=user_id,
                candidate=candidate,
                request_id=request_id,
                session_id=session_id,
            )

        candidate_token = discrete_value_token(candidate_fact)
        conflicting_records: list[StructuredMemoryRecord] = []
        for record, record_fact in comparable_records:
            if discrete_value_token(record_fact) == candidate_token:
                return self._none(
                    user_id=user_id,
                    candidate=candidate,
                    existing=record,
                    request_id=request_id,
                    session_id=session_id,
                    reason="candidate duplicates existing discrete memory",
                )
            conflicting_records.append(record)

        if (
            candidate_fact.certainty == "explicit"
            and candidate.confidence >= DISCRETE_CONFLICT_MIN_CONFIDENCE
        ):
            return self._supersede_discrete_conflicts(
                user_id=user_id,
                candidate=candidate,
                existing_records=conflicting_records,
                request_id=request_id,
                session_id=session_id,
            )
        return self._none(
            user_id=user_id,
            candidate=candidate,
            existing=conflicting_records[0],
            request_id=request_id,
            session_id=session_id,
            reason="candidate conflicts with active discrete memory but lacks confidence",
        )

    def _supersede_discrete_conflicts(
        self,
        *,
        user_id: str,
        candidate: MemoryCandidate,
        existing_records: list[StructuredMemoryRecord],
        request_id: str | None,
        session_id: str | None,
    ) -> MemoryOperationResult:
        now = datetime.now(UTC)
        new_record = self._build_new_record(
            user_id=user_id,
            candidate=candidate,
            request_id=request_id,
            session_id=session_id,
            supersedes=[record.memory_id for record in existing_records],
        )
        self.repository.upsert(new_record)
        self._upsert_index(new_record)
        for record in existing_records:
            superseded = record.model_copy(
                update={
                    "status": MemoryStatus.SUPERSEDED,
                    "superseded_by": new_record.memory_id,
                    "updated_at": now,
                    "version": record.version + 1,
                }
            )
            self.repository.upsert(superseded)
            self._upsert_index(superseded)
        event = self._record_event(
            user_id=user_id,
            operation=MemoryOperationName.UPDATE,
            candidate=candidate,
            old_record=existing_records[0],
            new_record=new_record,
            request_id=request_id,
            session_id=session_id,
            reason="candidate supersedes conflicting discrete memory",
        )
        return MemoryOperationResult(
            operation=MemoryOperationName.UPDATE,
            memory_id=new_record.memory_id,
            record=new_record,
            event_id=event.event_id,
            reason=event.reason,
        )

    def _update(
        self,
        *,
        user_id: str,
        candidate: MemoryCandidate,
        existing: StructuredMemoryRecord,
        request_id: str | None,
        session_id: str | None,
    ) -> MemoryOperationResult:
        updated = existing.model_copy(
            update={
                "session_id": session_id or existing.session_id,
                "request_id": request_id or existing.request_id,
                "memory_type": _updated_memory_type(existing, candidate),
                "content": candidate.content,
                "structured": candidate.structured,
                "keywords": candidate.keywords,
                "source": candidate.source,
                "evidence": candidate.evidence,
                "importance": candidate.importance,
                "confidence": candidate.confidence,
                "status": MemoryStatus.ACTIVE,
                "updated_at": datetime.now(UTC),
                "version": existing.version + 1,
            }
        )
        self.repository.upsert(updated)
        self._upsert_index(updated)
        event = self._record_event(
            user_id=user_id,
            operation=MemoryOperationName.UPDATE,
            candidate=candidate,
            old_record=existing,
            new_record=updated,
            request_id=request_id,
            session_id=session_id,
            reason="candidate updates semantically matching memory",
        )
        return MemoryOperationResult(
            operation=MemoryOperationName.UPDATE,
            memory_id=updated.memory_id,
            record=updated,
            event_id=event.event_id,
            reason=event.reason,
        )

    def _delete(
        self,
        *,
        user_id: str,
        candidate: MemoryCandidate,
        existing: StructuredMemoryRecord | None,
        request_id: str | None,
        session_id: str | None,
    ) -> MemoryOperationResult:
        if existing is None:
            return self._none(
                user_id=user_id,
                candidate=candidate,
                existing=None,
                request_id=request_id,
                session_id=session_id,
                reason="delete candidate has no active target",
            )
        deleted = existing.model_copy(
            update={
                "session_id": session_id or existing.session_id,
                "request_id": request_id or existing.request_id,
                "status": MemoryStatus.DELETED,
                "updated_at": datetime.now(UTC),
                "version": existing.version + 1,
            }
        )
        self.repository.upsert(deleted)
        self._upsert_index(deleted)
        event = self._record_event(
            user_id=user_id,
            operation=MemoryOperationName.DELETE,
            candidate=candidate,
            old_record=existing,
            new_record=deleted,
            request_id=request_id,
            session_id=session_id,
            reason="candidate requests deletion",
        )
        return MemoryOperationResult(
            operation=MemoryOperationName.DELETE,
            memory_id=deleted.memory_id,
            record=deleted,
            event_id=event.event_id,
            reason=event.reason,
        )

    def _archive(
        self,
        *,
        user_id: str,
        candidate: MemoryCandidate,
        existing: StructuredMemoryRecord | None,
        request_id: str | None,
        session_id: str | None,
    ) -> MemoryOperationResult:
        if existing is None:
            return self._none(
                user_id=user_id,
                candidate=candidate,
                existing=None,
                request_id=request_id,
                session_id=session_id,
                reason="archive candidate has no active target",
            )
        archived = existing.model_copy(
            update={
                "session_id": session_id or existing.session_id,
                "request_id": request_id or existing.request_id,
                "status": MemoryStatus.ARCHIVED,
                "updated_at": datetime.now(UTC),
                "version": existing.version + 1,
            }
        )
        self.repository.upsert(archived)
        self._upsert_index(archived)
        event = self._record_event(
            user_id=user_id,
            operation=MemoryOperationName.ARCHIVE,
            candidate=candidate,
            old_record=existing,
            new_record=archived,
            request_id=request_id,
            session_id=session_id,
            reason="candidate requests archive",
        )
        return MemoryOperationResult(
            operation=MemoryOperationName.ARCHIVE,
            memory_id=archived.memory_id,
            record=archived,
            event_id=event.event_id,
            reason=event.reason,
        )

    def _restore(
        self,
        *,
        user_id: str,
        candidate: MemoryCandidate,
        archived: StructuredMemoryRecord | None,
        request_id: str | None,
        session_id: str | None,
    ) -> MemoryOperationResult:
        if archived is None:
            return self._none(
                user_id=user_id,
                candidate=candidate,
                existing=None,
                request_id=request_id,
                session_id=session_id,
                reason="restore candidate has no archived target",
            )
        restored = archived.model_copy(
            update={
                "session_id": session_id or archived.session_id,
                "request_id": request_id or archived.request_id,
                "status": MemoryStatus.ACTIVE,
                "updated_at": datetime.now(UTC),
                "version": archived.version + 1,
            }
        )
        self.repository.upsert(restored)
        self._upsert_index(restored)
        event = self._record_event(
            user_id=user_id,
            operation=MemoryOperationName.RESTORE,
            candidate=candidate,
            old_record=archived,
            new_record=restored,
            request_id=request_id,
            session_id=session_id,
            reason="candidate requests restore",
        )
        return MemoryOperationResult(
            operation=MemoryOperationName.RESTORE,
            memory_id=restored.memory_id,
            record=restored,
            event_id=event.event_id,
            reason=event.reason,
        )

    def _none(
        self,
        *,
        user_id: str,
        candidate: MemoryCandidate,
        existing: StructuredMemoryRecord | None,
        request_id: str | None,
        session_id: str | None,
        reason: str,
    ) -> MemoryOperationResult:
        event = self._record_event(
            user_id=user_id,
            operation=MemoryOperationName.NONE,
            candidate=candidate,
            old_record=existing,
            request_id=request_id,
            session_id=session_id,
            reason=reason,
        )
        return MemoryOperationResult(
            operation=MemoryOperationName.NONE,
            memory_id=existing.memory_id if existing is not None else None,
            event_id=event.event_id,
            reason=event.reason,
        )

    def _record_event(
        self,
        *,
        user_id: str,
        operation: MemoryOperationName,
        candidate: MemoryCandidate,
        old_record: StructuredMemoryRecord | None = None,
        new_record: StructuredMemoryRecord | None = None,
        request_id: str | None,
        session_id: str | None,
        reason: str,
    ) -> MemoryOperationEvent:
        event = MemoryOperationEvent(
            event_id=f"memop_{uuid4().hex}",
            user_id=user_id,
            operation=operation,
            memory_id=(
                new_record.memory_id
                if new_record is not None
                else old_record.memory_id
                if old_record is not None
                else None
            ),
            session_id=session_id,
            request_id=request_id,
            candidate=candidate.model_dump(mode="json"),
            old_record=old_record.model_dump(mode="json") if old_record is not None else None,
            new_record=new_record.model_dump(mode="json") if new_record is not None else None,
            reason=reason,
            model="rule",
            created_at=datetime.now(UTC),
        )
        self.repository.record_operation(event)
        return event

    def _upsert_index(self, record: StructuredMemoryRecord) -> None:
        try:
            vector = self.embedder.embed_query(record.content)
            self.index.upsert_memory(record, vector)
        except Exception:
            logger.exception(
                "Derived memory index upsert failed: memory_id=%s",
                record.memory_id,
            )

    def _find_matching_record(
        self,
        *,
        user_id: str,
        candidate: MemoryCandidate,
        status: MemoryStatus,
    ) -> MemoryMatch | None:
        candidate_type = _enum_value(candidate.memory_type)
        match_types = _candidate_match_types(candidate_type)
        records = [
            record
            for record in self.repository.list_by_status(user_id, status)
            if _enum_value(record.memory_type) in match_types
        ]
        if not records:
            return None

        candidate_text = _normalize_memory_text(candidate.content)
        for record in records:
            if _normalize_memory_text(record.content) == candidate_text:
                return MemoryMatch(
                    record=record,
                    relation=_direct_match_relation(record, candidate),
                    source="exact",
                    score=1.0,
                    reason="candidate exactly matches existing memory",
                )

        lexical_match = self._find_lexical_match(candidate, records)
        if lexical_match is not None:
            return lexical_match

        candidate_vector = self._safe_embed(candidate.content)
        if candidate_vector is None:
            return None

        best_record: StructuredMemoryRecord | None = None
        best_score = -1.0
        for record in records:
            record_vector = self._safe_embed(record.content)
            if record_vector is None:
                continue
            score = _cosine_similarity(candidate_vector, record_vector)
            if score > best_score:
                best_score = score
                best_record = record
        if best_record is None:
            return None
        if best_score >= EMBEDDING_AUTO_MATCH_THRESHOLD:
            return MemoryMatch(
                record=best_record,
                relation="update",
                source="embedding",
                score=best_score,
                reason="candidate updates semantically matching memory",
            )
        if best_score < EMBEDDING_JUDGE_MIN_THRESHOLD:
            return None
        return self._judge_gray_zone_match(
            candidate=candidate,
            record=best_record,
            score=best_score,
        )

    def _safe_embed(self, text: str) -> list[float] | None:
        try:
            vector = self.embedder.embed_query(text)
        except Exception:
            logger.exception("Memory semantic match embedding failed")
            return None
        try:
            return [float(value) for value in vector]
        except (TypeError, ValueError):
            return None

    def _find_lexical_match(
        self,
        candidate: MemoryCandidate,
        records: list[StructuredMemoryRecord],
    ) -> MemoryMatch | None:
        best_record: StructuredMemoryRecord | None = None
        best_score = -1.0
        for record in records:
            score = _lexical_similarity(candidate, record)
            if score > best_score:
                best_score = score
                best_record = record
        if best_record is None:
            return None
        if best_score >= LEXICAL_DUPLICATE_THRESHOLD:
            return MemoryMatch(
                record=best_record,
                relation=_direct_match_relation(best_record, candidate),
                source="lexical",
                score=best_score,
                reason="candidate lexically duplicates existing memory",
            )
        if best_score >= LEXICAL_UPDATE_THRESHOLD:
            return MemoryMatch(
                record=best_record,
                relation="update",
                source="lexical",
                score=best_score,
                reason="candidate updates lexically matching memory",
            )
        return None

    def _judge_gray_zone_match(
        self,
        *,
        candidate: MemoryCandidate,
        record: StructuredMemoryRecord,
        score: float,
    ) -> MemoryMatch | None:
        if self.llm is None:
            return None
        prompt = _build_match_judge_prompt(candidate=candidate, record=record, score=score)
        parsed, _raw, error = invoke_structured_output_once(
            self.llm,
            prompt,
            MemoryMatchJudgeOutput,
        )
        if error is not None:
            logger.warning("Memory match judge failed: %s", error)
            return None
        try:
            judgment = (
                parsed
                if isinstance(parsed, MemoryMatchJudgeOutput)
                else MemoryMatchJudgeOutput.model_validate(parsed)
            )
        except Exception:
            logger.warning("Memory match judge returned invalid output: %r", parsed)
            return None
        if judgment.confidence < JUDGE_MIN_CONFIDENCE:
            return None
        if judgment.relation == "distinct":
            return None
        return MemoryMatch(
            record=record,
            relation=judgment.relation,
            source="judge",
            score=judgment.confidence,
            reason=judgment.reason,
        )


def _requested_operation(candidate: MemoryCandidate) -> MemoryOperationName | None:
    operation = str(candidate.structured.get("operation", "")).upper()
    try:
        return MemoryOperationName(operation)
    except ValueError:
        return None


def _normalize_candidate_structured(candidate: MemoryCandidate) -> MemoryCandidate:
    structured = normalize_discrete_structured(candidate.structured)
    if structured == candidate.structured:
        return candidate
    return candidate.model_copy(update={"structured": structured})


def _candidate_adds_discrete_structure(
    existing: StructuredMemoryRecord,
    candidate: MemoryCandidate,
) -> bool:
    return (
        parse_discrete_fact(candidate.structured) is not None
        and parse_discrete_fact(existing.structured) is None
    )


def _normalize_memory_text(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = re.sub(r"\s+", "", cleaned)
    return re.sub(r"[，。！？!?,.:：;；、\"'“”‘’()（）\[\]{}<>《》]", "", cleaned)


def _candidate_match_types(memory_type: str) -> set[str]:
    if memory_type == GENERIC_MEMORY_TYPE:
        return {GENERIC_MEMORY_TYPE, *CONCRETE_MEMORY_TYPES}
    return {memory_type, GENERIC_MEMORY_TYPE}


def _updated_memory_type(existing: StructuredMemoryRecord, candidate: MemoryCandidate):
    existing_type = _enum_value(existing.memory_type)
    candidate_type = _enum_value(candidate.memory_type)
    if candidate_type == GENERIC_MEMORY_TYPE and existing_type != GENERIC_MEMORY_TYPE:
        return existing.memory_type
    return candidate.memory_type


def _direct_match_relation(
    record: StructuredMemoryRecord,
    candidate: MemoryCandidate,
) -> Literal["duplicate", "update"]:
    record_type = _enum_value(record.memory_type)
    candidate_type = _enum_value(candidate.memory_type)
    if record_type == GENERIC_MEMORY_TYPE and candidate_type != GENERIC_MEMORY_TYPE:
        return "update"
    return "duplicate"


def _lexical_similarity(
    candidate: MemoryCandidate,
    record: StructuredMemoryRecord,
) -> float:
    left = _normalize_memory_text(candidate.content)
    right = _normalize_memory_text(record.content)
    if not left or not right:
        return 0.0
    sequence_score = SequenceMatcher(None, left, right).ratio()
    char_score = _overlap_coefficient(set(left), set(right))
    bigram_score = _overlap_coefficient(_ngrams(left, 2), _ngrams(right, 2))
    keyword_score = _keyword_overlap(candidate, record)
    return (
        0.20 * sequence_score
        + 0.35 * char_score
        + 0.30 * bigram_score
        + 0.15 * keyword_score
    )


def _keyword_overlap(candidate: MemoryCandidate, record: StructuredMemoryRecord) -> float:
    left = {
        _normalize_memory_text(keyword)
        for keyword in candidate.keywords
        if _normalize_memory_text(keyword)
    }
    right = {
        _normalize_memory_text(keyword)
        for keyword in record.keywords
        if _normalize_memory_text(keyword)
    }
    if not left or not right:
        return 0.0
    return _overlap_coefficient(left, right)


def _ngrams(text: str, size: int) -> set[str]:
    if len(text) < size:
        return {text} if text else set()
    return {text[idx : idx + size] for idx in range(len(text) - size + 1)}


def _overlap_coefficient(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _build_match_judge_prompt(
    *,
    candidate: MemoryCandidate,
    record: StructuredMemoryRecord,
    score: float,
) -> str:
    payload = {
        "task": (
            "Decide whether the candidate memory expresses the same durable user fact "
            "as the existing memory."
        ),
        "labels": {
            "duplicate": "Same fact with no meaningful new information.",
            "update": "Same evolving fact with new, corrected, or more specific information.",
            "distinct": "Different durable facts that should both be kept.",
        },
        "candidate": {
            "memory_type": _enum_value(candidate.memory_type),
            "content": candidate.content,
            "keywords": candidate.keywords,
            "structured": candidate.structured,
        },
        "existing": {
            "memory_type": _enum_value(record.memory_type),
            "content": record.content,
            "keywords": record.keywords,
            "structured": record.structured,
        },
        "embedding_similarity": round(score, 4),
        "output": "Return JSON with relation, confidence, and reason.",
    }
    return json.dumps(payload, ensure_ascii=False)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _enum_value(value) -> str:
    return value.value if hasattr(value, "value") else str(value)
