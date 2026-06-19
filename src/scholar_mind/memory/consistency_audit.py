"""Daily consistency audit for discrete structured memories."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from scholar_mind.memory.discrete import normalize_discrete_structured, parse_discrete_fact
from scholar_mind.memory.repository import MemoryRepository
from scholar_mind.models.domain import (
    MemoryOperationEvent,
    MemoryOperationName,
    StructuredMemoryRecord,
)
from scholar_mind.models.structured_output import invoke_structured_output_once

logger = logging.getLogger(__name__)

AUDIT_MODEL_NAME = "daily_consistency_audit"
AUDIT_CHECKER_VERSION = "memory_consistency_v1"


class MemoryConsistencyJudgeOutput(BaseModel):
    verdict: Literal["consistent", "inconsistent", "insufficient_evidence"]
    source_of_truth: str = "evidence"
    corrected_content: str = ""
    corrected_structured: dict[str, Any] = Field(default_factory=dict)
    corrected_keywords: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class MemoryConsistencyAuditItem(BaseModel):
    memory_id: str
    checked: bool
    verdict: str = "skipped"
    reason: str = ""
    correction: MemoryConsistencyJudgeOutput | None = None


class MemoryConsistencyAuditor:
    def __init__(
        self,
        *,
        repository: MemoryRepository,
        index,
        embedder,
        llm=None,
        min_confidence: float = 0.85,
        auto_fix: bool = True,
        batch_size: int = 500,
    ):
        self.repository = repository
        self.index = index
        self.embedder = embedder
        self.llm = llm
        self.min_confidence = min_confidence
        self.auto_fix = auto_fix
        self.batch_size = batch_size

    def run(self, user_id: str | None = None, dry_run: bool = False) -> dict[str, Any]:
        run_id = _new_run_id()
        records = self.repository.list_active_records(user_id=user_id)[: self.batch_size]
        checked_count = 0
        repaired_count = 0
        skipped_count = 0
        inconsistent_count = 0
        would_repair_count = 0
        repaired_memory_ids: list[str] = []
        skipped: list[dict[str, str]] = []

        for record in records:
            item = self.audit_record(record)
            if not item.checked:
                skipped_count += 1
                skipped.append({"memory_id": record.memory_id, "reason": item.reason})
                continue

            checked_count += 1
            correction = item.correction
            if item.verdict != "inconsistent" or correction is None:
                continue

            inconsistent_count += 1
            if correction.confidence < self.min_confidence:
                skipped_count += 1
                skipped.append(
                    {
                        "memory_id": record.memory_id,
                        "reason": "correction confidence below threshold",
                    }
                )
                continue

            if dry_run or not self.auto_fix:
                would_repair_count += 1
                continue

            repaired = self.repair_record(record, correction, run_id=run_id)
            if repaired is None:
                skipped_count += 1
                skipped.append({"memory_id": record.memory_id, "reason": "invalid correction"})
                continue

            repaired_count += 1
            repaired_memory_ids.append(repaired.memory_id)

        return {
            "run_id": run_id,
            "checked_count": checked_count,
            "inconsistent_count": inconsistent_count,
            "repaired_count": repaired_count,
            "would_repair_count": would_repair_count,
            "skipped_count": skipped_count,
            "repaired_memory_ids": repaired_memory_ids,
            "skipped": skipped,
        }

    def audit_record(self, record: StructuredMemoryRecord) -> MemoryConsistencyAuditItem:
        if parse_discrete_fact(record.structured) is None:
            return MemoryConsistencyAuditItem(
                memory_id=record.memory_id,
                checked=False,
                reason="memory is not a memory_fact_v1 discrete fact",
            )
        if not record.evidence:
            return MemoryConsistencyAuditItem(
                memory_id=record.memory_id,
                checked=False,
                reason="memory has no original evidence",
            )
        if self.llm is None:
            return MemoryConsistencyAuditItem(
                memory_id=record.memory_id,
                checked=False,
                reason="consistency judge llm is unavailable",
            )

        output, _raw, error = invoke_structured_output_once(
            self.llm,
            _build_audit_prompt(record),
            MemoryConsistencyJudgeOutput,
        )
        if error is not None or output is None:
            return MemoryConsistencyAuditItem(
                memory_id=record.memory_id,
                checked=False,
                reason=f"consistency judge failed: {error}",
            )
        if not isinstance(output, MemoryConsistencyJudgeOutput):
            try:
                output = MemoryConsistencyJudgeOutput.model_validate(output)
            except Exception as exc:
                return MemoryConsistencyAuditItem(
                    memory_id=record.memory_id,
                    checked=False,
                    reason=f"consistency judge returned invalid output: {exc}",
                )
        return MemoryConsistencyAuditItem(
            memory_id=record.memory_id,
            checked=True,
            verdict=output.verdict,
            reason=output.reason,
            correction=output,
        )

    def repair_record(
        self,
        record: StructuredMemoryRecord,
        correction: MemoryConsistencyJudgeOutput,
        *,
        run_id: str,
    ) -> StructuredMemoryRecord | None:
        corrected_content = correction.corrected_content.strip()
        corrected_structured = normalize_discrete_structured(correction.corrected_structured)
        if not corrected_content or parse_discrete_fact(corrected_structured) is None:
            return None

        updated = record.model_copy(
            update={
                "content": corrected_content,
                "structured": corrected_structured,
                "keywords": correction.corrected_keywords or record.keywords,
                "confidence": correction.confidence,
                "updated_at": datetime.now(UTC),
                "version": int(record.version or 1) + 1,
            }
        )
        self.repository.upsert(updated)
        self._record_repair_event(record, updated, correction, run_id)
        self._upsert_index(updated)
        return updated

    def _record_repair_event(
        self,
        old_record: StructuredMemoryRecord,
        new_record: StructuredMemoryRecord,
        correction: MemoryConsistencyJudgeOutput,
        run_id: str,
    ) -> None:
        event = MemoryOperationEvent(
            event_id=f"memop_{uuid4().hex}",
            user_id=old_record.user_id,
            operation=MemoryOperationName.UPDATE,
            memory_id=old_record.memory_id,
            session_id=old_record.session_id,
            request_id=old_record.request_id,
            candidate={
                "audit_run_id": run_id,
                "checker_version": AUDIT_CHECKER_VERSION,
                "source_of_truth": "evidence_json",
                "evidence_hash": _evidence_hash(old_record.evidence),
                "verdict": correction.verdict,
                "confidence": correction.confidence,
                "reason": correction.reason,
                "diff": {
                    "content_changed": old_record.content != new_record.content,
                    "structured_changed": old_record.structured != new_record.structured,
                    "keywords_changed": old_record.keywords != new_record.keywords,
                },
            },
            old_record=old_record.model_dump(mode="json"),
            new_record=new_record.model_dump(mode="json"),
            reason="daily_consistency_audit repaired memory from original evidence",
            model=AUDIT_MODEL_NAME,
            created_at=datetime.now(UTC),
        )
        self.repository.record_operation(event)

    def _upsert_index(self, record: StructuredMemoryRecord) -> None:
        try:
            vector = self.embedder.embed_query(record.content)
            self.index.upsert_memory(record, vector)
        except Exception:
            logger.exception(
                "Memory consistency audit index upsert failed: memory_id=%s",
                record.memory_id,
            )


def _new_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    return f"memaudit_{stamp}_{uuid4().hex[:6]}"


def _evidence_hash(evidence: list[dict[str, Any]]) -> str:
    payload = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_audit_prompt(record: StructuredMemoryRecord) -> str:
    payload = {
        "memory_id": record.memory_id,
        "content": record.content,
        "structured": record.structured,
        "evidence": record.evidence,
    }
    return (
        "You are auditing a user memory record for consistency.\n"
        "Compare three fields: `structured` memory_fact_v1, natural-language "
        "`content`, and original `evidence`.\n"
        "Treat `evidence` as the source of truth. If content or structured "
        "conflicts with evidence, return verdict `inconsistent` and rebuild "
        "corrected_content and corrected_structured from evidence only.\n"
        "If evidence is insufficient, return `insufficient_evidence` and do not "
        "invent facts. Use schema_version memory_fact_v1 for corrected_structured.\n\n"
        f"Record JSON:\n{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )
