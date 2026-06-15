from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import delete, desc, or_, select
from sqlalchemy.orm import Session, sessionmaker

from scholar_mind.db.models import (
    MemoryEvalAnnotationBatchModel,
    MemoryEvalAnnotationV2Model,
    MemoryEvalReportV2Model,
    MemoryEvalRunV2Model,
    MemoryExtractionEventV2Model,
    MemoryLibraryAuditBatchModel,
    MemoryLibraryAuditReportModel,
    MemoryRetrievalEventV2Model,
    RequestRunModel,
)

_VERSION = "memory_eval_v2"
_L_INJ_REF_MS = 100.0
_T_INJ_REF = 256.0
_L_EXT_REF_MS = 1000.0
_T_EXT_REF = 256.0
_MEMORY_HEADER_RE = re.compile(r"^##\s+(?P<record_id>\S+)\s*$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _dump_jsonl(path: Path, rows: list[dict]) -> None:
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _normalized_text_set(values: list[str]) -> set[str]:
    return {
        " ".join(str(value).strip().lower().split())
        for value in values
        if str(value).strip()
    }


def _mean_or_none(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 4)


class MemoryEvalV2Repository:
    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    # ------------------------------------------------------------------
    # Raw event persistence
    # ------------------------------------------------------------------

    def save_memory_retrieval_event(self, event: dict) -> None:
        with self.session_factory() as session:
            session.add(
                MemoryRetrievalEventV2Model(
                    event_id=event.get("event_id", f"memret_{uuid4().hex}"),
                    request_id=event["request_id"],
                    user_id=event.get("user_id", ""),
                    query=event.get("query", ""),
                    embedding_latency_ms=int(event.get("embedding_latency_ms", 0) or 0),
                    vector_search_latency_ms=int(event.get("vector_search_latency_ms", 0) or 0),
                    retrieved_memory_ids_json=json.dumps(event.get("retrieved_memory_ids", [])),
                    retrieved_scores_json=json.dumps(event.get("retrieved_scores", [])),
                    retrieved_count=int(event.get("retrieved_count", 0) or 0),
                    injected_memory_ids_json=json.dumps(event.get("injected_memory_ids", [])),
                    injected_count=int(event.get("injected_count", 0) or 0),
                    injected_text=event.get("injected_text", ""),
                    injected_tokens=int(event.get("injected_tokens", 0) or 0),
                    created_at=_utcnow(),
                )
            )
            session.commit()

    def save_memory_extraction_dispatch(
        self,
        *,
        request_id: str,
        user_id: str,
        dispatch_latency_ms: int,
        dispatch_success: bool,
    ) -> None:
        with self.session_factory() as session:
            row = session.scalars(
                select(MemoryExtractionEventV2Model).where(
                    MemoryExtractionEventV2Model.request_id == request_id
                )
            ).first()
            if row is None:
                row = MemoryExtractionEventV2Model(
                    event_id=f"memext_{uuid4().hex}",
                    request_id=request_id,
                    user_id=user_id,
                    created_at=_utcnow(),
                )
                session.add(row)
            elif user_id:
                row.user_id = user_id
            row.dispatch_latency_ms = dispatch_latency_ms
            row.dispatch_success = dispatch_success
            session.commit()

    def update_memory_extraction_result(
        self,
        *,
        request_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        written_memory_ids: list[str],
        written_memory_texts: list[str],
    ) -> None:
        with self.session_factory() as session:
            row = session.scalars(
                select(MemoryExtractionEventV2Model).where(
                    MemoryExtractionEventV2Model.request_id == request_id
                )
            ).first()
            if row is None:
                row = MemoryExtractionEventV2Model(
                    event_id=f"memext_{uuid4().hex}",
                    request_id=request_id,
                    user_id="",
                    created_at=_utcnow(),
                )
                session.add(row)
            row.prompt_tokens = prompt_tokens
            row.completion_tokens = completion_tokens
            row.total_tokens = total_tokens
            row.written_memory_ids_json = json.dumps(written_memory_ids)
            row.written_memory_texts_json = json.dumps(written_memory_texts)
            session.commit()

    # ------------------------------------------------------------------
    # Raw event reads
    # ------------------------------------------------------------------

    def get_memory_retrieval_event(self, request_id: str) -> dict | None:
        with self.session_factory() as session:
            row = session.scalars(
                select(MemoryRetrievalEventV2Model)
                .where(MemoryRetrievalEventV2Model.request_id == request_id)
                .order_by(desc(MemoryRetrievalEventV2Model.created_at))
            ).first()
            if row is None:
                return None
            return self._retrieval_event_to_dict(row)

    def get_memory_extraction_event(self, request_id: str) -> dict | None:
        with self.session_factory() as session:
            row = session.scalars(
                select(MemoryExtractionEventV2Model).where(
                    MemoryExtractionEventV2Model.request_id == request_id
                )
            ).first()
            if row is None:
                return None
            return self._extraction_event_to_dict(row)

    # ------------------------------------------------------------------
    # Batch / evaluation persistence
    # ------------------------------------------------------------------

    def create_annotation_batch(self, batch: dict) -> None:
        with self.session_factory() as session:
            session.merge(
                MemoryEvalAnnotationBatchModel(
                    batch_id=batch["batch_id"],
                    version=batch.get("version", _VERSION),
                    status=batch.get("status", "exported"),
                    k=int(batch.get("k", 5)),
                    created_at=batch.get("created_at") or _utcnow(),
                    annotated_at=batch.get("annotated_at"),
                    evaluated_at=batch.get("evaluated_at"),
                    report_id=batch.get("report_id"),
                )
            )
            session.commit()

    def update_annotation_batch(self, batch_id: str, **fields) -> None:
        with self.session_factory() as session:
            row = session.get(MemoryEvalAnnotationBatchModel, batch_id)
            if row is None:
                return
            for key, value in fields.items():
                if hasattr(row, key):
                    setattr(row, key, value)
            session.commit()

    def replace_annotations(self, batch_id: str, annotations: list[dict]) -> None:
        with self.session_factory() as session:
            session.execute(
                delete(MemoryEvalAnnotationV2Model).where(
                    MemoryEvalAnnotationV2Model.batch_id == batch_id
                )
            )
            for annotation in annotations:
                session.add(
                    MemoryEvalAnnotationV2Model(
                        annotation_id=f"ann_{uuid4().hex}",
                        batch_id=batch_id,
                        request_id=annotation["request_id"],
                        relevant_memory_ids_json=json.dumps(annotation["relevant_memory_ids"]),
                        stale_memory_ids_json=json.dumps(annotation["stale_memory_ids"]),
                        claims_json=json.dumps(annotation["claims"], ensure_ascii=False),
                        expected_extracted_memories_json=json.dumps(
                            annotation["expected_extracted_memories"], ensure_ascii=False
                        ),
                        annotator=annotation.get("annotator", ""),
                        created_at=_utcnow(),
                    )
                )
            session.commit()

    def replace_eval_runs(self, batch_id: str, runs: list[dict]) -> None:
        with self.session_factory() as session:
            session.execute(
                delete(MemoryEvalRunV2Model).where(MemoryEvalRunV2Model.batch_id == batch_id)
            )
            for run in runs:
                session.add(
                    MemoryEvalRunV2Model(
                        run_id=f"memrun_{uuid4().hex}",
                        batch_id=batch_id,
                        request_id=run["request_id"],
                        user_id=run.get("user_id", ""),
                        session_id=run.get("session_id", ""),
                        memory_score=run["memory_score"],
                        memory_injected_count=run["memory_injected_count"],
                        memory_injected_latency_ms=run["memory_injected_latency_ms"],
                        memory_injected_tokens=run["memory_injected_tokens"],
                        memory_hit_at_k=run.get("memory_hit_at_k"),
                        memory_relevant_recall=run.get("memory_relevant_recall"),
                        memory_relevant_precision=run.get("memory_relevant_precision"),
                        first_relevant_rank=run.get("first_relevant_rank"),
                        memory_stale_retrieval_rate=run.get("memory_stale_retrieval_rate"),
                        memory_answer_relevance=run.get("memory_answer_relevance"),
                        memory_extraction_precision=run.get("memory_extraction_precision"),
                        memory_extraction_latency_ms=run.get("memory_extraction_latency_ms"),
                        memory_extraction_tokens=run.get("memory_extraction_tokens"),
                        score_breakdown_json=json.dumps(run.get("score_breakdown", {}), ensure_ascii=False),
                        created_at=_utcnow(),
                    )
                )
            session.commit()

    def save_eval_report(self, report: dict) -> None:
        with self.session_factory() as session:
            session.merge(
                MemoryEvalReportV2Model(
                    report_id=report["report_id"],
                    batch_id=report["batch_id"],
                    sample_count=int(report.get("sample_count", 0)),
                    avg_memory_score=float(report.get("avg_memory_score", 0.0)),
                    avg_memory_hit_at_k=report.get("avg_memory_hit_at_k"),
                    avg_memory_relevant_recall=report.get("avg_memory_relevant_recall"),
                    avg_memory_relevant_precision=report.get("avg_memory_relevant_precision"),
                    avg_first_relevant_rank=report.get("avg_first_relevant_rank"),
                    avg_memory_stale_retrieval_rate=report.get("avg_memory_stale_retrieval_rate"),
                    avg_memory_answer_relevance=report.get("avg_memory_answer_relevance"),
                    avg_memory_extraction_precision=report.get("avg_memory_extraction_precision"),
                    summary_json=json.dumps(report.get("summary", {}), ensure_ascii=False),
                    created_at=_utcnow(),
                )
            )
            session.commit()

    def create_library_audit_batch(self, batch: dict) -> None:
        with self.session_factory() as session:
            session.merge(
                MemoryLibraryAuditBatchModel(
                    batch_id=batch["batch_id"],
                    status=batch.get("status", "exported"),
                    memory_count=int(batch.get("memory_count", 0)),
                    created_at=batch.get("created_at") or _utcnow(),
                    evaluated_at=batch.get("evaluated_at"),
                    report_id=batch.get("report_id"),
                )
            )
            session.commit()

    def update_library_audit_batch(self, batch_id: str, **fields) -> None:
        with self.session_factory() as session:
            row = session.get(MemoryLibraryAuditBatchModel, batch_id)
            if row is None:
                return
            for key, value in fields.items():
                if hasattr(row, key):
                    setattr(row, key, value)
            session.commit()

    def save_library_audit_report(self, report: dict) -> None:
        with self.session_factory() as session:
            session.merge(
                MemoryLibraryAuditReportModel(
                    report_id=report["report_id"],
                    batch_id=report["batch_id"],
                    memory_count=int(report.get("memory_count", 0)),
                    duplicate_pair_count=int(report.get("duplicate_pair_count", 0)),
                    duplicate_memory_count=int(report.get("duplicate_memory_count", 0)),
                    duplicate_memory_ratio=report.get("duplicate_memory_ratio"),
                    conflict_pair_count=int(report.get("conflict_pair_count", 0)),
                    conflict_memory_count=int(report.get("conflict_memory_count", 0)),
                    conflict_memory_ratio=report.get("conflict_memory_ratio"),
                    summary_json=json.dumps(report.get("summary", {}), ensure_ascii=False),
                    created_at=_utcnow(),
                )
            )
            session.commit()

    def backfill_request_memory_score(self, request_id: str, memory_score: float) -> None:
        with self.session_factory() as session:
            row = session.get(RequestRunModel, request_id)
            if row is None:
                return
            row.memory_score = round(memory_score, 4)
            session.commit()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_annotation_batch(self, batch_id: str) -> dict | None:
        with self.session_factory() as session:
            row = session.get(MemoryEvalAnnotationBatchModel, batch_id)
            if row is None:
                return None
            return self._batch_to_dict(row)

    def get_eval_report(self, report_id: str) -> dict | None:
        with self.session_factory() as session:
            row = session.get(MemoryEvalReportV2Model, report_id)
            if row is None:
                return None
            runs = session.scalars(
                select(MemoryEvalRunV2Model)
                .where(MemoryEvalRunV2Model.batch_id == row.batch_id)
                .order_by(MemoryEvalRunV2Model.created_at.desc())
            ).all()
            return {
                **self._report_to_dict(row),
                "runs": [self._run_to_dict(item) for item in runs],
            }

    def get_eval_request(self, request_id: str) -> dict | None:
        with self.session_factory() as session:
            run = session.scalars(
                select(MemoryEvalRunV2Model)
                .where(MemoryEvalRunV2Model.request_id == request_id)
                .order_by(MemoryEvalRunV2Model.created_at.desc())
            ).first()
            retrieval = session.scalars(
                select(MemoryRetrievalEventV2Model)
                .where(MemoryRetrievalEventV2Model.request_id == request_id)
                .order_by(desc(MemoryRetrievalEventV2Model.created_at))
            ).first()
            extraction = session.scalars(
                select(MemoryExtractionEventV2Model).where(
                    MemoryExtractionEventV2Model.request_id == request_id
                )
            ).first()
            if run is None and retrieval is None and extraction is None:
                return None
            batch = session.get(MemoryEvalAnnotationBatchModel, run.batch_id) if run else None
            return {
                "request_id": request_id,
                "run": self._run_to_dict(run) if run else None,
                "batch": self._batch_to_dict(batch) if batch else None,
                "retrieval_event": self._retrieval_event_to_dict(retrieval) if retrieval else None,
                "extraction_event": self._extraction_event_to_dict(extraction) if extraction else None,
            }

    def get_batch_report_by_batch_id(self, batch_id: str) -> dict | None:
        with self.session_factory() as session:
            row = session.scalars(
                select(MemoryEvalReportV2Model).where(MemoryEvalReportV2Model.batch_id == batch_id)
            ).first()
            if row is None:
                return None
            return self._report_to_dict(row)

    def get_library_audit_batch(self, batch_id: str) -> dict | None:
        with self.session_factory() as session:
            row = session.get(MemoryLibraryAuditBatchModel, batch_id)
            if row is None:
                return None
            return self._library_audit_batch_to_dict(row)

    def get_library_audit_report(self, report_id: str) -> dict | None:
        with self.session_factory() as session:
            row = session.get(MemoryLibraryAuditReportModel, report_id)
            if row is None:
                return None
            return self._library_audit_report_to_dict(row)

    def get_latest_library_audit_report(self) -> dict | None:
        with self.session_factory() as session:
            row = session.scalars(
                select(MemoryLibraryAuditReportModel)
                .order_by(desc(MemoryLibraryAuditReportModel.created_at))
            ).first()
            if row is None:
                return None
            return self._library_audit_report_to_dict(row)

    def list_exportable_requests(self, from_request_id: str, limit: int) -> list[dict]:
        with self.session_factory() as session:
            anchor = session.get(RequestRunModel, from_request_id)
            if anchor is None:
                raise ValueError(f"REQUEST_NOT_FOUND: {from_request_id}")
            rows = session.execute(
                select(RequestRunModel)
                .outerjoin(
                    MemoryRetrievalEventV2Model,
                    MemoryRetrievalEventV2Model.request_id == RequestRunModel.request_id,
                )
                .outerjoin(
                    MemoryExtractionEventV2Model,
                    MemoryExtractionEventV2Model.request_id == RequestRunModel.request_id,
                )
                .where(RequestRunModel.created_at <= anchor.created_at)
                .where(
                    or_(
                        MemoryRetrievalEventV2Model.event_id.is_not(None),
                        MemoryExtractionEventV2Model.total_tokens.is_not(None),
                    )
                )
                .distinct()
                .order_by(RequestRunModel.created_at.desc())
                .limit(limit)
            ).scalars().all()
            return [self._request_eval_row_to_dict(row) for row in rows]

    @staticmethod
    def _request_eval_row_to_dict(row: RequestRunModel) -> dict:
        return {
            "request_id": row.request_id,
            "user_id": row.user_id,
            "session_id": row.session_id,
            "query": row.query,
            "final_answer": row.final_answer,
            "created_at": _iso(row.created_at),
        }

    @staticmethod
    def _batch_to_dict(row: MemoryEvalAnnotationBatchModel) -> dict:
        return {
            "batch_id": row.batch_id,
            "version": row.version,
            "status": row.status,
            "k": row.k,
            "created_at": _iso(row.created_at),
            "annotated_at": _iso(row.annotated_at),
            "evaluated_at": _iso(row.evaluated_at),
            "report_id": row.report_id,
        }

    @staticmethod
    def _retrieval_event_to_dict(row: MemoryRetrievalEventV2Model) -> dict:
        return {
            "event_id": row.event_id,
            "request_id": row.request_id,
            "user_id": row.user_id,
            "query": row.query,
            "embedding_latency_ms": row.embedding_latency_ms,
            "vector_search_latency_ms": row.vector_search_latency_ms,
            "retrieved_memory_ids": json.loads(row.retrieved_memory_ids_json or "[]"),
            "retrieved_scores": json.loads(row.retrieved_scores_json or "[]"),
            "retrieved_count": row.retrieved_count,
            "injected_memory_ids": json.loads(row.injected_memory_ids_json or "[]"),
            "injected_count": row.injected_count,
            "injected_text": row.injected_text,
            "injected_tokens": row.injected_tokens,
            "created_at": _iso(row.created_at),
        }

    @staticmethod
    def _extraction_event_to_dict(row: MemoryExtractionEventV2Model) -> dict:
        return {
            "event_id": row.event_id,
            "request_id": row.request_id,
            "user_id": row.user_id,
            "dispatch_latency_ms": row.dispatch_latency_ms,
            "dispatch_success": row.dispatch_success,
            "prompt_tokens": row.prompt_tokens,
            "completion_tokens": row.completion_tokens,
            "total_tokens": row.total_tokens,
            "written_memory_ids": json.loads(row.written_memory_ids_json or "[]"),
            "written_memory_texts": json.loads(row.written_memory_texts_json or "[]"),
            "created_at": _iso(row.created_at),
        }

    @staticmethod
    def _run_to_dict(row: MemoryEvalRunV2Model) -> dict:
        return {
            "run_id": row.run_id,
            "batch_id": row.batch_id,
            "request_id": row.request_id,
            "user_id": row.user_id,
            "session_id": row.session_id,
            "memory_score": round(row.memory_score, 4),
            "memory_injected_count": row.memory_injected_count,
            "memory_injected_latency_ms": row.memory_injected_latency_ms,
            "memory_injected_tokens": row.memory_injected_tokens,
            "memory_hit_at_k": _round_or_none(row.memory_hit_at_k),
            "memory_relevant_recall": _round_or_none(row.memory_relevant_recall),
            "memory_relevant_precision": _round_or_none(row.memory_relevant_precision),
            "first_relevant_rank": row.first_relevant_rank,
            "memory_stale_retrieval_rate": _round_or_none(row.memory_stale_retrieval_rate),
            "memory_answer_relevance": _round_or_none(row.memory_answer_relevance),
            "memory_extraction_precision": _round_or_none(row.memory_extraction_precision),
            "memory_extraction_latency_ms": row.memory_extraction_latency_ms,
            "memory_extraction_tokens": row.memory_extraction_tokens,
            "score_breakdown": json.loads(row.score_breakdown_json or "{}"),
            "created_at": _iso(row.created_at),
        }

    @staticmethod
    def _report_to_dict(row: MemoryEvalReportV2Model) -> dict:
        return {
            "report_id": row.report_id,
            "batch_id": row.batch_id,
            "sample_count": row.sample_count,
            "avg_memory_score": round(row.avg_memory_score, 4),
            "avg_memory_hit_at_k": _round_or_none(row.avg_memory_hit_at_k),
            "avg_memory_relevant_recall": _round_or_none(row.avg_memory_relevant_recall),
            "avg_memory_relevant_precision": _round_or_none(row.avg_memory_relevant_precision),
            "avg_first_relevant_rank": _round_or_none(row.avg_first_relevant_rank),
            "avg_memory_stale_retrieval_rate": _round_or_none(
                row.avg_memory_stale_retrieval_rate
            ),
            "avg_memory_answer_relevance": _round_or_none(row.avg_memory_answer_relevance),
            "avg_memory_extraction_precision": _round_or_none(
                row.avg_memory_extraction_precision
            ),
            "summary": json.loads(row.summary_json or "{}"),
            "created_at": _iso(row.created_at),
        }

    @staticmethod
    def _library_audit_batch_to_dict(row: MemoryLibraryAuditBatchModel) -> dict:
        return {
            "batch_id": row.batch_id,
            "status": row.status,
            "memory_count": row.memory_count,
            "created_at": _iso(row.created_at),
            "evaluated_at": _iso(row.evaluated_at),
            "report_id": row.report_id,
        }

    @staticmethod
    def _library_audit_report_to_dict(row: MemoryLibraryAuditReportModel) -> dict:
        return {
            "report_id": row.report_id,
            "batch_id": row.batch_id,
            "memory_count": row.memory_count,
            "duplicate_pair_count": row.duplicate_pair_count,
            "duplicate_memory_count": row.duplicate_memory_count,
            "duplicate_memory_ratio": _round_or_none(row.duplicate_memory_ratio),
            "conflict_pair_count": row.conflict_pair_count,
            "conflict_memory_count": row.conflict_memory_count,
            "conflict_memory_ratio": _round_or_none(row.conflict_memory_ratio),
            "summary": json.loads(row.summary_json or "{}"),
            "created_at": _iso(row.created_at),
        }


class MemoryEvalServiceV2:
    def __init__(self, settings, repository: MemoryEvalV2Repository):
        self.settings = settings
        self.repository = repository
        self.eval_root = settings.resolve_path(settings.eval_root_dir)
        self.memory_root = settings.resolve_path(settings.memory_root_dir)
        self.eval_root.mkdir(parents=True, exist_ok=True)

    def export_batch(self, *, from_request_id: str, limit: int) -> dict:
        requests = self.repository.list_exportable_requests(from_request_id, limit)
        if not requests:
            raise ValueError("NO_EXPORTABLE_REQUESTS")

        batch_id = f"memeval_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:6]}"
        batch_dir = self._batch_dir(batch_id)
        batch_dir.mkdir(parents=True, exist_ok=True)

        request_rows: list[dict] = []
        memory_ids: set[str] = set()
        for request in requests:
            retrieval = self.repository.get_memory_retrieval_event(request["request_id"])
            extraction = self.repository.get_memory_extraction_event(request["request_id"])
            if retrieval is None and (
                extraction is None or extraction.get("total_tokens") is None
            ):
                continue
            retrieved_memory_ids = retrieval["retrieved_memory_ids"] if retrieval else []
            retrieved_scores = retrieval["retrieved_scores"] if retrieval else []
            injected_memory_ids = retrieval["injected_memory_ids"] if retrieval else []
            injected_text = retrieval["injected_text"] if retrieval else ""
            request_rows.append(
                {
                    "request_id": request["request_id"],
                    "user_id": request["user_id"],
                    "session_id": request["session_id"],
                    "query": request["query"],
                    "final_answer": request["final_answer"],
                    "retrieved_memory_ids": retrieved_memory_ids,
                    "retrieved_memory_scores": retrieved_scores,
                    "injected_memory_ids": injected_memory_ids,
                    "injected_memory_text": injected_text,
                    "memory_extraction_dispatch_latency_ms": extraction["dispatch_latency_ms"]
                    if extraction
                    else None,
                    "memory_extraction_tokens": extraction["total_tokens"] if extraction else None,
                }
            )
            memory_ids.update(retrieved_memory_ids)
            memory_ids.update(injected_memory_ids)
            if extraction:
                memory_ids.update(extraction["written_memory_ids"])

        if not request_rows:
            raise ValueError("NO_EXPORTABLE_REQUESTS")

        batch_payload = {
            "batch_id": batch_id,
            "version": _VERSION,
            "status": "exported",
            "k": int(self.settings.memory_top_k),
            "created_at": _iso(_utcnow()),
        }
        memory_catalog = self._build_memory_catalog(
            user_ids={row["user_id"] for row in request_rows},
            memory_ids=memory_ids,
        )

        _dump_json(batch_dir / "batch.json", batch_payload)
        _dump_jsonl(batch_dir / "requests.jsonl", request_rows)
        _dump_jsonl(batch_dir / "memory_catalog.jsonl", memory_catalog)
        _dump_jsonl(batch_dir / "annotations.jsonl", [])

        self.repository.create_annotation_batch(
            {
                "batch_id": batch_id,
                "version": _VERSION,
                "status": "exported",
                "k": int(self.settings.memory_top_k),
                "created_at": _utcnow(),
            }
        )
        return {
            **batch_payload,
            "path": str(batch_dir),
            "request_count": len(request_rows),
            "memory_count": len(memory_catalog),
        }

    def evaluate_batch(self, *, batch_id: str) -> dict:
        batch_dir = self._batch_dir(batch_id)
        batch_path = batch_dir / "batch.json"
        if not batch_path.exists():
            raise ValueError(f"BATCH_NOT_FOUND: {batch_id}")

        batch = _load_json(batch_path)
        request_rows = _load_jsonl(batch_dir / "requests.jsonl")
        annotations = _load_jsonl(batch_dir / "annotations.jsonl")
        memory_catalog = _load_jsonl(batch_dir / "memory_catalog.jsonl")

        request_index = {row["request_id"]: row for row in request_rows}
        memory_catalog_ids = {row["memory_id"] for row in memory_catalog}
        validated_annotations = self._validate_annotations(
            request_ids=set(request_index),
            memory_catalog_ids=memory_catalog_ids,
            annotations=annotations,
        )

        self.repository.replace_annotations(batch_id, validated_annotations)

        runs: list[dict] = []
        for annotation in validated_annotations:
            request_id = annotation["request_id"]
            request_row = request_index[request_id]
            retrieval = self.repository.get_memory_retrieval_event(request_id)
            extraction = self.repository.get_memory_extraction_event(request_id)
            run = self._build_run(
                batch_k=int(batch.get("k", self.settings.memory_top_k)),
                request_row=request_row,
                annotation=annotation,
                retrieval=retrieval,
                extraction=extraction,
            )
            runs.append(run)

        self.repository.replace_eval_runs(batch_id, runs)
        for run in runs:
            self.repository.backfill_request_memory_score(run["request_id"], run["memory_score"])

        report_id = f"memreport_{uuid4().hex}"
        report = self._build_report(report_id=report_id, batch_id=batch_id, runs=runs)
        self.repository.save_eval_report(report)
        now = _utcnow()
        self.repository.update_annotation_batch(
            batch_id,
            status="evaluated",
            annotated_at=now,
            evaluated_at=now,
            report_id=report_id,
        )

        result_payload = {
            "batch_id": batch_id,
            "report_id": report_id,
            "report": report,
            "runs": runs,
        }
        _dump_json(batch_dir / "result.json", result_payload)
        batch["status"] = "evaluated"
        batch["annotated_at"] = _iso(now)
        batch["evaluated_at"] = _iso(now)
        batch["report_id"] = report_id
        _dump_json(batch_path, batch)
        return result_payload

    def get_batch(self, batch_id: str) -> dict | None:
        batch = self.repository.get_annotation_batch(batch_id)
        if batch is None:
            return None
        report = None
        if batch.get("report_id"):
            report = self.repository.get_eval_report(batch["report_id"])
        return {
            "batch": batch,
            "report": report,
        }

    def get_report(self, report_id: str) -> dict | None:
        return self.repository.get_eval_report(report_id)

    def get_request(self, request_id: str) -> dict | None:
        return self.repository.get_eval_request(request_id)

    def export_library_audit_batch(self) -> dict:
        memory_catalog = self._build_full_memory_catalog()
        batch_id = f"memlibaudit_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:6]}"
        batch_dir = self.eval_root / "memory_library_audits" / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)

        batch_payload = {
            "batch_id": batch_id,
            "status": "exported",
            "memory_count": len(memory_catalog),
            "created_at": _iso(_utcnow()),
        }
        annotation_template = {
            "duplicate_pairs": [],
            "conflict_pairs": [],
        }

        _dump_json(batch_dir / "batch.json", batch_payload)
        _dump_jsonl(batch_dir / "memory_catalog.jsonl", memory_catalog)
        _dump_json(batch_dir / "annotations.json", annotation_template)

        self.repository.create_library_audit_batch(
            {
                "batch_id": batch_id,
                "status": "exported",
                "memory_count": len(memory_catalog),
                "created_at": _utcnow(),
            }
        )
        return {
            **batch_payload,
            "path": str(batch_dir),
        }

    def evaluate_library_audit_batch(self, *, batch_id: str) -> dict:
        batch_dir = self.eval_root / "memory_library_audits" / batch_id
        batch_path = batch_dir / "batch.json"
        if not batch_path.exists():
            raise ValueError(f"BATCH_NOT_FOUND: {batch_id}")

        batch = _load_json(batch_path)
        memory_catalog = _load_jsonl(batch_dir / "memory_catalog.jsonl")
        annotations = _load_json(batch_dir / "annotations.json")
        validated = self._validate_library_audit_annotations(
            memory_catalog_ids={row["memory_id"] for row in memory_catalog},
            annotations=annotations,
        )
        report_id = f"memlibreport_{uuid4().hex}"
        report = self._build_library_audit_report(
            report_id=report_id,
            batch_id=batch_id,
            memory_count=len(memory_catalog),
            duplicate_pairs=validated["duplicate_pairs"],
            conflict_pairs=validated["conflict_pairs"],
        )
        self.repository.save_library_audit_report(report)
        now = _utcnow()
        self.repository.update_library_audit_batch(
            batch_id,
            status="evaluated",
            evaluated_at=now,
            report_id=report_id,
        )
        result_payload = {
            "batch_id": batch_id,
            "report_id": report_id,
            "report": report,
        }
        _dump_json(batch_dir / "result.json", result_payload)
        batch["status"] = "evaluated"
        batch["report_id"] = report_id
        batch["evaluated_at"] = _iso(now)
        _dump_json(batch_path, batch)
        return result_payload

    def get_library_audit_report(self, report_id: str) -> dict | None:
        return self.repository.get_library_audit_report(report_id)

    def get_dashboard_memory_stats(self) -> dict:
        memory_count = len(self._build_full_memory_catalog())
        latest_report = self.repository.get_latest_library_audit_report()
        return {
            "recorded_memory_count": memory_count,
            "memory_duplicate_ratio": (
                latest_report.get("duplicate_memory_ratio") if latest_report else None
            ),
            "memory_conflict_ratio": (
                latest_report.get("conflict_memory_ratio") if latest_report else None
            ),
            "memory_library_audit_report_id": (
                latest_report.get("report_id") if latest_report else None
            ),
        }

    def _batch_dir(self, batch_id: str) -> Path:
        return self.eval_root / "memory_batches" / batch_id

    def _library_audit_batch_dir(self, batch_id: str) -> Path:
        return self.eval_root / "memory_library_audits" / batch_id

    def _build_memory_catalog(self, *, user_ids: set[str], memory_ids: set[str]) -> list[dict]:
        catalog = self._parse_memory_files(user_ids)
        rows = [catalog[memory_id] for memory_id in memory_ids if memory_id in catalog]
        missing_ids = sorted(memory_ids - set(catalog))
        for memory_id in missing_ids:
            rows.append(
                {
                    "memory_id": memory_id,
                    "user_id": "",
                    "content": "",
                    "created_at": None,
                    "source": "unknown",
                }
            )
        rows.sort(key=lambda row: (row.get("user_id", ""), row["memory_id"]))
        return rows

    def _build_full_memory_catalog(self) -> list[dict]:
        user_ids = {
            path.name for path in self.memory_root.iterdir()
            if path.is_dir()
        } if self.memory_root.exists() else set()
        rows = list(self._parse_memory_files(user_ids).values())
        rows.sort(key=lambda row: (row.get("user_id", ""), row["memory_id"]))
        return rows

    def _parse_memory_files(self, user_ids: set[str]) -> dict[str, dict]:
        catalog: dict[str, dict] = {}
        for user_id in user_ids:
            path = self.memory_root / user_id / "MEMORY.md"
            if not path.exists():
                continue
            current: dict | None = None
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                match = _MEMORY_HEADER_RE.match(line)
                if match:
                    if current is not None:
                        catalog[current["memory_id"]] = current
                    current = {
                        "memory_id": match.group("record_id"),
                        "user_id": user_id,
                        "content": "",
                        "created_at": None,
                        "source": "",
                    }
                    continue
                if current is None or not line.startswith("- "):
                    continue
                key, _, value = line[2:].partition(":")
                normalized_key = key.strip()
                normalized_value = value.strip()
                if normalized_key == "created_at":
                    current["created_at"] = normalized_value or None
                elif normalized_key == "source":
                    current["source"] = normalized_value
                elif normalized_key == "content":
                    current["content"] = normalized_value
            if current is not None:
                catalog[current["memory_id"]] = current
        return catalog

    def _validate_annotations(
        self,
        *,
        request_ids: set[str],
        memory_catalog_ids: set[str],
        annotations: list[dict],
    ) -> list[dict]:
        if not annotations:
            raise ValueError("ANNOTATIONS_EMPTY")
        if len(annotations) != len(request_ids):
            raise ValueError("ANNOTATIONS_INCOMPLETE")

        seen_request_ids: set[str] = set()
        validated: list[dict] = []
        for annotation in annotations:
            request_id = str(annotation.get("request_id", "")).strip()
            if not request_id or request_id not in request_ids:
                raise ValueError(f"UNKNOWN_REQUEST_ID: {request_id or '<empty>'}")
            if request_id in seen_request_ids:
                raise ValueError(f"DUPLICATE_REQUEST_ID: {request_id}")
            seen_request_ids.add(request_id)

            required_list_fields = [
                "relevant_memory_ids",
                "stale_memory_ids",
                "claims",
                "expected_extracted_memories",
            ]
            for field_name in required_list_fields:
                if field_name not in annotation or not isinstance(annotation[field_name], list):
                    raise ValueError(f"INVALID_ANNOTATION_FIELD: {request_id}:{field_name}")

            claims = annotation["claims"]
            all_memory_ids = set(annotation["relevant_memory_ids"])
            all_memory_ids.update(annotation["stale_memory_ids"])
            for claim in claims:
                if not isinstance(claim, dict):
                    raise ValueError(f"INVALID_CLAIM: {request_id}")
                if not str(claim.get("text", "")).strip():
                    raise ValueError(f"CLAIM_TEXT_EMPTY: {request_id}")
                if "supported_by_retrieved_memory" not in claim:
                    raise ValueError(f"CLAIM_SUPPORT_MISSING: {request_id}")
                support_memory_ids = claim.get("support_memory_ids", [])
                if not isinstance(support_memory_ids, list):
                    raise ValueError(f"INVALID_SUPPORT_MEMORY_IDS: {request_id}")
                all_memory_ids.update(support_memory_ids)

            unknown_memory_ids = {
                memory_id
                for memory_id in all_memory_ids
                if memory_id and memory_id not in memory_catalog_ids
            }
            if unknown_memory_ids:
                joined = ",".join(sorted(unknown_memory_ids))
                raise ValueError(f"UNKNOWN_MEMORY_IDS: {request_id}:{joined}")

            validated.append(
                {
                    "request_id": request_id,
                    "relevant_memory_ids": list(annotation["relevant_memory_ids"]),
                    "stale_memory_ids": list(annotation["stale_memory_ids"]),
                    "claims": claims,
                    "expected_extracted_memories": list(annotation["expected_extracted_memories"]),
                    "expected_extracted_memory_ids": list(
                        annotation.get("expected_extracted_memory_ids", [])
                    ),
                    "annotator": str(annotation.get("annotator", "")).strip(),
                }
            )
        return validated

    def _validate_library_audit_annotations(
        self,
        *,
        memory_catalog_ids: set[str],
        annotations: dict,
    ) -> dict[str, list[dict]]:
        if not isinstance(annotations, dict):
            raise ValueError("INVALID_LIBRARY_AUDIT_ANNOTATIONS")
        duplicate_pairs = self._validate_memory_pairs(
            annotations.get("duplicate_pairs", []),
            memory_catalog_ids=memory_catalog_ids,
            field_name="duplicate_pairs",
        )
        conflict_pairs = self._validate_memory_pairs(
            annotations.get("conflict_pairs", []),
            memory_catalog_ids=memory_catalog_ids,
            field_name="conflict_pairs",
        )
        duplicate_norm = {
            tuple(sorted((pair["memory_id_1"], pair["memory_id_2"])))
            for pair in duplicate_pairs
        }
        conflict_norm = {
            tuple(sorted((pair["memory_id_1"], pair["memory_id_2"])))
            for pair in conflict_pairs
        }
        overlap = sorted(duplicate_norm & conflict_norm)
        if overlap:
            raise ValueError(f"PAIR_MARKED_AS_BOTH_DUPLICATE_AND_CONFLICT: {overlap[0][0]} {overlap[0][1]}")
        return {
            "duplicate_pairs": duplicate_pairs,
            "conflict_pairs": conflict_pairs,
        }

    def _validate_memory_pairs(
        self,
        pairs: list[dict],
        *,
        memory_catalog_ids: set[str],
        field_name: str,
    ) -> list[dict]:
        if not isinstance(pairs, list):
            raise ValueError(f"INVALID_{field_name.upper()}")
        validated: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for pair in pairs:
            if not isinstance(pair, dict):
                raise ValueError(f"INVALID_{field_name.upper()}_ITEM")
            memory_id_1 = str(pair.get("memory_id_1", "")).strip()
            memory_id_2 = str(pair.get("memory_id_2", "")).strip()
            if not memory_id_1 or not memory_id_2:
                raise ValueError(f"MISSING_{field_name.upper()}_MEMORY_ID")
            if memory_id_1 == memory_id_2:
                raise ValueError(f"SELF_{field_name.upper()}_PAIR: {memory_id_1}")
            if memory_id_1 not in memory_catalog_ids:
                raise ValueError(f"UNKNOWN_MEMORY_ID: {memory_id_1}")
            if memory_id_2 not in memory_catalog_ids:
                raise ValueError(f"UNKNOWN_MEMORY_ID: {memory_id_2}")
            normalized = tuple(sorted((memory_id_1, memory_id_2)))
            if normalized in seen:
                raise ValueError(f"DUPLICATE_{field_name.upper()}_PAIR: {normalized[0]} {normalized[1]}")
            seen.add(normalized)
            validated.append(
                {
                    "memory_id_1": memory_id_1,
                    "memory_id_2": memory_id_2,
                    "reason": str(pair.get("reason", "")).strip(),
                }
            )
        return validated

    def _build_run(
        self,
        *,
        batch_k: int,
        request_row: dict,
        annotation: dict,
        retrieval: dict | None,
        extraction: dict | None,
    ) -> dict:
        retrieved_ids = list(retrieval["retrieved_memory_ids"]) if retrieval else []
        relevant_ids = set(annotation["relevant_memory_ids"])
        retrieved_set = set(retrieved_ids)
        relevant_hit_ids = relevant_ids & retrieved_set
        has_relevant_gold = bool(relevant_ids and retrieved_ids)

        memory_hit_at_k = None
        memory_relevant_recall = None
        memory_relevant_precision = None
        first_relevant_rank = None
        if has_relevant_gold:
            top_k_ids = set(retrieved_ids[:batch_k])
            memory_hit_at_k = 1.0 if (relevant_ids & top_k_ids) else 0.0
            if relevant_ids:
                memory_relevant_recall = _safe_ratio(len(relevant_hit_ids), len(relevant_ids))
            memory_relevant_precision = _safe_ratio(len(relevant_hit_ids), len(retrieved_ids))
            for index, memory_id in enumerate(retrieved_ids, start=1):
                if memory_id in relevant_ids:
                    first_relevant_rank = index
                    break

        stale_ids = set(annotation["stale_memory_ids"])
        memory_stale_retrieval_rate = (
            _safe_ratio(len(stale_ids & retrieved_set), len(retrieved_ids))
            if retrieval
            else None
        )

        claims = annotation["claims"]
        supported_claims = sum(
            1 for claim in claims if bool(claim.get("supported_by_retrieved_memory"))
        )
        memory_answer_relevance = _safe_ratio(supported_claims, len(claims)) if claims else None

        memory_extraction_precision = None
        memory_extraction_latency_ms = None
        memory_extraction_tokens = None
        if extraction and extraction.get("dispatch_success"):
            memory_extraction_latency_ms = extraction.get("dispatch_latency_ms")
            memory_extraction_tokens = extraction.get("total_tokens")
            written_ids = set(extraction.get("written_memory_ids", []))
            written_texts = _normalized_text_set(extraction.get("written_memory_texts", []))
            expected_ids = {
                str(memory_id).strip()
                for memory_id in annotation.get("expected_extracted_memory_ids", [])
                if str(memory_id).strip()
            }
            expected_texts = _normalized_text_set(annotation["expected_extracted_memories"])
            if written_ids and expected_ids:
                correct_written = len(written_ids & expected_ids)
                memory_extraction_precision = _safe_ratio(correct_written, len(written_ids))
            else:
                written_count = len(written_texts)
                correct_written = len(written_texts & expected_texts)
                if written_count == 0:
                    memory_extraction_precision = 1.0 if not expected_texts else 0.0
                else:
                    memory_extraction_precision = _safe_ratio(correct_written, written_count)

        s_rank = 0.0
        if first_relevant_rank:
            s_rank = 1.0 / math.log2(first_relevant_rank + 1)
        s_fresh = (
            1.0 - min(max(memory_stale_retrieval_rate, 0.0), 1.0)
            if memory_stale_retrieval_rate is not None
            else None
        )

        s_retrieval = None
        s_effective = None
        if has_relevant_gold and s_fresh is not None:
            s_retrieval = (
                0.20 * float(memory_hit_at_k or 0.0)
                + 0.25 * float(memory_relevant_recall or 0.0)
                + 0.25 * float(memory_relevant_precision or 0.0)
                + 0.15 * s_rank
                + 0.15 * s_fresh
            )
            if memory_answer_relevance is not None:
                s_effective = (
                    2.0 * s_retrieval * memory_answer_relevance
                    / (s_retrieval + memory_answer_relevance + 1e-8)
                )

        s_inject = None
        if retrieval is not None:
            c_target = max(batch_k, 1)
            s_count = math.exp(-max(0, int(retrieval["injected_count"]) - c_target) / c_target)
            s_inj_lat = math.exp(
                -float(retrieval["vector_search_latency_ms"] or 0) / _L_INJ_REF_MS
            )
            s_inj_tok = math.exp(-float(retrieval["injected_tokens"] or 0) / _T_INJ_REF)
            s_inject = (0.20 * s_count) + (0.40 * s_inj_lat) + (0.40 * s_inj_tok)

        s_extract = None
        if (
            memory_extraction_precision is not None
            and memory_extraction_latency_ms is not None
            and memory_extraction_tokens is not None
        ):
            s_ext_lat = math.exp(-float(memory_extraction_latency_ms) / _L_EXT_REF_MS)
            s_ext_tok = math.exp(-float(memory_extraction_tokens) / _T_EXT_REF)
            s_extract = (
                0.60 * memory_extraction_precision
                + 0.20 * s_ext_lat
                + 0.20 * s_ext_tok
            )
        weighted_components: list[tuple[float, float]] = []
        if s_inject is not None:
            weighted_components.append((0.15, s_inject))
        if s_effective is not None:
            weighted_components.append((0.65, s_effective))
        if s_extract is not None:
            weighted_components.append((0.20, s_extract))
        if not weighted_components:
            raise ValueError(f"MEMORY_SCORE_COMPONENTS_UNAVAILABLE: {request_row['request_id']}")
        total_weight = sum(weight for weight, _ in weighted_components)
        memory_score = sum(weight * score for weight, score in weighted_components) / total_weight

        score_breakdown = {
            "s_rank": _round_or_none(s_rank),
            "s_fresh": _round_or_none(s_fresh),
            "s_retrieval": _round_or_none(s_retrieval),
            "s_effective": _round_or_none(s_effective),
            "s_inject": _round_or_none(s_inject),
            "s_extract": _round_or_none(s_extract),
            "components": {
                "retrieval_used": s_effective is not None,
                "injection_used": s_inject is not None,
                "extraction_used": s_extract is not None,
            },
        }

        return {
            "request_id": request_row["request_id"],
            "user_id": request_row["user_id"],
            "session_id": request_row["session_id"],
            "memory_score": round(memory_score, 4),
            "memory_injected_count": int(retrieval["injected_count"]) if retrieval else 0,
            "memory_injected_latency_ms": int(retrieval["vector_search_latency_ms"])
            if retrieval
            else 0,
            "memory_injected_tokens": int(retrieval["injected_tokens"]) if retrieval else 0,
            "memory_hit_at_k": _round_or_none(memory_hit_at_k),
            "memory_relevant_recall": _round_or_none(memory_relevant_recall),
            "memory_relevant_precision": _round_or_none(memory_relevant_precision),
            "first_relevant_rank": first_relevant_rank,
            "memory_stale_retrieval_rate": _round_or_none(memory_stale_retrieval_rate),
            "memory_answer_relevance": _round_or_none(memory_answer_relevance),
            "memory_extraction_precision": _round_or_none(memory_extraction_precision),
            "memory_extraction_latency_ms": memory_extraction_latency_ms,
            "memory_extraction_tokens": memory_extraction_tokens,
            "score_breakdown": score_breakdown,
        }

    def _build_report(self, *, report_id: str, batch_id: str, runs: list[dict]) -> dict:
        sorted_runs = sorted(runs, key=lambda item: item["memory_score"])
        return {
            "report_id": report_id,
            "batch_id": batch_id,
            "sample_count": len(runs),
            "avg_memory_score": round(
                sum(item["memory_score"] for item in runs) / max(len(runs), 1),
                4,
            ),
            "avg_memory_hit_at_k": _mean_or_none([item["memory_hit_at_k"] for item in runs]),
            "avg_memory_relevant_recall": _mean_or_none(
                [item["memory_relevant_recall"] for item in runs]
            ),
            "avg_memory_relevant_precision": _mean_or_none(
                [item["memory_relevant_precision"] for item in runs]
            ),
            "avg_first_relevant_rank": _mean_or_none(
                [
                    float(item["first_relevant_rank"])
                    for item in runs
                    if item["first_relevant_rank"] is not None
                ]
            ),
            "avg_memory_stale_retrieval_rate": _mean_or_none(
                [item["memory_stale_retrieval_rate"] for item in runs]
            ),
            "avg_memory_answer_relevance": _mean_or_none(
                [item["memory_answer_relevance"] for item in runs]
            ),
            "avg_memory_extraction_precision": _mean_or_none(
                [item["memory_extraction_precision"] for item in runs]
            ),
            "summary": {
                "lowest_scoring_requests": [
                    {
                        "request_id": item["request_id"],
                        "memory_score": item["memory_score"],
                    }
                    for item in sorted_runs[:10]
                ],
            },
        }

    def _build_library_audit_report(
        self,
        *,
        report_id: str,
        batch_id: str,
        memory_count: int,
        duplicate_pairs: list[dict],
        conflict_pairs: list[dict],
    ) -> dict:
        duplicate_memory_ids = sorted(
            {
                memory_id
                for pair in duplicate_pairs
                for memory_id in (pair["memory_id_1"], pair["memory_id_2"])
            }
        )
        conflict_memory_ids = sorted(
            {
                memory_id
                for pair in conflict_pairs
                for memory_id in (pair["memory_id_1"], pair["memory_id_2"])
            }
        )
        return {
            "report_id": report_id,
            "batch_id": batch_id,
            "memory_count": memory_count,
            "duplicate_pair_count": len(duplicate_pairs),
            "duplicate_memory_count": len(duplicate_memory_ids),
            "duplicate_memory_ratio": _round_or_none(
                _safe_ratio(len(duplicate_memory_ids), memory_count)
            ),
            "conflict_pair_count": len(conflict_pairs),
            "conflict_memory_count": len(conflict_memory_ids),
            "conflict_memory_ratio": _round_or_none(
                _safe_ratio(len(conflict_memory_ids), memory_count)
            ),
            "summary": {
                "duplicate_pairs": duplicate_pairs,
                "conflict_pairs": conflict_pairs,
            },
        }
