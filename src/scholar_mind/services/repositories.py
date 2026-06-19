from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from statistics import quantiles
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from scholar_mind.db.models import (
    ConversationMetricModel,
    EvalReportModel,
    MemoryEvalRunV2Model,
    MemoryExtractionEventV2Model,
    MemoryMetricModel,
    MemoryRecordModel,
    MemoryRetrievalEventV2Model,
    RequestRunModel,
    SessionModel,
)
from scholar_mind.eval.answer_quality import compute_answer_quality_score
from scholar_mind.models.domain import (
    ReportSummary,
    SessionInfo,
)
from scholar_mind.utils.text import top_keywords


class SessionRepository:
    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    def create_or_get(self, user_id: str, session_id: str) -> SessionInfo:
        with self.session_factory() as session:
            row = session.get(SessionModel, session_id)
            if row is None:
                row = SessionModel(
                    session_id=session_id,
                    user_id=user_id,
                    created_at=datetime.now(UTC),
                    message_count=0,
                    topics_json="[]",
                    memory_context_loaded=False,
                    last_state_json="{}",
                )
                session.add(row)
                session.commit()
                session.refresh(row)
            return self._to_model(row)

    def get(self, session_id: str) -> SessionInfo | None:
        with self.session_factory() as session:
            row = session.get(SessionModel, session_id)
            return None if row is None else self._to_model(row)

    def get_last_state(self, session_id: str) -> dict[str, Any]:
        with self.session_factory() as session:
            row = session.get(SessionModel, session_id)
            if row is None or not row.last_state_json:
                return {}
            state = json.loads(row.last_state_json)
            if state.get("messages"):
                from scholar_mind.utils.messages import deserialize_messages

                state["messages"] = deserialize_messages(state["messages"])
            return state

    def update_from_state(
        self, user_id: str, session_id: str, state: dict[str, Any]
    ) -> SessionInfo:
        with self.session_factory() as session:
            row = session.get(SessionModel, session_id)
            if row is None:
                row = SessionModel(
                    session_id=session_id,
                    user_id=user_id,
                    created_at=datetime.now(UTC),
                    topics_json="[]",
                )
                session.add(row)

            row.user_id = user_id
            row.message_count = len(state.get("messages", []))
            row.topics_json = json.dumps(top_keywords(state.get("query", ""), limit=4))
            row.memory_context_loaded = bool(state.get("memory_context"))
            row.last_state_json = json.dumps(state, default=str)
            session.commit()
            session.refresh(row)
            return self._to_model(row)

    def close(self, session_id: str) -> SessionInfo | None:
        with self.session_factory() as session:
            row = session.get(SessionModel, session_id)
            if row is None:
                return None
            row.closed_at = datetime.now(UTC)
            session.commit()
            session.refresh(row)
            return self._to_model(row)

    @staticmethod
    def _to_model(row: SessionModel) -> SessionInfo:
        return SessionInfo(
            session_id=row.session_id,
            user_id=row.user_id,
            created_at=row.created_at,
            closed_at=row.closed_at,
            message_count=row.message_count,
            topics_discussed=json.loads(row.topics_json or "[]"),
            memory_context_loaded=row.memory_context_loaded,
        )


class EvalRepository:
    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    def save_report(
        self, report_id: str, report_type: str, config: dict[str, Any], results: dict[str, Any]
    ) -> None:
        with self.session_factory() as session:
            session.add(
                EvalReportModel(
                    report_id=report_id,
                    type=report_type,
                    created_at=datetime.now(UTC),
                    config_json=json.dumps(config),
                    results_json=json.dumps(results),
                )
            )
            session.commit()

    def get_report(self, report_id: str) -> ReportSummary | None:
        with self.session_factory() as session:
            row = session.get(EvalReportModel, report_id)
            if row is None:
                return None
            return ReportSummary(
                report_id=row.report_id,
                type=row.type,
                created_at=row.created_at,
                config=json.loads(row.config_json),
                results=json.loads(row.results_json),
            )

    def list_reports_by_type(
        self, report_type: str, limit: int = 20
    ) -> list[ReportSummary]:
        """List recent reports of a given type, newest first."""
        with self.session_factory() as session:
            rows = session.scalars(
                select(EvalReportModel)
                .where(EvalReportModel.type == report_type)
                .order_by(EvalReportModel.created_at.desc())
                .limit(limit)
            ).all()
            return [
                ReportSummary(
                    report_id=row.report_id,
                    type=row.type,
                    created_at=row.created_at,
                    config=json.loads(row.config_json),
                    results=json.loads(row.results_json),
                )
                for row in rows
            ]


class MetricsRepository:
    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    def record_round(
        self,
        request_id: str,
        user_id: str,
        session_id: str,
        query_type: str,
        success: bool,
        retrieval_latency_ms: int,
        latency_ms: int,
        citations_count: int,
        retrieved_chunks_count: int,
        output_length: int,
        agent_path: list[str],
        error_summary: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        with self.session_factory() as session:
            session.merge(
                ConversationMetricModel(
                    request_id=request_id,
                    user_id=user_id,
                    session_id=session_id,
                    query_type=query_type,
                    success=success,
                    retrieval_latency_ms=retrieval_latency_ms,
                    latency_ms=latency_ms,
                    citations_count=citations_count,
                    retrieved_chunks_count=retrieved_chunks_count,
                    output_length=output_length,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    agent_path_json=json.dumps(agent_path),
                    error_summary=error_summary,
                    created_at=datetime.now(UTC),
                )
            )
            session.commit()

    def record_memory_run(
        self,
        *,
        user_id: str,
        success: bool,
        extracted_count: int,
        latency_ms: int,
        error_summary: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        with self.session_factory() as session:
            session.add(
                MemoryMetricModel(
                    metric_id=f"mem_metric_{uuid4().hex}",
                    user_id=user_id,
                    success=success,
                    extracted_count=extracted_count,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    error_summary=error_summary,
                    created_at=datetime.now(UTC),
                )
            )
            session.commit()

    def health_stats(self) -> dict[str, Any]:
        with self.session_factory() as session:
            memory_records = session.query(MemoryRecordModel).count()
            active_sessions = (
                session.query(SessionModel).filter(SessionModel.closed_at.is_(None)).count()
            )
            return {
                "memory_records": memory_records,
                "active_sessions": active_sessions,
            }

    def latency_p95(self, report_type: str) -> float:
        with self.session_factory() as session:
            latencies = [
                row.latency_ms
                for row in session.scalars(
                    select(ConversationMetricModel).where(
                        ConversationMetricModel.query_type == report_type
                    )
                ).all()
            ]
        if len(latencies) < 2:
            return float(latencies[0]) if latencies else 0.0
        return float(quantiles(latencies, n=20)[-1])


# ---------------------------------------------------------------------------
# Request audit repository (Document 23)
# ---------------------------------------------------------------------------


class OnlineEvalRepository:
    """Repository for memory-only request audit data."""

    HEALTH_SCORE_PENALTIES = {
        "has_error": 0.45,
        "timeout": 0.25,
        "has_fallback": 0.20,
        "has_retry": 0.10,
    }

    OVERALL_SCORE_WEIGHTS = {
        "answer_quality_score": 0.50,
        "memory_score": 0.50,
    }

    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    def save_request_run(self, payload: dict[str, Any]) -> None:
        execution_health = dict(payload.get("execution_health", {}))
        execution_health_score = self._compute_execution_health_score(
            has_error=execution_health.get("has_error", False),
            timeout=execution_health.get("timeout", False),
            has_fallback=payload.get(
                "has_fallback",
                execution_health.get("has_fallback", False),
            ),
            has_retry=payload.get("has_retry", execution_health.get("has_retry", False)),
        )
        execution_health["execution_health_score"] = execution_health_score
        with self.session_factory() as session:
            session.merge(
                RequestRunModel(
                    request_id=payload["request_id"],
                    session_id=payload.get("session_id", ""),
                    user_id=payload.get("user_id", ""),
                    query=payload.get("query", ""),
                    query_type=payload.get("query_type", ""),
                    final_answer=payload.get("final_answer", ""),
                    memory_score=payload.get("memory_score"),
                    execution_health_score=execution_health_score,
                    has_retry=payload.get("has_retry", execution_health.get("has_retry", False)),
                    has_fallback=payload.get(
                        "has_fallback",
                        execution_health.get("has_fallback", False),
                    ),
                    execution_health_json=json.dumps(execution_health),
                    runtime_metrics_json=json.dumps(payload.get("runtime_metrics", {})),
                    agent_trace_json=json.dumps(payload.get("agent_trace", [])),
                    agent_events_json=json.dumps(payload.get("agent_events", [])),
                    answer_event_json=json.dumps(payload.get("answer_event", {})),
                    created_at=datetime.now(UTC),
                )
            )
            session.commit()

    def get_request_eval(self, request_id: str) -> dict[str, Any] | None:
        with self.session_factory() as session:
            row = session.get(RequestRunModel, request_id)
            if row is None:
                return None
            return self._request_to_dict(row)

    def get_request_diagnosis(self, request_id: str) -> dict[str, Any] | None:
        request = self.get_request_eval(request_id)
        if request is None:
            return None
        return self._request_diagnosis_from_scores(request)

    def get_session_evals(self, session_id: str) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(RequestRunModel)
                .where(RequestRunModel.session_id == session_id)
                .order_by(RequestRunModel.created_at.desc())
            ).all()
            return [self._request_to_dict(row) for row in rows]

    def get_request_events(self, request_id: str) -> dict[str, Any]:
        with self.session_factory() as session:
            request = session.get(RequestRunModel, request_id)
            return {
                "request": self._request_to_dict(request) if request else None,
                "event_summary": {
                    "request_id": request_id,
                    "memory_event_count": 0,
                },
                "memory_events": [],
            }

    def get_recent_complete_requests(
        self,
        *,
        hours: int = 168,
        query_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        from datetime import timedelta

        since = datetime.now(UTC) - timedelta(hours=hours)
        with self.session_factory() as session:
            stmt = (
                select(RequestRunModel)
                .where(RequestRunModel.created_at >= since, RequestRunModel.final_answer != "")
                .order_by(RequestRunModel.created_at.desc())
                .limit(limit)
            )
            if query_type:
                stmt = stmt.where(RequestRunModel.query_type == query_type)
            rows = list(session.scalars(stmt).all())
            return [self._request_to_dict(row) for row in rows]

    def get_dashboard_stats(
        self,
        *,
        hours: int = 24,
        query_type: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        from datetime import timedelta

        since = datetime.now(UTC) - timedelta(hours=hours)
        with self.session_factory() as session:
            rows = self._filtered_request_rows(
                session, since=since, query_type=query_type, user_id=user_id
            )
            if not rows:
                return {
                    "total_requests": 0,
                    "avg_memory_score": 0.0,
                    "avg_overall_score": None,
                    "avg_answer_quality_score": 0.0,
                    "avg_latency_ms": 0,
                    "avg_total_tokens": 0,
                    "has_error_count": 0,
                    "timeout_count": 0,
                    "has_retry_count": 0,
                    "has_fallback_count": 0,
                    "low_score_count": 0,
                    "by_query_type": {},
                    "recent_scores": [],
                }
            details = [self._request_to_dict(row) for row in rows]
        memory_scores = [
            item["memory_score"] for item in details if item.get("memory_score") is not None
        ]
        answer_quality_scores = [
            item["answer_quality_score"]
            for item in details
            if item.get("answer_quality_score") is not None
        ]
        overall_scores = [
            item["overall_score"] for item in details if item.get("overall_score") is not None
        ]
        latencies = [item["runtime_metrics"].get("latency_ms", 0) for item in details]
        token_counts = [item["runtime_metrics"].get("total_tokens", 0) for item in details]
        by_query_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in details:
            by_query_type[item.get("query_type", "unknown")].append(item)
        return {
            "total_requests": len(details),
            "avg_memory_score": (
                round(sum(memory_scores) / len(memory_scores), 4) if memory_scores else 0.0
            ),
            "avg_overall_score": (
                round(sum(overall_scores) / len(overall_scores), 4) if overall_scores else 0.0
            ),
            "avg_answer_quality_score": (
                round(sum(answer_quality_scores) / len(answer_quality_scores), 4)
                if answer_quality_scores
                else 0.0
            ),
            "avg_latency_ms": int(sum(latencies) / len(latencies)) if latencies else 0,
            "avg_total_tokens": int(sum(token_counts) / len(token_counts)) if token_counts else 0,
            "has_error_count": sum(
                1 for item in details if item["execution_health"].get("has_error")
            ),
            "timeout_count": sum(1 for item in details if item["execution_health"].get("timeout")),
            "has_retry_count": sum(1 for item in details if item.get("has_retry")),
            "has_fallback_count": sum(1 for item in details if item.get("has_fallback")),
            "low_score_count": sum(
                1
                for item in details
                if item.get("overall_score") is not None and item["overall_score"] <= 0.4
            ),
            "by_query_type": {
                key: {"count": len(items)} for key, items in by_query_type.items()
            },
            "recent_scores": [],
        }

    def get_low_score_requests(
        self, threshold: float = 0.4, limit: int = 20, offset: int = 0
    ) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(RequestRunModel)
                .order_by(RequestRunModel.created_at.desc())
            ).all()
            requests = [self._request_to_dict(row) for row in rows]
            filtered = [
                item
                for item in requests
                if item.get("overall_score") is not None and item["overall_score"] <= threshold
            ]
            return filtered[offset : offset + limit]

    def get_all_requests(self, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(RequestRunModel)
                .order_by(RequestRunModel.created_at.desc())
                .offset(offset)
                .limit(limit)
            ).all()
            return [self._request_to_dict(row) for row in rows]

    def get_score_trend(
        self,
        hours: int = 168,
        granularity: str = "hourly",
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        from datetime import timedelta

        since = datetime.now(UTC) - timedelta(hours=hours)
        with self.session_factory() as session:
            rows = self._filtered_request_rows(session, since=since, user_id=user_id)
        buckets: dict[str, list[RequestRunModel]] = defaultdict(list)
        for row in rows:
            buckets[self._time_bucket_label(row.created_at, granularity)].append(row)
        return [
            {
                "period": period,
                "count": len(items),
                "avg_overall_score": self._avg(
                    [
                        self._compute_overall_score(
                            answer_quality_score=compute_answer_quality_score(
                                query=item.query,
                                query_type=item.query_type,
                                final_answer=item.final_answer,
                            ),
                            memory_score=item.memory_score,
                        )
                        for item in items
                    ]
                ),
                "avg_memory_score": self._avg([item.memory_score for item in items]),
                "avg_answer_quality_score": self._avg(
                    [
                        compute_answer_quality_score(
                            query=item.query,
                            query_type=item.query_type,
                            final_answer=item.final_answer,
                        )
                        for item in items
                    ]
                ),
            }
            for period, items in sorted(buckets.items())
        ]

    def get_distinct_users(self) -> list[str]:
        with self.session_factory() as session:
            rows = session.scalars(select(RequestRunModel.user_id).distinct()).all()
            return sorted(set(rows))

    def get_eval_rows_for_export(
        self, *, hours: int = 168, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        from datetime import timedelta

        since = datetime.now(UTC) - timedelta(hours=hours)
        with self.session_factory() as session:
            rows = self._filtered_request_rows(session, since=since, user_id=user_id)
            return [
                self._flatten_export_row(
                    row,
                    self._latest_memory_eval_run(session, row.request_id),
                    self._latest_memory_retrieval_event(session, row.request_id),
                    self._memory_extraction_event(session, row.request_id),
                )
                for row in rows
            ]

    @staticmethod
    def _filtered_request_rows(
        session: Session,
        *,
        since: datetime,
        query_type: str | None = None,
        user_id: str | None = None,
    ) -> list[RequestRunModel]:
        stmt = select(RequestRunModel).where(RequestRunModel.created_at >= since)
        if query_type:
            stmt = stmt.where(RequestRunModel.query_type == query_type)
        if user_id:
            stmt = stmt.where(RequestRunModel.user_id == user_id)
        return list(session.scalars(stmt.order_by(RequestRunModel.created_at.desc())).all())

    @staticmethod
    def _latest_memory_eval_run(session: Session, request_id: str) -> MemoryEvalRunV2Model | None:
        return session.scalars(
            select(MemoryEvalRunV2Model)
            .where(MemoryEvalRunV2Model.request_id == request_id)
            .order_by(MemoryEvalRunV2Model.created_at.desc())
        ).first()

    @staticmethod
    def _latest_memory_retrieval_event(
        session: Session, request_id: str
    ) -> MemoryRetrievalEventV2Model | None:
        return session.scalars(
            select(MemoryRetrievalEventV2Model)
            .where(MemoryRetrievalEventV2Model.request_id == request_id)
            .order_by(MemoryRetrievalEventV2Model.created_at.desc())
        ).first()

    @staticmethod
    def _memory_extraction_event(
        session: Session, request_id: str
    ) -> MemoryExtractionEventV2Model | None:
        return session.scalars(
            select(MemoryExtractionEventV2Model).where(
                MemoryExtractionEventV2Model.request_id == request_id
            )
        ).first()

    @classmethod
    def _request_to_dict(
        cls,
        row: RequestRunModel,
    ) -> dict[str, Any]:
        runtime = json.loads(row.runtime_metrics_json or "{}")
        execution_health = json.loads(row.execution_health_json or "{}")
        health_score = cls._compute_execution_health_score(
            has_error=execution_health.get("has_error", False),
            timeout=execution_health.get("timeout", False),
            has_fallback=row.has_fallback,
            has_retry=row.has_retry,
        )
        execution_health["execution_health_score"] = health_score
        answer_quality_score = compute_answer_quality_score(
            query=row.query,
            query_type=row.query_type,
            final_answer=row.final_answer,
        )
        memory_used = row.memory_score is not None
        used_modules = {
            "answer": answer_quality_score is not None,
            "memory": memory_used,
        }
        overall_score = cls._compute_overall_score(
            answer_quality_score=answer_quality_score,
            memory_score=row.memory_score if memory_used else None,
        )
        return {
            "request_id": row.request_id,
            "session_id": row.session_id,
            "user_id": row.user_id,
            "query": row.query,
            "query_type": row.query_type,
            "final_answer": row.final_answer,
            "memory_score": row.memory_score,
            "overall_score": overall_score,
            "answer_quality_score": answer_quality_score,
            "used_modules": used_modules,
            "has_retry": row.has_retry,
            "has_fallback": row.has_fallback,
            "execution_health_score": health_score,
            "memory_metrics": {},
            "execution_health": execution_health,
            "runtime_metrics": runtime,
            "agent_trace": json.loads(row.agent_trace_json or "[]"),
            "agent_events": json.loads(row.agent_events_json or "[]"),
            "answer_event": json.loads(row.answer_event_json or "{}"),
            "created_at": cls._datetime_to_local_iso(row.created_at),
        }

    @classmethod
    def _request_diagnosis_from_scores(cls, request: dict[str, Any]) -> dict[str, Any]:
        scores = {
            "overall_score": request.get("overall_score"),
            "memory_score": request.get("memory_score"),
            "answer_quality_score": request.get("answer_quality_score"),
        }
        used_modules = request.get("used_modules") or {}
        issues: list[str] = []
        strengths: list[str] = []
        recommendations: list[str] = []

        if used_modules.get("memory"):
            cls._add_score_diagnosis(
                label="Memory",
                score=scores["memory_score"],
                issues=issues,
                strengths=strengths,
                recommendations=recommendations,
                missing_recommendation="Run Memory evaluation if this request should use memory.",
                low_issue="memory retrieval or memory use quality needs review.",
                low_recommendation=(
                    "Check whether relevant memories were retrieved and used correctly."
                ),
                strong_detail="memory behavior is currently healthy.",
            )
        if used_modules.get("answer"):
            cls._add_score_diagnosis(
                label="Answer",
                score=scores["answer_quality_score"],
                issues=issues,
                strengths=strengths,
                recommendations=recommendations,
                missing_recommendation=(
                    "Check whether the final answer is empty or failed during generation."
                ),
                low_issue="the final answer may miss user intent, coverage, structure, or clarity.",
                low_recommendation=(
                    "Review the final answer against the query for coverage, specificity, "
                    "and format."
                ),
                strong_detail="the final answer is aligned, specific, and clear.",
            )

        if not issues and not strengths and not recommendations:
            recommendations.append("No score-based diagnosis is available for this request.")

        return {
            "request_id": request.get("request_id"),
            "scores": scores,
            "used_modules": used_modules,
            "issues": issues,
            "strengths": strengths,
            "recommendations": recommendations,
        }

    @classmethod
    def _compute_overall_score(
        cls,
        *,
        answer_quality_score: float | None,
        memory_score: float | None,
    ) -> float | None:
        scores = {
            "answer_quality_score": answer_quality_score,
            "memory_score": memory_score,
        }
        weighted = [
            (cls.OVERALL_SCORE_WEIGHTS[name], float(score))
            for name, score in scores.items()
            if score is not None
        ]
        if not weighted:
            return None
        total_weight = sum(weight for weight, _ in weighted)
        return round(sum(weight * score for weight, score in weighted) / total_weight, 4)

    @classmethod
    def _compute_execution_health_score(
        cls,
        *,
        has_error: bool,
        timeout: bool,
        has_fallback: bool,
        has_retry: bool,
    ) -> float:
        score = 1.0
        score -= cls.HEALTH_SCORE_PENALTIES["has_error"] * float(bool(has_error))
        score -= cls.HEALTH_SCORE_PENALTIES["timeout"] * float(bool(timeout))
        score -= cls.HEALTH_SCORE_PENALTIES["has_fallback"] * float(bool(has_fallback))
        score -= cls.HEALTH_SCORE_PENALTIES["has_retry"] * float(bool(has_retry))
        return round(min(max(score, 0.0), 1.0), 4)

    @staticmethod
    def _add_score_diagnosis(
        *,
        label: str,
        score: float | None,
        issues: list[str],
        strengths: list[str],
        recommendations: list[str],
        missing_recommendation: str,
        low_issue: str,
        low_recommendation: str,
        strong_detail: str,
    ) -> None:
        if score is None:
            issues.append(f"{label} score is not available for this request.")
            recommendations.append(missing_recommendation)
            return

        value = float(score)
        if value < 0.60:
            issues.append(f"{label} score is low ({value:.2f}); {low_issue}")
            recommendations.append(low_recommendation)
        elif value >= 0.75:
            strengths.append(f"{label} score is strong ({value:.2f}); {strong_detail}")
        else:
            recommendations.append(
                f"{label} score is moderate ({value:.2f}); monitor this request if it is important."
            )

    @classmethod
    def _memory_eval_run_to_dict(cls, row: MemoryEvalRunV2Model | None) -> dict[str, Any] | None:
        if row is None:
            return None
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
            "memory_hit_at_k": cls._round_or_none(row.memory_hit_at_k),
            "memory_relevant_recall": cls._round_or_none(row.memory_relevant_recall),
            "memory_relevant_precision": cls._round_or_none(row.memory_relevant_precision),
            "first_relevant_rank": row.first_relevant_rank,
            "memory_stale_retrieval_rate": cls._round_or_none(row.memory_stale_retrieval_rate),
            "memory_answer_relevance": cls._round_or_none(row.memory_answer_relevance),
            "memory_extraction_precision": cls._round_or_none(row.memory_extraction_precision),
            "memory_extraction_latency_ms": row.memory_extraction_latency_ms,
            "memory_extraction_tokens": row.memory_extraction_tokens,
            "score_breakdown": json.loads(row.score_breakdown_json or "{}"),
            "created_at": cls._datetime_to_local_iso(row.created_at),
        }

    @classmethod
    def _memory_retrieval_event_to_dict(
        cls, row: MemoryRetrievalEventV2Model | None
    ) -> dict[str, Any] | None:
        if row is None:
            return None
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
            "created_at": cls._datetime_to_local_iso(row.created_at),
        }

    @classmethod
    def _memory_extraction_event_to_dict(
        cls, row: MemoryExtractionEventV2Model | None
    ) -> dict[str, Any] | None:
        if row is None:
            return None
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
            "created_at": cls._datetime_to_local_iso(row.created_at),
        }

    @staticmethod
    def _round_or_none(value: float | None) -> float | None:
        return round(float(value), 4) if value is not None else None

    @classmethod
    def _flatten_export_row(
        cls,
        row: RequestRunModel,
        memory_run: MemoryEvalRunV2Model | None = None,
        memory_retrieval: MemoryRetrievalEventV2Model | None = None,
        memory_extraction: MemoryExtractionEventV2Model | None = None,
    ) -> dict[str, Any]:
        payload = cls._request_to_dict(row)
        runtime = payload["runtime_metrics"]
        health = payload["execution_health"]
        total_latency_ms = health.get("total_latency_ms", runtime.get("latency_ms", 0))
        request_overview = {
            "request_id": payload["request_id"],
            "user_id": payload["user_id"],
            "session_id": payload["session_id"],
            "query_type": payload["query_type"],
            "query": payload["query"],
            "final_answer": payload["final_answer"],
            "total_latency_ms": total_latency_ms,
            "execution_health_score": payload["execution_health_score"],
            "prompt_tokens": runtime.get("prompt_tokens", 0),
            "completion_tokens": runtime.get("completion_tokens", 0),
            "total_tokens": runtime.get("total_tokens", 0),
            "overall_score": payload["overall_score"],
            "answer_quality_score": payload["answer_quality_score"],
            "has_error": health.get("has_error", False),
            "has_retry": payload["has_retry"],
            "has_fallback": payload["has_fallback"],
            "timeout": health.get("timeout", False),
            "created_at": payload["created_at"],
        }
        memory_data = {
            "run": cls._memory_eval_run_to_dict(memory_run),
            "retrieval_event": cls._memory_retrieval_event_to_dict(memory_retrieval),
            "extraction_event": cls._memory_extraction_event_to_dict(memory_extraction),
        }
        return {
            "request_overview": request_overview,
            "memory_data": memory_data,
            "request_overview_json": json.dumps(request_overview, ensure_ascii=False),
            "memory_data_json": json.dumps(memory_data, ensure_ascii=False),
        }

    @classmethod
    def _time_bucket_label(cls, dt: datetime, granularity: str) -> str:
        dt = cls._to_local_datetime(dt)
        if granularity == "daily":
            return dt.strftime("%Y-%m-%d")
        if granularity == "weekly":
            iso = dt.isocalendar()
            return f"{iso[0]}-W{iso[1]:02d}"
        return dt.strftime("%Y-%m-%dT%H:00")

    @staticmethod
    def _local_timezone():
        return datetime.now().astimezone().tzinfo or UTC

    @classmethod
    def _to_local_datetime(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(cls._local_timezone())

    @classmethod
    def _datetime_to_local_iso(cls, value: datetime | None) -> str | None:
        localized = cls._to_local_datetime(value)
        return localized.isoformat() if localized else None

    @staticmethod
    def _avg(values: list[float | None]) -> float:
        scored = [float(value) for value in values if value is not None]
        return round(sum(scored) / len(scored), 4) if scored else 0.0
