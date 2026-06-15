"""Persistence helpers for Document 23 RAG evaluation datasets, runs, and results."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from statistics import mean
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from scholar_mind.db.models import (
    RagEvalCaseModel,
    RagEvalResultV2Model,
    RagEvalRunV2Model,
)
from scholar_mind.models.rag_eval_models import (
    RagEvalAggregate,
    RagEvalCase,
    RagEvalResult,
    RagEvalRunSummary,
    RagLatencyStats,
    RagMetricStats,
    RagScoreStats,
)

MAX_STORED_CONTEXTS = 10
MAX_STORED_CONTEXT_CHARS = 2000


class RagEvalRepository:
    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    def upsert_cases(self, cases: list[RagEvalCase]) -> None:
        now = _utcnow()
        with self.session_factory() as session:
            for case in cases:
                row = session.get(RagEvalCaseModel, case.case_id)
                created_at = row.created_at if row is not None else case.created_at
                session.merge(
                    RagEvalCaseModel(
                        case_id=case.case_id,
                        user_input=case.user_input,
                        reference=case.reference,
                        required_points_json=json.dumps(case.required_points, ensure_ascii=False),
                        tags_json=json.dumps(case.tags, ensure_ascii=False),
                        expected_source_ids_json=json.dumps(
                            case.expected_source_ids, ensure_ascii=False
                        ),
                        created_at=created_at,
                        updated_at=now,
                    )
                )
            session.commit()

    def list_cases(self, *, limit: int | None = None) -> list[RagEvalCase]:
        with self.session_factory() as session:
            stmt = select(RagEvalCaseModel).order_by(RagEvalCaseModel.case_id)
            if limit is not None:
                stmt = stmt.limit(limit)
            return [self._case_to_model(row) for row in session.scalars(stmt).all()]

    def create_run(
        self,
        *,
        dataset_name: str,
        strategies: list[str],
        metrics: list[str],
        ragas_model: str,
        embedding_model: str,
        sample_count: int,
    ) -> RagEvalRunSummary:
        run_id = f"rageval_{uuid4().hex}"
        started_at = _utcnow()
        row = RagEvalRunV2Model(
            run_id=run_id,
            dataset_name=dataset_name,
            strategies_json=json.dumps(strategies),
            metrics_json=json.dumps(metrics),
            ragas_model=ragas_model,
            embedding_model=embedding_model,
            status="running",
            sample_count=sample_count,
            started_at=started_at,
        )
        with self.session_factory() as session:
            session.add(row)
            session.commit()
        return RagEvalRunSummary(
            run_id=run_id,
            dataset_name=dataset_name,
            strategies=strategies,
            metrics=metrics,
            ragas_model=ragas_model,
            embedding_model=embedding_model,
            status="running",
            sample_count=sample_count,
            started_at=started_at,
        )

    def save_result(self, payload: dict) -> RagEvalResult:
        result = RagEvalResult.model_validate(
            {
                **payload,
                "result_id": payload.get("result_id") or f"ragres_{uuid4().hex}",
            }
        )
        with self.session_factory() as session:
            session.add(self._result_to_row(result))
            session.commit()
        return result

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        error_summary: str | None = None,
    ) -> None:
        with self.session_factory() as session:
            row = session.get(RagEvalRunV2Model, run_id)
            if row is None:
                return
            row.status = status
            row.error_summary = error_summary
            row.finished_at = _utcnow()
            session.commit()

    def list_runs(self, *, limit: int = 20, offset: int = 0) -> list[RagEvalRunSummary]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(RagEvalRunV2Model)
                .order_by(RagEvalRunV2Model.started_at.desc())
                .offset(offset)
                .limit(limit)
            ).all()
            return [self._run_to_summary(row, aggregates={}) for row in rows]

    def get_run_summary(self, run_id: str) -> RagEvalRunSummary | None:
        with self.session_factory() as session:
            row = session.get(RagEvalRunV2Model, run_id)
            if row is None:
                return None
            results = session.scalars(
                select(RagEvalResultV2Model).where(RagEvalResultV2Model.run_id == run_id)
            ).all()
            aggregates = self._aggregate_results(
                [self._result_to_model(result) for result in results]
            )
            return self._run_to_summary(row, aggregates=aggregates)

    def list_results(
        self, run_id: str, *, limit: int = 200, offset: int = 0
    ) -> list[RagEvalResult]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(RagEvalResultV2Model)
                .where(RagEvalResultV2Model.run_id == run_id)
                .order_by(RagEvalResultV2Model.created_at.desc())
                .offset(offset)
                .limit(limit)
            ).all()
            return [self._result_to_model(row) for row in rows]

    @staticmethod
    def _aggregate_results(results: list[RagEvalResult]) -> dict[str, RagEvalAggregate]:
        groups: dict[str, list[RagEvalResult]] = defaultdict(list)
        for result in results:
            groups[result.strategy].append(result)
        return {strategy: _aggregate_group(rows) for strategy, rows in groups.items()}

    @staticmethod
    def _case_to_model(row: RagEvalCaseModel) -> RagEvalCase:
        return RagEvalCase(
            case_id=row.case_id,
            user_input=row.user_input,
            reference=row.reference,
            required_points=json.loads(row.required_points_json or "[]"),
            tags=json.loads(row.tags_json or "{}"),
            expected_source_ids=json.loads(row.expected_source_ids_json or "[]"),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    @staticmethod
    def _run_to_summary(
        row: RagEvalRunV2Model,
        *,
        aggregates: dict[str, RagEvalAggregate],
    ) -> RagEvalRunSummary:
        return RagEvalRunSummary(
            run_id=row.run_id,
            dataset_name=row.dataset_name,
            strategies=json.loads(row.strategies_json or "[]"),
            metrics=json.loads(row.metrics_json or "[]"),
            ragas_model=row.ragas_model,
            embedding_model=row.embedding_model,
            status=row.status,
            sample_count=row.sample_count,
            started_at=row.started_at,
            finished_at=row.finished_at,
            error_summary=row.error_summary,
            aggregates=aggregates,
        )

    @staticmethod
    def _result_to_row(result: RagEvalResult) -> RagEvalResultV2Model:
        return RagEvalResultV2Model(
            result_id=result.result_id,
            run_id=result.run_id,
            case_id=result.case_id,
            strategy=result.strategy,
            user_input=result.user_input,
            response=result.response,
            retrieved_chunk_ids_json=json.dumps(
                result.retrieved_chunk_ids[:MAX_STORED_CONTEXTS], ensure_ascii=False
            ),
            retrieved_contexts_json=json.dumps(
                _truncate_contexts(result.retrieved_contexts), ensure_ascii=False
            ),
            faithfulness=result.faithfulness,
            answer_relevancy=result.answer_relevancy,
            context_precision=result.context_precision,
            context_recall=result.context_recall,
            noise_sensitivity=result.noise_sensitivity,
            semantic_similarity=result.semantic_similarity,
            retrieval_latency_ms=result.retrieval_latency_ms,
            redundancy=result.redundancy,
            completeness=result.completeness,
            rag_score=result.rag_score,
            metric_errors_json=json.dumps(result.metric_errors, ensure_ascii=False),
            generated_at=result.generated_at,
            created_at=result.created_at,
        )

    @staticmethod
    def _result_to_model(row: RagEvalResultV2Model) -> RagEvalResult:
        return RagEvalResult(
            result_id=row.result_id,
            run_id=row.run_id,
            case_id=row.case_id,
            strategy=row.strategy,
            user_input=row.user_input,
            response=row.response,
            retrieved_chunk_ids=json.loads(row.retrieved_chunk_ids_json or "[]"),
            retrieved_contexts=json.loads(row.retrieved_contexts_json or "[]"),
            faithfulness=row.faithfulness,
            answer_relevancy=row.answer_relevancy,
            context_precision=row.context_precision,
            context_recall=row.context_recall,
            noise_sensitivity=row.noise_sensitivity,
            semantic_similarity=row.semantic_similarity,
            retrieval_latency_ms=row.retrieval_latency_ms,
            redundancy=row.redundancy,
            completeness=row.completeness,
            rag_score=row.rag_score,
            metric_errors=json.loads(row.metric_errors_json or "{}"),
            generated_at=_ensure_utc(row.generated_at or row.created_at),
            created_at=_ensure_utc(row.created_at),
        )


def _aggregate_group(rows: list[RagEvalResult]) -> RagEvalAggregate:
    return RagEvalAggregate(
        sample_count=len(rows),
        faithfulness=_metric_stats([row.faithfulness for row in rows]),
        answer_relevancy=_metric_stats([row.answer_relevancy for row in rows]),
        context_precision=_metric_stats([row.context_precision for row in rows]),
        context_recall=_metric_stats([row.context_recall for row in rows]),
        noise_sensitivity=_metric_stats([row.noise_sensitivity for row in rows]),
        semantic_similarity=_metric_stats([row.semantic_similarity for row in rows]),
        redundancy=_metric_stats([row.redundancy for row in rows]),
        completeness=_metric_stats([row.completeness for row in rows]),
        rag_score=_rag_score_stats([row.rag_score for row in rows]),
        retrieval_latency_ms=_latency_stats([row.retrieval_latency_ms for row in rows]),
    )


def _metric_stats(values: list[float | None]) -> RagMetricStats:
    scored = [float(value) for value in values if value is not None]
    if not scored:
        return RagMetricStats()
    return RagMetricStats(
        avg=round(mean(scored), 4),
        min=round(min(scored), 4),
        max=round(max(scored), 4),
    )


def _rag_score_stats(values: list[float | None]) -> RagScoreStats:
    base = _metric_stats(values)
    scored = [value for value in values if value is not None]
    return RagScoreStats(
        avg=base.avg,
        min=base.min,
        max=base.max,
        scored_count=len(scored),
        missing_score_count=len(values) - len(scored),
    )


def _latency_stats(values: list[int]) -> RagLatencyStats:
    if not values:
        return RagLatencyStats()
    ordered = sorted(values)
    return RagLatencyStats(
        avg=round(mean(ordered), 4),
        p50=_percentile(ordered, 0.50),
        p95=_percentile(ordered, 0.95),
        max=max(ordered),
    )


def _percentile(ordered_values: list[int], percentile: float) -> float:
    if len(ordered_values) == 1:
        return float(ordered_values[0])
    index = int(round((len(ordered_values) - 1) * percentile))
    return float(ordered_values[min(max(index, 0), len(ordered_values) - 1)])


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _truncate_contexts(contexts: list[str]) -> list[str]:
    return [
        context[:MAX_STORED_CONTEXT_CHARS]
        for context in contexts[:MAX_STORED_CONTEXTS]
    ]
