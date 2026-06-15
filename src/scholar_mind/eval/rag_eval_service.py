"""Application service for creating and reading Document 23 RAG eval runs."""

from __future__ import annotations

import logging

from scholar_mind.eval.rag_custom_metrics import (
    RequiredPointCoverageJudge,
    apply_empty_retrieval_defaults,
    compute_completeness,
    compute_rag_score,
    compute_redundancy,
    compute_retrieval_latency,
    extract_strategy,
)
from scholar_mind.eval.rag_dataset import RagEvalDatasetLoader
from scholar_mind.eval.rag_runner import RagEvalRunner
from scholar_mind.eval.ragas_official import (
    OfficialRagasEvaluator,
    ProjectRagasEmbeddings,
    build_ragas_llm,
)
from scholar_mind.models.rag_eval_models import (
    OFFICIAL_RAGAS_METRICS,
    RagEvalCase,
    RagEvalRunRequest,
    RagEvalRunSummary,
)
from scholar_mind.services.rag_eval_repository import RagEvalRepository

logger = logging.getLogger(__name__)


class RagEvalService:
    def __init__(
        self,
        settings,
        repository: RagEvalRepository,
        rag_engine,
        *,
        llm=None,
        embedding_service=None,
        ragas_evaluator: OfficialRagasEvaluator | None = None,
        online_eval_repository=None,
    ):
        self.settings = settings
        self.repository = repository
        self.runner = RagEvalRunner(rag_engine, llm=llm)
        self.embedding_service = embedding_service
        self.llm = llm
        self.ragas_evaluator = ragas_evaluator
        self.online_eval_repository = online_eval_repository
        self._default_ragas_evaluator: OfficialRagasEvaluator | None = None
        self._completeness_judge: RequiredPointCoverageJudge | None = None

    def list_cases(self, *, limit: int | None = None) -> list[RagEvalCase]:
        cases = self.repository.list_cases(limit=limit)
        if cases:
            return cases
        loaded = self._load_default_cases(limit=limit)
        self.repository.upsert_cases(loaded)
        return loaded

    def list_runs(self, *, limit: int = 20, offset: int = 0) -> list[RagEvalRunSummary]:
        return self.repository.list_runs(limit=limit, offset=offset)

    def get_run(self, run_id: str) -> RagEvalRunSummary | None:
        return self.repository.get_run_summary(run_id)

    def list_results(self, run_id: str, *, limit: int = 200, offset: int = 0):
        return self.repository.list_results(run_id, limit=limit, offset=offset)

    async def create_run(self, request: RagEvalRunRequest) -> RagEvalRunSummary:
        cases = self.list_cases(limit=request.case_limit)
        metric_names = list(request.metrics)
        strategies = [strategy.value for strategy in request.strategies]
        run = self.repository.create_run(
            dataset_name=request.dataset_name,
            strategies=strategies,
            metrics=metric_names,
            ragas_model=self._ragas_model_name(),
            embedding_model=self._embedding_model_name(),
            sample_count=len(cases),
        )
        try:
            for case in cases:
                for strategy in request.strategies:
                    observation = await self.runner.run_case(
                        case, strategy=strategy, top_k=request.top_k
                    )
                    scores = await self._score_official_metrics(
                        observation.user_input,
                        observation.response,
                        case.reference,
                        observation.retrieved_contexts,
                        metric_names,
                    )
                    embeddings = self._embed_contexts(observation.retrieved_contexts)
                    redundancy = compute_redundancy(
                        observation.retrieved_contexts,
                        observation.retrieved_chunk_ids,
                        embeddings=embeddings,
                        threshold=self.settings.rag_eval_redundancy_similarity_threshold,
                    )
                    metric_errors = dict(scores.errors)
                    try:
                        completeness = compute_completeness(
                            case.required_points,
                            observation.retrieved_contexts,
                            llm=self._get_completeness_judge(),
                        )
                    except Exception as exc:
                        completeness = None
                        metric_errors["completeness"] = str(exc)
                    metric_values = {
                        **scores.scores,
                        "retrieval_latency_ms": compute_retrieval_latency(
                            observation.retrieval_latency_ms
                        ),
                        "strategy": extract_strategy(observation.strategy),
                        "redundancy": redundancy,
                        "completeness": completeness,
                    }
                    metric_values = apply_empty_retrieval_defaults(
                        metric_values,
                        observation.retrieved_contexts,
                    )
                    rag_score, missing = compute_rag_score(metric_values)
                    if missing:
                        metric_errors["rag_score"] = "Missing metrics: " + ", ".join(missing)
                    self.repository.save_result(
                        {
                            "run_id": run.run_id,
                            "case_id": case.case_id,
                            "strategy": observation.strategy,
                            "user_input": observation.user_input,
                            "response": observation.response,
                            "retrieved_chunk_ids": observation.retrieved_chunk_ids,
                            "retrieved_contexts": observation.retrieved_contexts,
                            **{
                                name: metric_values.get(name)
                                for name in OFFICIAL_RAGAS_METRICS
                            },
                            "retrieval_latency_ms": metric_values["retrieval_latency_ms"],
                            "redundancy": redundancy,
                            "completeness": metric_values["completeness"],
                            "rag_score": rag_score,
                            "metric_errors": metric_errors,
                            "generated_at": observation.generated_at,
                        }
                    )
            self.repository.finish_run(run.run_id, status="succeeded")
        except Exception as exc:
            logger.exception("RAG eval run failed: run_id=%s", run.run_id)
            self.repository.finish_run(run.run_id, status="failed", error_summary=str(exc))
        summary = self.repository.get_run_summary(run.run_id)
        if summary is None:
            raise RuntimeError(f"RAG eval run disappeared: {run.run_id}")
        return summary

    async def score_online_request_if_annotated(
        self,
        request_id: str,
        *,
        force: bool = False,
    ) -> bool:
        if self.online_eval_repository is None:
            return False
        annotation = self.online_eval_repository.get_request_rag_eval_annotation(request_id)
        if annotation is None:
            return False
        source = self.online_eval_repository.get_online_rag_eval_source(request_id)
        if source is None or (source.get("metrics_complete") and not force):
            return False
        required_points = [
            item.strip()
            for item in annotation.get("required_points", [])
            if isinstance(item, str) and item.strip()
        ]
        reference = str(annotation.get("reference") or "").strip()
        if not reference or not required_points:
            return False

        retrieved_contexts = list(source.get("retrieved_contexts") or [])
        retrieved_chunk_ids = list(source.get("retrieved_chunk_ids") or [])
        official_contexts = self._online_official_contexts(retrieved_contexts)
        scores = await self._score_official_metrics(
            source.get("user_input", ""),
            source.get("response", ""),
            reference,
            official_contexts,
            list(OFFICIAL_RAGAS_METRICS),
        )
        embeddings = self._embed_contexts(retrieved_contexts)
        redundancy = compute_redundancy(
            retrieved_contexts,
            retrieved_chunk_ids,
            embeddings=embeddings,
            threshold=self.settings.rag_eval_redundancy_similarity_threshold,
        )
        try:
            completeness = compute_completeness(
                required_points,
                retrieved_contexts,
                llm=self._get_completeness_judge(),
            )
        except Exception:
            logger.exception("Online RAG completeness failed: request_id=%s", request_id)
            completeness = None

        metric_values = {
            **scores.scores,
            "redundancy": redundancy,
            "completeness": completeness,
        }
        metric_values = apply_empty_retrieval_defaults(metric_values, retrieved_contexts)
        rag_score, _ = compute_rag_score(metric_values)
        self.online_eval_repository.save_online_rag_metrics(
            request_id,
            {
                **{name: metric_values.get(name) for name in OFFICIAL_RAGAS_METRICS},
                "redundancy": metric_values["redundancy"],
                "completeness": metric_values["completeness"],
                "rag_score": rag_score,
                "event_id": source.get("event_id"),
            },
        )
        return True

    def _online_official_contexts(self, contexts: list[str]) -> list[str]:
        limit = int(getattr(self.settings, "rag_eval_context_top_k", len(contexts)))
        if limit <= 0:
            return contexts
        return contexts[:limit]

    async def _score_official_metrics(
        self,
        user_input: str,
        response: str,
        reference: str,
        retrieved_contexts: list[str],
        metric_names: list[str],
    ):
        evaluator = self.ragas_evaluator or self._get_default_ragas_evaluator()
        return await evaluator.score(
            user_input=user_input,
            response=response,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
            metric_names=[name for name in metric_names if name in OFFICIAL_RAGAS_METRICS],
        )

    def _get_default_ragas_evaluator(self) -> OfficialRagasEvaluator:
        if self._default_ragas_evaluator is None:
            embeddings = (
                ProjectRagasEmbeddings(self.embedding_service)
                if self.embedding_service is not None
                else None
            )
            self._default_ragas_evaluator = OfficialRagasEvaluator(
                llm=build_ragas_llm(self.settings),
                embeddings=embeddings,
            )
        return self._default_ragas_evaluator

    def _get_completeness_judge(self):
        if self.llm is not None and hasattr(self.llm, "covers_required_point"):
            return self.llm
        if self._completeness_judge is None:
            self._completeness_judge = RequiredPointCoverageJudge(self.llm)
        return self._completeness_judge

    def _load_default_cases(self, *, limit: int | None) -> list[RagEvalCase]:
        dataset_path = self.settings.resolve_path(self.settings.rag_eval_dataset_path)
        return RagEvalDatasetLoader(dataset_path).load_cases(limit=limit)

    def _embed_contexts(self, contexts: list[str]):
        if not contexts:
            return []
        if self.embedding_service is None:
            raise RuntimeError("Embedding service is required for RAG redundancy evaluation")
        return self.embedding_service.embed_documents(contexts)

    def _ragas_model_name(self) -> str:
        return (
            self.settings.rag_eval_llm_model
            or getattr(self.settings, "llm_light_model", "")
            or getattr(self.settings, "llm_reasoning_model", "")
        )

    def _embedding_model_name(self) -> str:
        return self.settings.rag_eval_embedding_model or getattr(
            self.settings, "embedding_model", ""
        )
