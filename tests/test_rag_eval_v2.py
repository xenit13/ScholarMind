from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from scholar_mind.api.routes.eval import (
    export_dashboard_csv,
    get_request_diagnosis,
    get_request_eval,
)
from scholar_mind.asgi import create_app
from scholar_mind.config.settings import get_settings
from scholar_mind.db.models import Base, RequestRagEvalAnnotationModel
from scholar_mind.db.session import init_database
from scholar_mind.eval.answer_quality import compute_answer_quality_score
from scholar_mind.eval.rag_custom_metrics import (
    compute_completeness,
    compute_rag_score,
    compute_redundancy,
)
from scholar_mind.eval.rag_dataset import RagEvalDatasetLoader, RagEvalDatasetValidationError
from scholar_mind.eval.rag_eval_service import RagEvalService
from scholar_mind.eval.rag_runner import RagEvalObservation, RagEvalRunner
from scholar_mind.eval.ragas_official import OfficialRagasEvaluator
from scholar_mind.models.domain import RetrievalStrategyName
from scholar_mind.models.rag_eval_models import (
    OfficialRagasScores,
    RagEvalCase,
    RagEvalRunRequest,
)
from scholar_mind.services.memory_eval_v2 import MemoryEvalV2Repository
from scholar_mind.services.rag_eval_repository import RagEvalRepository
from scholar_mind.services.repositories import OnlineEvalRepository


def test_rag_eval_case_requires_reference_and_required_points():
    with pytest.raises(ValueError):
        RagEvalCase(case_id="c1", user_input="What is RAG?", reference="", required_points=["a"])

    with pytest.raises(ValueError):
        RagEvalCase(case_id="c1", user_input="What is RAG?", reference="ref", required_points=[])


def test_dataset_loader_rejects_duplicate_case_ids(tmp_path):
    dataset = tmp_path / "rag_eval.jsonl"
    rows = [
        {
            "case_id": "case_1",
            "user_input": "What improves recall?",
            "reference": "Hybrid retrieval improves recall.",
            "required_points": ["hybrid retrieval improves recall"],
        },
        {
            "case_id": "case_1",
            "user_input": "Duplicate",
            "reference": "Duplicate reference.",
            "required_points": ["duplicate point"],
        },
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    loader = RagEvalDatasetLoader(dataset)

    with pytest.raises(RagEvalDatasetValidationError, match="duplicate case_id"):
        loader.load_cases()


def test_redundancy_requires_embeddings_for_non_empty_contexts():
    assert compute_redundancy([], [], embeddings=None, threshold=0.90) == 0.0

    with pytest.raises(ValueError, match="embeddings are required"):
        compute_redundancy(
            ["dense retrieval captures semantics"],
            ["c1"],
            embeddings=None,
            threshold=0.90,
        )

    with pytest.raises(ValueError, match="embeddings length"):
        compute_redundancy(
            ["dense retrieval captures semantics", "sparse retrieval preserves exact terms"],
            ["c1", "c2"],
            embeddings=[[1.0, 0.0]],
            threshold=0.90,
        )

    unique = compute_redundancy(
        ["dense retrieval captures semantics", "sparse retrieval preserves exact terms"],
        ["c1", "c2"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        threshold=0.90,
    )
    assert unique == 0.0

    duplicate_id = compute_redundancy(
        ["first context", "different text"],
        ["c1", "c1"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        threshold=0.90,
    )
    assert duplicate_id == 0.5

    duplicate_text = compute_redundancy(
        ["Hybrid retrieval improves recall.", " hybrid retrieval improves recall. "],
        ["c1", "c2"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        threshold=0.90,
    )
    assert duplicate_text == 0.5

    semantic_duplicate = compute_redundancy(
        ["alpha beta gamma", "alpha beta gamma delta"],
        ["c1", "c2"],
        embeddings=[[1.0, 0.0], [0.95, 0.05]],
        threshold=0.90,
    )
    assert semantic_duplicate == 0.5


def test_completeness_and_rag_score_boundaries():
    class _LLM:
        def covers_required_point(self, point: str, context_text: str) -> bool:
            return point.lower() in context_text.lower()

    assert compute_completeness(
        ["dense retrieval", "sparse retrieval"],
        ["Dense retrieval captures semantic similarity."],
        llm=_LLM(),
    ) == 0.5
    assert compute_completeness(["missing point"], [], llm=_LLM()) == 0.0
    with pytest.raises(ValueError, match="standard completeness judge"):
        compute_completeness(["dense retrieval"], ["Dense retrieval captures semantic similarity."])
    with pytest.raises(ValueError, match="required_points"):
        compute_completeness([], ["context"], llm=_LLM())

    score, missing = compute_rag_score(
        {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "semantic_similarity": 0.7,
            "context_recall": 0.6,
            "context_precision": 0.5,
            "completeness": 0.4,
            "noise_sensitivity": 0.2,
            "redundancy": 0.1,
        }
    )
    assert score == pytest.approx(0.693)
    assert missing == []

    missing_score, missing_fields = compute_rag_score({"faithfulness": 1.0})
    assert missing_score is None
    assert "answer_relevancy" in missing_fields


def test_answer_quality_score_uses_answer_only_formula():
    query = (
        "Analyze hybrid retrieval: compare dense and sparse methods, explain tradeoffs, "
        "and list recommendations."
    )
    answer = (
        "Hybrid retrieval combines dense semantic matching with sparse exact-term matching. "
        "It compares two methods: dense retrieval improves semantic recall for paraphrased "
        "questions, while sparse retrieval keeps API names and identifiers precise. "
        "The tradeoff is broader recall versus stricter lexical precision. "
        "Recommendations:\n"
        "1. Use hybrid retrieval when queries mix concepts and exact terms.\n"
        "2. Rerank the combined results when precision matters."
    )

    score = compute_answer_quality_score(query=query, query_type="qa", final_answer=answer)

    assert score is not None
    assert score > 0.7
    assert compute_answer_quality_score(query=query, query_type="qa", final_answer="") is None
    assert (
        compute_answer_quality_score(
            query=query,
            query_type="qa",
            final_answer="Error: request timed out while generating the answer.",
        )
        is None
    )


def test_overall_score_reweights_available_request_scores(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'overall_eval.db'}", future=True)
    Base.metadata.create_all(engine)
    repo = OnlineEvalRepository(
        sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    )
    query = "Analyze hybrid retrieval and list recommendations."
    final_answer = (
        "Hybrid retrieval combines dense semantic search with sparse exact matching. "
        "Recommendations:\n"
        "1. Use hybrid retrieval for mixed semantic and exact-term queries.\n"
        "2. Rerank results when precision matters."
    )

    def save_request(
        request_id: str,
        *,
        rag_score: float | None = None,
        memory_score: float | None = None,
        save_rag_event: bool = False,
    ) -> dict:
        repo.save_request_run(
            {
                "request_id": request_id,
                "session_id": "sess_overall",
                "user_id": "user_1",
                "query": query,
                "query_type": "qa",
                "final_answer": final_answer,
                "rag_score": rag_score,
                "memory_score": memory_score,
                "runtime_metrics": {"latency_ms": 100},
                "execution_health": {"has_error": False},
            }
        )
        if save_rag_event:
            repo.save_rag_retrieval_event(
                {
                    "event_id": f"event_{request_id}",
                    "request_id": request_id,
                    "query": query,
                    "strategy": "hybrid",
                    "latency_ms": 10,
                    "returned_contexts": ["Hybrid retrieval improves recall."],
                    "returned_chunk_ids": ["chunk_1"],
                    "rag_score": rag_score,
                }
            )
        request = repo.get_request_eval(request_id)
        assert request is not None
        return request

    answer_only = save_request("req_answer_only")
    answer_score = answer_only["answer_quality_score"]
    assert answer_score is not None
    assert answer_only["overall_score"] == answer_score

    rag_answer = save_request("req_rag_answer", rag_score=0.2, save_rag_event=True)
    assert rag_answer["overall_score"] == round(((0.50 * answer_score) + (0.30 * 0.2)) / 0.80, 4)

    memory_answer = save_request("req_memory_answer", memory_score=0.4)
    assert memory_answer["overall_score"] == round(
        ((0.50 * answer_score) + (0.20 * 0.4)) / 0.70,
        4,
    )

    all_scores = save_request(
        "req_all_scores",
        rag_score=0.2,
        memory_score=0.4,
        save_rag_event=True,
    )
    assert all_scores["overall_score"] == round(
        (0.50 * answer_score) + (0.30 * 0.2) + (0.20 * 0.4),
        4,
    )

    diagnosis = repo.get_request_diagnosis("req_answer_only")
    assert diagnosis is not None
    joined = " ".join(diagnosis["issues"] + diagnosis["strengths"] + diagnosis["recommendations"])
    assert "Answer" in joined
    assert "RAG" not in joined
    assert "Memory" not in joined


def test_execution_health_score_uses_error_timeout_fallback_retry_formula(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'health_eval.db'}", future=True)
    Base.metadata.create_all(engine)
    repo = OnlineEvalRepository(
        sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    )

    repo.save_request_run(
        {
            "request_id": "req_health",
            "session_id": "sess_health",
            "user_id": "user_1",
            "query": "What happened?",
            "query_type": "qa",
            "final_answer": "The request returned a partial answer.",
            "execution_health_score": 1.0,
            "has_retry": True,
            "has_fallback": True,
            "execution_health": {
                "has_error": True,
                "timeout": True,
                "has_retry": True,
                "has_fallback": True,
            },
        }
    )

    request = repo.get_request_eval("req_health")

    assert request is not None
    assert request["execution_health_score"] == 0.0
    assert request["execution_health"]["execution_health_score"] == 0.0


@pytest.mark.asyncio
async def test_official_ragas_evaluator_maps_fields_and_isolates_errors():
    class _Metric:
        def __init__(self, value=None, error: Exception | None = None):
            self.value = value
            self.error = error
            self.calls = []

        async def ascore(self, **kwargs):
            self.calls.append(kwargs)
            if self.error:
                raise self.error
            return SimpleNamespace(value=self.value)

    metrics = {
        "faithfulness": _Metric(0.9),
        "answer_relevancy": _Metric(RuntimeError("wrong type"), error=RuntimeError("boom")),
        "context_precision": _Metric(0.7),
    }
    evaluator = OfficialRagasEvaluator(metrics=metrics)

    result = await evaluator.score(
        user_input="What improves recall?",
        response="Hybrid retrieval improves recall.",
        reference="Hybrid retrieval improves recall.",
        retrieved_contexts=["Hybrid retrieval improves recall."],
        metric_names=list(metrics),
    )

    assert result.scores == {
        "faithfulness": 0.9,
        "answer_relevancy": None,
        "context_precision": 0.7,
    }
    assert "boom" in result.errors["answer_relevancy"]
    assert metrics["faithfulness"].calls[0]["user_input"] == "What improves recall?"
    assert metrics["faithfulness"].calls[0]["retrieved_contexts"] == [
        "Hybrid retrieval improves recall."
    ]


@pytest.mark.asyncio
async def test_official_ragas_evaluator_uses_metric_specific_payloads():
    calls = {}

    class _Faithfulness:
        async def ascore(self, user_input, response, retrieved_contexts):
            calls["faithfulness"] = {
                "user_input": user_input,
                "response": response,
                "retrieved_contexts": retrieved_contexts,
            }
            return SimpleNamespace(value=0.1)

    class _AnswerRelevancy:
        async def ascore(self, user_input, response):
            calls["answer_relevancy"] = {
                "user_input": user_input,
                "response": response,
            }
            return SimpleNamespace(value=0.2)

    class _ContextPrecision:
        async def ascore(self, user_input, reference, retrieved_contexts):
            calls["context_precision"] = {
                "user_input": user_input,
                "reference": reference,
                "retrieved_contexts": retrieved_contexts,
            }
            return SimpleNamespace(value=0.3)

    class _ContextRecall:
        async def ascore(self, user_input, retrieved_contexts, reference):
            calls["context_recall"] = {
                "user_input": user_input,
                "retrieved_contexts": retrieved_contexts,
                "reference": reference,
            }
            return SimpleNamespace(value=0.4)

    class _NoiseSensitivity:
        async def ascore(self, user_input, response, reference, retrieved_contexts):
            calls["noise_sensitivity"] = {
                "user_input": user_input,
                "response": response,
                "reference": reference,
                "retrieved_contexts": retrieved_contexts,
            }
            return SimpleNamespace(value=0.5)

    class _SemanticSimilarity:
        async def ascore(self, reference, response):
            calls["semantic_similarity"] = {
                "reference": reference,
                "response": response,
            }
            return SimpleNamespace(value=0.6)

    evaluator = OfficialRagasEvaluator(
        metrics={
            "faithfulness": _Faithfulness(),
            "answer_relevancy": _AnswerRelevancy(),
            "context_precision": _ContextPrecision(),
            "context_recall": _ContextRecall(),
            "noise_sensitivity": _NoiseSensitivity(),
            "semantic_similarity": _SemanticSimilarity(),
        }
    )

    result = await evaluator.score(
        user_input="What improves recall?",
        response="Hybrid retrieval improves recall.",
        reference="Hybrid retrieval improves recall.",
        retrieved_contexts=["Hybrid retrieval improves recall."],
        metric_names=list(evaluator.metrics),
    )

    assert result.scores == {
        "faithfulness": 0.1,
        "answer_relevancy": 0.2,
        "context_precision": 0.3,
        "context_recall": 0.4,
        "noise_sensitivity": 0.5,
        "semantic_similarity": 0.6,
    }
    assert calls["faithfulness"] == {
        "user_input": "What improves recall?",
        "response": "Hybrid retrieval improves recall.",
        "retrieved_contexts": ["Hybrid retrieval improves recall."],
    }
    assert calls["semantic_similarity"] == {
        "reference": "Hybrid retrieval improves recall.",
        "response": "Hybrid retrieval improves recall.",
    }


@pytest.mark.asyncio
async def test_official_ragas_evaluator_reports_adapter_setup_errors(monkeypatch):
    def _raise_setup_error(*_args):
        raise ValueError("unsupported llm adapter")

    monkeypatch.setattr(
        OfficialRagasEvaluator,
        "_build_metrics",
        staticmethod(_raise_setup_error),
    )
    evaluator = OfficialRagasEvaluator(llm=object())

    result = await evaluator.score(
        user_input="What improves recall?",
        response="Hybrid retrieval improves recall.",
        reference="Hybrid retrieval improves recall.",
        retrieved_contexts=["Hybrid retrieval improves recall."],
        metric_names=["faithfulness", "context_precision"],
    )

    assert result.scores == {"faithfulness": None, "context_precision": None}
    assert result.errors == {
        "faithfulness": "unsupported llm adapter",
        "context_precision": "unsupported llm adapter",
    }


@pytest.mark.asyncio
async def test_project_ragas_embeddings_wraps_existing_embedding_service():
    from scholar_mind.eval.ragas_official import ProjectRagasEmbeddings

    class _EmbeddingService:
        def embed_query(self, text: str):
            return [float(len(text))]

        def embed_documents(self, texts: list[str]):
            return [[float(index)] for index, _ in enumerate(texts)]

        async def aembed_query(self, text: str):
            return [float(len(text)) + 1.0]

        async def aembed_documents(self, texts: list[str]):
            return [[float(index) + 1.0] for index, _ in enumerate(texts)]

    embeddings = ProjectRagasEmbeddings(_EmbeddingService())

    assert embeddings.embed_text("abc") == [3.0]
    assert embeddings.embed_texts(["a", "b"]) == [[0.0], [1.0]]
    assert await embeddings.aembed_text("abc") == [4.0]
    assert await embeddings.aembed_texts(["a", "b"]) == [[1.0], [2.0]]


def test_build_ragas_llm_uses_configured_max_tokens(monkeypatch):
    from openai import AsyncOpenAI
    from ragas import llms as ragas_llms

    from scholar_mind.eval.ragas_official import build_ragas_llm

    captured = {}

    class _Client:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

    def _llm_factory(model, **kwargs):
        captured["model"] = model
        captured["factory_kwargs"] = kwargs
        return object()

    monkeypatch.setattr("openai.AsyncOpenAI", _Client)
    monkeypatch.setattr(ragas_llms, "llm_factory", _llm_factory)

    settings = SimpleNamespace(
        rag_eval_llm_model="mock-ragas-model",
        llm_light_model="mock-light",
        llm_reasoning_model="mock-reasoning",
        llm_api_key="key",
        llm_base_url="https://example.test/v1",
        llm_request_timeout_seconds=11.0,
        llm_max_retries=2,
        rag_eval_llm_max_tokens=8192,
    )

    llm = build_ragas_llm(settings)

    assert llm is not None
    assert captured["model"] == "mock-ragas-model"
    assert captured["client_kwargs"]["timeout"] == 11.0
    assert captured["factory_kwargs"]["client"].__class__ is _Client
    assert captured["factory_kwargs"]["temperature"] == 0.0
    assert captured["factory_kwargs"]["max_tokens"] == 8192
    assert AsyncOpenAI is not _Client


@pytest.mark.asyncio
async def test_rag_eval_service_uses_modern_ragas_llm_and_embedding_adapter(monkeypatch):
    from scholar_mind.eval import rag_eval_service as service_module

    ragas_llm = object()
    captured = {}

    class _EmbeddingService:
        def embed_query(self, text: str):
            return [0.1, 0.2]

        def embed_documents(self, texts: list[str]):
            return [[0.1, 0.2] for _ in texts]

        async def aembed_query(self, text: str):
            return [0.1, 0.2]

        async def aembed_documents(self, texts: list[str]):
            return [[0.1, 0.2] for _ in texts]

    class _Evaluator:
        def __init__(self, *, llm, embeddings):
            captured["llm"] = llm
            captured["embeddings"] = embeddings

        async def score(self, **kwargs):
            captured["metric_names"] = kwargs["metric_names"]
            return OfficialRagasScores(scores={"faithfulness": 0.9})

    monkeypatch.setattr(service_module, "build_ragas_llm", lambda settings: ragas_llm)
    monkeypatch.setattr(service_module, "OfficialRagasEvaluator", _Evaluator)

    settings = SimpleNamespace(
        rag_eval_llm_model="mock-llm",
        llm_light_model="mock-light",
        llm_reasoning_model="mock-reasoning",
        rag_eval_embedding_model="mock-embedding",
        embedding_model="mock-embedding",
        rag_eval_redundancy_similarity_threshold=0.90,
        rag_eval_dataset_path="unused.jsonl",
        resolve_path=lambda value: value,
    )
    service = RagEvalService(
        settings,
        repository=object(),
        rag_engine=object(),
        embedding_service=_EmbeddingService(),
    )

    result = await service._score_official_metrics(
        "What improves recall?",
        "Hybrid retrieval improves recall.",
        "Hybrid retrieval improves recall.",
        ["Hybrid retrieval improves recall."],
        ["faithfulness", "retrieval_latency"],
    )

    assert result.scores == {"faithfulness": 0.9}
    assert captured["llm"] is ragas_llm
    assert captured["embeddings"].embed_text("query") == [0.1, 0.2]
    assert captured["metric_names"] == ["faithfulness"]


@pytest.mark.asyncio
async def test_rag_eval_service_scores_empty_retrieval_as_valid_rag_score(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'rag_eval.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    repo = RagEvalRepository(factory)
    case = RagEvalCase(
        case_id="case_empty",
        user_input="What improves recall?",
        reference="Hybrid retrieval improves recall.",
        required_points=["hybrid retrieval improves recall"],
    )
    repo.upsert_cases([case])

    class _Runner:
        async def run_case(self, *_args, **_kwargs):
            return RagEvalObservation(
                case=case,
                strategy="hybrid",
                user_input=case.user_input,
                response="Hybrid retrieval improves recall.",
                retrieved_contexts=[],
                retrieved_chunk_ids=[],
                retrieval_latency_ms=42,
                generated_at=datetime(2026, 1, 1, tzinfo=UTC),
            )

    class _Evaluator:
        async def score(self, **_kwargs):
            return OfficialRagasScores(
                scores={
                    "faithfulness": None,
                    "answer_relevancy": 0.4,
                    "context_precision": None,
                    "context_recall": None,
                    "noise_sensitivity": None,
                    "semantic_similarity": 0.5,
                }
            )

    settings = SimpleNamespace(
        rag_eval_llm_model="mock-llm",
        llm_light_model="mock-light",
        llm_reasoning_model="mock-reasoning",
        rag_eval_embedding_model="mock-embedding",
        embedding_model="mock-embedding",
        rag_eval_redundancy_similarity_threshold=0.90,
        rag_eval_dataset_path="unused.jsonl",
        resolve_path=lambda value: value,
    )
    service = RagEvalService(
        settings,
        repo,
        rag_engine=object(),
        ragas_evaluator=_Evaluator(),
        llm=object(),
        embedding_service=object(),
    )
    service.runner = _Runner()

    summary = await service.create_run(
        RagEvalRunRequest(strategies=[RetrievalStrategyName.HYBRID])
    )
    result = repo.list_results(summary.run_id)[0]

    assert result.rag_score is not None
    assert result.faithfulness == 0.0
    assert result.context_precision == 0.0
    assert result.context_recall == 0.0
    assert result.noise_sensitivity == 1.0
    assert result.completeness == 0.0
    assert "rag_score" not in result.metric_errors


@pytest.mark.asyncio
async def test_rag_eval_service_keeps_rag_score_null_when_completeness_fails(
    tmp_path, monkeypatch
):
    engine = create_engine(f"sqlite:///{tmp_path / 'rag_eval.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    repo = RagEvalRepository(factory)
    case = RagEvalCase(
        case_id="case_1",
        user_input="What improves recall?",
        reference="Hybrid retrieval improves recall.",
        required_points=["hybrid retrieval improves recall"],
    )
    repo.upsert_cases([case])

    class _Runner:
        async def run_case(self, *_args, **_kwargs):
            return RagEvalObservation(
                case=case,
                strategy="hybrid",
                user_input=case.user_input,
                response="Hybrid retrieval improves recall.",
                retrieved_contexts=["Hybrid retrieval improves recall."],
                retrieved_chunk_ids=["chunk_1"],
                retrieval_latency_ms=42,
                generated_at=datetime(2026, 1, 1, tzinfo=UTC),
            )

    class _Evaluator:
        async def score(self, **_kwargs):
            return OfficialRagasScores(
                scores={
                    "faithfulness": 1.0,
                    "answer_relevancy": 1.0,
                    "context_precision": 1.0,
                    "context_recall": 1.0,
                    "noise_sensitivity": 0.0,
                    "semantic_similarity": 1.0,
                }
            )

    class _EmbeddingService:
        def embed_documents(self, texts: list[str]):
            vectors = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
            return vectors[: len(texts)]

    class _CompletenessJudge:
        def covers_required_point(self, point: str, context_text: str) -> bool:
            return False

    settings = SimpleNamespace(
        rag_eval_llm_model="mock-llm",
        llm_light_model="mock-light",
        llm_reasoning_model="mock-reasoning",
        rag_eval_embedding_model="mock-embedding",
        embedding_model="mock-embedding",
        rag_eval_redundancy_similarity_threshold=0.90,
        rag_eval_dataset_path="unused.jsonl",
        resolve_path=lambda value: value,
    )
    service = RagEvalService(
        settings,
        repo,
        rag_engine=object(),
        ragas_evaluator=_Evaluator(),
        llm=_CompletenessJudge(),
        embedding_service=_EmbeddingService(),
    )
    service.runner = _Runner()
    monkeypatch.setattr(
        "scholar_mind.eval.rag_eval_service.compute_completeness",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("judge failed")),
    )

    summary = await service.create_run(
        RagEvalRunRequest(strategies=[RetrievalStrategyName.HYBRID])
    )
    result = repo.list_results(summary.run_id)[0]

    assert result.completeness is None
    assert result.rag_score is None
    assert result.metric_errors["completeness"] == "judge failed"
    assert "completeness" in result.metric_errors["rag_score"]


@pytest.mark.asyncio
async def test_rag_eval_runner_requires_llm_answer_generation():
    class _RagEngine:
        async def retrieve(self, *_args, **_kwargs):
            return [SimpleNamespace(content="retrieved evidence", chunk_id="chunk_1")], 3

    case = RagEvalCase(
        case_id="case_1",
        user_input="What improves recall?",
        reference="Hybrid retrieval improves recall.",
        required_points=["hybrid retrieval improves recall"],
    )
    runner = RagEvalRunner(_RagEngine(), llm=None)

    with pytest.raises(RuntimeError, match="LLM is required"):
        await runner.run_case(case, strategy=RetrievalStrategyName.HYBRID, top_k=5)


def test_rag_eval_service_requires_embedding_service_for_redundancy():
    settings = SimpleNamespace(
        rag_eval_llm_model="mock-llm",
        llm_light_model="mock-light",
        llm_reasoning_model="mock-reasoning",
        rag_eval_embedding_model="mock-embedding",
        embedding_model="mock-embedding",
        rag_eval_redundancy_similarity_threshold=0.90,
        rag_eval_dataset_path="unused.jsonl",
        resolve_path=lambda value: value,
    )
    service = RagEvalService(settings, repository=object(), rag_engine=object())

    with pytest.raises(RuntimeError, match="Embedding service is required"):
        service._embed_contexts(["context"])


def test_rag_eval_repository_persists_runs_results_and_aggregates(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'rag_eval.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    repo = RagEvalRepository(factory)
    case = RagEvalCase(
        case_id="case_1",
        user_input="What improves recall?",
        reference="Hybrid retrieval improves recall.",
        required_points=["hybrid retrieval improves recall"],
    )
    repo.upsert_cases([case])
    run = repo.create_run(
        dataset_name="rag_eval_v2",
        strategies=["hybrid"],
        metrics=["faithfulness", "rag_score"],
        ragas_model="mock-llm",
        embedding_model="mock-embedding",
        sample_count=1,
    )
    repo.save_result(
        {
            "run_id": run.run_id,
            "case_id": "case_1",
            "strategy": "hybrid",
            "user_input": case.user_input,
            "response": "Hybrid retrieval improves recall.",
            "retrieved_chunk_ids": ["chunk_1"],
            "retrieved_contexts": ["Hybrid retrieval improves recall."],
            "faithfulness": 1.0,
            "answer_relevancy": 1.0,
            "context_precision": 1.0,
            "context_recall": 1.0,
            "noise_sensitivity": 0.0,
            "semantic_similarity": 1.0,
            "retrieval_latency_ms": 42,
            "redundancy": 0.0,
            "completeness": 1.0,
            "rag_score": 1.0,
            "metric_errors": {},
            "generated_at": datetime(2026, 1, 1, tzinfo=UTC),
        }
    )
    repo.finish_run(run.run_id, status="succeeded")

    summary = repo.get_run_summary(run.run_id)
    results = repo.list_results(run.run_id)

    assert summary is not None
    assert summary.status == "succeeded"
    assert summary.aggregates["hybrid"].rag_score.avg == 1.0
    assert summary.aggregates["hybrid"].retrieval_latency_ms.p95 == 42
    assert len(results) == 1
    assert results[0].retrieved_chunk_ids == ["chunk_1"]
    assert results[0].generated_at == datetime(2026, 1, 1, tzinfo=UTC)


@pytest.mark.asyncio
async def test_rag_eval_runner_offloads_llm_answer_generation(monkeypatch):
    class _Retrieved:
        content = "Hybrid retrieval improves recall."
        chunk_id = "chunk_1"

    class _RagEngine:
        async def retrieve(self, *_args, **_kwargs):
            return [_Retrieved()], 12

    calls = []

    async def _to_thread(func, *args, **kwargs):
        calls.append((func, args, kwargs))
        return "Threaded answer."

    monkeypatch.setattr("scholar_mind.eval.rag_runner.asyncio.to_thread", _to_thread)
    runner = RagEvalRunner(_RagEngine(), llm=object())
    case = RagEvalCase(
        case_id="case_1",
        user_input="What improves recall?",
        reference="Hybrid retrieval improves recall.",
        required_points=["hybrid retrieval improves recall"],
    )

    observation = await runner.run_case(
        case,
        strategy=RetrievalStrategyName.HYBRID,
        top_k=5,
    )

    assert calls
    assert observation.response == "Threaded answer."
    assert observation.generated_at.tzinfo is not None


def test_online_request_dashboard_fields_and_low_score_filter(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'online_eval.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    repo = OnlineEvalRepository(factory)

    def save_request(request_id: str, rag_score: float, memory_score: float):
        query = (
            "Analyze hybrid retrieval: compare dense and sparse methods, explain tradeoffs, "
            "and list recommendations."
        )
        final_answer = (
            "Hybrid retrieval combines dense semantic matching with sparse exact-term matching. "
            "It compares two methods: dense retrieval improves semantic recall for paraphrased "
            "questions, while sparse retrieval keeps API names and identifiers precise. "
            "The tradeoff is broader recall versus stricter lexical precision. "
            "Recommendations:\n"
            "1. Use hybrid retrieval when queries mix concepts and exact terms.\n"
            "2. Rerank the combined results when precision matters."
        )
        repo.save_request_run(
            {
                "request_id": request_id,
                "session_id": "sess_1",
                "user_id": "user_1",
                "query": query,
                "query_type": "qa",
                "final_answer": final_answer,
                "memory_score": memory_score,
                "execution_health_score": 1.0,
                "rag_score": rag_score,
                "faithfulness": rag_score,
                "answer_relevancy": 0.1,
                "context_precision": 0.8,
                "context_recall": 0.9,
                "noise_sensitivity": 0.1,
                "semantic_similarity": 0.75,
                "redundancy": 0.2,
                "completeness": 0.85,
                "runtime_metrics": {"latency_ms": 100, "total_tokens": 20},
                "execution_health": {"has_error": False, "timeout": False},
            }
        )
        repo.save_rag_retrieval_event(
            {
                "event_id": f"event_{request_id}",
                "request_id": request_id,
                "query": "What improves recall?",
                "strategy": "hybrid",
                "top_k": 5,
                "latency_ms": 12,
                "returned_contexts": ["Hybrid retrieval improves recall."],
                "returned_chunk_ids": ["chunk_1"],
                "rag_score": rag_score,
                "faithfulness": rag_score,
                "answer_relevancy": 0.7,
                "context_precision": 0.8,
                "context_recall": 0.9,
                "noise_sensitivity": 0.1,
                "semantic_similarity": 0.75,
                "redundancy": 0.2,
                "completeness": 0.85,
            }
        )

    save_request("req_low", rag_score=0.2, memory_score=0.4)
    save_request("req_high", rag_score=0.8, memory_score=0.6)

    request = repo.get_request_eval("req_low")
    assert request is not None
    expected_answer_score = compute_answer_quality_score(
        query=request["query"],
        query_type=request["query_type"],
        final_answer=request["final_answer"],
    )
    assert request["rag_score"] == 0.2
    expected_low_overall = round(
        (0.50 * expected_answer_score) + (0.30 * 0.2) + (0.20 * 0.4),
        4,
    )
    expected_high_overall = round(
        (0.50 * expected_answer_score) + (0.30 * 0.8) + (0.20 * 0.6),
        4,
    )
    assert request["overall_score"] == expected_low_overall
    assert request["answer_quality_score"] == expected_answer_score
    assert request["answer_quality_score"] != request["memory_score"]
    assert request["answer_quality_score"] != request["rag_metrics"]["answer_relevancy"]
    assert request["faithfulness_score"] == 0.2
    assert request["rag_metrics"] == {
        "rag_score": 0.2,
        "faithfulness": 0.2,
        "answer_relevancy": 0.7,
        "context_precision": 0.8,
        "context_recall": 0.9,
        "noise_sensitivity": 0.1,
        "semantic_similarity": 0.75,
        "retrieval_latency_ms": 12,
        "strategy": "hybrid",
        "caller_agent": "researcher",
        "redundancy": 0.2,
        "completeness": 0.85,
        "returned_chunks_count": 1,
        "retrieved_contexts": ["Hybrid retrieval improves recall."],
        "retrieved_chunk_ids": ["chunk_1"],
    }

    stats = repo.get_dashboard_stats(hours=1)
    assert stats["avg_rag_score"] == 0.5
    assert stats["avg_memory_score"] == 0.5
    assert stats["avg_overall_score"] == round(
        (expected_low_overall + expected_high_overall) / 2,
        4,
    )
    assert stats["avg_answer_quality_score"] == expected_answer_score
    assert stats["low_score_count"] == 0

    trend = repo.get_score_trend(hours=1)
    assert trend
    assert {
        "avg_overall_score",
        "avg_rag_score",
        "avg_memory_score",
        "avg_answer_quality_score",
    } <= set(trend[0])
    assert trend[0]["avg_answer_quality_score"] == expected_answer_score
    assert trend[0]["avg_memory_score"] == 0.5
    assert trend[0]["avg_overall_score"] == stats["avg_overall_score"]

    events = repo.get_request_events("req_low")
    assert events["rag_events"][0]["rag_score"] == 0.2
    assert events["rag_events"][0]["faithfulness"] == 0.2
    assert events["rag_events"][0]["answer_relevancy"] == 0.7
    assert events["rag_events"][0]["context_precision"] == 0.8
    assert events["rag_events"][0]["context_recall"] == 0.9
    assert events["rag_events"][0]["noise_sensitivity"] == 0.1
    assert events["rag_events"][0]["semantic_similarity"] == 0.75
    assert events["rag_events"][0]["redundancy"] == 0.2
    assert events["rag_events"][0]["completeness"] == 0.85
    assert events["rag_events"][0]["caller_agent"] == "researcher"

    all_requests = repo.get_all_requests()
    assert {"rag_score", "overall_score", "answer_quality_score", "faithfulness_score"} <= set(
        all_requests[0]
    )
    assert [item["request_id"] for item in repo.get_low_score_requests(threshold=0.6)] == [
        "req_low"
    ]

    diagnosis = repo.get_request_diagnosis("req_low")
    assert diagnosis is not None
    assert diagnosis["scores"]["rag_score"] == 0.2
    assert diagnosis["scores"]["memory_score"] == 0.4
    assert diagnosis["scores"]["answer_quality_score"] == expected_answer_score
    assert any("RAG score is low" in item for item in diagnosis["issues"])
    assert any("Memory score is low" in item for item in diagnosis["issues"])
    assert any("Answer score is strong" in item for item in diagnosis["strengths"])
    assert any("retrieved contexts" in item for item in diagnosis["recommendations"])


@pytest.mark.asyncio
async def test_dashboard_export_includes_current_frontend_request_fields(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'online_eval_export.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    repo = OnlineEvalRepository(factory)
    memory_repo = MemoryEvalV2Repository(factory)
    query = "Analyze hybrid retrieval and explain when it helps."
    final_answer = (
        "Hybrid retrieval helps when a query needs both semantic matching and exact terms. "
        "It improves recall for paraphrases while sparse matching keeps important identifiers. "
        "Use reranking when precision matters."
    )
    repo.save_request_run(
        {
            "request_id": "req_export",
            "session_id": "sess_1",
            "user_id": "user_1",
            "query": query,
            "query_type": "qa",
            "final_answer": final_answer,
            "memory_score": 0.4,
            "rag_score": 0.7,
            "faithfulness": 0.65,
            "runtime_metrics": {
                "latency_ms": 123,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "execution_health": {"has_error": False, "timeout": False},
        }
    )
    repo.save_rag_retrieval_event(
        {
            "event_id": "event_req_export",
            "request_id": "req_export",
            "query": "hybrid retrieval",
            "strategy": "hybrid",
            "latency_ms": 12,
            "returned_contexts": ["Hybrid retrieval improves recall."],
            "returned_chunk_ids": ["chunk_1"],
            "rag_score": 0.7,
            "faithfulness": 0.65,
            "answer_relevancy": 0.6,
            "context_precision": 0.8,
            "context_recall": 0.9,
            "noise_sensitivity": 0.1,
            "semantic_similarity": 0.75,
            "redundancy": 0.2,
            "completeness": 0.85,
            "tool_name": "rag_retrieve",
        }
    )
    memory_repo.save_memory_retrieval_event(
        {
            "event_id": "memret_req_export",
            "request_id": "req_export",
            "user_id": "user_1",
            "query": query,
            "embedding_latency_ms": 6,
            "vector_search_latency_ms": 9,
            "retrieved_memory_ids": ["mem_1", "mem_2"],
            "retrieved_scores": [0.92, 0.81],
            "retrieved_count": 2,
            "injected_memory_ids": ["mem_1"],
            "injected_count": 1,
            "injected_text": "- User prefers concise RAG analysis.",
            "injected_tokens": 7,
        }
    )
    memory_repo.save_memory_extraction_dispatch(
        request_id="req_export",
        user_id="user_1",
        dispatch_latency_ms=14,
        dispatch_success=True,
    )
    memory_repo.update_memory_extraction_result(
        request_id="req_export",
        prompt_tokens=20,
        completion_tokens=8,
        total_tokens=28,
        written_memory_ids=["mem_written_1"],
        written_memory_texts=["User evaluates RAG exports."],
    )
    memory_repo.replace_eval_runs(
        "batch_req_export",
        [
            {
                "request_id": "req_export",
                "user_id": "user_1",
                "session_id": "sess_1",
                "memory_score": 0.4,
                "memory_injected_count": 1,
                "memory_injected_latency_ms": 9,
                "memory_injected_tokens": 7,
                "memory_hit_at_k": 1.0,
                "memory_relevant_recall": 0.5,
                "memory_relevant_precision": 0.5,
                "first_relevant_rank": 1,
                "memory_stale_retrieval_rate": 0.0,
                "memory_answer_relevance": 0.75,
                "memory_extraction_precision": 1.0,
                "memory_extraction_latency_ms": 14,
                "memory_extraction_tokens": 28,
                "score_breakdown": {"s_retrieval": 0.5, "s_answer": 0.75},
            }
        ],
    )

    container = SimpleNamespace(online_eval_repository=repo)
    json_response = await export_dashboard_csv(
        container=container,
        hours=1,
        user_id=None,
        format="json",
    )
    csv_response = await export_dashboard_csv(
        container=container,
        hours=1,
        user_id=None,
        format="csv",
    )
    json_payload = json.loads(json_response.body.decode("utf-8"))
    csv_rows = list(csv.DictReader(io.StringIO(csv_response.body.decode("utf-8"))))

    expected_answer_score = compute_answer_quality_score(
        query=query,
        query_type="qa",
        final_answer=final_answer,
    )
    request_payload = repo.get_request_eval("req_export")
    assert request_payload is not None
    expected_overall_score = request_payload["overall_score"]
    exported = json_payload["requests"][0]
    assert set(exported) == {"request_overview", "memory_data", "rag_data"}
    assert exported["request_overview"]["created_at"]
    assert exported["request_overview"] == {
        "request_id": "req_export",
        "user_id": "user_1",
        "session_id": "sess_1",
        "query_type": "qa",
        "query": query,
        "final_answer": final_answer,
        "total_latency_ms": 123,
        "execution_health_score": 1.0,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "overall_score": expected_overall_score,
        "answer_quality_score": expected_answer_score,
        "faithfulness_score": 0.65,
        "has_error": False,
        "has_retry": False,
        "has_fallback": False,
        "timeout": False,
        "created_at": exported["request_overview"]["created_at"],
    }
    assert exported["memory_data"]["run"]["memory_score"] == 0.4
    assert exported["memory_data"]["run"]["memory_hit_at_k"] == 1.0
    assert exported["memory_data"]["run"]["memory_relevant_recall"] == 0.5
    assert exported["memory_data"]["run"]["memory_answer_relevance"] == 0.75
    assert exported["memory_data"]["run"]["score_breakdown"] == {
        "s_retrieval": 0.5,
        "s_answer": 0.75,
    }
    assert exported["memory_data"]["retrieval_event"]["retrieved_memory_ids"] == [
        "mem_1",
        "mem_2",
    ]
    assert exported["memory_data"]["retrieval_event"]["injected_text"] == (
        "- User prefers concise RAG analysis."
    )
    assert exported["memory_data"]["extraction_event"]["total_tokens"] == 28
    assert exported["memory_data"]["extraction_event"]["written_memory_ids"] == [
        "mem_written_1"
    ]
    assert exported["rag_data"]["metrics"]["rag_score"] == 0.7
    assert exported["rag_data"]["metrics"]["faithfulness"] == 0.65
    assert exported["rag_data"]["metrics"]["answer_relevancy"] == 0.6
    assert exported["rag_data"]["metrics"]["context_precision"] == 0.8
    assert exported["rag_data"]["metrics"]["context_recall"] == 0.9
    assert exported["rag_data"]["metrics"]["noise_sensitivity"] == 0.1
    assert exported["rag_data"]["metrics"]["semantic_similarity"] == 0.75
    assert exported["rag_data"]["metrics"]["redundancy"] == 0.2
    assert exported["rag_data"]["metrics"]["completeness"] == 0.85
    assert exported["rag_data"]["metrics"]["caller_agent"] == "researcher"
    assert exported["rag_data"]["events"][0]["returned_chunk_ids"] == ["chunk_1"]
    assert exported["rag_data"]["events"][0]["returned_contexts"] == [
        "Hybrid retrieval improves recall."
    ]

    assert csv_rows
    row = csv_rows[0]
    assert set(row) == {"request_overview_json", "memory_data_json", "rag_data_json"}
    assert json.loads(row["request_overview_json"])["request_id"] == "req_export"
    assert json.loads(row["request_overview_json"])["overall_score"] == expected_overall_score
    assert json.loads(row["request_overview_json"])["answer_quality_score"] == expected_answer_score
    assert json.loads(row["memory_data_json"])["run"]["memory_hit_at_k"] == 1.0
    assert json.loads(row["rag_data_json"])["events"][0]["returned_chunk_ids"] == ["chunk_1"]


@pytest.mark.asyncio
async def test_online_request_rag_metrics_are_computed_from_user_inserted_reference(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'online_rag_eval.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    online_repo = OnlineEvalRepository(factory)
    rag_repo = RagEvalRepository(factory)

    online_repo.save_request_run(
        {
            "request_id": "req_annotated",
            "session_id": "sess_1",
            "user_id": "user_1",
            "query": "What does memory-augmented RAG improve?",
            "query_type": "qa",
            "final_answer": "Memory improves citation-grounded answers.",
            "runtime_metrics": {"latency_ms": 100, "total_tokens": 20},
            "execution_health": {"has_error": False, "timeout": False},
        }
    )
    online_repo.save_rag_retrieval_event(
        {
            "event_id": "event_req_annotated",
            "request_id": "req_annotated",
            "query": "What does memory-augmented RAG improve?",
            "strategy": "hybrid",
            "top_k": 3,
            "latency_ms": 12,
            "returned_contexts": [
                "Memory improves citation-grounded answers.",
                "Context outside the online scoring window.",
                "Another context outside the online scoring window.",
            ],
            "returned_chunk_ids": ["chunk_1", "chunk_2", "chunk_3"],
        }
    )
    with factory() as session:
        session.add(
            RequestRagEvalAnnotationModel(
                request_id="req_annotated",
                reference="Memory improves citation-grounded answers.",
                required_points_json=json.dumps(["memory improves citation-grounded answers"]),
            )
        )
        session.commit()

    captured = {}

    class _Evaluator:
        async def score(self, **kwargs):
            captured["retrieved_contexts"] = kwargs["retrieved_contexts"]
            return OfficialRagasScores(
                scores={
                    "faithfulness": 1.0,
                    "answer_relevancy": 0.9,
                    "context_precision": 0.8,
                    "context_recall": 0.7,
                    "noise_sensitivity": 0.1,
                    "semantic_similarity": 0.95,
                }
            )

    class _EmbeddingService:
        def embed_documents(self, texts: list[str]):
            vectors = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
            return vectors[: len(texts)]

    class _CompletenessJudge:
        def covers_required_point(self, point: str, context_text: str) -> bool:
            return point.lower() in context_text.lower()

    settings = SimpleNamespace(
        rag_eval_llm_model="mock-llm",
        llm_light_model="mock-light",
        llm_reasoning_model="mock-reasoning",
        rag_eval_embedding_model="mock-embedding",
        embedding_model="mock-embedding",
        rag_eval_redundancy_similarity_threshold=0.90,
        rag_eval_context_top_k=1,
        rag_eval_dataset_path="unused.jsonl",
        resolve_path=lambda value: value,
    )
    service = RagEvalService(
        settings,
        rag_repo,
        rag_engine=object(),
        ragas_evaluator=_Evaluator(),
        llm=_CompletenessJudge(),
        embedding_service=_EmbeddingService(),
        online_eval_repository=online_repo,
    )

    computed = await service.score_online_request_if_annotated("req_annotated")
    request = online_repo.get_request_eval("req_annotated")
    events = online_repo.get_request_events("req_annotated")

    assert computed is True
    assert captured["retrieved_contexts"] == ["Memory improves citation-grounded answers."]
    assert request["rag_metrics"]["faithfulness"] == 1.0
    assert request["rag_metrics"]["answer_relevancy"] == 0.9
    assert request["rag_metrics"]["context_precision"] == 0.8
    assert request["rag_metrics"]["context_recall"] == 0.7
    assert request["rag_metrics"]["noise_sensitivity"] == 0.1
    assert request["rag_metrics"]["semantic_similarity"] == 0.95
    assert request["rag_metrics"]["redundancy"] == 0.0
    assert request["rag_metrics"]["completeness"] == 1.0
    assert request["rag_metrics"]["rag_score"] is not None
    assert events["rag_events"][0]["rag_score"] == request["rag_metrics"]["rag_score"]


@pytest.mark.asyncio
async def test_online_request_empty_retrieval_gets_valid_rag_score(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'online_rag_eval.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    online_repo = OnlineEvalRepository(factory)
    rag_repo = RagEvalRepository(factory)

    online_repo.save_request_run(
        {
            "request_id": "req_empty",
            "session_id": "sess_empty",
            "user_id": "user_1",
            "query": "What does memory-augmented RAG improve?",
            "query_type": "qa",
            "final_answer": "Memory improves citation-grounded answers.",
            "runtime_metrics": {"latency_ms": 100, "total_tokens": 20},
            "execution_health": {"has_error": False, "timeout": False},
        }
    )
    online_repo.save_rag_retrieval_event(
        {
            "event_id": "event_req_empty",
            "request_id": "req_empty",
            "query": "What does memory-augmented RAG improve?",
            "strategy": "hybrid",
            "top_k": 3,
            "latency_ms": 12,
            "returned_contexts": [],
            "returned_chunk_ids": [],
        }
    )
    with factory() as session:
        session.add(
            RequestRagEvalAnnotationModel(
                request_id="req_empty",
                reference="Memory improves citation-grounded answers.",
                required_points_json=json.dumps(["memory improves citation-grounded answers"]),
            )
        )
        session.commit()

    class _Evaluator:
        async def score(self, **_kwargs):
            return OfficialRagasScores(
                scores={
                    "faithfulness": None,
                    "answer_relevancy": 0.4,
                    "context_precision": None,
                    "context_recall": None,
                    "noise_sensitivity": None,
                    "semantic_similarity": 0.5,
                }
            )

    settings = SimpleNamespace(
        rag_eval_llm_model="mock-llm",
        llm_light_model="mock-light",
        llm_reasoning_model="mock-reasoning",
        rag_eval_embedding_model="mock-embedding",
        embedding_model="mock-embedding",
        rag_eval_redundancy_similarity_threshold=0.90,
        rag_eval_context_top_k=3,
        rag_eval_dataset_path="unused.jsonl",
        resolve_path=lambda value: value,
    )
    service = RagEvalService(
        settings,
        rag_repo,
        rag_engine=object(),
        ragas_evaluator=_Evaluator(),
        llm=object(),
        embedding_service=object(),
        online_eval_repository=online_repo,
    )

    computed = await service.score_online_request_if_annotated("req_empty")
    request = online_repo.get_request_eval("req_empty")
    events = online_repo.get_request_events("req_empty")

    assert computed is True
    assert request["rag_metrics"]["rag_score"] is not None
    assert request["rag_metrics"]["faithfulness"] == 0.0
    assert request["rag_metrics"]["context_precision"] == 0.0
    assert request["rag_metrics"]["context_recall"] == 0.0
    assert request["rag_metrics"]["noise_sensitivity"] == 1.0
    assert request["rag_metrics"]["completeness"] == 0.0
    assert events["rag_events"][0]["rag_score"] == request["rag_metrics"]["rag_score"]


def test_online_eval_repository_does_not_expose_legacy_write_aliases(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'online_eval.db'}", future=True)
    Base.metadata.create_all(engine)
    repo = OnlineEvalRepository(
        sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    )

    assert not hasattr(repo, "save_request_eval_run")
    assert not hasattr(repo, "save_rag_call_event")


def test_online_rag_eval_source_treats_saved_null_metrics_as_scored(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'online_eval_status.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    repo = OnlineEvalRepository(factory)

    repo.save_request_run(
        {
            "request_id": "req_partial_null",
            "session_id": "sess_1",
            "user_id": "user_1",
            "query": "What does memory improve?",
            "query_type": "qa",
            "final_answer": "Memory improves citation-grounded answers.",
            "runtime_metrics": {"latency_ms": 100},
            "execution_health": {"has_error": False, "timeout": False},
        }
    )
    repo.save_online_rag_metrics(
        "req_partial_null",
        {
            "faithfulness": None,
            "answer_relevancy": 0.5,
            "context_precision": None,
            "context_recall": None,
            "noise_sensitivity": None,
            "semantic_similarity": 0.6,
            "redundancy": 0.0,
            "completeness": None,
            "rag_score": None,
        },
    )

    source = repo.get_online_rag_eval_source("req_partial_null")
    request = repo.get_request_eval("req_partial_null")

    assert source is not None
    assert source["metrics_complete"] is True
    assert request["rag_eval_status"] == "scored"
    assert request["rag_scored_at"] is not None


def test_rag_eval_repository_truncates_persisted_contexts(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'rag_eval.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    repo = RagEvalRepository(factory)
    run = repo.create_run(
        dataset_name="rag_eval_v2",
        strategies=["hybrid"],
        metrics=["rag_score"],
        ragas_model="mock-llm",
        embedding_model="mock-embedding",
        sample_count=1,
    )
    repo.save_result(
        {
            "run_id": run.run_id,
            "case_id": "case_1",
            "strategy": "hybrid",
            "user_input": "What improves recall?",
            "response": "Hybrid retrieval improves recall.",
            "retrieved_chunk_ids": [f"chunk_{index}" for index in range(12)],
            "retrieved_contexts": ["x" * 3000 for _ in range(12)],
            "retrieval_latency_ms": 42,
            "redundancy": 0.0,
            "completeness": 1.0,
            "metric_errors": {},
        }
    )

    result = repo.list_results(run.run_id)[0]

    assert len(result.retrieved_contexts) == 10
    assert all(len(context) == 2000 for context in result.retrieved_contexts)


@pytest.mark.asyncio
async def test_rag_eval_api_uses_new_routes_and_old_routes_are_absent(monkeypatch):
    app = create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        cases_response = await client.get("/api/v1/eval/rag/cases")
        old_response = await client.post("/api/v1/eval/ragas", json={})

    assert cases_response.status_code == 200
    assert old_response.status_code == 404


@pytest.mark.asyncio
async def test_request_diagnosis_api_returns_score_based_summary():
    app = create_app()
    assert any(
        getattr(route, "path", "") == "/api/v1/eval/requests/{request_id}/diagnosis"
        for route in app.routes
    )

    class _Repo:
        def get_request_diagnosis(self, request_id: str):
            assert request_id == "req_diag"
            return {
                "request_id": request_id,
                "scores": {
                    "rag_score": 0.82,
                    "memory_score": None,
                    "answer_quality_score": 0.78,
                },
                "issues": ["Memory score is not available for this request."],
                "strengths": [
                    "RAG score is strong (0.82).",
                    "Answer score is strong (0.78).",
                ],
                "recommendations": ["Run Memory evaluation if this request should use memory."],
            }

    payload = await get_request_diagnosis(
        "req_diag",
        container=SimpleNamespace(online_eval_repository=_Repo(), rag_eval_service=None),
    )

    assert payload["success"] is True
    assert payload["data"]["issues"] == ["Memory score is not available for this request."]
    assert payload["data"]["strengths"] == [
        "RAG score is strong (0.82).",
        "Answer score is strong (0.78).",
    ]


@pytest.mark.asyncio
async def test_request_detail_and_diagnosis_do_not_score_rag_by_default():
    class _Repo:
        def get_request_eval(self, request_id: str):
            assert request_id == "req_detail"
            return {
                "request_id": request_id,
                "rag_score": None,
                "memory_score": None,
                "answer_quality_score": 0.5,
                "used_modules": {"answer": True, "rag": False, "memory": False},
            }

        def get_request_diagnosis(self, request_id: str):
            assert request_id == "req_detail"
            return {"request_id": request_id, "issues": [], "strengths": [], "recommendations": []}

    class _RagService:
        def __init__(self):
            self.calls = 0

        async def score_online_request_if_annotated(self, request_id: str):
            self.calls += 1
            return True

    rag_service = _RagService()
    container = SimpleNamespace(
        online_eval_repository=_Repo(),
        rag_eval_service=rag_service,
    )

    detail = await get_request_eval("req_detail", container=container)
    diagnosis = await get_request_diagnosis("req_detail", container=container)

    assert detail["success"] is True
    assert diagnosis["success"] is True
    assert rag_service.calls == 0


def test_init_database_removes_old_online_eval_schema(tmp_path):
    db_path = tmp_path / "legacy_eval.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE request_eval_runs (
                    request_id VARCHAR(64) NOT NULL PRIMARY KEY,
                    answer_quality_score FLOAT,
                    faithfulness_score FLOAT,
                    rag_score FLOAT,
                    rag_metrics_json TEXT
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE rag_call_events (
                    event_id VARCHAR(64) NOT NULL PRIMARY KEY,
                    request_id VARCHAR(64)
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE request_eval_judgements (
                    judgement_id VARCHAR(64) NOT NULL PRIMARY KEY,
                    request_id VARCHAR(64)
                )
                """
            )
        )

    class _Settings:
        database_url = f"sqlite:///{db_path}"

    init_database(_Settings())

    inspector = inspect(create_engine(f"sqlite:///{db_path}", future=True))
    tables = set(inspector.get_table_names())

    assert "request_eval_runs" not in tables
    assert "rag_call_events" not in tables
    assert "request_eval_judgements" not in tables
    assert "request_runs" in tables
    assert "rag_retrieval_events_v2" in tables
    assert "rag_eval_runs_v2" in tables


def test_run_request_defaults_to_new_metrics_only():
    request = RagEvalRunRequest()

    assert request.dataset_name == get_settings().rag_eval_default_dataset
    assert RetrievalStrategyName.HYBRID in request.strategies
    assert "answer_relevancy" in request.metrics
    assert "answer_quality_score" not in request.metrics


def test_run_request_rejects_empty_metrics():
    with pytest.raises(ValueError, match="metrics"):
        RagEvalRunRequest(metrics=[])
