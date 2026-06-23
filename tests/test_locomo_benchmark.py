from __future__ import annotations

import json
from collections import Counter
from datetime import date
from types import SimpleNamespace

import anyio
import pytest
from typer.testing import CliRunner

import scholar_mind.main as main_module
from scholar_mind.eval.locomo import (
    build_memory_locomo_dataset,
    normalize_answer,
    run_locomo_qa,
    score_answer,
    score_locomo_samples,
    validate_locomo_dataset,
)
from scholar_mind.main import cli_app
from scholar_mind.models.domain import QueryType, StructuredPaper
from scholar_mind.rag.top_k import FINAL_CITATION_TOP_K

_LOCOMO_TEST_INSTRUCTION = (
    "请只输出最终短答案，不要解释；如果答案包含多项，用英文逗号和空格分隔；"
    "如果没有足够记忆支持答案，回答 No information available."
)


def test_locomo_normalize_answer_matches_upstream_rules():
    assert normalize_answer("The RAG, and Re-Ranking!") == "rag reranking"


def test_locomo_score_answer_handles_category_specific_rules():
    assert score_answer("reranking, hybrid retrieval", "hybrid retrieval, reranking", 1) == 1.0
    assert score_answer("23 April 2026", "April 23, 2026", 2) == 1.0
    assert score_answer("No information available.", "No information available.", 5) == 1.0
    assert score_answer("无法从当前记忆中确定。", "", 5) == 1.0
    assert score_answer("It was GraphRAG.", "No information available.", 5) == 0.0


def test_score_locomo_samples_adds_f1_recall_and_report():
    samples = [
        {
            "sample_id": "sample_1",
            "conversation": {},
            "qa": [
                {
                    "question": "Which two techniques did the user compare?",
                    "answer": "hybrid retrieval, reranking",
                    "category": 1,
                    "evidence": ["paper_a::Methods", "paper_b::Results"],
                    "model_prediction": "reranking, hybrid retrieval",
                    "model_prediction_context": ["paper_b::Results"],
                },
                {
                    "question": "When was the paper published?",
                    "answer": "April 23, 2026",
                    "category": 2,
                    "evidence": ["paper_c::metadata"],
                    "model_prediction": "23 April 2026",
                    "model_prediction_context": ["paper_c::metadata"],
                },
                {
                    "question": "What private note did the user attach?",
                    "adversarial_answer": "GraphRAG",
                    "category": 5,
                    "evidence": ["paper_d::Methods"],
                    "model_prediction": "No information available.",
                    "model_prediction_context": ["paper_d::Methods"],
                },
            ],
        }
    ]

    scored_samples, report = score_locomo_samples(
        samples,
        prediction_key="model_prediction",
        model_name="model",
    )

    scored_qa = scored_samples[0]["qa"]
    assert scored_qa[0]["model_f1"] == 1.0
    assert scored_qa[0]["model_recall"] == 0.5
    assert scored_qa[1]["model_f1"] == 1.0
    assert scored_qa[1]["model_recall"] == 1.0
    assert scored_qa[2]["model_f1"] == 1.0
    assert scored_qa[2]["model_recall"] == 1.0

    assert report["model"]["question_count"] == 3
    assert report["model"]["category_counts"] == {"1": 1.0, "2": 1.0, "5": 1.0}
    assert report["model"]["accuracy_by_category"] == {"1": 1.0, "2": 1.0, "5": 1.0}
    assert report["model"]["overall_accuracy"] == 1.0
    assert report["model"]["recall_by_category"] == {"1": 0.5, "2": 1.0, "5": 1.0}


def test_score_locomo_samples_rejects_gold_prediction_key():
    samples = [
        {
            "sample_id": "sample_1",
            "conversation": {},
            "qa": [
                {
                    "question": "When was the paper published?",
                    "answer": "April 23, 2026",
                    "category": 2,
                    "evidence": ["paper_c::metadata"],
                }
            ],
        }
    ]

    try:
        score_locomo_samples(samples, prediction_key="answer", model_name="model")
    except ValueError as exc:
        assert "prediction_key must not be a gold answer field" in str(exc)
    else:
        raise AssertionError("score_locomo_samples accepted prediction_key='answer'")


class _PaperRepository:
    def __init__(self, papers: list[StructuredPaper]):
        self._papers = papers

    def all_papers(self):
        return list(self._papers)

    def list_chunks(self, _filters=None):
        return [
            {
                "chunk_id": f"{paper.paper_id}::{_section_for_paper(paper)}",
                "paper_id": paper.paper_id,
                "title": paper.title,
                "section": _section_for_paper(paper),
                "content": f"{paper.title} uses a deterministic evaluation protocol.",
                "categories": paper.categories,
                "publish_date": paper.publish_date,
            }
            for paper in self._papers
        ]


def test_build_memory_locomo_dataset_creates_balanced_business_memory_question_set():
    papers = [
        StructuredPaper(
            paper_id=f"2604.{index:05d}",
            title=f"Memory Evaluation Paper {index}",
            authors=["Researcher"],
            abstract=f"Paper {index} studies memory-aware research assistants.",
            categories=["cs.CL" if index % 2 else "cs.LG"],
            publish_date=date(2026, 4, 22 if index % 2 else 23),
        )
        for index in range(1, 81)
    ]

    dataset = build_memory_locomo_dataset(_PaperRepository(papers), question_count=150)
    summary = validate_locomo_dataset(dataset, expected_question_count=150)
    qa = dataset[0]["qa"]
    evidence_turns = {
        turn["dia_id"]: turn
        for key, value in dataset[0]["conversation"].items()
        if key.startswith("session_") and isinstance(value, list)
        for turn in value
    }
    evidence_id_list = [
        turn["dia_id"]
        for key, value in dataset[0]["conversation"].items()
        if key.startswith("session_") and isinstance(value, list)
        for turn in value
    ]

    assert summary == {
        "sample_count": 1,
        "question_count": 150,
        "category_counts": {"1": 30, "2": 30, "3": 30, "4": 30, "5": 30},
    }
    assert Counter(item["category"] for item in qa) == {1: 30, 2: 30, 3: 30, 4: 30, 5: 30}
    assert len(evidence_id_list) == len(evidence_turns)
    assert all(item["answer"] for item in qa if item["category"] != 5)
    assert all(
        evidence in evidence_turns
        for item in qa
        if item["category"] != 5
        for evidence in item["evidence"]
    )
    assert all("answer" not in item for item in qa if item["category"] == 5)
    assert all(item.get("adversarial_answer") for item in qa if item["category"] == 5)
    assert all(item["evidence"] for item in qa if item["category"] == 5)
    assert all(
        len(
            {
                item["metadata"]["template_id"]
                for item in qa
                if item["category"] == category
            }
        )
        >= 4
        for category in {1, 2, 3, 4, 5}
    )
    assert len({item["question"] for item in qa}) >= 130
    assert {
        item["metadata"]["question_kind"]
        for item in qa
    } == {
        "memory_multi_hop",
        "memory_temporal_update",
        "memory_inference",
        "memory_business_personalization",
        "memory_adversarial_confusable",
    }
    assert not any("paper id" in item["question"].lower() for item in qa)
    assert not any("Which remembered paper has paper id" in item["question"] for item in qa)
    assert any("请记住" in turn["text"] for turn in evidence_turns.values())
    assert any("默认输出" in turn["text"] for turn in evidence_turns.values())
    assert any("更新" in turn["text"] for turn in evidence_turns.values())
    assert any("我的背景" in turn["text"] for turn in evidence_turns.values())
    assert all(turn["metadata"]["case_id"] in turn["text"] for turn in evidence_turns.values())
    assert all(
        evidence_turns[evidence]["metadata"]["memory_type"]
        in {
            "preference",
            "research_interest",
            "knowledge_level",
            "goal",
            "workflow",
            "project_constraint",
            "paper_read",
            "feedback",
        }
        for item in qa
        if item["category"] != 5
        for evidence in item["evidence"]
    )


def test_build_memory_locomo_dataset_preserves_full_memory_evidence_text():
    long_abstract = " ".join(["abstract-token"] * 60) + " abstract-tail-marker"
    papers = [
        StructuredPaper(
            paper_id="2604.00001",
            title=f"Long Abstract Paper {long_abstract}",
            authors=["Researcher"],
            abstract=long_abstract,
            categories=["cs.CL"],
            publish_date=date(2026, 4, 22),
        ),
        StructuredPaper(
            paper_id="2604.00002",
            title="Long Section Paper",
            authors=["Researcher"],
            abstract="Short abstract.",
            categories=["cs.LG"],
            publish_date=date(2026, 4, 23),
        ),
    ]

    dataset = build_memory_locomo_dataset(_PaperRepository(papers), question_count=5)
    turns = [
        turn
        for key, value in dataset[0]["conversation"].items()
        if key.startswith("session_") and isinstance(value, list)
        for turn in value
    ]

    assert any("abstract-tail-marker" in turn["text"] for turn in turns)


class _StreamingResearchService:
    def __init__(self):
        self.calls = []

    async def stream(self, *, query, user_id, session_id, query_type, request_payload):
        self.calls.append(
            {
                "query": query,
                "user_id": user_id,
                "session_id": session_id,
                "query_type": query_type,
                "request_payload": request_payload,
            }
        )
        yield (
            "answer",
            {
                "answer": "streamed answer",
                "citations": [{"paper_id": "2604.00001", "section": "Methods"}],
            },
        )
        yield ("done", {"session_id": session_id})


class _FailingSecondQuestionService(_StreamingResearchService):
    async def stream(self, *, query, user_id, session_id, query_type, request_payload):
        if query.startswith("Question 2"):
            raise RuntimeError("stream failed")
        async for item in super().stream(
            query=query,
            user_id=user_id,
            session_id=session_id,
            query_type=query_type,
            request_payload=request_payload,
        ):
            yield item


def test_run_locomo_qa_uses_qa_stream_path_without_memory_extraction():
    service = _StreamingResearchService()
    samples = [
        {
            "sample_id": "sample_1",
            "conversation": {},
            "qa": [
                {
                    "question": "Which remembered paper matches this preference?",
                    "answer": "Memory Evaluation Paper",
                    "category": 2,
                    "evidence": ["2604.00001::Methods"],
                }
            ],
        }
    ]

    predicted = anyio.run(
        run_locomo_qa,
        service,
        samples,
        "locomo-user",
        "model_prediction",
    )

    assert service.calls == [
        {
            "query": (
                "Which remembered paper matches this preference?\n\n"
                f"{_LOCOMO_TEST_INSTRUCTION}"
            ),
            "user_id": "locomo-user",
            "session_id": "locomo-user-q001",
            "query_type": QueryType.QA,
            "request_payload": {
                "paper_ids": [],
                "rag_strategy": "hybrid",
                "top_k": FINAL_CITATION_TOP_K,
                "conditional_memory_injection": False,
                "memory_extraction_enabled": False,
            },
        }
    ]
    assert predicted[0]["qa"][0]["model_prediction"] == "streamed answer"
    assert predicted[0]["qa"][0]["model_prediction_context"] == ["2604.00001::Methods"]


def test_run_locomo_qa_writes_progress_after_each_question(tmp_path):
    service = _FailingSecondQuestionService()
    progress_file = tmp_path / "predictions.json"
    samples = [
        {
            "sample_id": "sample_1",
            "conversation": {},
            "qa": [
                {
                    "question": "Question 1",
                    "answer": "Answer 1",
                    "category": 2,
                    "evidence": ["2604.00001::Methods"],
                },
                {
                    "question": "Question 2",
                    "answer": "Answer 2",
                    "category": 2,
                    "evidence": ["2604.00002::Methods"],
                },
            ],
        }
    ]

    with pytest.raises(RuntimeError, match="stream failed"):
        anyio.run(
            run_locomo_qa,
            service,
            samples,
            "locomo-user",
            "model_prediction",
            None,
            progress_file,
        )

    predictions = json.loads(progress_file.read_text(encoding="utf-8"))
    assert predictions[0]["qa"][0]["model_prediction"] == "streamed answer"
    assert "model_prediction" not in predictions[0]["qa"][1]


def test_run_locomo_qa_rejects_gold_prediction_key_before_streaming():
    service = _StreamingResearchService()
    samples = [
        {
            "sample_id": "sample_1",
            "conversation": {},
            "qa": [
                {
                    "question": "Which remembered paper matches this preference?",
                    "answer": "Memory Evaluation Paper",
                    "category": 2,
                    "evidence": ["2604.00001::Methods"],
                }
            ],
        }
    ]

    try:
        anyio.run(run_locomo_qa, service, samples, "locomo-user", "answer")
    except ValueError as exc:
        assert "prediction_key must not be a gold answer field" in str(exc)
    else:
        raise AssertionError("run_locomo_qa accepted prediction_key='answer'")
    assert service.calls == []


def test_run_locomo_qa_limit_keeps_only_first_n_questions():
    service = _StreamingResearchService()
    samples = [
        {
            "sample_id": "sample_1",
            "conversation": {},
            "qa": [
                {
                    "question": f"Question {index}",
                    "answer": f"Answer {index}",
                    "category": 2,
                    "evidence": ["2604.00001::Methods"],
                }
                for index in range(1, 4)
            ],
        }
    ]

    predicted = anyio.run(
        run_locomo_qa,
        service,
        samples,
        "locomo-user",
        "model_prediction",
        2,
    )

    assert [qa["question"] for qa in predicted[0]["qa"]] == ["Question 1", "Question 2"]
    assert len(service.calls) == 2
    assert [call["session_id"] for call in service.calls] == [
        "locomo-user-q001",
        "locomo-user-q002",
    ]


def test_run_locomo_qa_limit_seeds_only_selected_question_evidence():
    service = _StreamingResearchService()
    samples = [
        {
            "sample_id": "sample_1",
            "conversation": {
                "session_1_date_time": "2026-05-01",
                "session_1": [
                    {
                        "speaker": "ScholarMind user",
                        "dia_id": "case_001::role",
                        "text": "请记住：我主要关注检索增强生成。",
                        "metadata": {"seed_memory": True, "case_id": "case_001"},
                    },
                    {
                        "speaker": "ScholarMind user",
                        "dia_id": "case_001::output",
                        "text": "请记住：回答时默认输出实验设计。",
                        "metadata": {"seed_memory": True, "case_id": "case_001"},
                    },
                    {
                        "speaker": "ScholarMind user",
                        "dia_id": "case_001::background",
                        "text": "请记住：我已经读过基础综述。",
                        "metadata": {"seed_memory": True, "case_id": "case_001"},
                    },
                    {
                        "speaker": "ScholarMind user",
                        "dia_id": "case_002::role",
                        "text": "请记住：另一个项目关注代码智能体。",
                        "metadata": {"seed_memory": True, "case_id": "case_002"},
                    },
                ],
            },
            "qa": [
                {
                    "question": "我关注什么方向，默认输出什么？",
                    "answer": "检索增强生成，实验设计",
                    "category": 1,
                    "evidence": ["case_001::role", "case_001::output"],
                    "metadata": {"case_id": "case_001"},
                },
                {
                    "question": "另一个项目关注什么？",
                    "answer": "代码智能体",
                    "category": 2,
                    "evidence": ["case_002::role"],
                    "metadata": {"case_id": "case_002"},
                },
            ],
        }
    ]

    predicted = anyio.run(
        run_locomo_qa,
        service,
        samples,
        "locomo-user",
        "model_prediction",
        1,
    )

    assert [qa["question"] for qa in predicted[0]["qa"]] == [
        "我关注什么方向，默认输出什么？"
    ]
    assert [call["query"] for call in service.calls] == [
        "请记住：我主要关注检索增强生成。",
        "请记住：回答时默认输出实验设计。",
        f"我关注什么方向，默认输出什么？\n\n{_LOCOMO_TEST_INSTRUCTION}",
    ]
    assert [call["session_id"] for call in service.calls] == [
        "locomo-user-seed-s001",
        "locomo-user-seed-s001",
        "locomo-user-q001",
    ]
    assert [call["query_type"] for call in service.calls] == [
        QueryType.QA,
        QueryType.QA,
        QueryType.QA,
    ]


def test_run_locomo_qa_uses_incrementing_user_prefixed_sessions():
    service = _StreamingResearchService()
    samples = [
        {
            "sample_id": "sample_1",
            "conversation": {},
            "qa": [
                {
                    "question": f"Question {index}",
                    "answer": f"Answer {index}",
                    "category": 2,
                    "evidence": ["2604.00001::Methods"],
                }
                for index in range(1, 3)
            ],
        }
    ]

    anyio.run(
        run_locomo_qa,
        service,
        samples,
        "locomo-user",
        "model_prediction",
    )

    assert [call["session_id"] for call in service.calls] == [
        "locomo-user-q001",
        "locomo-user-q002",
    ]


def test_run_locomo_qa_seeds_memory_history_before_questions():
    service = _StreamingResearchService()
    samples = [
        {
            "sample_id": "sample_1",
            "conversation": {
                "session_1_date_time": "2026-05-01",
                "session_1": [
                    {
                        "speaker": "ScholarMind user",
                        "dia_id": "D1:1",
                        "text": "请记住：我默认希望论文精读先讲方法，再讲局限。",
                        "metadata": {"seed_memory": True},
                    },
                    {
                        "speaker": "ScholarMind assistant",
                        "dia_id": "D1:2",
                        "text": "已记住。",
                    },
                ],
            },
            "qa": [
                {
                    "question": "基于我之前的偏好，我默认希望论文精读先讲什么？",
                    "answer": "方法",
                    "category": 2,
                    "evidence": ["D1:1"],
                }
            ],
        }
    ]

    anyio.run(
        run_locomo_qa,
        service,
        samples,
        "locomo-user",
        "model_prediction",
    )

    assert [call["query"] for call in service.calls] == [
        "请记住：我默认希望论文精读先讲方法，再讲局限。",
        f"基于我之前的偏好，我默认希望论文精读先讲什么？\n\n{_LOCOMO_TEST_INSTRUCTION}",
    ]
    assert [call["session_id"] for call in service.calls] == [
        "locomo-user-seed-s001",
        "locomo-user-q001",
    ]
    assert service.calls[0]["request_payload"]["conditional_memory_injection"] is False
    assert service.calls[1]["request_payload"]["conditional_memory_injection"] is False
    assert service.calls[0]["query_type"] == QueryType.QA
    assert service.calls[1]["query_type"] == QueryType.QA
    assert service.calls[0]["request_payload"]["memory_extraction_enabled"] is True
    assert service.calls[0]["request_payload"]["request_memory_extraction_enabled"] is False
    assert service.calls[1]["request_payload"]["memory_extraction_enabled"] is False


def test_locomo_run_cli_accepts_user_id_for_session_prefix(tmp_path, monkeypatch):
    service = _StreamingResearchService()
    monkeypatch.setattr(
        main_module,
        "get_container",
        lambda: SimpleNamespace(research_service=service),
    )
    data_file = tmp_path / "dataset.json"
    out_file = tmp_path / "predictions.json"
    data_file.write_text(
        json.dumps(
            [
                {
                    "sample_id": "sample_1",
                    "conversation": {},
                    "qa": [
                        {
                            "question": "Which remembered paper matches this preference?",
                            "answer": "Memory Evaluation Paper",
                            "category": 2,
                            "evidence": ["2604.00001::Methods"],
                        },
                        {
                            "question": "Which remembered paper matches this second preference?",
                            "answer": "Memory Evaluation Paper 2",
                            "category": 2,
                            "evidence": ["2604.00002::Methods"],
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli_app,
        [
            "eval",
            "locomo-run",
            "--data-file",
            str(data_file),
            "--out-file",
            str(out_file),
            "--user-id",
            "custom-user",
            "--limit",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert [call["user_id"] for call in service.calls] == ["custom-user", "custom-user"]
    assert [call["session_id"] for call in service.calls] == [
        "custom-user-q001",
        "custom-user-q002",
    ]
    predictions = json.loads(out_file.read_text(encoding="utf-8"))
    assert [qa["scholarmind_prediction"] for qa in predictions[0]["qa"]] == [
        "streamed answer",
        "streamed answer",
    ]


def test_locomo_run_cli_does_not_expose_session_id_option():
    result = CliRunner().invoke(cli_app, ["eval", "locomo-run", "--help"])

    assert result.exit_code == 0, result.output
    assert "--user-id" in result.output
    assert "--session-id" not in result.output


def _section_for_paper(paper: StructuredPaper) -> str:
    return "metadata" if paper.paper_id.endswith("00001") else "Methods"


def test_locomo_build_cli_writes_dataset(tmp_path, monkeypatch):
    papers = [
        StructuredPaper(
            paper_id=f"2604.{index:05d}",
            title=f"Memory Evaluation Paper {index}",
            authors=["Researcher"],
            abstract=f"Paper {index} studies memory-aware research assistants.",
            categories=["cs.CL" if index % 2 else "cs.LG"],
            publish_date=date(2026, 4, 22 if index % 2 else 23),
        )
        for index in range(1, 81)
    ]
    monkeypatch.setattr(
        main_module,
        "get_container",
        lambda: SimpleNamespace(paper_repository=_PaperRepository(papers)),
    )
    out_file = tmp_path / "scholarmind_locomo_150.json"

    result = CliRunner().invoke(
        cli_app,
        [
            "eval",
            "locomo-build",
            "--out-file",
            str(out_file),
            "--question-count",
            "150",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["path"] == str(out_file)
    assert payload["question_count"] == 150
    assert out_file.exists()


def test_locomo_score_cli_writes_scored_predictions_and_stats(tmp_path):
    prediction_file = tmp_path / "predictions.json"
    prediction_file.write_text(
        json.dumps(
            [
                {
                    "sample_id": "sample_1",
                    "conversation": {},
                    "qa": [
                        {
                            "question": "When was the paper published?",
                            "answer": "April 23, 2026",
                            "category": 2,
                            "evidence": ["paper_c::metadata"],
                            "model_prediction": "23 April 2026",
                            "model_prediction_context": ["paper_c::metadata"],
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    scored_file = tmp_path / "scored.json"
    stats_file = tmp_path / "stats.json"

    result = CliRunner().invoke(
        cli_app,
        [
            "eval",
            "locomo-score",
            "--prediction-file",
            str(prediction_file),
            "--out-file",
            str(scored_file),
            "--stats-file",
            str(stats_file),
            "--prediction-key",
            "model_prediction",
            "--model-name",
            "model",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["scored_file"] == str(scored_file)
    assert payload["stats_file"] == str(stats_file)
    assert json.loads(scored_file.read_text(encoding="utf-8"))[0]["qa"][0]["model_f1"] == 1.0
    assert json.loads(stats_file.read_text(encoding="utf-8"))["model"]["overall_accuracy"] == 1.0
