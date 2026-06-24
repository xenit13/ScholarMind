from __future__ import annotations

import pytest

from scholar_mind.eval.locomo import (
    normalize_answer,
    score_answer,
    score_locomo_samples,
)


def test_normalize_answer_lowercases_and_strips_punctuation():
    assert normalize_answer("Hello, World!") == "hello world"


def test_normalize_answer_removes_articles():
    assert normalize_answer("the anchor paper") == "anchor paper"


def test_score_answer_cat1_exact_match():
    assert score_answer("anchor paper", "anchor paper", category=1) == 1.0


def test_score_answer_cat1_partial_match():
    score = score_answer("anchor paper baseline candidate", "anchor paper", category=1)
    assert 0.0 < score < 1.0


def test_score_answer_cat2_uses_f1():
    score = score_answer("method assumptions failure modes", "method assumptions", category=2)
    assert 0.0 < score <= 1.0


def test_score_answer_cat5_no_info_match_when_gold_is_no_info():
    # Gold answer is "no information available" → prediction must also be a refusal
    assert score_answer("no information available", "no information available", category=5) == 1.0
    assert score_answer("无法确定", "no information available", category=5) == 1.0


def test_score_answer_cat5_actual_answer_scores_zero_when_gold_is_no_info():
    # Gold is no-info but prediction is a real answer → 0
    assert score_answer("anchor paper", "no information available", category=5) == 0.0


def test_score_answer_cat5_real_answer_uses_f1():
    # Gold is a real cross-case answer → use F1 like cat 2-4
    assert score_answer("anchor paper", "anchor paper", category=5) == 1.0
    score = score_answer("anchor paper baseline candidate", "anchor paper", category=5)
    assert 0.0 < score < 1.0
    assert score_answer("totally unrelated text", "anchor paper", category=5) == 0.0


def test_score_answer_unsupported_category():
    with pytest.raises(ValueError, match="Unsupported LOCOMO category"):
        score_answer("x", "y", category=6)


def _make_sample(qas: list[dict]) -> dict:
    return {"sample_id": "x", "persona": {}, "conversation": {}, "qa": qas}


def test_score_locomo_samples_gold_run_returns_perfect_accuracy():
    sample = _make_sample(
        [
            {
                "question": "q",
                "answer": "anchor paper",
                "category": 1,
                "evidence": ["s1:1"],
                "metadata": {},
                "gold_prediction": "anchor paper",
            }
        ]
    )
    scored, report = score_locomo_samples(
        [sample], prediction_key="gold_prediction", model_name="gold"
    )
    assert report["gold"]["overall_accuracy"] == 1.0
    assert scored[0]["qa"][0]["gold_f1"] == 1.0


def test_score_locomo_samples_rejects_gold_field_as_prediction_key():
    with pytest.raises(ValueError, match="prediction_key must not"):
        score_locomo_samples([_make_sample([])], prediction_key="answer", model_name="x")


def test_score_locomo_samples_rejects_adversarial_field():
    with pytest.raises(ValueError, match="prediction_key must not"):
        score_locomo_samples(
            [_make_sample([])], prediction_key="adversarial_answer", model_name="x"
        )
