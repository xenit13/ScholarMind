from __future__ import annotations

import pytest

from scholar_mind.eval.locomo_build.validate import (
    ValidationError,
    run_gold_check,
    run_random_check,
    run_structural_check,
    validate_samples,
)


def _turn(speaker: str, dia_id: str, text: str) -> dict:
    return {
        "speaker": speaker,
        "dia_id": dia_id,
        "text": text,
        "metadata": {
            "seed_id": None,
            "memory_type": None,
            "is_distractor": True,
        },
    }


def _make_minimal_sample(category=1) -> dict:
    return {
        "sample_id": "p01",
        "persona": {"persona_id": "p01", "user_id": "u", "background": "b"},
        "conversation": {
            "speaker_a": "user",
            "speaker_b": "assistant",
            "session_1_date_time": "2026-05-03",
            "session_1": [
                _turn("user", "s1:1", "x"),
                _turn("assistant", "s1:2", "y"),
            ],
        },
        "qa": [
            {
                "question": "q",
                "answer": "anchor paper",
                "category": category,
                "evidence": ["s1:1"],
                "metadata": {
                    "question_kind": "memory_single_hop",
                    "template_id": "t",
                    "memory_focus": ["paper_read"],
                    "case_id": "case_001",
                    "distractor_case_id": None,
                },
            }
        ],
    }


def _make_cat5_sample(no_info_count: int = 6, total: int = 12) -> dict:
    sample = _make_minimal_sample(category=5)
    qas = []
    for i in range(total):
        qas.append(
            {
                "question": f"q{i}",
                "answer": "no information available" if i < no_info_count else f"answer{i}",
                "category": 5,
                "evidence": ["s1:1"],
                "metadata": {
                    "question_kind": "memory_adversarial",
                    "template_id": f"t{i}",
                    "memory_focus": ["confusable_memory"],
                    "case_id": "case_001",
                    "distractor_case_id": "case_002",
                },
            }
        )
    sample["qa"] = qas
    return sample


def test_run_gold_check_passes_when_prediction_equals_answer():
    sample = _make_minimal_sample()
    for qa in sample["qa"]:
        qa["gold_prediction"] = qa["answer"]
    report = run_gold_check([sample])
    assert report["overall_accuracy"] == 1.0


def test_run_random_check_passes_when_score_low():
    sample = _make_minimal_sample()
    for qa in sample["qa"]:
        qa["random_prediction"] = "zzz totally unrelated random string zzz"
    report = run_random_check([sample])
    assert report["overall_accuracy"] <= 0.05


def test_run_structural_check_passes_on_valid_sample():
    sample = _make_minimal_sample()
    run_structural_check([sample])


def test_run_structural_check_fails_when_evidence_dia_id_missing():
    sample = _make_minimal_sample()
    sample["qa"][0]["evidence"] = ["s9:99"]
    with pytest.raises(ValidationError, match="evidence dia_id"):
        run_structural_check([sample])


def test_run_structural_check_passes_cat5_no_info_ratio_at_50_pct():
    sample = _make_cat5_sample(no_info_count=6, total=12)
    run_structural_check([sample])


def test_run_structural_check_fails_when_cat5_no_info_ratio_too_low():
    sample = _make_cat5_sample(no_info_count=3, total=12)
    with pytest.raises(ValidationError, match="cat5 no_info ratio"):
        run_structural_check([sample])


def test_validate_samples_aggregates_all_checks():
    sample = _make_minimal_sample()
    for qa in sample["qa"]:
        qa["gold_prediction"] = qa["answer"]
        qa["random_prediction"] = "unrelated random"
    report = validate_samples([sample])
    assert report["gold_overall_accuracy"] == 1.0
    assert report["random_overall_accuracy"] <= 0.05
    assert report["structural_check_passed"] is True
