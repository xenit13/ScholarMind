from __future__ import annotations

import copy
import random
from typing import Any

from scholar_mind.eval.locomo import _is_no_information_response, score_locomo_samples


class ValidationError(Exception):
    pass


_NO_INFO_MIN_RATIO = 0.40


def _all_conversation_dia_ids(sample: dict[str, Any]) -> set[str]:
    """Collect all dia_ids from session_1..session_N turn lists."""
    out: set[str] = set()
    conv = sample.get("conversation", {})
    for key, value in conv.items():
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        if isinstance(value, list):
            for turn in value:
                dia_id = turn.get("dia_id")
                if dia_id:
                    out.add(dia_id)
    return out


def run_gold_check(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Set each qa's gold_prediction = answer, score, return the gold report dict."""
    for sample in samples:
        for qa in sample.get("qa", []):
            qa["gold_prediction"] = qa["answer"]
    _, report = score_locomo_samples(
        samples, prediction_key="gold_prediction", model_name="gold"
    )
    return report["gold"]


def run_random_check(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Populate missing random_predictions by sampling non-no-info answers; score; return report.

    QAs that already have a ``random_prediction`` set are left untouched so callers may
    inject their own random predictions (e.g. for tests or controlled baselines).
    """
    all_answers = [
        qa["answer"]
        for sample in samples
        for qa in sample.get("qa", [])
        if qa.get("answer") and not _is_no_information_response(qa["answer"])
    ]
    rng = random.Random(0)
    for sample in samples:
        for qa in sample.get("qa", []):
            if qa.get("random_prediction") is None:
                qa["random_prediction"] = (
                    rng.choice(all_answers) if all_answers else "unknown"
                )
    _, report = score_locomo_samples(
        samples, prediction_key="random_prediction", model_name="random"
    )
    return report["random"]


def run_structural_check(samples: list[dict[str, Any]]) -> None:
    """Verify evidence dia_ids exist + cat5 no_info ratio >= 40%.

    Raises ``ValidationError`` on failure.
    """
    for sample_idx, sample in enumerate(samples):
        dia_ids = _all_conversation_dia_ids(sample)
        cat5_qas = []
        for qa_idx, qa in enumerate(sample.get("qa", [])):
            for eid in qa.get("evidence", []):
                if eid not in dia_ids:
                    raise ValidationError(
                        f"sample[{sample_idx}].qa[{qa_idx}] evidence dia_id"
                        f" {eid!r} not in conversation"
                    )
            if int(qa["category"]) == 5:
                cat5_qas.append(qa)
        if cat5_qas:
            no_info_count = sum(
                1 for qa in cat5_qas if _is_no_information_response(qa["answer"])
            )
            ratio = no_info_count / len(cat5_qas)
            if ratio < _NO_INFO_MIN_RATIO:
                raise ValidationError(
                    f"sample[{sample_idx}] cat5 no_info ratio {ratio:.2f} < {_NO_INFO_MIN_RATIO}"
                )


def validate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Run all 3 checks (gold, random, structural) on deep copies; return aggregated report."""
    gold_report = run_gold_check(copy.deepcopy(samples))
    random_report = run_random_check(copy.deepcopy(samples))
    structural_samples = copy.deepcopy(samples)
    structural_passed = True
    try:
        run_structural_check(structural_samples)
    except ValidationError:
        structural_passed = False
    return {
        "gold_overall_accuracy": gold_report["overall_accuracy"],
        "gold_by_category": gold_report["accuracy_by_category"],
        "random_overall_accuracy": random_report["overall_accuracy"],
        "random_by_category": random_report["accuracy_by_category"],
        "structural_check_passed": structural_passed,
    }
