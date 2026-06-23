from __future__ import annotations

import copy
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_GOLD_ANSWER_FIELDS = {"answer", "adversarial_answer"}
_CATEGORY_ORDER = (1, 2, 3, 4, 5)

_NO_INFORMATION_PATTERNS = (
    "no information available",
    "not mentioned",
    "cannot determine",
    "can't determine",
    "unable to determine",
    "not enough information",
    "no evidence",
    "don't know",
    "无法确定",
    "无法判断",
    "不能确定",
    "没有相关信息",
    "未提及",
    "没有提到",
)

_ARTICLE_RE = re.compile(r"\b(a|an|the|and)\b", flags=re.IGNORECASE)

try:
    from nltk.stem import PorterStemmer

    _STEMMER = PorterStemmer()
except Exception:  # pragma: no cover - nltk optional
    _STEMMER = None


def normalize_answer(value: Any) -> str:
    """Lowercase, strip punctuation, remove articles, collapse whitespace."""
    text = str(value).replace(",", "")
    text = text.lower()
    text = "".join(c for c in text if c not in set(string.punctuation))
    text = _ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def _stem(token: str) -> str:
    return _STEMMER.stem(token) if _STEMMER else token


def _f1_score(prediction: str, answer: str) -> float:
    pred_tokens = [_stem(t) for t in normalize_answer(prediction).split()]
    ans_tokens = [_stem(t) for t in normalize_answer(answer).split()]
    if not pred_tokens or not ans_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ans_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ans_tokens)
    return 2 * precision * recall / (precision + recall)


def _multi_answer_f1(prediction: str, answer: str) -> float:
    preds = [s.strip() for s in prediction.split(",") if s.strip()]
    answers = [s.strip() for s in answer.split(",") if s.strip()]
    if not answers:
        return 0.0
    if not preds:
        preds = [prediction]
    matched = sum(max(_f1_score(c, g) for c in preds) for g in answers)
    return matched / len(answers)


def _is_no_information_response(prediction: str) -> bool:
    lowered = str(prediction).lower()
    return any(p in lowered for p in _NO_INFORMATION_PATTERNS)


def score_answer(prediction: Any, answer: Any, category: int | str) -> float:
    """Score a single prediction against gold answer for given LOCOMO category (1-5).

    Category 5 (adversarial) has two valid answer shapes:
    - Gold answer is "no information available" → prediction must also be a refusal.
    - Gold answer is a real cross-case answer → prediction must match via F1.
    """
    cat = int(category)
    if cat == 5:
        if _is_no_information_response(answer):
            return 1.0 if _is_no_information_response(prediction) else 0.0
        return _f1_score(str(prediction), str(answer))
    if cat == 1:
        return _multi_answer_f1(str(prediction), str(answer))
    if cat in {2, 3, 4}:
        return _f1_score(str(prediction), str(answer))
    raise ValueError(f"Unsupported LOCOMO category: {category}")


def _validate_prediction_key(prediction_key: str) -> None:
    if prediction_key in _GOLD_ANSWER_FIELDS:
        raise ValueError(
            f"prediction_key must not be a gold answer field: {prediction_key}"
        )


def _evidence_recall(evidence: list[str], context: list[str]) -> float:
    if not evidence:
        return 0.0
    ctx = set(context)
    return sum(item in ctx for item in evidence) / len(evidence)


def score_locomo_samples(
    samples: list[dict[str, Any]],
    *,
    prediction_key: str,
    model_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score samples in-place (deep-copied) and return (scored_samples, report)."""
    _validate_prediction_key(prediction_key)
    scored = copy.deepcopy(samples)
    cat_counts: dict[str, float] = defaultdict(float)
    cat_scores: dict[str, float] = defaultdict(float)
    recall_counts: dict[str, float] = defaultdict(float)
    recall_scores: dict[str, float] = defaultdict(float)
    prediction_nonempty = 0
    question_count = 0
    for sample in scored:
        for qa in sample.get("qa", []):
            question_count += 1
            cat = str(int(qa["category"]))
            prediction = qa.get(prediction_key, "")
            if str(prediction).strip():
                prediction_nonempty += 1
            f1_value = round(
                score_answer(prediction, qa.get("answer", ""), qa["category"]), 3
            )
            qa[f"{model_name}_f1"] = f1_value
            cat_counts[cat] += 1.0
            cat_scores[cat] += f1_value
            ctx_key = f"{prediction_key}_context"
            evidence = [str(e) for e in qa.get("evidence", []) if str(e)]
            if ctx_key in qa and evidence:
                ctx = [str(c) for c in qa.get(ctx_key, []) if str(c)]
                recall = round(_evidence_recall(evidence, ctx), 3)
                qa[f"{model_name}_recall"] = recall
                recall_counts[cat] += 1.0
                recall_scores[cat] += recall

    total = sum(cat_scores.values())
    report: dict[str, Any] = {
        model_name: {
            "category_counts": dict(cat_counts),
            "cum_accuracy_by_category": dict(cat_scores),
            "accuracy_by_category": {
                c: round(cat_scores[c] / n, 3) for c, n in cat_counts.items() if n
            },
            "overall_accuracy": round(total / question_count, 3)
            if question_count
            else 0.0,
            "question_count": question_count,
            "prediction_nonempty": prediction_nonempty,
        }
    }
    if recall_counts:
        report[model_name]["recall_by_category"] = {
            c: round(recall_scores[c] / n, 3)
            for c, n in recall_counts.items()
            if n
        }
        report[model_name]["cum_recall_by_category"] = dict(recall_scores)
    return scored, report


def score_prediction_file(
    prediction_file: Path,
    *,
    out_file: Path | None = None,
    stats_file: Path | None = None,
    prediction_key: str,
    model_name: str,
) -> dict[str, Any]:
    """Score a prediction JSON file and write scored + stats files."""
    samples = json.loads(prediction_file.read_text(encoding="utf-8"))
    scored_samples, report = score_locomo_samples(
        samples, prediction_key=prediction_key, model_name=model_name
    )
    target_out = out_file or prediction_file.with_name(
        f"{prediction_file.stem}_scored{prediction_file.suffix}"
    )
    target_stats = stats_file or prediction_file.with_name(
        f"{prediction_file.stem}_stats{prediction_file.suffix}"
    )
    target_out.parent.mkdir(parents=True, exist_ok=True)
    target_stats.parent.mkdir(parents=True, exist_ok=True)
    target_out.write_text(
        json.dumps(scored_samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    target_stats.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "prediction_file": str(prediction_file),
        "scored_file": str(target_out),
        "stats_file": str(target_stats),
        "report": report,
    }
