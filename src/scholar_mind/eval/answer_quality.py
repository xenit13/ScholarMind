"""Answer-only quality scoring for online request dashboards."""

from __future__ import annotations

import re

from scholar_mind.utils.text import STOPWORDS

ANSWER_SCORE_WEIGHTS = {
    "intent_alignment": 0.25,
    "task_coverage": 0.25,
    "specificity": 0.15,
    "reasoning_quality": 0.15,
    "format_compliance": 0.10,
    "clarity": 0.10,
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*|[\u4e00-\u9fff]")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?。！？；;]\s*")
_TASK_SPLIT_RE = re.compile(r"[,;，；、]|(?:\s+and\s+)|(?:\s+or\s+)|和|并|以及")
_FAILURE_RE = re.compile(
    r"(traceback|exception|error:|timeout|timed out|request failed|internal server error|"
    r"service unavailable|抱歉.*出错|请求失败|超时|异常)",
    re.IGNORECASE,
)
_CJK_STOP_CHARS = set("的一是在和与或及并了为对就都而及其中请帮我你他她它们个")
_REASONING_MARKERS = {
    "because",
    "therefore",
    "thus",
    "however",
    "first",
    "second",
    "third",
    "tradeoff",
    "tradeoffs",
    "recommendation",
    "recommendations",
    "原因",
    "因此",
    "所以",
    "首先",
    "其次",
    "最后",
    "但是",
    "建议",
    "权衡",
}


def compute_answer_quality_score(
    *,
    query: str,
    query_type: str,
    final_answer: str,
) -> float | None:
    """Compute answer quality without using RAG or Memory signals."""
    query = (query or "").strip()
    answer = (final_answer or "").strip()
    if not answer or _is_failure_answer(answer):
        return None

    metrics = {
        "intent_alignment": _intent_alignment(query, answer),
        "task_coverage": _task_coverage(query, answer),
        "specificity": _specificity(answer),
        "reasoning_quality": _reasoning_quality(answer),
        "format_compliance": _format_compliance(query, answer),
        "clarity": _clarity(answer, query_type=query_type),
    }
    score = sum(ANSWER_SCORE_WEIGHTS[name] * metrics[name] for name in ANSWER_SCORE_WEIGHTS)
    return round(_clamp(score), 4)


def _is_failure_answer(answer: str) -> bool:
    return bool(_FAILURE_RE.search(answer))


def _intent_alignment(query: str, answer: str) -> float:
    query_terms = _keywords(query)
    if not query_terms:
        return 0.5
    answer_terms = set(_keywords(answer))
    return _clamp(len(set(query_terms) & answer_terms) / len(set(query_terms)))


def _task_coverage(query: str, answer: str) -> float:
    parts = [_keywords(part) for part in _TASK_SPLIT_RE.split(query)]
    tasks = [terms for terms in parts if terms]
    if not tasks:
        return _intent_alignment(query, answer)
    answer_terms = set(_keywords(answer))
    covered = 0
    for terms in tasks:
        unique_terms = set(terms)
        if not unique_terms:
            continue
        overlap = len(unique_terms & answer_terms) / len(unique_terms)
        covered += int(overlap >= 0.34)
    return _clamp(covered / len(tasks))


def _specificity(answer: str) -> float:
    terms = _keywords(answer)
    if not terms:
        return 0.0
    length_score = min(len(terms) / 60.0, 1.0)
    distinct_score = min(len(set(terms)) / 30.0, 1.0)
    concrete_marker_score = min(_concrete_marker_count(answer) / 4.0, 1.0)
    return _clamp((0.45 * length_score) + (0.35 * distinct_score) + (0.20 * concrete_marker_score))


def _reasoning_quality(answer: str) -> float:
    terms = set(_keywords(answer))
    marker_hits = len(terms & _REASONING_MARKERS)
    marker_score = min(marker_hits / 3.0, 1.0)
    structure_score = 1.0 if _has_list_or_steps(answer) else 0.5
    return _clamp((0.70 * marker_score) + (0.30 * structure_score))


def _format_compliance(query: str, answer: str) -> float:
    query_lower = query.lower()
    requirements = 0
    satisfied = 0
    if any(marker in query_lower for marker in ("list", "bullet", "列出", "清单", "要点")):
        requirements += 1
        satisfied += int(_has_list_or_steps(answer) or "recommendation" in answer.lower() or "建议" in answer)
    if any(marker in query_lower for marker in ("table", "表格")):
        requirements += 1
        satisfied += int("|" in answer or "\t" in answer)
    if any(marker in query_lower for marker in ("json", "JSON")):
        requirements += 1
        stripped = answer.strip()
        satisfied += int(stripped.startswith("{") or stripped.startswith("["))
    if any(marker in query_lower for marker in ("中文", "chinese")):
        requirements += 1
        satisfied += int(_cjk_count(answer) > 0)
    if any(marker in query_lower for marker in ("english", "英文")):
        requirements += 1
        satisfied += int(bool(re.search(r"[A-Za-z]", answer)))
    if requirements == 0:
        return 1.0
    return _clamp(satisfied / requirements)


def _clarity(answer: str, *, query_type: str) -> float:
    sentences = [item.strip() for item in _SENTENCE_SPLIT_RE.split(answer) if item.strip()]
    if not sentences:
        return 0.0
    average_sentence_chars = sum(len(sentence) for sentence in sentences) / len(sentences)
    sentence_score = 1.0 if 20 <= average_sentence_chars <= 220 else 0.65
    structure_score = 1.0 if len(sentences) > 1 or _has_list_or_steps(answer) else 0.7
    repetition_score = _repetition_score(answer)
    type_bonus = 0.05 if query_type and query_type != "unknown" else 0.0
    return _clamp((0.40 * sentence_score) + (0.30 * structure_score) + (0.30 * repetition_score) + type_bonus)


def _keywords(text: str) -> list[str]:
    terms: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        if len(token) == 1 and "\u4e00" <= token <= "\u9fff":
            if token not in _CJK_STOP_CHARS:
                terms.append(token)
            continue
        if token not in STOPWORDS and len(token) > 1:
            terms.append(token)
    return terms


def _has_list_or_steps(text: str) -> bool:
    return bool(re.search(r"(^|\n)\s*(?:[-*]|\d+[.)]|[一二三四五六七八九十]+[、.])", text))


def _concrete_marker_count(text: str) -> int:
    count = 0
    count += len(re.findall(r"\d", text))
    count += len(re.findall(r"\b[A-Z][A-Za-z0-9_\-]{2,}\b", text))
    count += len(re.findall(r"\b[A-Za-z0-9_\-]+(?:\.[A-Za-z0-9_\-]+)+\b", text))
    return count


def _cjk_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def _repetition_score(text: str) -> float:
    terms = _keywords(text)
    if not terms:
        return 0.0
    return _clamp(len(set(terms)) / len(terms))


def _clamp(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)
