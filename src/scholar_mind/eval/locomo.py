from __future__ import annotations

import copy
import json
import re
import string
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

from scholar_mind.models.domain import QueryType
from scholar_mind.rag.top_k import FINAL_CITATION_TOP_K

try:  # pragma: no cover - optional parity with upstream LOCOMO when available.
    from nltk.stem import PorterStemmer
except Exception:  # pragma: no cover
    PorterStemmer = None


_ARTICLE_RE = re.compile(r"\b(a|an|the|and)\b", flags=re.IGNORECASE)
_STEMMER = PorterStemmer() if PorterStemmer is not None else None
_CATEGORY_ORDER = (1, 2, 3, 4, 5)
_GOLD_ANSWER_FIELDS = {"answer", "adversarial_answer"}
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
    "无法从",
    "没有足够",
    "未提及",
    "没有提到",
    "无法可靠",
    "无法确认",
)
_LOCOMO_ANSWER_INSTRUCTION = (
    "请只输出最终短答案，不要解释；如果答案包含多项，用英文逗号和空格分隔；"
    "如果没有足够记忆支持答案，回答 No information available."
)


def normalize_answer(value: Any) -> str:
    text = str(value).replace(",", "")
    text = text.lower()
    text = "".join(character for character in text if character not in set(string.punctuation))
    text = _ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def score_answer(prediction: Any, answer: Any, category: int | str) -> float:
    category_int = int(category)
    prediction_text = str(prediction)
    answer_text = str(answer)
    if category_int == 5:
        return 1.0 if _is_no_information_response(prediction_text) else 0.0
    if category_int == 1:
        return _multi_answer_f1(prediction_text, answer_text)
    if category_int in {2, 3, 4}:
        return _f1_score(prediction_text, answer_text)
    raise ValueError(f"Unsupported LOCOMO category: {category}")


def score_locomo_samples(
    samples: list[dict[str, Any]],
    *,
    prediction_key: str,
    model_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_prediction_key(prediction_key)
    scored_samples = copy.deepcopy(samples)
    category_counts: dict[str, float] = defaultdict(float)
    category_scores: dict[str, float] = defaultdict(float)
    recall_counts: dict[str, float] = defaultdict(float)
    recall_scores: dict[str, float] = defaultdict(float)
    prediction_nonempty = 0
    question_count = 0

    for sample in scored_samples:
        for qa in sample.get("qa", []):
            question_count += 1
            category = str(int(qa["category"]))
            prediction = qa.get(prediction_key, "")
            if str(prediction).strip():
                prediction_nonempty += 1
            f1_value = round(score_answer(prediction, qa.get("answer", ""), category), 3)
            qa[f"{model_name}_f1"] = f1_value
            category_counts[category] += 1.0
            category_scores[category] += f1_value

            context_key = f"{prediction_key}_context"
            evidence = [str(item) for item in qa.get("evidence", []) if str(item)]
            if context_key in qa and evidence:
                context = [str(item) for item in qa.get(context_key, []) if str(item)]
                recall = round(_evidence_recall(evidence, context), 3)
                qa[f"{model_name}_recall"] = recall
                recall_counts[category] += 1.0
                recall_scores[category] += recall

    total_score = sum(category_scores.values())
    report = {
        model_name: {
            "category_counts": dict(category_counts),
            "cum_accuracy_by_category": dict(category_scores),
            "accuracy_by_category": {
                category: round(category_scores[category] / count, 3)
                for category, count in category_counts.items()
                if count
            },
            "overall_accuracy": round(total_score / question_count, 3)
            if question_count
            else 0.0,
            "question_count": question_count,
            "prediction_nonempty": prediction_nonempty,
        }
    }
    if recall_counts:
        report[model_name]["recall_by_category"] = {
            category: round(recall_scores[category] / count, 3)
            for category, count in recall_counts.items()
            if count
        }
        report[model_name]["cum_recall_by_category"] = dict(recall_scores)
    return scored_samples, report


def score_prediction_file(
    prediction_file: Path,
    *,
    out_file: Path | None = None,
    stats_file: Path | None = None,
    prediction_key: str,
    model_name: str,
) -> dict[str, Any]:
    samples = json.loads(prediction_file.read_text(encoding="utf-8"))
    scored_samples, report = score_locomo_samples(
        samples,
        prediction_key=prediction_key,
        model_name=model_name,
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
    target_stats.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "prediction_file": str(prediction_file),
        "scored_file": str(target_out),
        "stats_file": str(target_stats),
        "report": report,
    }


async def run_locomo_qa(
    research_service,
    samples: list[dict[str, Any]],
    user_id: str,
    prediction_key: str,
    limit: int | None = None,
    progress_file: Path | None = None,
) -> list[dict[str, Any]]:
    _validate_prediction_key(prediction_key)
    predicted = _limit_qa_samples(samples, limit)
    _write_prediction_progress(predicted, progress_file)
    settings = getattr(research_service, "settings", None)
    top_k = getattr(settings, "final_citation_top_k", FINAL_CITATION_TOP_K)
    question_index = 0
    for sample in predicted:
        await _seed_memory_history(
            research_service=research_service,
            sample=sample,
            user_id=user_id,
            top_k=top_k,
        )
        _write_prediction_progress(predicted, progress_file)
        for qa in sample.get("qa", []):
            question_index += 1
            active_session_id = _question_session_id(user_id, question_index)
            answer_text = ""
            citations = []
            async for event, data in research_service.stream(
                query=_locomo_question_text(str(qa["question"])),
                user_id=user_id,
                session_id=active_session_id,
                query_type=QueryType.QA,
                request_payload={
                    "paper_ids": [],
                    "rag_strategy": "hybrid",
                    "top_k": top_k,
                    "conditional_memory_injection": False,
                    "memory_extraction_enabled": False,
                },
            ):
                if event == "answer":
                    answer_text = str(_value(data, "answer", ""))
                    citations = list(_value(data, "citations", []) or [])
            if not answer_text:
                state = await _stream_final_state(research_service, active_session_id)
                answer_text = str(state.get("final_answer", ""))
                citations = list(state.get("citations", []) or [])
            qa[prediction_key] = answer_text
            qa[f"{prediction_key}_context"] = [
                _citation_evidence_id(citation)
                for citation in citations
                if _citation_evidence_id(citation)
            ]
            _write_prediction_progress(predicted, progress_file)
    return predicted


def _locomo_question_text(question: str) -> str:
    return f"{question.strip()}\n\n{_LOCOMO_ANSWER_INSTRUCTION}"


def _write_prediction_progress(samples: list[dict[str, Any]], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


async def _seed_memory_history(
    *,
    research_service,
    sample: dict[str, Any],
    user_id: str,
    top_k: int,
) -> None:
    for session_index, turn in _iter_seed_turns(sample.get("conversation", {})):
        async for _event, _data in research_service.stream(
            query=str(turn["text"]),
            user_id=user_id,
            session_id=f"{user_id}-seed-s{session_index:03d}",
            query_type=QueryType.QA,
            request_payload={
                "paper_ids": [],
                "rag_strategy": "hybrid",
                "top_k": top_k,
                "conditional_memory_injection": False,
                "memory_extraction_enabled": True,
                "request_memory_extraction_enabled": False,
            },
        ):
            continue
    await _flush_seeded_memory_extraction(research_service, user_id)


def _iter_seed_turns(conversation: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    turns = []
    session_numbers = sorted(
        int(match.group(1))
        for key in conversation
        if (match := re.fullmatch(r"session_(\d+)", key))
        and isinstance(conversation.get(key), list)
    )
    for session_index in session_numbers:
        for turn in conversation.get(f"session_{session_index}", []):
            if not isinstance(turn, dict):
                continue
            if not bool(_value(turn.get("metadata", {}), "seed_memory", False)):
                continue
            if "user" not in str(turn.get("speaker", "")).lower():
                continue
            if str(turn.get("text", "")).strip():
                turns.append((session_index, turn))
    return turns


async def _flush_seeded_memory_extraction(research_service, user_id: str) -> None:
    memory_manager = getattr(research_service, "memory_manager", None)
    extract_pending = getattr(memory_manager, "extract_pending_memories", None)
    if callable(extract_pending):
        extract_pending(user_id=user_id)


def _question_session_id(user_id: str, question_index: int) -> str:
    return f"{user_id}-q{question_index:03d}"


def _limit_qa_samples(
    samples: list[dict[str, Any]],
    limit: int | None,
) -> list[dict[str, Any]]:
    if limit is None:
        return copy.deepcopy(samples)

    remaining = max(limit, 0)
    limited_samples = []
    for sample in samples:
        if remaining <= 0:
            break
        sample_copy = copy.deepcopy(sample)
        selected_qa = list(sample_copy.get("qa", []))[:remaining]
        if not selected_qa:
            continue
        sample_copy["qa"] = selected_qa
        sample_copy["conversation"] = _filter_conversation_for_selected_qa(
            sample_copy.get("conversation", {}),
            selected_qa,
        )
        limited_samples.append(sample_copy)
        remaining -= len(selected_qa)
    return limited_samples


def _filter_conversation_for_selected_qa(
    conversation: Any,
    selected_qa: Sequence[dict[str, Any]],
) -> Any:
    if not isinstance(conversation, dict):
        return copy.deepcopy(conversation)

    evidence_ids, fallback_case_ids = _selected_conversation_filters(selected_qa)
    if not evidence_ids and not fallback_case_ids:
        return copy.deepcopy(conversation)

    filtered = {
        key: copy.deepcopy(value)
        for key, value in conversation.items()
        if not re.fullmatch(r"session_\d+", key)
        and not re.fullmatch(r"session_\d+_date_time", key)
    }
    session_numbers = sorted(
        int(match.group(1))
        for key in conversation
        if (match := re.fullmatch(r"session_(\d+)", key))
        and isinstance(conversation.get(key), list)
    )
    for session_index in session_numbers:
        session_key = f"session_{session_index}"
        selected_turns = [
            copy.deepcopy(turn)
            for turn in conversation.get(session_key, [])
            if isinstance(turn, dict)
            and _conversation_turn_matches_filters(
                turn,
                evidence_ids=evidence_ids,
                fallback_case_ids=fallback_case_ids,
            )
        ]
        if not selected_turns:
            continue
        date_key = f"{session_key}_date_time"
        if date_key in conversation:
            filtered[date_key] = copy.deepcopy(conversation[date_key])
        filtered[session_key] = selected_turns
    return filtered


def _selected_conversation_filters(
    selected_qa: Sequence[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    evidence_ids: set[str] = set()
    fallback_case_ids: set[str] = set()
    for qa in selected_qa:
        evidence = _qa_evidence_ids(qa)
        if evidence:
            evidence_ids.update(evidence)
            continue
        metadata = qa.get("metadata", {})
        for key in ("case_id", "distractor_case_id"):
            case_id = str(_value(metadata, key, "")).strip()
            if case_id:
                fallback_case_ids.add(case_id)
    return evidence_ids, fallback_case_ids


def _qa_evidence_ids(qa: dict[str, Any]) -> set[str]:
    raw_evidence = qa.get("evidence", [])
    if isinstance(raw_evidence, str):
        raw_items = [raw_evidence]
    elif isinstance(raw_evidence, Sequence):
        raw_items = raw_evidence
    else:
        raw_items = []
    return {str(item).strip() for item in raw_items if str(item).strip()}


def _conversation_turn_matches_filters(
    turn: dict[str, Any],
    *,
    evidence_ids: set[str],
    fallback_case_ids: set[str],
) -> bool:
    dia_id = str(turn.get("dia_id", "")).strip()
    if dia_id and dia_id in evidence_ids:
        return True
    case_id = str(_value(turn.get("metadata", {}), "case_id", "")).strip()
    return bool(case_id and case_id in fallback_case_ids)


async def _stream_final_state(research_service, session_id: str) -> dict[str, Any]:
    orchestrator = getattr(research_service, "orchestrator", None)
    get_state = getattr(orchestrator, "get_state", None)
    if not callable(get_state):
        return {}
    state = await get_state(session_id)
    return state or {}


_OUTPUT_SECTION_PAIRS = (
    ("method assumptions", "failure modes"),
    ("dataset fit", "implementation risks"),
    ("core contribution", "reproducibility checklist"),
    ("baseline comparison", "open questions"),
    ("math intuition", "engineering steps"),
    ("evidence quotes", "decision notes"),
)
_PAPER_ROLES = (
    "anchor paper",
    "negative example",
    "baseline candidate",
    "implementation reference",
    "survey seed",
    "ablation target",
)
_DEPTH_UPDATES = (
    ("survey-first overview", "implementation-first notes"),
    ("full derivation", "intuition-first summary"),
    ("broad landscape", "failure-mode focused reading"),
    ("paper-by-paper notes", "comparison table first"),
    ("theory-heavy reading", "experiment-first reading"),
    ("quick skim", "reproduction-oriented reading"),
)
_BACKGROUND_STYLES = (
    (
        "strong Python engineering background but limited causal inference math",
        "intuition-first formula explanation",
    ),
    (
        "comfortable with transformers but new to graph retrieval systems",
        "architecture-first explanation",
    ),
    (
        "experienced in product evaluation but weak on statistical testing",
        "metric-grounded explanation",
    ),
    (
        "strong systems background but limited HCI vocabulary",
        "example-driven explanation",
    ),
    (
        "new to academic writing but familiar with backend services",
        "plain-language first explanation",
    ),
    (
        "comfortable with embeddings but weak on benchmark design",
        "evaluation-protocol first explanation",
    ),
)
_BUSINESS_TASKS = (
    (
        "paper reading",
        "start with limitations before novelty",
        "avoid unsupported implementation claims",
    ),
    (
        "study plan",
        "include weekly reproduction checkpoints",
        "keep the plan within six weeks",
    ),
    (
        "cross-domain ideation",
        "map methods into education technology",
        "exclude healthcare examples",
    ),
    (
        "idea novelty review",
        "separate user hypothesis from retrieved evidence",
        "flag missing baselines explicitly",
    ),
    (
        "trend analysis",
        "compare quarterly momentum before naming hot topics",
        "avoid one-paper trend claims",
    ),
    (
        "experiment planning",
        "include ablations for memory retrieval and answer use",
        "avoid paid datasets",
    ),
)


def build_memory_locomo_dataset(
    paper_repository,
    *,
    question_count: int = 150,
    sample_id: str = "scholarmind_locomo_150",
) -> list[dict[str, Any]]:
    if question_count <= 0 or question_count % 5 != 0:
        raise ValueError("question_count must be a positive multiple of 5")

    papers = [_paper_profile(paper) for paper in paper_repository.all_papers()]
    papers = [paper for paper in papers if paper["paper_id"] and paper["title"]]
    if len(papers) < 2:
        raise ValueError("At least two papers are required to build the benchmark")

    papers.sort(key=lambda item: (item["publish_date"], item["paper_id"]))

    per_category = question_count // 5
    selected = _cycled(papers, max(per_category * 2, per_category + 1))
    cases = _build_memory_cases(selected, per_category)
    conversation = _build_memory_conversation(cases)

    qa: list[dict[str, Any]] = []
    qa.extend(_memory_multi_hop_questions(cases))
    qa.extend(_memory_temporal_questions(cases))
    qa.extend(_memory_inference_questions(cases))
    qa.extend(_memory_business_questions(cases))
    qa.extend(_memory_adversarial_questions(cases))

    dataset = [
        {
            "sample_id": sample_id,
            "conversation": conversation,
            "qa": qa[:question_count],
        }
    ]
    validate_locomo_dataset(dataset, expected_question_count=question_count)
    return dataset


def build_paper_locomo_dataset(
    paper_repository,
    *,
    question_count: int = 150,
    sample_id: str = "scholarmind_locomo_150",
) -> list[dict[str, Any]]:
    return build_memory_locomo_dataset(
        paper_repository,
        question_count=question_count,
        sample_id=sample_id,
    )


def validate_locomo_dataset(
    samples: list[dict[str, Any]],
    *,
    expected_question_count: int | None = None,
) -> dict[str, Any]:
    if not isinstance(samples, list) or not samples:
        raise ValueError("LOCOMO dataset must be a non-empty JSON array")
    category_counts: Counter[str] = Counter()
    question_count = 0
    for sample in samples:
        if not sample.get("sample_id"):
            raise ValueError("Each sample must include sample_id")
        if not isinstance(sample.get("conversation"), dict):
            raise ValueError(f"Sample {sample.get('sample_id')} has invalid conversation")
        qa_items = sample.get("qa")
        if not isinstance(qa_items, list) or not qa_items:
            raise ValueError(f"Sample {sample.get('sample_id')} has no qa items")
        for qa in qa_items:
            for field_name in ("question", "category", "evidence"):
                if field_name not in qa:
                    raise ValueError(f"QA item missing {field_name}")
            category = int(qa["category"])
            if category not in _CATEGORY_ORDER:
                raise ValueError(f"Unsupported LOCOMO category: {category}")
            if category == 5:
                if "adversarial_answer" not in qa:
                    raise ValueError("Category 5 QA item missing adversarial_answer")
            elif "answer" not in qa:
                raise ValueError("QA item missing answer")
            if not isinstance(qa["evidence"], list):
                raise ValueError("QA evidence must be a list")
            category_counts[str(category)] += 1
            question_count += 1
    if expected_question_count is not None and question_count != expected_question_count:
        raise ValueError(
            f"Expected {expected_question_count} questions, found {question_count}"
        )
    return {
        "sample_count": len(samples),
        "question_count": question_count,
        "category_counts": {
            str(category): category_counts[str(category)]
            for category in _CATEGORY_ORDER
            if category_counts[str(category)]
        },
    }


def _multi_answer_f1(prediction: str, answer: str) -> float:
    predictions = [item.strip() for item in prediction.split(",") if item.strip()]
    answers = [item.strip() for item in answer.split(",") if item.strip()]
    if not answers:
        return 0.0
    if not predictions:
        predictions = [prediction]
    matched = sum(max(_f1_score(candidate, gold) for candidate in predictions) for gold in answers)
    return matched / len(answers)


def _f1_score(prediction: str, answer: str) -> float:
    prediction_tokens = [_stem(token) for token in normalize_answer(prediction).split()]
    answer_tokens = [_stem(token) for token in normalize_answer(answer).split()]
    if not prediction_tokens or not answer_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(answer_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(answer_tokens)
    return (2 * precision * recall) / (precision + recall)


def _validate_prediction_key(prediction_key: str) -> None:
    if prediction_key in _GOLD_ANSWER_FIELDS:
        raise ValueError("prediction_key must not be a gold answer field")


def _is_no_information_response(prediction: str) -> bool:
    lowered = str(prediction).lower()
    return any(pattern in lowered for pattern in _NO_INFORMATION_PATTERNS)


def _evidence_recall(evidence: list[str], context: list[str]) -> float:
    if not evidence:
        return 0.0
    context_set = set(context)
    return sum(item in context_set for item in evidence) / len(evidence)


def _stem(token: str) -> str:
    if _STEMMER is None:
        return token
    return _STEMMER.stem(token)


def _citation_evidence_id(citation: Any) -> str:
    paper_id = str(_value(citation, "paper_id", "")).strip()
    section = str(_value(citation, "section", "")).strip()
    if not paper_id:
        return ""
    if not section:
        return f"{paper_id}::metadata"
    return f"{paper_id}::{section}"


def _build_memory_cases(
    papers: Sequence[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    base = list(papers)
    cases = []
    for index in range(count):
        paper = base[index % len(base)]
        partner = base[(index + count) % len(base)]
        output_a, output_b = _OUTPUT_SECTION_PAIRS[index % len(_OUTPUT_SECTION_PAIRS)]
        old_depth, new_depth = _DEPTH_UPDATES[index % len(_DEPTH_UPDATES)]
        background, formula_style = _BACKGROUND_STYLES[index % len(_BACKGROUND_STYLES)]
        business_task, business_requirement, business_constraint = _BUSINESS_TASKS[
            index % len(_BUSINESS_TASKS)
        ]
        cases.append(
            {
                "case_id": f"case_{index + 1:03d}",
                "paper": paper,
                "partner": partner,
                "topic": _memory_topic(paper, index),
                "paper_role": _PAPER_ROLES[index % len(_PAPER_ROLES)],
                "output_a": output_a,
                "output_b": output_b,
                "old_depth": old_depth,
                "new_depth": new_depth,
                "background": background,
                "formula_style": formula_style,
                "business_task": business_task,
                "business_requirement": business_requirement,
                "business_constraint": business_constraint,
            }
        )
    return cases


def _build_memory_conversation(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    conversation: dict[str, Any] = {
        "speaker_a": "ScholarMind user",
        "speaker_b": "ScholarMind assistant",
    }
    sessions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, case in enumerate(cases, start=1):
        session_index = ((index - 1) // 5) + 1
        case["evidence"] = {
            "role": _case_evidence_id(case, "role"),
            "output": _case_evidence_id(case, "output"),
            "old_depth": _case_evidence_id(case, "old-depth"),
            "new_depth": _case_evidence_id(case, "new-depth"),
            "background": _case_evidence_id(case, "background"),
            "business": _case_evidence_id(case, "business"),
        }
        sessions[session_index].extend(
            [
                _memory_turn(
                    dia_id=case["evidence"]["role"],
                    case=case,
                    memory_type="paper_read",
                    text=(
                        "请记住：在 ScholarMind 项目 "
                        f"{case['case_id']} 中，我把论文《{case['paper']['title']}》"
                        f"标记为 `{case['paper_role']}`，用于 {case['topic']}。"
                    ),
                ),
                _memory_turn(
                    dia_id=case["evidence"]["output"],
                    case=case,
                    memory_type="workflow",
                    text=(
                        f"请记住：在 ScholarMind 项目 {case['case_id']} 的 "
                        f"{case['topic']} 比较设置中，比较《{case['paper']['title']}》"
                        f"和《{case['partner']['title']}》时，"
                        "我的默认输出要包含英文标签 "
                        f"`{case['output_a']}` 和 `{case['output_b']}`。"
                    ),
                ),
                _memory_turn(
                    dia_id=case["evidence"]["old_depth"],
                    case=case,
                    memory_type="preference",
                    text=(
                        f"请记住：2026-05-{(index % 20) + 1:02d} 时，"
                        f"我在 ScholarMind 项目 {case['case_id']} 中对 "
                        f"{case['topic']} 的默认阅读深度是 `{case['old_depth']}`。"
                    ),
                ),
                _memory_turn(
                    dia_id=case["evidence"]["new_depth"],
                    case=case,
                    memory_type="feedback",
                    text=(
                        f"请记住：2026-06-{(index % 20) + 1:02d} 更新，"
                        f"以后处理 ScholarMind 项目 {case['case_id']} 的 "
                        f"{case['topic']} 时，默认阅读深度改为 "
                        f"`{case['new_depth']}`。"
                    ),
                ),
                _memory_turn(
                    dia_id=case["evidence"]["background"],
                    case=case,
                    memory_type="knowledge_level",
                    text=(
                        f"请记住：在 ScholarMind 项目 {case['case_id']} 中，"
                        f"我的背景是 {case['background']}；解释 {case['topic']} "
                        "的公式或指标时使用英文风格标签 "
                        f"`{case['formula_style']}`。"
                    ),
                ),
                _memory_turn(
                    dia_id=case["evidence"]["business"],
                    case=case,
                    memory_type="project_constraint",
                    text=(
                        f"请记住：在 ScholarMind 项目 {case['case_id']} 中，"
                        "当 ScholarMind 为我执行 "
                        f"`{case['business_task']}` 时，个性化要求是 "
                        f"`{case['business_requirement']}`；硬约束是 "
                        f"`{case['business_constraint']}`。"
                    ),
                ),
            ]
        )
    for session_index in sorted(sessions):
        conversation[f"session_{session_index}_date_time"] = (
            f"2026-05-{min(28, session_index * 3):02d}"
        )
        conversation[f"session_{session_index}"] = sessions[session_index]
    return conversation


def _memory_multi_hop_questions(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    questions = []
    for index, case in enumerate(cases, start=1):
        template_id = (
            "multi_hop_topic_role_outputs",
            "multi_hop_paper_compare_labels",
            "multi_hop_case_report_columns",
            "multi_hop_title_role_and_sections",
            "multi_hop_project_setup",
        )[(index - 1) % 5]
        question = {
            "multi_hop_topic_role_outputs": (
                f"基于我之前对 {case['topic']} 的比较设置，请用英文标签回答："
                "这个项目的 paper role 和两个默认输出项是什么？"
            ),
            "multi_hop_paper_compare_labels": (
                f"我在读《{case['paper']['title']}》时，把它设成了什么 paper role？"
                f"和《{case['partner']['title']}》比较时还要求哪两个英文标签？"
            ),
            "multi_hop_case_report_columns": (
                f"回忆 ScholarMind 项目 {case['case_id']}："
                "生成比较报告时，论文定位和两个默认栏目分别是什么？"
            ),
            "multi_hop_title_role_and_sections": (
                f"针对《{case['paper']['title']}》这个 {case['topic']} 项目，"
                "请列出我记录的 role 以及默认比较输出里的两个 section label。"
            ),
            "multi_hop_project_setup": (
                f"如果现在继续处理 {case['case_id']}，"
                "需要沿用的论文角色和两个英文输出标签是什么？"
            ),
        }[template_id]
        questions.append(
            {
                "question": question,
                "answer": (
                    f"{case['paper_role']}, {case['output_a']}, {case['output_b']}"
                ),
                "category": 1,
                "evidence": [case["evidence"]["role"], case["evidence"]["output"]],
                "metadata": {
                    "question_kind": "memory_multi_hop",
                    "template_id": template_id,
                    "case_id": case["case_id"],
                    "memory_focus": ["paper_read", "workflow"],
                    "question_index": index,
                },
            }
        )
    return questions


def _memory_temporal_questions(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    questions = []
    for index, case in enumerate(cases, start=1):
        template_id = (
            "temporal_latest_depth_by_topic",
            "temporal_updated_depth_by_title",
            "temporal_ignore_old_depth",
            "temporal_june_update",
            "temporal_case_current_depth",
        )[(index - 1) % 5]
        question = {
            "temporal_latest_depth_by_topic": (
                f"后来我把 {case['topic']} 的默认阅读深度改成了哪个英文标签？"
            ),
            "temporal_updated_depth_by_title": (
                f"关于《{case['paper']['title']}》，最新记录里的默认阅读深度是什么英文标签？"
            ),
            "temporal_ignore_old_depth": (
                f"不要沿用旧的 `{case['old_depth']}`；"
                f"{case['case_id']} 现在应该使用哪个 reading-depth label？"
            ),
            "temporal_june_update": (
                f"2026-06 的更新之后，处理 {case['topic']} 时默认采用哪个深度标签？"
            ),
            "temporal_case_current_depth": (
                f"我在 {case['case_id']} 里最后确认的阅读深度偏好是什么？"
            ),
        }[template_id]
        questions.append(
            {
                "question": question,
                "answer": case["new_depth"],
                "category": 2,
                "evidence": [case["evidence"]["old_depth"], case["evidence"]["new_depth"]],
                "metadata": {
                    "question_kind": "memory_temporal_update",
                    "template_id": template_id,
                    "case_id": case["case_id"],
                    "memory_focus": ["preference", "feedback"],
                    "question_index": index,
                },
            }
        )
    return questions


def _memory_inference_questions(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    questions = []
    for index, case in enumerate(cases, start=1):
        template_id = (
            "inference_formula_style_by_topic",
            "inference_background_to_style",
            "inference_title_metric_explanation",
            "inference_reader_profile",
            "inference_explanation_mode",
        )[(index - 1) % 5]
        question = {
            "inference_formula_style_by_topic": (
                f"按我记录的背景，解释 {case['topic']} 的公式或指标时，"
                "应采用哪个英文风格标签？"
            ),
            "inference_background_to_style": (
                f"考虑我在 {case['case_id']} 里记录的能力背景，"
                "这类论文指标说明应套用哪个 explanation style label？"
            ),
            "inference_title_metric_explanation": (
                f"如果要解释《{case['paper']['title']}》相关指标，"
                "我偏好的英文说明风格标签是什么？"
            ),
            "inference_reader_profile": (
                f"基于我的背景 `{case['background']}`，"
                f"处理 {case['topic']} 的公式时应选择哪个风格标签？"
            ),
            "inference_explanation_mode": (
                f"为 {case['case_id']} 生成面向我的方法/指标解释时，"
                "应该使用哪个英文 mode label？"
            ),
        }[template_id]
        questions.append(
            {
                "question": question,
                "answer": case["formula_style"],
                "category": 3,
                "evidence": [case["evidence"]["background"]],
                "metadata": {
                    "question_kind": "memory_inference",
                    "template_id": template_id,
                    "case_id": case["case_id"],
                    "memory_focus": ["knowledge_level"],
                    "question_index": index,
                },
            }
        )
    return questions


def _memory_business_questions(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    questions = []
    for index, case in enumerate(cases, start=1):
        template_id = (
            "business_requirement_by_topic",
            "business_task_personalization",
            "business_title_workflow_requirement",
            "business_case_preference",
            "business_apply_requirement",
        )[(index - 1) % 5]
        question = {
            "business_requirement_by_topic": (
                f"如果现在为 {case['topic']} 生成 `{case['business_task']}`，"
                "应优先遵守哪个英文个性化要求？"
            ),
            "business_task_personalization": (
                f"做 `{case['business_task']}` 时，我记录的 personalization requirement 是什么？"
            ),
            "business_title_workflow_requirement": (
                f"围绕《{case['paper']['title']}》继续产出业务结果时，"
                f"`{case['business_task']}` 的个性化要求是哪条？"
            ),
            "business_case_preference": (
                f"在 {case['case_id']} 的 ScholarMind 工作流里，"
                f"`{case['business_task']}` 应先满足哪个用户偏好？"
            ),
            "business_apply_requirement": (
                f"请只回答 requirement：为 {case['topic']} 做 "
                f"`{case['business_task']}` 时要应用哪条英文偏好？"
            ),
        }[template_id]
        questions.append(
            {
                "question": question,
                "answer": case["business_requirement"],
                "category": 4,
                "evidence": [case["evidence"]["business"]],
                "metadata": {
                    "question_kind": "memory_business_personalization",
                    "template_id": template_id,
                    "case_id": case["case_id"],
                    "memory_focus": ["project_constraint"],
                    "business_task": case["business_task"],
                    "question_index": index,
                },
            }
        )
    return questions


def _memory_adversarial_questions(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    questions = []
    for index, case in enumerate(cases, start=1):
        template_id = (
            "adversarial_cross_paper_outputs",
            "adversarial_wrong_task_requirement",
            "adversarial_wrong_depth_update",
            "adversarial_wrong_formula_style",
            "adversarial_wrong_role",
        )[(index - 1) % 5]
        distractor = _distractor_case(cases, index - 1, template_id)
        question = {
            "adversarial_cross_paper_outputs": (
                f"我是否为《{case['paper']['title']}》和"
                f"《{distractor['partner']['title']}》这组论文设置过 "
                f"`{distractor['output_a']}` / `{distractor['output_b']}`？"
                "如果只是别的项目里的记录，请回答没有信息。"
            ),
            "adversarial_wrong_task_requirement": (
                f"我给《{case['paper']['title']}》对应项目设置的 "
                f"`{distractor['business_task']}` 个性化要求是什么？"
                "如果该任务只属于别的项目，请回答没有信息。"
            ),
            "adversarial_wrong_depth_update": (
                f"关于 {case['topic']}，我有没有把默认阅读深度更新为 "
                f"`{distractor['new_depth']}`？如果没有直接记录，请回答没有信息。"
            ),
            "adversarial_wrong_formula_style": (
                f"解释《{case['paper']['title']}》的公式时，我是否记录过使用 "
                f"`{distractor['formula_style']}`？如果这是别的 case 的背景，请回答没有信息。"
            ),
            "adversarial_wrong_role": (
                f"在 {case['case_id']} 中，我是否把这篇论文标记为 "
                f"`{distractor['paper_role']}`？如果该 role 来自其他论文，请回答没有信息。"
            ),
        }[template_id]
        adversarial_answer, evidence = _adversarial_payload(distractor, template_id)
        questions.append(
            {
                "question": question,
                "adversarial_answer": adversarial_answer,
                "category": 5,
                "evidence": [evidence],
                "metadata": {
                    "question_kind": "memory_adversarial_confusable",
                    "template_id": template_id,
                    "case_id": case["case_id"],
                    "distractor_case_id": distractor["case_id"],
                    "memory_focus": ["confusable_memory"],
                    "question_index": index,
                },
            }
        )
    return questions


def _distractor_case(
    cases: Sequence[dict[str, Any]],
    case_index: int,
    template_id: str,
) -> dict[str, Any]:
    current = cases[case_index]
    predicates = {
        "adversarial_cross_paper_outputs": lambda item: item["partner"]["title"]
        != current["partner"]["title"],
        "adversarial_wrong_task_requirement": lambda item: item["business_task"]
        != current["business_task"],
        "adversarial_wrong_depth_update": lambda item: item["new_depth"] != current["new_depth"],
        "adversarial_wrong_formula_style": lambda item: item["formula_style"]
        != current["formula_style"],
        "adversarial_wrong_role": lambda item: item["paper_role"] != current["paper_role"],
    }
    predicate = predicates[template_id]
    for offset in range(1, len(cases)):
        candidate = cases[(case_index + offset) % len(cases)]
        if predicate(candidate):
            return candidate
    return current


def _adversarial_payload(distractor: dict[str, Any], template_id: str) -> tuple[str, str]:
    if template_id == "adversarial_cross_paper_outputs":
        return (
            f"{distractor['output_a']}, {distractor['output_b']}",
            distractor["evidence"]["output"],
        )
    if template_id == "adversarial_wrong_task_requirement":
        return distractor["business_requirement"], distractor["evidence"]["business"]
    if template_id == "adversarial_wrong_depth_update":
        return distractor["new_depth"], distractor["evidence"]["new_depth"]
    if template_id == "adversarial_wrong_formula_style":
        return distractor["formula_style"], distractor["evidence"]["background"]
    if template_id == "adversarial_wrong_role":
        return distractor["paper_role"], distractor["evidence"]["role"]
    raise ValueError(f"Unsupported adversarial template: {template_id}")


def _memory_turn(
    *,
    dia_id: str,
    case: dict[str, Any],
    memory_type: str,
    text: str,
) -> dict[str, Any]:
    return {
        "speaker": "ScholarMind user",
        "dia_id": dia_id,
        "text": text,
        "metadata": {
            "seed_memory": True,
            "memory_type": memory_type,
            "case_id": case["case_id"],
            "paper_id": case["paper"]["paper_id"],
            "topic": case["topic"],
        },
    }


def _case_evidence_id(case: dict[str, Any], suffix: str) -> str:
    return f"{case['case_id']}::{suffix}"


def _memory_topic(paper: dict[str, Any], index: int) -> str:
    focus = (
        "memory evaluation",
        "paper reading personalization",
        "research workflow automation",
        "cross-domain hypothesis design",
        "long-context assistant reliability",
        "retrieval-grounded study planning",
    )[index % 6]
    return f"{paper['primary_category']} {focus}"


def _paper_profile(paper: Any) -> dict[str, Any]:
    categories = list(_value(paper, "categories", []) or [])
    publish_date = _date_text(_value(paper, "publish_date", ""))
    return {
        "paper_id": str(_value(paper, "paper_id", "")).strip(),
        "title": str(_value(paper, "title", "")).strip(),
        "abstract": " ".join(str(_value(paper, "abstract", "")).split()),
        "categories": categories,
        "primary_category": str(categories[0]) if categories else "unknown",
        "publish_date": publish_date,
    }


def _chunk_profile(chunk: dict[str, Any]) -> dict[str, Any]:
    section = str(chunk.get("section") or "section").strip()
    return {
        "chunk_id": str(chunk.get("chunk_id") or "").strip(),
        "paper_id": str(chunk.get("paper_id") or "").strip(),
        "section": section,
        "content": " ".join(str(chunk.get("content") or "").split()),
    }


def _metadata_evidence_id(paper: dict[str, Any]) -> str:
    return f"{paper['paper_id']}::metadata"


def _section_evidence_id(chunk: dict[str, Any]) -> str:
    if chunk.get("chunk_id"):
        return str(chunk["chunk_id"])
    return f"{chunk['paper_id']}::{chunk['section']}"


def _date_text(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _cycled(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if not items:
        return []
    return [items[index % len(items)] for index in range(count)]
