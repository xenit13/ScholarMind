from __future__ import annotations

import re
from datetime import date
from time import perf_counter
from typing import Any

from langchain_core.messages import AIMessage

from scholar_mind.agents.common import (
    ainvoke_structured_output_with_raw,
    extract_json_candidate,
    merge_usage,
    raw_output_text,
)
from scholar_mind.agents.state import reading_value, request_value, telemetry_value
from scholar_mind.models.domain import PlannerOutput, QueryType
from scholar_mind.rag.query_transform import QueryTransformer
from scholar_mind.utils.text import overlap_score

_ARXIV_ID_PATTERN = re.compile(r"\b(\d{4}\.\d{5})(?:v\d+)?\b", flags=re.IGNORECASE)

_MEMORY_SAVE_PATTERNS = [
    re.compile(
        r"^\s*(?:请|请你|麻烦|麻烦你|帮我|帮忙|请帮我)?\s*(?:记住|记下|记一下|记着|记好)\s*(?:这件事|这点|这个|以下内容)?[\s:：,，]*(?P<content>.+?)\s*$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^\s*remember\s+(?:that\s+)?(?P<content>.+?)\s*$",
        flags=re.IGNORECASE,
    ),
]

_MEMORY_FOLLOWUP_TASK_PATTERN = re.compile(
    r"(?:。|！|!|？|\?)?\s*(?:顺便|另外|然后|接下来|再帮我|请帮我|帮我|并帮我|并给我|同时帮我|同时给我).*$",
    flags=re.IGNORECASE,
)

_GENERAL_MEMORY_INTENT_PATTERNS = [
    re.compile(
        r"(?:结合|根据|基于|按|参考).*(?:我之前的|刚才这些|这些|我的).*(?:偏好|习惯|背景|研究方向|目标|需求)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:我最近在研究|我长期关注|我的研究方向|我的偏好|我的习惯|我的背景|我的目标|我的需求)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:结合|根据|基于|按).*(?:刚才|之前|以前|过往|历史).*(?:偏好|习惯|背景|研究|目标|需求)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        (
            r"(?:remember|based on|according to).*(?:my preferences|my habits|"
            r"my background|my research focus|my goals?)"
        ),
        flags=re.IGNORECASE,
    ),
]

_CONTEXT_HEAVY_MEMORY_INTENT_PATTERNS = [
    re.compile(
        r"(?:继续上次|延续上次|接着上次|继续之前|上次读到|读到哪里|还记得我)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:我的基础|我的进度|我的背景|我的目标|按我的习惯|按我的偏好)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:continue from last time|pick up where we left off|where did we stop last time)",
        flags=re.IGNORECASE,
    ),
]


def make_planner_node(llm, memory_manager, prompt_catalog, paper_repository=None):
    return make_planner_primary_node(llm, memory_manager, prompt_catalog, paper_repository)


def make_planner_primary_node(llm, memory_manager, prompt_catalog, paper_repository=None):
    planner_prompt = prompt_catalog.get("planner")
    query_transformer = QueryTransformer()

    async def planner_node(state):
        started = perf_counter()
        query = request_value(state, "query", "")
        explicit_memory_candidates = _extract_explicit_memory_candidates(query)
        memory_context, hits = "", 0
        memory_notices: list[str] = []
        existing_payload = request_value(state, "payload")
        hint = request_value(state, "query_type_hint")
        structured = None
        response = None
        usage = merge_usage()
        if hint:
            query_type = QueryType(hint)
            sub_queries = []
            source_papers = []
            target_domains = []
        else:
            if llm is None:
                raise RuntimeError(
                    "Planner primary requires an LLM when no fixed hint is available"
                )
            prompt = (
                f"{planner_prompt}\n\n"
                f"Query: {query}\n"
                f"Hint: {hint or ''}\n"
                f"Memory:\n{memory_context or '(none)'}"
            )
            structured, usage, response = await ainvoke_structured_output_with_raw(
                llm,
                prompt,
                PlannerOutput,
                recover=_recover_planner_output,
            )
            query_type = QueryType(hint) if hint else (
                structured.query_type if structured else planner_fallback(query, hint)
            )
            if (
                not hint
                and query_type == QueryType.QA
                and (
                    _looks_like_paper_reading_request(query, structured)
                    or _looks_like_active_reading_followup(state)
                )
            ):
                query_type = QueryType.PAPER_READING
            sub_queries = structured.sub_queries if structured else []
            source_papers = structured.source_papers if structured else []
            target_domains = structured.target_domains if structured else []
        if query_type == QueryType.IDEA_NOVELTY and not sub_queries:
            sub_queries = query_transformer.expand_for_idea_novelty(query, limit=4)
        elif query_type == QueryType.CROSS_DOMAIN:
            fallback_intent = _heuristic_crossdomain_intent(query, paper_repository)
            source_papers = source_papers or fallback_intent["source_papers"]
            target_domains = target_domains or fallback_intent["target_domains"]
            sub_queries = []
        if _should_retrieve_memory_context(
            query=query,
            query_type=query_type,
            explicit_memory_candidates=explicit_memory_candidates,
            conditional_memory_injection=bool(
                (existing_payload or {}).get("conditional_memory_injection", False)
            ),
        ):
            context_payload = getattr(memory_manager, "get_context_payload", None)
            if callable(context_payload):
                payload = await context_payload(
                    user_id=request_value(state, "user_id", ""),
                    session_id=request_value(state, "session_id", None),
                    current_query=query,
                )
                memory_context = payload.context
                hits = payload.hit_count
                memory_notices = list(getattr(payload, "notices", []))
            else:
                memory_context, hits = await memory_manager.get_context(
                    user_id=request_value(state, "user_id", ""),
                    current_query=query,
                )
        request_payload = _build_request_payload(
            existing_payload,
            query,
            query_type,
            structured,
            paper_repository,
        )
        duration = int((perf_counter() - started) * 1000)
        cross_domain_intent = None
        if query_type == QueryType.CROSS_DOMAIN:
            cross_domain_intent = {
                "source_papers": source_papers,
                "target_domains": target_domains,
                "sources": [
                    {
                        "kind": "planner_llm" if structured else "planner_heuristic",
                        "label": "cross-domain intent extraction",
                        "metadata": {"query": query},
                    }
                ],
            }
        result = {
            "request": {"query_type": query_type.value, "payload": request_payload},
            "planning": {"sub_queries": sub_queries},
            "memory": {
                "explicit_candidates": explicit_memory_candidates,
                "context": memory_context,
                "hit_count": hits,
                "notices": memory_notices,
            },
            "cross_domain": {"intent": cross_domain_intent},
            "telemetry": {
                "llm_usage": merge_usage(telemetry_value(state, "llm_usage"), usage),
                "agent_trace": [{"agent": "planner", "duration_ms": duration}],
            },
        }
        if response is not None:
            result["messages"] = [response]
        return result

    return planner_node


def make_planner_fallback_node(memory_manager, paper_repository=None):
    query_transformer = QueryTransformer()

    async def planner_fallback_node(state):
        started = perf_counter()
        query = request_value(state, "query", "")
        explicit_memory_candidates = _extract_explicit_memory_candidates(query)
        memory_context, hits = "", 0
        memory_notices: list[str] = []
        existing_payload = request_value(state, "payload")
        hint = request_value(state, "query_type_hint")
        query_type = QueryType(hint) if hint else planner_fallback(query, hint)
        if not hint and query_type == QueryType.QA and _looks_like_active_reading_followup(state):
            query_type = QueryType.PAPER_READING
        sub_queries: list[str] = []
        source_papers: list[str] = []
        target_domains: list[str] = []
        if query_type == QueryType.IDEA_NOVELTY:
            sub_queries = query_transformer.expand_for_idea_novelty(query, limit=4)
        elif query_type == QueryType.CROSS_DOMAIN:
            fallback_intent = _heuristic_crossdomain_intent(query, paper_repository)
            source_papers = fallback_intent["source_papers"]
            target_domains = fallback_intent["target_domains"]
        if _should_retrieve_memory_context(
            query=query,
            query_type=query_type,
            explicit_memory_candidates=explicit_memory_candidates,
            conditional_memory_injection=bool(
                (existing_payload or {}).get("conditional_memory_injection", False)
            ),
        ):
            context_payload = getattr(memory_manager, "get_context_payload", None)
            if callable(context_payload):
                payload = await context_payload(
                    user_id=request_value(state, "user_id", ""),
                    session_id=request_value(state, "session_id", None),
                    current_query=query,
                )
                memory_context = payload.context
                hits = payload.hit_count
                memory_notices = list(getattr(payload, "notices", []))
            else:
                memory_context, hits = await memory_manager.get_context(
                    user_id=request_value(state, "user_id", ""),
                    current_query=query,
                )
        request_payload = _build_request_payload(
            existing_payload,
            query,
            query_type,
            None,
            paper_repository,
        )
        duration = int((perf_counter() - started) * 1000)
        cross_domain_intent = None
        if query_type == QueryType.CROSS_DOMAIN:
            cross_domain_intent = {
                "source_papers": source_papers,
                "target_domains": target_domains,
                "sources": [
                    {
                        "kind": "planner_heuristic",
                        "label": "cross-domain intent extraction",
                        "metadata": {"query": query},
                    }
                ],
            }
        result = {
            "messages": [AIMessage(content=f"Planner selected path: {query_type.value}")],
            "request": {"query_type": query_type.value, "payload": request_payload},
            "planning": {"sub_queries": sub_queries},
            "memory": {
                "explicit_candidates": explicit_memory_candidates,
                "context": memory_context,
                "hit_count": hits,
                "notices": memory_notices,
            },
            "cross_domain": {"intent": cross_domain_intent},
            "telemetry": {"agent_trace": [{"agent": "planner", "duration_ms": duration}]},
        }
        return result

    return planner_fallback_node


def planner_fallback(query: str, query_type_hint: str | None = None) -> QueryType:
    if query_type_hint:
        return QueryType(query_type_hint)
    lowered = query.lower()
    if any(
        keyword in lowered
        for keyword in [
            "study plan",
            "roadmap",
            "学习计划",
            "安排学习",
            "系统学习",
            "帮我制定一个计划",
        ]
    ):
        return QueryType.STUDY_PLAN
    if any(
        keyword in lowered
        for keyword in ["paper reading", "论文精读", "逐段", "讲这一段", "读这篇论文", "继续读"]
    ):
        return QueryType.PAPER_READING
    if _looks_like_paper_reading_request(query):
        return QueryType.PAPER_READING
    if any(
        keyword in lowered
        for keyword in [
            "novelty",
            "idea",
            "新颖性",
            "重叠",
            "现有工作",
            "survey",
            "综述",
            "literature review",
        ]
    ):
        return QueryType.IDEA_NOVELTY
    if "cross" in lowered or "跨" in lowered or "transfer" in lowered:
        return QueryType.CROSS_DOMAIN
    if any(
        keyword in lowered
        for keyword in ["趋势", "trend analysis", "trend report", "over time", "growth"]
    ):
        return QueryType.TREND
    return QueryType.QA


def _recover_planner_output(raw) -> PlannerOutput | None:
    text = raw_output_text(raw).strip()
    payload = extract_json_candidate(text)
    if isinstance(payload, dict):
        query_type = payload.get("query_type") or payload.get("classification")
        query_type_aliases = {
            "idea_novelty_analysis": QueryType.IDEA_NOVELTY.value,
            "novelty_analysis": QueryType.IDEA_NOVELTY.value,
            "idea_novelty": QueryType.IDEA_NOVELTY.value,
            "survey": QueryType.IDEA_NOVELTY.value,
            "paper_reading_mode": QueryType.PAPER_READING.value,
            "paper_reader": QueryType.PAPER_READING.value,
            "study_plan_assistant": QueryType.STUDY_PLAN.value,
            "chat": QueryType.QA.value,
            "casual_chat": QueryType.QA.value,
            "chitchat": QueryType.QA.value,
            "smalltalk": QueryType.QA.value,
        }
        sub_queries = (
            payload.get("sub_queries")
            or payload.get("subqueries")
            or payload.get("queries")
            or []
        )
        source_papers = (
            payload.get("source_papers")
            or payload.get("source_paper_titles")
            or payload.get("papers")
            or []
        )
        target_domains = (
            payload.get("target_domains")
            or payload.get("requested_domains")
            or payload.get("domains")
            or []
        )
        paper_ids = (
            payload.get("paper_ids")
            or payload.get("paper_id_filters")
            or payload.get("filter_paper_ids")
            or []
        )
        read_papers = payload.get("read_papers") or payload.get("completed_papers") or []
        known_topics = payload.get("known_topics") or payload.get("topics_known") or []
        constraints = payload.get("constraints") or payload.get("requirements") or []
        categories = payload.get("categories") or payload.get("domains_filter") or []
        if isinstance(sub_queries, str):
            sub_queries = [sub_queries]
        if isinstance(source_papers, str):
            source_papers = [source_papers]
        if isinstance(target_domains, str):
            target_domains = [target_domains]
        if isinstance(paper_ids, str):
            paper_ids = [paper_ids]
        if isinstance(read_papers, str):
            read_papers = [read_papers]
        if isinstance(known_topics, str):
            known_topics = [known_topics]
        if isinstance(constraints, str):
            constraints = [constraints]
        if isinstance(categories, str):
            categories = [categories]
        adapted = dict(payload)
        if query_type:
            normalized = str(query_type).strip().lower()
            mapped = query_type_aliases.get(normalized, normalized)
            if mapped not in {item.value for item in QueryType}:
                mapped = planner_fallback(mapped).value
            adapted["query_type"] = mapped
        adapted["sub_queries"] = list(sub_queries)
        adapted["source_papers"] = list(source_papers)
        adapted["target_domains"] = list(target_domains)
        adapted["paper_ids"] = list(paper_ids)
        adapted["read_papers"] = list(read_papers)
        adapted["known_topics"] = list(known_topics)
        adapted["constraints"] = list(constraints)
        adapted["categories"] = list(categories)
        if "paper_id" not in adapted:
            adapted["paper_id"] = payload.get("arxiv_id") or payload.get("paper")
        if "paper_title" not in adapted:
            adapted["paper_title"] = payload.get("title") or payload.get("paper_name")
        time_range = payload.get("time_range") or payload.get("date_range")
        if isinstance(time_range, dict):
            adapted["date_from"] = adapted.get("date_from") or time_range.get("start")
            adapted["date_to"] = adapted.get("date_to") or time_range.get("end")
        adapted.pop("classification", None)
        try:
            return PlannerOutput.model_validate(adapted)
        except Exception:
            return None
    if not text:
        return None
    query_type = planner_fallback(text)
    return PlannerOutput(query_type=query_type.value, sub_queries=[])


def _build_request_payload(
    existing_payload: dict[str, Any] | None,
    query: str,
    query_type: QueryType,
    structured: PlannerOutput | None,
    paper_repository,
) -> dict[str, Any]:
    payload = dict(existing_payload or {})
    arxiv_ids = _extract_arxiv_ids(query)

    _merge_common_retrieval_payload(payload, structured)
    if query_type in {QueryType.QA, QueryType.IDEA_NOVELTY, QueryType.TREND} and arxiv_ids:
        _set_payload_list(payload, "paper_ids", arxiv_ids)

    if query_type == QueryType.PAPER_READING:
        _merge_paper_reading_payload(payload, query, structured, arxiv_ids, paper_repository)
    elif query_type == QueryType.STUDY_PLAN:
        _merge_study_plan_payload(payload, structured, arxiv_ids)
    elif query_type == QueryType.TREND:
        _set_payload_string(payload, "topic", _structured_value(structured, "topic"))
        granularity = _optional_string(_structured_value(structured, "granularity"))
        if granularity in {"yearly", "quarterly", "monthly"}:
            _set_payload_string(payload, "granularity", granularity)
    elif query_type == QueryType.CROSS_DOMAIN:
        _set_payload_int(payload, "max_hypotheses", _structured_value(structured, "max_hypotheses"))

    return payload


def _merge_common_retrieval_payload(
    payload: dict[str, Any],
    structured: PlannerOutput | None,
) -> None:
    _set_payload_list(payload, "paper_ids", _structured_value(structured, "paper_ids"))
    _set_payload_list(payload, "categories", _structured_value(structured, "categories"))
    _set_payload_date(payload, "date_from", _structured_value(structured, "date_from"))
    _set_payload_date(payload, "date_to", _structured_value(structured, "date_to"))
    _set_payload_int(payload, "max_papers", _structured_value(structured, "max_papers"))
    rag_strategy = _optional_string(_structured_value(structured, "rag_strategy"))
    if rag_strategy in {"dense", "sparse", "hybrid", "reranked_hybrid"}:
        _set_payload_string(payload, "rag_strategy", rag_strategy)


def _merge_paper_reading_payload(
    payload: dict[str, Any],
    query: str,
    structured: PlannerOutput | None,
    arxiv_ids: list[str],
    paper_repository,
) -> None:
    paper_id = _clean_arxiv_id(_structured_value(structured, "paper_id"))
    if not paper_id and arxiv_ids:
        paper_id = arxiv_ids[0]

    paper_title = _optional_string(_structured_value(structured, "paper_title"))
    if not paper_id:
        candidates = [paper_title] if paper_title else []
        if _has_title_like_text(query):
            candidates.append(query)
        paper_id = _resolve_paper_id(candidates, paper_repository)

    _set_payload_string(payload, "paper_id", paper_id)
    _set_payload_string(payload, "paper_title", paper_title)
    _set_payload_string(
        payload,
        "section",
        _normalize_section(_structured_value(structured, "section")),
    )
    _set_payload_int(
        payload,
        "paragraph_index",
        _structured_value(structured, "paragraph_index"),
        minimum=0,
    )
    _set_payload_string(payload, "depth", _normalize_depth(_structured_value(structured, "depth")))
    _set_payload_string(
        payload,
        "instruction",
        _structured_value(structured, "instruction") or query,
    )


def _merge_study_plan_payload(
    payload: dict[str, Any],
    structured: PlannerOutput | None,
    arxiv_ids: list[str],
) -> None:
    _set_payload_string(payload, "goal", _structured_value(structured, "goal"))
    _set_payload_string(
        payload,
        "current_progress",
        _structured_value(structured, "current_progress"),
    )
    _set_payload_list(
        payload,
        "read_papers",
        _structured_value(structured, "read_papers") or arxiv_ids,
    )
    _set_payload_list(payload, "known_topics", _structured_value(structured, "known_topics"))
    _set_payload_int(payload, "timeline_weeks", _structured_value(structured, "timeline_weeks"))
    _set_payload_int(payload, "weekly_hours", _structured_value(structured, "weekly_hours"))
    _set_payload_list(payload, "constraints", _structured_value(structured, "constraints"))


def _looks_like_paper_reading_request(query: str, structured: PlannerOutput | None = None) -> bool:
    lowered = query.lower()
    read_signal = any(
        keyword in lowered
        for keyword in [
            "paper reading",
            "read this paper",
            "read the paper",
            "论文精读",
            "精读",
            "逐段",
            "阅读",
            "读这篇",
            "继续读",
            "讲这一段",
        ]
    )
    paper_signal = (
        bool(_extract_arxiv_ids(query))
        or any(keyword in lowered for keyword in ["论文", "文章", "paper", "arxiv"])
        or bool(_structured_value(structured, "paper_id"))
        or bool(_structured_value(structured, "paper_title"))
    )
    return read_signal and paper_signal


def _looks_like_active_reading_followup(state: dict[str, Any]) -> bool:
    if not (reading_value(state, "active_paper_id") or reading_value(state, "cursor")):
        return False
    query = str(request_value(state, "query", "")).lower()
    if not query:
        return False
    return any(
        keyword in query
        for keyword in [
            "继续",
            "下一段",
            "下一个段落",
            "接着",
            "往下",
            "后面",
            "再讲",
            "讲解",
            "总结目前",
            "目前读过",
            "跳到",
            "实验部分",
            "方法部分",
            "摘要",
            "引言",
            "next",
            "continue",
        ]
    )


def _extract_arxiv_ids(query: str) -> list[str]:
    seen: set[str] = set()
    paper_ids: list[str] = []
    for match in _ARXIV_ID_PATTERN.findall(query):
        if match in seen:
            continue
        paper_ids.append(match)
        seen.add(match)
    return paper_ids


def _resolve_paper_id(candidates: list[str | None], paper_repository) -> str | None:
    clean_candidates = [
        candidate
        for candidate in (_optional_string(item) for item in candidates)
        if candidate
    ]
    if not clean_candidates or paper_repository is None:
        return None

    for candidate in clean_candidates:
        ids = _extract_arxiv_ids(candidate)
        if ids:
            return ids[0]

    resolver = getattr(paper_repository, "resolve_paper_queries", None)
    if callable(resolver):
        try:
            for item in resolver(clean_candidates):
                paper_id = (
                    item.get("paper_id")
                    if isinstance(item, dict)
                    else getattr(item, "paper_id", None)
                )
                cleaned = _clean_arxiv_id(paper_id)
                if cleaned:
                    return cleaned
        except Exception:
            pass

    search = getattr(paper_repository, "search_papers", None)
    if not callable(search):
        return None
    for candidate in clean_candidates:
        try:
            matches, _ = search(candidate, page=1, page_size=1)
        except Exception:
            continue
        if not matches:
            continue
        match = matches[0]
        title = match.get("title", "") if isinstance(match, dict) else getattr(match, "title", "")
        paper_id = (
            match.get("paper_id")
            if isinstance(match, dict)
            else getattr(match, "paper_id", None)
        )
        if title and overlap_score(candidate, title) <= 0:
            continue
        cleaned = _clean_arxiv_id(paper_id)
        if cleaned:
            return cleaned
    return None


def _set_payload_string(payload: dict[str, Any], key: str, value: Any) -> None:
    if _payload_has_value(payload, key):
        return
    text = _optional_string(value)
    if text:
        payload[key] = text


def _set_payload_list(payload: dict[str, Any], key: str, value: Any) -> None:
    if _payload_has_value(payload, key):
        return
    items = _normalize_list(value)
    if items:
        payload[key] = items


def _set_payload_int(
    payload: dict[str, Any],
    key: str,
    value: Any,
    *,
    minimum: int = 1,
) -> None:
    if _payload_has_value(payload, key):
        return
    if value is None or value == "":
        return
    try:
        number = int(value)
    except (TypeError, ValueError):
        return
    if number < minimum:
        return
    payload[key] = number


def _set_payload_date(payload: dict[str, Any], key: str, value: Any) -> None:
    if _payload_has_value(payload, key):
        return
    parsed = _coerce_date(value)
    if parsed is not None:
        payload[key] = parsed


def _payload_has_value(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _structured_value(structured: PlannerOutput | None, field: str) -> Any:
    if structured is None:
        return None
    return getattr(structured, field, None)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set)) else [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _optional_string(item)
        if not text or text in seen:
            continue
        result.append(text)
        seen.add(text)
    return result


def _coerce_date(value: Any) -> date | None:
    if value is None or isinstance(value, date):
        return value
    text = _optional_string(value)
    if text is None:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _clean_arxiv_id(value: Any) -> str | None:
    text = _optional_string(value)
    if not text:
        return None
    match = _ARXIV_ID_PATTERN.search(text)
    return match.group(1) if match else None


def _normalize_section(value: Any) -> str | None:
    text = _optional_string(value)
    if not text:
        return None
    lowered = text.lower()
    aliases = {
        "摘要": "abstract",
        "abstract": "abstract",
        "引言": "introduction",
        "介绍": "introduction",
        "introduction": "introduction",
        "方法": "method",
        "method": "method",
        "methodology": "method",
        "实验": "experiment",
        "结果": "experiment",
        "experiment": "experiment",
        "experiments": "experiment",
        "讨论": "discussion",
        "discussion": "discussion",
        "结论": "conclusion",
        "conclusion": "conclusion",
    }
    return aliases.get(lowered, text)


def _normalize_depth(value: Any) -> str | None:
    text = _optional_string(value)
    if not text:
        return None
    lowered = text.lower()
    aliases = {
        "brief": "brief",
        "简要": "brief",
        "简单": "brief",
        "standard": "standard",
        "普通": "standard",
        "正常": "standard",
        "deep": "deep",
        "深入": "deep",
        "详细": "deep",
    }
    return aliases.get(lowered)


def _has_title_like_text(query: str) -> bool:
    return bool(re.search(r"[A-Za-z][A-Za-z0-9:\- ]{6,}", query))


def _heuristic_crossdomain_intent(query: str, paper_repository) -> dict[str, list[str]]:
    lowered = query.lower()
    source_papers: list[str] = []
    target_domains: list[str] = []
    seen_sources: set[str] = set()
    seen_domains: set[str] = set()

    paper_id_matches = re.findall(r"\b\d{4}\.\d{5}\b", query)
    for paper_id in paper_id_matches:
        if paper_id not in seen_sources:
            source_papers.append(paper_id)
            seen_sources.add(paper_id)

    if paper_repository is not None:
        try:
            for paper in paper_repository.all_papers():
                title = getattr(paper, "title", "")
                if not title:
                    continue
                if title.lower() in lowered and title not in seen_sources:
                    source_papers.append(title)
                    seen_sources.add(title)
            if not source_papers:
                papers, _ = paper_repository.search_papers(query, page=1, page_size=3)
                for paper in papers:
                    title = paper["title"]
                    if overlap_score(query, title) <= 0:
                        continue
                    if title not in seen_sources:
                        source_papers.append(title)
                        seen_sources.add(title)
                        break
        except Exception:
            pass

    patterns = [
        r"(?:迁移到|应用到|用于|尝试应用到|适配到)(.+)",
        r"(?:transfer to|apply to|use in)\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        fragment = match.group(1)
        fragment = re.split(r"[。？！!?]| with | 并 | ，但 |, but ", fragment, maxsplit=1)[0]
        candidates = re.split(r"[、,，/]| and | 与 | 和 ", fragment)
        for candidate in candidates:
            cleaned = candidate.strip(" “”…\"'：:;；")
            if len(cleaned) < 2:
                continue
            if cleaned.lower() in {
                "cross domain",
                "cross-domain",
                "paper",
                "papers",
                "领域",
                "方向",
            }:
                continue
            if cleaned not in seen_domains:
                target_domains.append(cleaned)
                seen_domains.add(cleaned)
        if target_domains:
            break

    return {"source_papers": source_papers, "target_domains": target_domains}


def _extract_explicit_memory_candidates(query: str) -> list[str]:
    for pattern in _MEMORY_SAVE_PATTERNS:
        match = pattern.match(query.strip())
        if not match:
            continue
        content = _clean_explicit_memory_content(match.group("content"))
        if content:
            return [content]
    return []


def _clean_explicit_memory_content(content: str) -> str:
    content = _MEMORY_FOLLOWUP_TASK_PATTERN.sub("", content.strip())
    return content.strip(" \t\n\r。！!？?")


def _should_retrieve_memory_context(
    *,
    query: str,
    query_type: QueryType,
    explicit_memory_candidates: list[str],
    conditional_memory_injection: bool = False,
) -> bool:
    if not conditional_memory_injection:
        return True
    if explicit_memory_candidates:
        return False
    if query_type in {QueryType.PAPER_READING, QueryType.STUDY_PLAN}:
        return True
    return _matches_memory_injection_intent(query, query_type)


def _matches_memory_injection_intent(query: str, query_type: QueryType) -> bool:
    patterns = list(_GENERAL_MEMORY_INTENT_PATTERNS)
    if query_type in {QueryType.PAPER_READING, QueryType.STUDY_PLAN}:
        patterns.extend(_CONTEXT_HEAVY_MEMORY_INTENT_PATTERNS)
    return any(pattern.search(query) for pattern in patterns)
