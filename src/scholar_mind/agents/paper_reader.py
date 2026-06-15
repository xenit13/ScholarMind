from __future__ import annotations

import json
from time import perf_counter

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from scholar_mind.agents.common import (
    ainvoke_structured_output_with_raw,
    ainvoke_text_output_with_raw,
    merge_usage,
)
from scholar_mind.agents.state import memory_value, reading_value, request_value, telemetry_value
from scholar_mind.models.domain import PaperReadingDecision
from scholar_mind.utils.text import overlap_score, top_keywords, truncate

_DEFAULT_READING_PLANNER_PROMPT = """
You are the Paper Reading Planner for ScholarMind.
Decide the next reading action from the user's request, paper outline, and current reading state.
Return JSON only matching the schema. Use an existing section and a valid paragraph index when possible.
For a first request to read a paper, create a concise whole-paper reading plan and target the first useful passage.
For implicit follow-ups like "continue" or "next paragraph", continue from the current reading_cursor.
Do not invent sections or paper content.
""".strip()


def make_paper_reader_node(paper_repository, primary_llm, secondary_llm, prompt_catalog):
    return make_paper_reader_primary_node(paper_repository, primary_llm, secondary_llm, prompt_catalog)


def make_paper_reader_primary_node(paper_repository, primary_llm, secondary_llm, prompt_catalog):
    prompt = prompt_catalog.get("paper_reader") or prompt_catalog.get("writer")
    planner_prompt = prompt_catalog.get("paper_reading_planner")

    async def paper_reader_node(state):
        started = perf_counter()
        payload = request_value(state, "payload", {})
        paper_id = payload.get("paper_id") or reading_value(state, "active_paper_id")
        paper = paper_repository.get_paper(paper_id) if paper_id else None
        if paper is None:
            duration = int((perf_counter() - started) * 1000)
            empty_report = {
                "paper": {
                    "paper_id": paper_id or "",
                    "title": "",
                    "current_section": "",
                    "current_paragraph_index": 0,
                },
                "current_passage": {"section": "", "paragraph_index": 0, "text": ""},
                "explanation": {
                    "plain_language": "未找到指定论文。",
                    "technical_detail": "请确认 paper_id 是否存在于当前索引语料中。",
                    "formula_notes": [],
                    "figure_notes": [],
                    "algorithm_notes": [],
                },
                "knowledge_links": [],
                "notes": {
                    "contribution": [],
                    "methodology": [],
                    "key_results": [],
                    "limitations": [],
                },
                "next_step": {"section": "", "paragraph_index": 0, "suggestion": "请重新指定论文。"},
            }
            return {
                "output": {"report_payload": empty_report, "draft": "未找到指定论文。"},
                "telemetry": {
                    "llm_usage": merge_usage(telemetry_value(state, "llm_usage")),
                    "agent_trace": telemetry_value(state, "agent_trace", [])
                    + [{"agent": "paper_reader", "duration_ms": duration}],
                },
            }

        outline = paper_repository.paper_outline(paper.paper_id)
        decision, decision_usage, decision_response = await _invoke_reading_decision(
            [primary_llm, secondary_llm],
            prompt=planner_prompt,
            state=state,
            payload=payload,
            paper=paper,
            outline=outline,
        )
        decision = _validated_decision(decision, state, payload, outline)
        section, paragraph_index = _resolve_decision_target(decision, state, payload, outline)
        passage = paper_repository.paper_read_passage(
            paper.paper_id,
            section,
            paragraph_index,
            window=0,
        )
        if passage is None:
            section = outline[0]["title"] if outline else (paper.sections[0].title if paper.sections else "")
            passage = paper_repository.paper_read_passage(
                paper.paper_id,
                section,
                0,
                window=0,
            ) or {
                "section": section,
                "paragraph_index": 0,
                "text": paper.abstract,
                "paragraphs": [paper.abstract],
                "section_paragraph_count": 1,
            }
        assets = paper_repository.paper_section_assets(
            paper.paper_id,
            passage["section"],
            ["formula", "algorithm", "table", "figure_desc"],
        )
        memory_context = memory_value(state, "context", "")
        explanation = _build_explanation(passage["text"], assets)
        knowledge_links = _build_knowledge_links(memory_context, passage["text"])
        notes = _build_notes(reading_value(state, "notes"), passage["text"], passage["section"])
        next_step = _next_step(outline, passage["section"], passage["paragraph_index"])
        reading_plan = _reading_plan(decision, state)
        reading_action = _reading_action_payload(decision, passage, next_step)
        prompt_messages = _paper_reader_messages(
            prompt=prompt,
            paper_title=paper.title,
            instruction=payload.get("instruction", ""),
            depth=decision.depth or payload.get("depth", "standard"),
            reading_action=reading_action,
            reading_plan=reading_plan,
            section=passage["section"],
            paragraph_index=passage["paragraph_index"],
            passage_text=passage["text"],
            assets=assets,
            memory_context=memory_context,
            next_step=next_step,
        )
        if primary_llm is None and secondary_llm is None:
            raise RuntimeError("Paper reader primary requires at least one LLM")
        generated_explanation, usage, response = await _invoke_text_output_with_fallback(
            [primary_llm, secondary_llm],
            prompt_messages,
        )
        if generated_explanation:
            explanation["plain_language"] = generated_explanation.strip()
        explanation["plain_language"] = _compose_reader_answer(
            explanation["plain_language"],
            decision,
            reading_plan,
        )
        report = {
            "paper": {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "current_section": passage["section"],
                "current_paragraph_index": passage["paragraph_index"],
            },
            "current_passage": {
                "section": passage["section"],
                "paragraph_index": passage["paragraph_index"],
                "text": passage["text"],
            },
            "explanation": explanation,
            "knowledge_links": knowledge_links,
            "notes": notes,
            "next_step": next_step,
            "reading_action": reading_action,
            "reading_plan": reading_plan,
        }
        draft = explanation["plain_language"]
        duration = int((perf_counter() - started) * 1000)
        result = {
            "reading": {
                "active_paper_id": paper.paper_id,
                "plan": reading_plan,
                "completed_steps": _completed_reading_steps(state, reading_action),
                "action": reading_action,
                "outline": outline,
                "cursor": {
                    "section": passage["section"],
                    "paragraph_index": passage["paragraph_index"],
                    "last_action": payload.get("instruction", ""),
                },
                "current_passage": report["current_passage"],
                "notes": notes,
                "knowledge_links": knowledge_links,
            },
            "output": {"report_payload": report, "draft": draft},
            "telemetry": {
                "llm_usage": merge_usage(
                    telemetry_value(state, "llm_usage"), decision_usage, usage
                ),
                "agent_trace": telemetry_value(state, "agent_trace", [])
                + [{"agent": "paper_reader", "duration_ms": duration}],
            },
        }
        messages = []
        if decision_response is not None:
            messages.append(decision_response)
        if response is not None:
            messages.append(response)
        if messages:
            result["messages"] = messages
        return result

    return paper_reader_node


def make_paper_reader_fallback_node(paper_repository):
    async def paper_reader_fallback(state):
        started = perf_counter()
        payload = request_value(state, "payload", {})
        paper_id = payload.get("paper_id") or reading_value(state, "active_paper_id")
        paper = paper_repository.get_paper(paper_id) if paper_id else None
        if paper is None:
            duration = int((perf_counter() - started) * 1000)
            empty_report = {
                "paper": {
                    "paper_id": paper_id or "",
                    "title": "",
                    "current_section": "",
                    "current_paragraph_index": 0,
                },
                "current_passage": {"section": "", "paragraph_index": 0, "text": ""},
                "explanation": {
                    "plain_language": "未找到指定论文。",
                    "technical_detail": "请确认 paper_id 是否存在于当前索引语料中。",
                    "formula_notes": [],
                    "figure_notes": [],
                    "algorithm_notes": [],
                },
                "knowledge_links": [],
                "notes": {
                    "contribution": [],
                    "methodology": [],
                    "key_results": [],
                    "limitations": [],
                },
                "next_step": {"section": "", "paragraph_index": 0, "suggestion": "请重新指定论文。"},
            }
            return {
                "messages": [AIMessage(content="Paper reading failed: missing paper")],
                "output": {"report_payload": empty_report, "draft": "未找到指定论文。"},
                "telemetry": {
                    "agent_trace": telemetry_value(state, "agent_trace", [])
                    + [{"agent": "paper_reader", "duration_ms": duration}]
                },
            }

        outline = paper_repository.paper_outline(paper.paper_id)
        section, paragraph_index = _resolve_cursor(state, payload, outline)
        passage = paper_repository.paper_read_passage(
            paper.paper_id,
            section,
            paragraph_index,
            window=0,
        )
        if passage is None:
            section = outline[0]["title"] if outline else (paper.sections[0].title if paper.sections else "")
            passage = paper_repository.paper_read_passage(
                paper.paper_id,
                section,
                0,
                window=0,
            ) or {
                "section": section,
                "paragraph_index": 0,
                "text": paper.abstract,
                "paragraphs": [paper.abstract],
                "section_paragraph_count": 1,
            }
        assets = paper_repository.paper_section_assets(
            paper.paper_id,
            passage["section"],
            ["formula", "algorithm", "table", "figure_desc"],
        )
        memory_context = memory_value(state, "context", "")
        explanation = _build_explanation(passage["text"], assets)
        knowledge_links = _build_knowledge_links(memory_context, passage["text"])
        notes = _build_notes(reading_value(state, "notes"), passage["text"], passage["section"])
        next_step = _next_step(outline, passage["section"], passage["paragraph_index"])
        report = {
            "paper": {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "current_section": passage["section"],
                "current_paragraph_index": passage["paragraph_index"],
            },
            "current_passage": {
                "section": passage["section"],
                "paragraph_index": passage["paragraph_index"],
                "text": passage["text"],
            },
            "explanation": explanation,
            "knowledge_links": knowledge_links,
            "notes": notes,
            "next_step": next_step,
        }
        duration = int((perf_counter() - started) * 1000)
        return {
            "messages": [AIMessage(content="Paper reading step generated")],
            "reading": {
                "active_paper_id": paper.paper_id,
                "outline": outline,
                "cursor": {
                    "section": passage["section"],
                    "paragraph_index": passage["paragraph_index"],
                    "last_action": payload.get("instruction", ""),
                },
                "current_passage": report["current_passage"],
                "notes": notes,
                "knowledge_links": knowledge_links,
            },
            "output": {"report_payload": report, "draft": explanation["plain_language"]},
            "telemetry": {
                "llm_usage": merge_usage(telemetry_value(state, "llm_usage")),
                "agent_trace": telemetry_value(state, "agent_trace", [])
                + [{"agent": "paper_reader", "duration_ms": duration}],
            },
        }

    return paper_reader_fallback


async def _invoke_reading_decision(
    llms: list[object],
    *,
    prompt: str,
    state: dict,
    payload: dict,
    paper,
    outline: list[dict],
) -> tuple[PaperReadingDecision | None, dict, object | None]:
    merged_usage = merge_usage()
    decision_prompt = _reading_decision_messages(
        prompt=prompt,
        state=state,
        payload=payload,
        paper=paper,
        outline=outline,
    )
    last_response = None
    for llm in llms:
        if llm is None:
            continue
        decision, usage, response = await ainvoke_structured_output_with_raw(
            llm,
            decision_prompt,
            PaperReadingDecision,
        )
        merged_usage = merge_usage(merged_usage, usage)
        last_response = response or last_response
        if decision is not None:
            return decision, merged_usage, response
    return None, merged_usage, last_response


def _reading_decision_messages(
    *,
    prompt: str,
    state: dict,
    payload: dict,
    paper,
    outline: list[dict],
) -> list:
    system_prompt = prompt or _DEFAULT_READING_PLANNER_PROMPT
    context = {
        "user_query": request_value(state, "query", ""),
        "instruction": payload.get("instruction", ""),
        "active_paper_id": reading_value(state, "active_paper_id") or payload.get("paper_id"),
        "paper": {
            "paper_id": getattr(paper, "paper_id", ""),
            "title": getattr(paper, "title", ""),
            "abstract": truncate(getattr(paper, "abstract", ""), 1200),
        },
        "outline": [
            {
                "section": item.get("title", ""),
                "paragraph_count": item.get("paragraph_count", 0),
            }
            for item in outline
        ],
        "reading_cursor": reading_value(state, "cursor", {}),
        "reading_plan": reading_value(state, "plan", []),
        "completed_reading_steps": reading_value(state, "completed_steps", []),
        "memory_context": truncate(memory_value(state, "context", ""), 800),
    }
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(context, ensure_ascii=False, default=str)),
    ]


def _validated_decision(
    decision: PaperReadingDecision | None,
    state: dict,
    payload: dict,
    outline: list[dict],
) -> PaperReadingDecision:
    if decision is None:
        return _fallback_reading_decision(state, payload, outline)
    action = decision.action.strip().lower() if decision.action else "start_reading"
    aliases = {
        "start": "start_reading",
        "plan": "start_plan",
        "next": "continue",
        "jump": "jump_to_section",
        "qa": "local_qa",
        "question_answering": "local_qa",
        "summary": "summarize_so_far",
    }
    allowed = {
        "start_plan",
        "start_reading",
        "continue",
        "jump_to_section",
        "explain_current",
        "local_qa",
        "summarize_so_far",
        "adjust_plan",
        "clarify",
    }
    decision.action = aliases.get(action, action)
    if decision.action not in allowed:
        decision.action = "start_reading"
    if decision.target_section:
        decision.target_section = _resolve_outline_section(decision.target_section, outline)
    if _is_first_broad_reading_request(state, payload):
        decision.action = "start_plan"
        if not decision.plan:
            decision.plan = _build_outline_plan(outline)
        if not decision.target_section and outline:
            decision.target_section = outline[0]["title"]
        if decision.target_paragraph_index is None:
            decision.target_paragraph_index = 0
    return decision


def _is_first_broad_reading_request(state: dict, payload: dict) -> bool:
    if reading_value(state, "plan") or reading_value(state, "cursor"):
        return False
    text = f"{request_value(state, 'query', '')} {payload.get('instruction', '')}".lower()
    read_signal = any(keyword in text for keyword in ["阅读", "精读", "读", "read"])
    paper_signal = any(keyword in text for keyword in ["论文", "文章", "paper"]) or bool(
        payload.get("paper_id")
    )
    return read_signal and paper_signal


def _fallback_reading_decision(
    state: dict,
    payload: dict,
    outline: list[dict],
) -> PaperReadingDecision:
    instruction = str(payload.get("instruction") or request_value(state, "query", "")).lower()
    cursor = reading_value(state, "cursor", {})
    current_section = cursor.get("section") or (outline[0]["title"] if outline else "")
    current_index = int(cursor.get("paragraph_index", 0) or 0)
    if any(keyword in instruction for keyword in ["继续", "下一段", "next", "continue"]):
        return PaperReadingDecision(
            action="continue",
            target_section=current_section,
            target_paragraph_index=current_index + 1,
            reason="用户要求沿当前阅读位置继续。",
        )
    if not reading_value(state, "plan") and not reading_value(state, "cursor"):
        return PaperReadingDecision(
            action="start_reading",
            target_section=outline[0]["title"] if outline else "",
            target_paragraph_index=0,
            reason="首次阅读，开始第一段。",
        )
    return PaperReadingDecision(
        action="explain_current",
        target_section=current_section,
        target_paragraph_index=current_index,
        reason="未识别到明确跳转，解释当前位置。",
    )


def _resolve_decision_target(
    decision: PaperReadingDecision,
    state: dict,
    payload: dict,
    outline: list[dict],
) -> tuple[str, int]:
    if decision.target_section:
        section = decision.target_section
    else:
        section, _ = _resolve_cursor(state, payload, outline)
    paragraph_index = (
        decision.target_paragraph_index
        if decision.target_paragraph_index is not None
        else _resolve_cursor(state, payload, outline)[1]
    )
    section = _resolve_outline_section(section, outline) or (outline[0]["title"] if outline else section)
    if decision.action == "continue":
        return _continue_target(section, int(paragraph_index or 0), outline)
    return section, _clamp_paragraph_index(section, int(paragraph_index or 0), outline)


def _resolve_outline_section(section: str, outline: list[dict]) -> str:
    if not section:
        return ""
    for item in outline:
        title = item.get("title", "")
        if title.lower() == section.lower():
            return title
    ranked = sorted(
        outline,
        key=lambda item: overlap_score(section, item.get("title", "")),
        reverse=True,
    )
    if ranked and overlap_score(section, ranked[0].get("title", "")) > 0:
        return ranked[0].get("title", section)
    return section


def _clamp_paragraph_index(section: str, paragraph_index: int, outline: list[dict]) -> int:
    paragraph_count = 1
    for item in outline:
        if item.get("title") == section:
            paragraph_count = int(item.get("paragraph_count") or 1)
            break
    return min(max(paragraph_index, 0), max(paragraph_count - 1, 0))


def _continue_target(section: str, paragraph_index: int, outline: list[dict]) -> tuple[str, int]:
    for index, item in enumerate(outline):
        if item.get("title") != section:
            continue
        paragraph_count = int(item.get("paragraph_count") or 1)
        if paragraph_index < paragraph_count:
            return section, max(paragraph_index, 0)
        if index + 1 < len(outline):
            return outline[index + 1].get("title", section), 0
        return section, max(paragraph_count - 1, 0)
    return section, _clamp_paragraph_index(section, paragraph_index, outline)


def _reading_plan(decision: PaperReadingDecision, state: dict) -> list[dict]:
    if decision.plan:
        return _normalize_plan(decision.plan)
    existing = reading_value(state, "plan") or []
    if existing:
        return _normalize_plan(existing)
    return []


def _normalize_plan(plan: list[dict]) -> list[dict]:
    normalized = []
    for index, item in enumerate(plan, start=1):
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "step": item.get("step", index),
                "section": item.get("section", ""),
                "purpose": item.get("purpose", ""),
            }
        )
    return normalized


def _build_outline_plan(outline: list[dict]) -> list[dict]:
    plan = []
    for index, item in enumerate(outline[:5], start=1):
        section = item.get("title", "")
        if not section:
            continue
        plan.append(
            {
                "step": index,
                "section": section,
                "purpose": "按论文结构逐步建立理解",
            }
        )
    return plan


def _reading_action_payload(
    decision: PaperReadingDecision,
    passage: dict,
    next_step: dict,
) -> dict:
    return {
        "action": decision.action,
        "reason": decision.reason,
        "reading_mode": decision.reading_mode,
        "depth": decision.depth or "standard",
        "target": {
            "section": passage["section"],
            "paragraph_index": passage["paragraph_index"],
        },
        "next_step": next_step,
        "needs_clarification": decision.needs_clarification,
        "clarification_question": decision.clarification_question,
        "reading_goal": decision.reading_goal,
    }


def _completed_reading_steps(state: dict, reading_action: dict) -> list[dict]:
    completed = list(reading_value(state, "completed_steps", []) or [])
    if reading_action.get("action") in {"start_plan", "start_reading", "continue", "jump_to_section"}:
        completed.append(
            {
                "action": reading_action.get("action"),
                "section": reading_action.get("target", {}).get("section", ""),
                "paragraph_index": reading_action.get("target", {}).get("paragraph_index", 0),
            }
        )
    return completed[-20:]


def _compose_reader_answer(
    explanation: str,
    decision: PaperReadingDecision,
    reading_plan: list[dict],
) -> str:
    if decision.action not in {"start_plan", "adjust_plan"} or not reading_plan:
        return explanation
    plan_text = _format_reading_plan(reading_plan)
    if not plan_text:
        return explanation
    return f"阅读计划：\n{plan_text}\n\n现在开始第一步。\n\n{explanation}"


def _format_reading_plan(reading_plan: list[dict]) -> str:
    lines = []
    for index, item in enumerate(reading_plan, start=1):
        step = item.get("step", index)
        section = item.get("section", "")
        purpose = item.get("purpose", "")
        if section and purpose:
            lines.append(f"{step}. {section}: {purpose}")
        elif section:
            lines.append(f"{step}. {section}")
    return "\n".join(lines)


async def _invoke_text_output_with_fallback(llms: list[object], prompt: object):
    merged_usage = merge_usage()
    for llm in llms:
        if llm is None:
            continue
        text, usage, response = await ainvoke_text_output_with_raw(llm, prompt)
        merged_usage = merge_usage(merged_usage, usage)
        if text:
            return text, merged_usage, response
    return None, merged_usage, None


def _resolve_cursor(state: dict, payload: dict, outline: list[dict]) -> tuple[str, int]:
    explicit_section = payload.get("section")
    explicit_index = payload.get("paragraph_index")
    if explicit_section is not None:
        return explicit_section, int(explicit_index or 0)

    cursor = reading_value(state, "cursor", {})
    current_section = cursor.get("section") or (outline[0]["title"] if outline else "")
    current_index = int(cursor.get("paragraph_index", 0))
    instruction = (payload.get("instruction") or "").lower()
    if any(keyword in instruction for keyword in ["继续", "next", "continue"]):
        return current_section, current_index + 1
    if any(keyword in instruction for keyword in ["摘要", "abstract"]) and outline:
        return outline[0]["title"], 0
    return current_section, current_index


def _paper_reader_messages(
    *,
    prompt: str,
    paper_title: str,
    instruction: str,
    depth: str,
    reading_action: dict,
    reading_plan: list[dict],
    section: str,
    paragraph_index: int,
    passage_text: str,
    assets: list[dict],
    memory_context: str,
    next_step: dict,
) -> list:
    asset_lines = []
    for asset in assets[:4]:
        asset_lines.append(
            f"- {asset['chunk_type']}: {truncate(asset.get('content', ''), 120)}"
        )
    memory_lines = [
        line.strip()
        for line in memory_context.splitlines()
        if line.strip()
    ][:2]
    plan_text = _format_reading_plan(reading_plan) or "(none)"
    human_prompt = (
        f"Paper title: {paper_title}\n"
        f"User instruction: {instruction or '开始精读'}\n"
        f"Reading action: {reading_action.get('action', '')}\n"
        f"Reading action reason: {reading_action.get('reason', '')}\n"
        f"Reading plan:\n{plan_text}\n"
        f"Depth: {depth}\n"
        f"Section: {section}\n"
        f"Paragraph index: {paragraph_index}\n"
        f"Current passage:\n{passage_text}\n\n"
        f"Section assets:\n{chr(10).join(asset_lines) if asset_lines else '(none)'}\n\n"
        f"Relevant memory:\n{chr(10).join(memory_lines) if memory_lines else '(none)'}\n\n"
        f"Next reading suggestion: {next_step.get('suggestion', '')}"
    )
    return [SystemMessage(content=prompt), HumanMessage(content=human_prompt)]


def _build_explanation(text: str, assets: list[dict]) -> dict:
    formula_notes = [truncate(item["content"], 120) for item in assets if item["chunk_type"] == "formula"]
    figure_notes = [truncate(item["content"], 120) for item in assets if item["chunk_type"] == "figure_desc"]
    algorithm_notes = [
        truncate(item["content"], 120)
        for item in assets
        if item["chunk_type"] in {"algorithm", "table"}
    ]
    keywords = top_keywords(text, limit=4)
    return {
        "plain_language": truncate(text, 180),
        "technical_detail": (
            f"这一段重点围绕 {', '.join(keywords) or '核心方法'} 展开，"
            "适合先抓任务设定、方法机制和实验信号。"
        ),
        "formula_notes": formula_notes,
        "figure_notes": figure_notes or ["当前语料未提供稳定的图像描述。"],
        "algorithm_notes": algorithm_notes,
    }


def _build_knowledge_links(memory_context: str, text: str) -> list[dict]:
    if not memory_context.strip():
        return []
    first_memory = memory_context.splitlines()[0].strip("- ").strip()
    return [
        {
            "related_memory": first_memory,
            "connection": (
                "当前段落与已有记忆中的主题存在关联，可以优先对照任务设定和方法假设来理解。"
            ),
        }
    ]


def _build_notes(existing_notes: dict | None, text: str, section: str) -> dict:
    notes = dict(existing_notes or {})
    notes.setdefault("contribution", [])
    notes.setdefault("methodology", [])
    notes.setdefault("key_results", [])
    notes.setdefault("limitations", [])
    snippet = truncate(text, 120)
    lowered = section.lower()
    if "intro" in lowered:
        notes["contribution"] = _append_unique(notes["contribution"], snippet)
    elif "method" in lowered or "approach" in lowered:
        notes["methodology"] = _append_unique(notes["methodology"], snippet)
    elif "experiment" in lowered or "result" in lowered:
        notes["key_results"] = _append_unique(notes["key_results"], snippet)
    else:
        notes["limitations"] = _append_unique(notes["limitations"], snippet)
    return notes


def _next_step(outline: list[dict], section: str, paragraph_index: int) -> dict:
    for index, item in enumerate(outline):
        if item["title"] != section:
            continue
        if paragraph_index + 1 < item["paragraph_count"]:
            return {
                "section": section,
                "paragraph_index": paragraph_index + 1,
                "suggestion": "继续读下一段，重点看这一段如何承接前文。",
            }
        if index + 1 < len(outline):
            next_section = outline[index + 1]["title"]
            return {
                "section": next_section,
                "paragraph_index": 0,
                "suggestion": f"当前段结束，下一步切到 {next_section}。",
            }
    return {
        "section": section,
        "paragraph_index": paragraph_index,
        "suggestion": "已到当前可读范围末尾，可改为提问公式、图表或实验细节。",
    }


def _append_unique(items: list[str], value: str) -> list[str]:
    if value and value not in items:
        items.append(value)
    return items[-3:]
