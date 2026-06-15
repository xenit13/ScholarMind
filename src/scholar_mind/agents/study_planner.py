from __future__ import annotations

from time import perf_counter

from langchain_core.messages import AIMessage

from scholar_mind.agents.common import ainvoke_text_output_with_raw, merge_usage
from scholar_mind.agents.state import memory_value, request_value, telemetry_value
from scholar_mind.utils.text import top_keywords


def make_study_planner_node(paper_repository, llm, prompt_catalog):
    return make_study_planner_primary_node(paper_repository, llm, prompt_catalog)


def make_study_planner_primary_node(paper_repository, llm, prompt_catalog):
    prompt = prompt_catalog.get("writer")

    async def study_planner_node(state):
        started = perf_counter()
        payload = request_value(state, "payload", {})
        memory_context = memory_value(state, "context", "")
        plan_basis = _plan_basis(payload, memory_context)
        goal_summary = (
            payload.get("goal")
            or payload.get("request")
            or _goal_from_memory(memory_context)
            or "建立一个可执行的研究学习路线"
        )
        horizon = int(payload.get("timeline_weeks") or 8)
        topics = _plan_topics(payload, memory_context, goal_summary)
        recommended = _recommended_papers(
            paper_repository,
            goal_summary,
            payload.get("read_papers", []),
        )
        phases = _build_phases(topics, recommended, horizon)
        checkpoints = _build_checkpoints(horizon, topics)
        report = {
            "goal_summary": goal_summary,
            "plan_basis": plan_basis,
            "plan_horizon_weeks": horizon,
            "phases": phases,
            "weekly_checkpoints": checkpoints,
            "risks": _build_risks(payload, memory_context),
        }
        if llm is None:
            raise RuntimeError("Study planner primary requires an LLM")
        draft_prompt = f"{prompt}\n\nStudy plan: {report}"
        draft, usage, response = await ainvoke_text_output_with_raw(llm, draft_prompt)
        draft = draft or (
            f"已生成 {horizon} 周学习计划，覆盖 {len(phases)} 个阶段。"
        )
        duration = int((perf_counter() - started) * 1000)
        result = {
            "output": {"study_plan": report, "report_payload": report, "draft": draft},
            "telemetry": {
                "llm_usage": merge_usage(telemetry_value(state, "llm_usage"), usage),
                "agent_trace": telemetry_value(state, "agent_trace", [])
                + [{"agent": "study_planner", "duration_ms": duration}],
            },
        }
        if response is not None:
            result["messages"] = [response]
        return result

    return study_planner_node


def make_study_planner_fallback_node(paper_repository):
    async def study_planner_fallback(state):
        started = perf_counter()
        payload = request_value(state, "payload", {})
        memory_context = memory_value(state, "context", "")
        plan_basis = _plan_basis(payload, memory_context)
        goal_summary = (
            payload.get("goal")
            or payload.get("request")
            or _goal_from_memory(memory_context)
            or "建立一个可执行的研究学习路线"
        )
        horizon = int(payload.get("timeline_weeks") or 8)
        topics = _plan_topics(payload, memory_context, goal_summary)
        recommended = _recommended_papers(
            paper_repository,
            goal_summary,
            payload.get("read_papers", []),
        )
        phases = _build_phases(topics, recommended, horizon)
        checkpoints = _build_checkpoints(horizon, topics)
        report = {
            "goal_summary": goal_summary,
            "plan_basis": plan_basis,
            "plan_horizon_weeks": horizon,
            "phases": phases,
            "weekly_checkpoints": checkpoints,
            "risks": _build_risks(payload, memory_context),
        }
        duration = int((perf_counter() - started) * 1000)
        return {
            "messages": [AIMessage(content="Study plan generated")],
            "output": {
                "study_plan": report,
                "report_payload": report,
                "draft": f"已生成 {horizon} 周学习计划，覆盖 {len(phases)} 个阶段。",
            },
            "telemetry": {
                "llm_usage": merge_usage(telemetry_value(state, "llm_usage")),
                "agent_trace": telemetry_value(state, "agent_trace", [])
                + [{"agent": "study_planner", "duration_ms": duration}],
            },
        }

    return study_planner_fallback


def _plan_basis(payload: dict, memory_context: str) -> str:
    explicit_fields = [
        payload.get("goal"),
        payload.get("current_progress"),
        payload.get("read_papers"),
        payload.get("known_topics"),
        payload.get("timeline_weeks"),
        payload.get("weekly_hours"),
        payload.get("constraints"),
    ]
    if any(value for value in explicit_fields):
        return "input_grounded"
    if memory_context.strip():
        return "memory_grounded"
    return "exploratory"


def _goal_from_memory(memory_context: str) -> str:
    lines = [line.strip("- ").strip() for line in memory_context.splitlines() if line.strip()]
    return lines[0] if lines else ""


def _plan_topics(payload: dict, memory_context: str, goal_summary: str) -> list[str]:
    topics = list(payload.get("known_topics", []))
    if not topics:
        topics = top_keywords(" ".join(filter(None, [goal_summary, memory_context])), limit=4)
    return topics or ["foundations", "methods", "evaluation"]


def _recommended_papers(paper_repository, goal_summary: str, read_papers: list[str]) -> list[str]:
    papers, _ = paper_repository.search_papers(goal_summary, page=1, page_size=5)
    already_read = set(read_papers or [])
    return [
        paper["paper_id"]
        for paper in papers
        if paper["paper_id"] not in already_read
    ][:4]


def _build_phases(topics: list[str], recommended_papers: list[str], horizon: int) -> list[dict]:
    phase_count = 3 if horizon >= 6 else 2
    weeks_per_phase = max(horizon // phase_count, 1)
    phases = []
    for index in range(phase_count):
        topic = topics[index] if index < len(topics) else topics[-1]
        start_week = index * weeks_per_phase + 1
        end_week = horizon if index == phase_count - 1 else min(horizon, (index + 1) * weeks_per_phase)
        phases.append(
            {
                "title": f"阶段 {index + 1}：{topic}",
                "weeks": f"{start_week}-{end_week}",
                "objectives": [
                    f"理解 {topic} 的核心概念",
                    f"把 {topic} 放到整体研究路线中定位",
                ],
                "tasks": [
                    f"阅读并标注与 {topic} 最相关的论文",
                    f"输出一份关于 {topic} 的结构化笔记",
                ],
                "recommended_papers": recommended_papers[index : index + 2],
                "deliverables": [
                    f"{topic} 概念图",
                    f"{topic} 阅读笔记",
                ],
            }
        )
    return phases


def _build_checkpoints(horizon: int, topics: list[str]) -> list[dict]:
    checkpoints = []
    for week in range(1, horizon + 1):
        topic = topics[min(len(topics) - 1, max(0, week - 1) % len(topics))]
        checkpoints.append(
            {
                "week": week,
                "checkpoint": f"能用自己的话解释 {topic} 的关键假设与适用边界。",
            }
        )
    return checkpoints


def _build_risks(payload: dict, memory_context: str) -> list[str]:
    risks = []
    if not payload.get("timeline_weeks"):
        risks.append("未提供时间跨度，当前默认按 8 周规划，节奏可能需要再校准。")
    if not payload.get("weekly_hours"):
        risks.append("未提供每周投入时间，任务密度需结合实际精力调整。")
    if not memory_context.strip():
        risks.append("长期记忆信息较少，计划中的个性化程度有限。")
    return risks
