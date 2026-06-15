from __future__ import annotations

import json
from datetime import date

import pytest

from scholar_mind.agents.paper_reader import make_paper_reader_node as _make_paper_reader_node
from scholar_mind.agents.state import flatten_graph_state
from scholar_mind.models.domain import PaperSection, StructuredPaper

pytestmark = pytest.mark.asyncio


def make_paper_reader_node(*args, **kwargs):
    node = _make_paper_reader_node(*args, **kwargs)

    async def _node(state):
        return flatten_graph_state(await node(state))

    return _node


class _PromptCatalog:
    def get(self, name: str) -> str:
        return f"{name} prompt"


class _LLMResult:
    def __init__(self, content: str, *, input_tokens: int = 10, output_tokens: int = 6):
        self.content = content
        self.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }


class _FailingLLM:
    def invoke(self, _prompt):
        raise RuntimeError("timeout")

    async def ainvoke(self, prompt):
        return self.invoke(prompt)


class _WorkingLLM:
    def __init__(self, content: str):
        self.content = content
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return _LLMResult(self.content)

    async def ainvoke(self, prompt):
        return self.invoke(prompt)


class _StructuredRunnable:
    def __init__(self, llm, payload: dict):
        self.llm = llm
        self.payload = payload

    def invoke(self, prompt):
        self.llm.structured_prompts.append(prompt)
        return {
            "parsed": None,
            "raw": _LLMResult(json.dumps(self.payload)),
            "parsing_error": ValueError("force recovery path"),
        }

    async def ainvoke(self, prompt):
        return self.invoke(prompt)


class _PlanningLLM(_WorkingLLM):
    def __init__(self, decision: dict, content: str):
        super().__init__(content)
        self.decision = decision
        self.structured_prompts = []

    def with_structured_output(self, _schema, include_raw: bool = False):
        assert include_raw is True
        return _StructuredRunnable(self, self.decision)


class _PaperRepository:
    def __init__(self):
        self.paragraphs = [
            "Scientific question answering requires precise evidence.",
            "Existing RAG systems often forget earlier researcher context.",
        ]
        self.paper = StructuredPaper(
            paper_id="p1",
            title="Memory-Augmented RAG",
            authors=["a"],
            abstract="abstract",
            categories=["cs.AI"],
            publish_date=date(2024, 1, 1),
            sections=[
                PaperSection(
                    section_id="s1",
                    title="Introduction",
                    content="\n\n".join(self.paragraphs),
                )
            ],
        )

    def get_paper(self, _paper_id: str):
        return self.paper

    def paper_outline(self, _paper_id: str):
        return [
            {
                "section_id": "s1",
                "title": "Introduction",
                "level": 1,
                "paragraph_count": len(self.paragraphs),
            }
        ]

    def paper_read_passage(self, _paper_id: str, section: str, paragraph_index: int, window: int = 0):
        paragraph_index = min(max(paragraph_index, 0), len(self.paragraphs) - 1)
        return {
            "section": section,
            "paragraph_index": paragraph_index,
            "text": self.paragraphs[paragraph_index],
            "paragraphs": self.paragraphs,
            "section_paragraph_count": len(self.paragraphs),
        }

    def paper_section_assets(self, _paper_id: str, _section: str, _chunk_types: list[str]):
        return [{"chunk_type": "formula", "content": "score = retrieve(query, memory)"}]


async def test_paper_reader_falls_back_to_secondary_model_and_updates_explanation():
    node = make_paper_reader_node(
        _PaperRepository(),
        _FailingLLM(),
        _WorkingLLM("这一段在说明科学问答需要稳定证据，同时指出传统 RAG 容易丢失历史上下文。"),
        _PromptCatalog(),
    )

    result = await node(
        {
            "query": "开始精读，先讲摘要和引言",
            "query_type": "paper_reading",
            "request_payload": {
                "paper_id": "p1",
                "instruction": "开始精读，先讲摘要和引言",
                "depth": "standard",
            },
            "agent_trace": [],
            "messages": [],
        }
    )

    assert result["report_payload"]["paper"]["current_section"] == "Introduction"
    assert result["report_payload"]["explanation"]["plain_language"].startswith("这一段在说明")
    assert result["draft"].startswith("这一段在说明")
    assert result["llm_usage"]["total_tokens"] > 0


async def test_paper_reader_starts_with_llm_reading_plan_and_records_state():
    decision = {
        "action": "start_plan",
        "target_section": "Introduction",
        "target_paragraph_index": 0,
        "reading_mode": "paragraph_explain",
        "depth": "standard",
        "reason": "首次阅读需要先建立整体计划再开始第一段",
        "needs_clarification": False,
        "reading_goal": "理解论文问题、方法和限制",
        "plan": [
            {
                "step": 1,
                "section": "Introduction",
                "purpose": "理解研究动机",
            },
            {
                "step": 2,
                "section": "Method",
                "purpose": "理解方法设计",
            },
        ],
    }
    llm = _PlanningLLM(decision, "这一段在说明科学问答需要稳定证据。")
    node = make_paper_reader_node(_PaperRepository(), llm, None, _PromptCatalog())

    result = await node(
        {
            "query": "帮我阅读 p1 这篇文章",
            "query_type": "paper_reading",
            "request_payload": {
                "paper_id": "p1",
                "instruction": "帮我阅读 p1 这篇文章",
                "depth": "standard",
            },
            "agent_trace": [],
            "messages": [],
        }
    )

    assert result["reading_plan"] == decision["plan"]
    assert result["reading_cursor"]["section"] == "Introduction"
    assert result["reading_cursor"]["paragraph_index"] == 0
    assert result["report_payload"]["reading_action"]["action"] == "start_plan"
    assert result["report_payload"]["explanation"]["plain_language"].startswith("阅读计划")


async def test_paper_reader_enforces_plan_for_first_broad_reading_request():
    decision = {
        "action": "start_reading",
        "target_section": "Introduction",
        "target_paragraph_index": 0,
        "reading_mode": "paragraph_explain",
        "depth": "standard",
        "reason": "开始阅读",
        "needs_clarification": False,
        "plan": [],
    }
    llm = _PlanningLLM(decision, "这一段在说明科学问答需要稳定证据。")
    node = make_paper_reader_node(_PaperRepository(), llm, None, _PromptCatalog())

    result = await node(
        {
            "query": "帮我阅读 p1 这篇文章",
            "query_type": "paper_reading",
            "request_payload": {
                "paper_id": "p1",
                "instruction": "帮我阅读 p1 这篇文章",
                "depth": "standard",
            },
            "agent_trace": [],
            "messages": [],
        }
    )

    assert result["report_payload"]["reading_action"]["action"] == "start_plan"
    assert result["reading_plan"]
    assert result["report_payload"]["explanation"]["plain_language"].startswith("阅读计划")


async def test_paper_reader_continues_to_llm_selected_next_paragraph():
    decision = {
        "action": "continue",
        "target_section": "Introduction",
        "target_paragraph_index": 1,
        "reading_mode": "paragraph_explain",
        "depth": "standard",
        "reason": "用户要求继续下一段",
        "needs_clarification": False,
        "plan": [],
    }
    llm = _PlanningLLM(decision, "这一段指出系统会忘记早先的研究者上下文。")
    node = make_paper_reader_node(_PaperRepository(), llm, None, _PromptCatalog())

    result = await node(
        {
            "query": "继续讲解下一段",
            "query_type": "paper_reading",
            "active_paper_id": "p1",
            "reading_cursor": {"section": "Introduction", "paragraph_index": 0},
            "reading_plan": [{"step": 1, "section": "Introduction", "purpose": "理解研究动机"}],
            "request_payload": {
                "instruction": "继续讲解下一段",
                "depth": "standard",
            },
            "agent_trace": [],
            "messages": [],
        }
    )

    assert result["reading_cursor"]["section"] == "Introduction"
    assert result["reading_cursor"]["paragraph_index"] == 1
    assert result["current_passage"]["text"] == "Existing RAG systems often forget earlier researcher context."
    assert result["report_payload"]["reading_action"]["action"] == "continue"
