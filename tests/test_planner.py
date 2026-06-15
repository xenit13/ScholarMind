from __future__ import annotations

import pytest

from scholar_mind.agents.planner import make_planner_node as _make_planner_node
from scholar_mind.agents.state import flatten_graph_state, merge_state_dict

pytestmark = pytest.mark.asyncio


def make_planner_node(*args, **kwargs):
    node = _make_planner_node(*args, **kwargs)

    async def _node(state):
        return flatten_graph_state(await node(state))

    return _node


class _RawResult:
    def __init__(self, content: str, usage_metadata: dict[str, int]):
        self.content = content
        self.usage_metadata = usage_metadata


class _Runnable:
    def __init__(self, llm, payload):
        self.llm = llm
        self.payload = payload

    def invoke(self, prompt: str):
        self.llm.prompts.append(prompt)
        return self.payload

    async def ainvoke(self, prompt: str):
        return self.invoke(prompt)


class _StructuredOutputLLM:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.prompts: list[str] = []

    def with_structured_output(self, _schema, include_raw: bool = False):
        assert include_raw is True
        return _Runnable(self, self.payloads.pop(0))


class _FailingLLM:
    def with_structured_output(self, *_args, **_kwargs):
        raise AssertionError("planner should not call the model when query_type_hint is present")


class _MemoryManager:
    async def get_context(self, *, user_id: str, current_query: str):
        return self.get_context_sync(user_id=user_id, current_query=current_query)

    def get_context_sync(self, *, user_id: str, current_query: str):
        return f"{user_id}:{current_query}", 1


class _TrackingMemoryManager:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    async def get_context(self, *, user_id: str, current_query: str):
        self.calls.append((user_id, current_query))
        return "memory-context", 1


class _PromptCatalog:
    def get(self, name: str) -> str:
        return f"{name} prompt"


class _Paper:
    def __init__(self, paper_id: str, title: str):
        self.paper_id = paper_id
        self.title = title


class _PaperRepository:
    def all_papers(self):
        return [
            _Paper(
                "2405.00005",
                "Cross-Domain Transfer of Planning Algorithms from Reinforcement Learning to NLP",
            ),
            _Paper(
                "2604.20779",
                "SWE-chat: Coding Agent Interactions From Real Users in the Wild",
            ),
        ]

    def search_papers(self, query: str, page: int = 1, page_size: int = 3):
        if "SWE-chat" in query or "2604.20779" in query:
            return (
                [
                    {
                        "paper_id": "2604.20779",
                        "title": "SWE-chat: Coding Agent Interactions From Real Users in the Wild",
                        "abstract": query,
                    }
                ],
                1,
            )
        return (
            [
                {
                    "paper_id": "2405.00005",
                    "title": (
                        "Cross-Domain Transfer of Planning Algorithms "
                        "from Reinforcement Learning to NLP"
                    ),
                    "abstract": query,
                }
            ],
            1,
        )

    def resolve_paper_queries(self, queries):
        resolved = []
        for query in queries:
            if "SWE-chat" in query or "2604.20779" in query:
                resolved.append(
                    {
                        "requested": query,
                        "paper_id": "2604.20779",
                        "title": "SWE-chat: Coding Agent Interactions From Real Users in the Wild",
                    }
                )
        return resolved


async def test_planner_uses_query_type_hint_without_llm_call():
    node = make_planner_node(_FailingLLM(), _MemoryManager(), _PromptCatalog())

    result = await node(
        {
            "query": "开始精读，先讲摘要和引言",
            "user_id": "u1",
            "query_type_hint": "paper_reading",
        }
    )

    assert result["query_type"] == "paper_reading"
    assert result["sub_queries"] == []


async def test_planner_extracts_crossdomain_intent_from_hint_without_llm():
    node = make_planner_node(
        _FailingLLM(),
        _MemoryManager(),
        _PromptCatalog(),
        _PaperRepository(),
    )

    result = await node(
        {
            "query": (
                "把 Cross-Domain Transfer of Planning Algorithms from Reinforcement Learning "
                "to NLP 尝试应用到机器人控制"
            ),
            "user_id": "u1",
            "query_type_hint": "cross_domain",
            "request_payload": {"conditional_memory_injection": True},
        }
    )

    assert result["query_type"] == "cross_domain"
    assert result["cross_domain_intent"]["source_papers"]
    assert result["cross_domain_intent"]["target_domains"] == ["机器人控制"]
    assert result["memory_hit_count"] == 0


async def test_planner_overrides_qa_for_explicit_paper_reading_id():
    llm = _StructuredOutputLLM(
        [
            {
                "parsed": None,
                "raw": _RawResult(
                    '{"classification":"qa"}',
                    {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10},
                ),
                "parsing_error": ValueError("invalid json"),
            }
        ]
    )
    node = make_planner_node(llm, _MemoryManager(), _PromptCatalog(), _PaperRepository())

    result = await node(
        {
            "query": "帮我阅读 2604.20779 这篇文章",
            "user_id": "u1",
            "agent_trace": [],
            "request_payload": {},
        }
    )

    assert result["query_type"] == "paper_reading"
    assert result["request_payload"]["paper_id"] == "2604.20779"
    assert result["request_payload"]["instruction"] == "帮我阅读 2604.20779 这篇文章"


async def test_planner_resolves_paper_reading_title_to_id():
    llm = _StructuredOutputLLM(
        [
            {
                "parsed": None,
                "raw": _RawResult(
                    (
                        '{"query_type":"paper_reading",'
                        '"paper_title":"SWE-chat: Coding Agent Interactions From Real Users in the Wild"}'
                    ),
                    {"input_tokens": 14, "output_tokens": 9, "total_tokens": 23},
                ),
                "parsing_error": ValueError("invalid json"),
            }
        ]
    )
    node = make_planner_node(llm, _MemoryManager(), _PromptCatalog(), _PaperRepository())

    result = await node(
        {
            "query": "帮我阅读 SWE-chat: Coding Agent Interactions From Real Users in the Wild",
            "user_id": "u1",
            "agent_trace": [],
            "request_payload": {},
        }
    )

    assert result["query_type"] == "paper_reading"
    assert result["request_payload"]["paper_id"] == "2604.20779"
    assert (
        result["request_payload"]["paper_title"]
        == "SWE-chat: Coding Agent Interactions From Real Users in the Wild"
    )


async def test_planner_retrieves_memory_after_final_type_selection():
    memory_manager = _TrackingMemoryManager()
    llm = _StructuredOutputLLM(
        [
            {
                "parsed": None,
                "raw": _RawResult(
                    '{"query_type":"paper_reading"}',
                    {"input_tokens": 6, "output_tokens": 2, "total_tokens": 8},
                ),
                "parsing_error": ValueError("invalid json"),
            }
        ]
    )
    node = make_planner_node(llm, memory_manager, _PromptCatalog())

    result = await node(
        {
            "query": "下一步请详细解释",
            "user_id": "u1",
            "agent_trace": [],
            "request_payload": {},
        }
    )

    assert result["query_type"] == "paper_reading"
    assert result["memory_context"] == "memory-context"
    assert result["memory_hit_count"] == 1
    assert memory_manager.calls == [("u1", "下一步请详细解释")]


async def test_planner_routes_implicit_continuation_with_active_reading_state():
    llm = _StructuredOutputLLM(
        [
            {
                "parsed": None,
                "raw": _RawResult(
                    '{"query_type":"qa"}',
                    {"input_tokens": 6, "output_tokens": 2, "total_tokens": 8},
                ),
                "parsing_error": ValueError("invalid json"),
            }
        ]
    )
    node = make_planner_node(llm, _MemoryManager(), _PromptCatalog())

    result = await node(
        {
            "query": "继续讲解下一段",
            "user_id": "u1",
            "agent_trace": [],
            "request_payload": {},
            "active_paper_id": "2604.20779",
            "reading_cursor": {"section": "Introduction", "paragraph_index": 0},
        }
    )

    assert result["query_type"] == "paper_reading"
    assert result["request_payload"]["instruction"] == "继续讲解下一段"


async def test_planner_expands_idea_novelty_sub_queries_from_hint():
    node = make_planner_node(_FailingLLM(), _MemoryManager(), _PromptCatalog())

    result = await node(
        {
            "query": "将检索增强规划策略迁移到多智能体代码修复场景",
            "user_id": "u1",
            "query_type_hint": "idea_novelty",
        }
    )

    assert result["query_type"] == "idea_novelty"
    assert len(result["sub_queries"]) >= 1


async def test_planner_recovers_fenced_json_alias_payload():
    llm = _StructuredOutputLLM(
        [
            {
                "parsed": None,
                "raw": _RawResult(
                    '```json\n{"classification":"idea_novelty","sub_queries":["q1","q2"]}\n```',
                    {"input_tokens": 8, "output_tokens": 5, "total_tokens": 13},
                ),
                "parsing_error": ValueError("invalid json"),
            }
        ]
    )
    node = make_planner_node(llm, _MemoryManager(), _PromptCatalog())

    result = await node(
        {
            "query": "评估这个 idea 的新颖性",
            "user_id": "u1",
            "agent_trace": [],
        }
    )

    assert result["query_type"] == "idea_novelty"
    assert result["sub_queries"] == ["q1", "q2"]
    assert result["llm_usage"]["total_tokens"] == 13
    assert len(llm.prompts) == 1


async def test_planner_recovers_plain_text_query_type():
    llm = _StructuredOutputLLM(
        [
            {
                "parsed": None,
                "raw": _RawResult(
                    "论文精读",
                    {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
                ),
                "parsing_error": ValueError("invalid json"),
            }
        ]
    )
    node = make_planner_node(llm, _MemoryManager(), _PromptCatalog())

    result = await node(
        {
            "query": "先逐段讲解这篇论文",
            "user_id": "u1",
            "agent_trace": [],
        }
    )

    assert result["query_type"] == "paper_reading"
    assert result["sub_queries"] == []


async def test_planner_maps_chat_alias_to_qa():
    llm = _StructuredOutputLLM(
        [
            {
                "parsed": None,
                "raw": _RawResult(
                    '{"classification":"chat"}',
                    {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
                ),
                "parsing_error": ValueError("invalid json"),
            }
        ]
    )
    node = make_planner_node(llm, _MemoryManager(), _PromptCatalog())

    result = await node(
        {
            "query": "今天天气不错",
            "user_id": "u1",
            "agent_trace": [],
        }
    )

    assert result["query_type"] == "qa"
    assert result["sub_queries"] == []


async def test_planner_extracts_explicit_memory_candidates():
    node = make_planner_node(_FailingLLM(), _MemoryManager(), _PromptCatalog())

    result = await node(
        {
            "query": "记住我偏好中文回答",
            "user_id": "u1",
            "query_type_hint": "qa",
        }
    )

    assert result["query_type"] == "qa"
    assert result["explicit_memory_candidates"] == ["我偏好中文回答"]


async def test_planner_trims_followup_task_from_explicit_memory_candidate():
    node = make_planner_node(_FailingLLM(), _MemoryManager(), _PromptCatalog())

    result = await node(
        {
            "query": "请记住：我的偏好是以后回答请简洁，关键结论要带引用。顺便给我制定学习计划。",
            "user_id": "u1",
            "query_type_hint": "study_plan",
        }
    )

    assert result["explicit_memory_candidates"] == [
        "我的偏好是以后回答请简洁，关键结论要带引用"
    ]


async def test_planner_clears_stale_explicit_memory_candidates():
    node = _make_planner_node(_FailingLLM(), _MemoryManager(), _PromptCatalog())

    update = await node(
        {
            "query": "帮我分析这篇论文的主要贡献",
            "user_id": "u1",
            "query_type_hint": "qa",
        }
    )
    merged_memory = merge_state_dict(
        {"explicit_candidates": ["我偏好中文回答"]},
        update["memory"],
    )

    assert merged_memory["explicit_candidates"] == []


async def test_planner_ignores_non_memory_requests():
    node = make_planner_node(_FailingLLM(), _MemoryManager(), _PromptCatalog())

    result = await node(
        {
            "query": "帮我分析这篇论文的主要贡献",
            "user_id": "u1",
            "query_type_hint": "qa",
        }
    )

    assert result["query_type"] == "qa"
    assert result.get("explicit_memory_candidates", []) == []


async def test_planner_retrieves_memory_by_default_for_generic_qa_query():
    memory_manager = _TrackingMemoryManager()
    node = make_planner_node(_FailingLLM(), memory_manager, _PromptCatalog())

    result = await node(
        {
            "query": "帮我分析这篇论文的主要贡献",
            "user_id": "u1",
            "query_type_hint": "qa",
        }
    )

    assert result["memory_context"] == "memory-context"
    assert result["memory_hit_count"] == 1
    assert memory_manager.calls == [("u1", "帮我分析这篇论文的主要贡献")]


async def test_planner_skips_memory_retrieval_for_generic_qa_query_when_conditional():
    memory_manager = _TrackingMemoryManager()
    node = make_planner_node(_FailingLLM(), memory_manager, _PromptCatalog())

    result = await node(
        {
            "query": "帮我分析这篇论文的主要贡献",
            "user_id": "u1",
            "query_type_hint": "qa",
            "request_payload": {"conditional_memory_injection": True},
        }
    )

    assert result["memory_context"] == ""
    assert result["memory_hit_count"] == 0
    assert memory_manager.calls == []


async def test_planner_retrieves_memory_for_preference_driven_qa_query():
    memory_manager = _TrackingMemoryManager()
    node = make_planner_node(_FailingLLM(), memory_manager, _PromptCatalog())

    result = await node(
        {
            "query": "请结合我之前的偏好回答这个问题",
            "user_id": "u1",
            "query_type_hint": "qa",
        }
    )

    assert result["memory_context"] == "memory-context"
    assert result["memory_hit_count"] == 1
    assert memory_manager.calls == [("u1", "请结合我之前的偏好回答这个问题")]


async def test_planner_retrieves_memory_for_based_on_recent_preferences_when_conditional():
    memory_manager = _TrackingMemoryManager()
    node = make_planner_node(_FailingLLM(), memory_manager, _PromptCatalog())

    result = await node(
        {
            "query": "基于刚才这些偏好，帮我回答一篇论文的方法主线时应该怎么组织？",
            "user_id": "u1",
            "query_type_hint": "qa",
            "request_payload": {"conditional_memory_injection": True},
        }
    )

    assert result["memory_context"] == "memory-context"
    assert result["memory_hit_count"] == 1
    assert memory_manager.calls == [
        ("u1", "基于刚才这些偏好，帮我回答一篇论文的方法主线时应该怎么组织？")
    ]


async def test_planner_always_retrieves_memory_for_paper_reading_query():
    memory_manager = _TrackingMemoryManager()
    node = make_planner_node(_FailingLLM(), memory_manager, _PromptCatalog())

    result = await node(
        {
            "query": "先逐段讲解这篇论文的方法部分",
            "user_id": "u1",
            "query_type_hint": "paper_reading",
        }
    )

    assert result["query_type"] == "paper_reading"
    assert result["memory_context"] == "memory-context"
    assert result["memory_hit_count"] == 1
    assert memory_manager.calls == [("u1", "先逐段讲解这篇论文的方法部分")]


async def test_planner_retrieves_memory_for_continuation_paper_reading_query():
    memory_manager = _TrackingMemoryManager()
    node = make_planner_node(_FailingLLM(), memory_manager, _PromptCatalog())

    result = await node(
        {
            "query": "继续上次那篇论文精读，先讲方法部分",
            "user_id": "u1",
            "query_type_hint": "paper_reading",
        }
    )

    assert result["query_type"] == "paper_reading"
    assert result["memory_context"] == "memory-context"
    assert result["memory_hit_count"] == 1
    assert memory_manager.calls == [("u1", "继续上次那篇论文精读，先讲方法部分")]


async def test_planner_always_retrieves_memory_for_study_plan_query():
    memory_manager = _TrackingMemoryManager()
    node = make_planner_node(_FailingLLM(), memory_manager, _PromptCatalog())

    result = await node(
        {
            "query": "帮我制定一个两周的学习计划",
            "user_id": "u1",
            "query_type_hint": "study_plan",
        }
    )

    assert result["query_type"] == "study_plan"
    assert result["memory_context"] == "memory-context"
    assert result["memory_hit_count"] == 1
    assert memory_manager.calls == [("u1", "帮我制定一个两周的学习计划")]


async def test_planner_retrieves_memory_by_default_for_explicit_memory_write_request():
    memory_manager = _TrackingMemoryManager()
    node = make_planner_node(_FailingLLM(), memory_manager, _PromptCatalog())

    result = await node(
        {
            "query": "记住我偏好中文回答",
            "user_id": "u1",
            "query_type_hint": "qa",
        }
    )

    assert result["explicit_memory_candidates"] == ["我偏好中文回答"]
    assert result["memory_context"] == "memory-context"
    assert result["memory_hit_count"] == 1
    assert memory_manager.calls == [("u1", "记住我偏好中文回答")]


async def test_planner_skips_memory_retrieval_for_explicit_memory_write_request_when_conditional():
    memory_manager = _TrackingMemoryManager()
    node = make_planner_node(_FailingLLM(), memory_manager, _PromptCatalog())

    result = await node(
        {
            "query": "记住我偏好中文回答",
            "user_id": "u1",
            "query_type_hint": "qa",
            "request_payload": {"conditional_memory_injection": True},
        }
    )

    assert result["explicit_memory_candidates"] == ["我偏好中文回答"]
    assert result["memory_context"] == ""
    assert result["memory_hit_count"] == 0
    assert memory_manager.calls == []
