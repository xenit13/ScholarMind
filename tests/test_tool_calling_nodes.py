from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from scholar_mind.agents.crossdomain import (
    make_crossdomain_fallback_node as _make_crossdomain_fallback_node,
)
from scholar_mind.agents.crossdomain import (
    make_crossdomain_primary_node as _make_crossdomain_primary_node,
)
from scholar_mind.agents.hypothesis import (
    make_hypothesis_fallback_node as _make_hypothesis_fallback_node,
)
from scholar_mind.agents.hypothesis import (
    make_hypothesis_primary_node as _make_hypothesis_primary_node,
)
from scholar_mind.agents.researcher import make_idea_research_node as _make_idea_research_node
from scholar_mind.agents.researcher import (
    make_research_fallback_node as _make_research_fallback_node,
)
from scholar_mind.agents.researcher import make_research_node as _make_research_node
from scholar_mind.agents.state import flatten_graph_state
from scholar_mind.agents.tools.analytics import build_analytics_tools
from scholar_mind.agents.tools.papers import build_paper_tools
from scholar_mind.agents.tools.retrieval import (
    build_retrieval_tools,
    retrieve_top10_similar_papers_payload,
)
from scholar_mind.agents.trend import make_trend_primary_node as _make_trend_primary_node
from scholar_mind.agents.writer import make_writer_node as _make_writer_node
from scholar_mind.eval.context import finish_eval_context, init_eval_context

pytestmark = pytest.mark.asyncio


def _flattening_node(node):
    async def _node(state):
        return flatten_graph_state(await node(state))

    return _node


def make_research_node(*args, **kwargs):
    return _flattening_node(_make_research_node(*args, **kwargs))


def make_research_fallback_node(*args, **kwargs):
    return _flattening_node(_make_research_fallback_node(*args, **kwargs))


def make_idea_research_node(*args, **kwargs):
    return _flattening_node(_make_idea_research_node(*args, **kwargs))


def make_writer_node(*args, **kwargs):
    return _flattening_node(_make_writer_node(*args, **kwargs))


def make_trend_primary_node(*args, **kwargs):
    return _flattening_node(_make_trend_primary_node(*args, **kwargs))


def make_crossdomain_primary_node(*args, **kwargs):
    return _flattening_node(_make_crossdomain_primary_node(*args, **kwargs))


def make_crossdomain_fallback_node(*args, **kwargs):
    return _flattening_node(_make_crossdomain_fallback_node(*args, **kwargs))


def make_hypothesis_primary_node(*args, **kwargs):
    return _flattening_node(_make_hypothesis_primary_node(*args, **kwargs))


def make_hypothesis_fallback_node(*args, **kwargs):
    return _flattening_node(_make_hypothesis_fallback_node(*args, **kwargs))


class _PromptCatalog:
    def get(self, name: str) -> str:
        return f"{name} prompt"


class _ToolCallingLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.bound_tools = None
        self.invocations = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.invocations.append(messages)
        return self._responses.pop(0)

    async def ainvoke(self, messages):
        return self.invoke(messages)


class _PaperRepository:
    def get_paper(self, _paper_id: str):
        return None

    def related_papers(self, _paper_id: str, limit: int = 5):
        return [{"paper_id": "p2", "title": "Paper 2"}][:limit]


class _ToolRepository:
    def __init__(self):
        self._papers = {
            "p1": SimpleNamespace(
                paper_id="p1",
                title="Paper 1",
                abstract="Hybrid retrieval combines sparse and dense evidence.",
                categories=["cs.IR"],
                authors=["A", "B"],
                publish_date=date(2024, 1, 1),
                citation_count=5,
            ),
            "p2": SimpleNamespace(
                paper_id="p2",
                title="Paper 2",
                abstract="A related paper.",
                categories=["cs.IR"],
                authors=["C"],
                publish_date=date(2023, 6, 1),
                citation_count=2,
            ),
            "x1": SimpleNamespace(
                paper_id="x1",
                title="Candidate Paper 1",
                abstract="Robotics control uses retrieval-guided planning loops.",
                categories=["cs.RO"],
                authors=["D"],
                publish_date=date(2024, 2, 1),
                citation_count=7,
            ),
            "x2": SimpleNamespace(
                paper_id="x2",
                title="Candidate Paper 2",
                abstract="A second robotics planning method.",
                categories=["cs.RO"],
                authors=["E"],
                publish_date=date(2024, 3, 1),
                citation_count=4,
            ),
            "2405.00005": SimpleNamespace(
                paper_id="2405.00005",
                title="Source Paper",
                abstract="Planning with iterative correction.",
                categories=["cs.AI"],
                authors=["S"],
                publish_date=date(2024, 5, 1),
                citation_count=3,
            ),
        }

    def get_paper(self, paper_id):
        return self._papers.get(paper_id)

    def related_papers(self, _paper_id: str, limit: int = 5):
        return [
            {
                "paper_id": "p2",
                "title": "Paper 2",
                "categories": ["cs.IR"],
                "summary": "A related paper.",
            }
        ][:limit]

    def paper_methodology_details(self, paper_id: str):
        paper = self.get_paper(paper_id)
        return {
            "paper_id": paper_id,
            "title": paper.title if paper else paper_id,
            "methodology_summary": f"Methodology details for {paper_id}",
            "sources": [{"kind": "paper_methodology_lookup", "paper_id": paper_id}],
        }

    def top_keywords_for_topic(self, _topic: str, limit: int = 4):
        return ["retrieval", "ranking"][:limit]

    def paper_count_stats(self, **_kwargs):
        return [{"period": "2024-Q1", "count": 3}]

    def keyword_trend_stats(self, keywords, **_kwargs):
        return [
            {"keyword": keyword, "count": 2, "growth_rate": 0.5}
            for keyword in keywords
        ]

    def search_papers(self, _query: str, **_kwargs):
        return (
            [
                {"paper_id": "p1", "title": "Paper 1"},
                {"paper_id": "p2", "title": "Paper 2"},
            ],
            2,
        )


class _ResearchRagEngine:
    def retrieve_sync(self, **_kwargs):
        class _Chunk(SimpleNamespace):
            def model_dump(self, mode="json"):
                return dict(self.__dict__)

        return [
            _Chunk(
                chunk_id="c1",
                paper_id="p1",
                title="Paper 1",
                section="method",
                content="Hybrid retrieval combines lexical and dense evidence.",
                score=0.8,
            )
        ], 8


async def test_researcher_node_runs_tool_loop_inside_private_subgraph():
    repository = _PaperRepository()
    rag_tool = build_retrieval_tools(_ResearchRagEngine())["rag_retrieve"]
    related_tool = build_paper_tools(repository)["related_papers"]
    llm = _ToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "rag_retrieve",
                        "args": {"query": "What does hybrid retrieval improve?"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Collected enough evidence."),
        ]
    )
    node = make_research_node(repository, llm, [rag_tool, related_tool], _PromptCatalog())

    result = await node(
        {
            "query": "What does hybrid retrieval improve?",
            "query_type": "qa",
            "request_payload": {"rag_strategy": "hybrid", "top_k": 4},
            "agent_trace": [],
            "messages": [AIMessage(content="previous turn context")],
        }
    )

    assert llm.bound_tools == [rag_tool, related_tool]
    assert len(llm.invocations) == 2
    assert llm.invocations[0][-1].content == "previous turn context"
    assert result["active_agent"] is None
    assert result["retrieved_chunks"][0]["chunk_id"] == "c1"
    assert result["rag_latency_ms"] == 8
    assert result["messages"][0].content == "Collected enough evidence."
    assert "Hybrid retrieval combines lexical and dense evidence." in result["draft"]
    assert isinstance(llm.invocations[1][-2], AIMessage)
    assert llm.invocations[1][-2].tool_calls[0]["name"] == "rag_retrieve"
    assert isinstance(llm.invocations[1][-1], ToolMessage)


async def test_researcher_rag_retrieve_records_caller_agent_in_eval_context():
    repository = _PaperRepository()
    rag_tool = build_retrieval_tools(_ResearchRagEngine())["rag_retrieve"]
    llm = _ToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "rag_retrieve",
                        "args": {"query": "What does hybrid retrieval improve?"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Collected enough evidence."),
        ]
    )
    node = make_research_node(repository, llm, [rag_tool], _PromptCatalog())
    ctx = init_eval_context(
        request_id="req_rag_caller",
        session_id="sess_1",
        user_id="user_1",
        query="What does hybrid retrieval improve?",
        query_type="qa",
    )

    try:
        await node(
            {
                "query": "What does hybrid retrieval improve?",
                "query_type": "qa",
                "request_payload": {"rag_strategy": "hybrid", "top_k": 4},
                "agent_trace": [],
                "messages": [],
            }
        )
    finally:
        finish_eval_context(ctx, {"final_answer": ""})

    assert ctx.rag_events[0].caller_agent == "researcher"


async def test_researcher_fallback_records_rag_event_in_eval_context():
    repository = _PaperRepository()
    node = make_research_fallback_node(repository, _ResearchRagEngine())
    ctx = init_eval_context(
        request_id="req_research_fallback",
        session_id="sess_1",
        user_id="user_1",
        query="What does hybrid retrieval improve?",
        query_type="qa",
    )

    try:
        result = await node(
            {
                "query": "What does hybrid retrieval improve?",
                "query_type": "qa",
                "request_payload": {"rag_strategy": "hybrid", "top_k": 4},
                "agent_trace": [],
                "messages": [],
            }
        )
    finally:
        finish_eval_context(ctx, {"final_answer": ""})

    assert result["retrieved_chunks"][0]["chunk_id"] == "c1"
    assert ctx.rag_events[0].request_id == "req_research_fallback"
    assert ctx.rag_events[0].caller_agent == "researcher"
    assert ctx.rag_events[0].returned_chunk_ids == ["c1"]


async def test_idea_research_records_rag_event_in_eval_context():
    node = make_idea_research_node(_ResearchRagEngine())
    ctx = init_eval_context(
        request_id="req_idea_research",
        session_id="sess_1",
        user_id="user_1",
        query="Assess novelty for hybrid retrieval",
        query_type="idea_novelty",
    )

    try:
        result = await node(
            {
                "request": {"query": "Assess novelty for hybrid retrieval"},
                "request_payload": {"rag_strategy": "hybrid", "top_k": 4},
            }
        )
    finally:
        finish_eval_context(ctx, {"final_answer": ""})

    assert result["idea_chunk_batches"][0][0]["chunk_id"] == "c1"
    assert ctx.rag_events[0].request_id == "req_idea_research"
    assert ctx.rag_events[0].caller_agent == "idea_research"
    assert ctx.rag_events[0].returned_chunk_ids == ["c1"]


async def test_rag_retrieve_defaults_to_final_citation_top_k_when_tool_call_omits_top_k():
    class _TrackingRagEngine:
        def __init__(self):
            self.calls = []

        def retrieve_sync(self, **kwargs):
            self.calls.append(kwargs)

            class _Chunk(SimpleNamespace):
                def model_dump(self, mode="json"):
                    return dict(self.__dict__)

            return [
                _Chunk(
                    chunk_id="c1",
                    paper_id="p1",
                    title="Paper 1",
                    section="method",
                    content="Hybrid retrieval combines lexical and dense evidence.",
                    score=0.8,
                )
            ], 8

    repository = _PaperRepository()
    rag_engine = _TrackingRagEngine()
    rag_tool = build_retrieval_tools(rag_engine)["rag_retrieve"]
    related_tool = build_paper_tools(repository)["related_papers"]
    llm = _ToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "rag_retrieve",
                        "args": {"query": "What does hybrid retrieval improve?"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Collected enough evidence."),
        ]
    )
    node = make_research_node(repository, llm, [rag_tool, related_tool], _PromptCatalog())

    await node(
        {
            "query": "What does hybrid retrieval improve?",
            "query_type": "qa",
            "request_payload": {"rag_strategy": "hybrid", "top_k": 4},
            "agent_trace": [],
            "messages": [],
        }
    )

    assert rag_engine.calls[0]["top_k"] == 4


async def test_researcher_node_can_answer_qa_without_using_tools():
    repository = _PaperRepository()
    rag_tool = build_retrieval_tools(_ResearchRagEngine())["rag_retrieve"]
    related_tool = build_paper_tools(repository)["related_papers"]
    llm = _ToolCallingLLM([AIMessage(content="你好，我可以直接和你聊天。")])
    node = make_research_node(repository, llm, [rag_tool, related_tool], _PromptCatalog())

    result = await node(
        {
            "query": "你好呀",
            "query_type": "qa",
            "request_payload": {"rag_strategy": "hybrid", "top_k": 4},
            "agent_trace": [],
            "messages": [],
        }
    )

    assert llm.bound_tools == [rag_tool, related_tool]
    assert len(llm.invocations) == 1
    assert result["retrieved_chunks"] == []
    assert result["rag_latency_ms"] == 0
    assert result["messages"][0].content == "你好，我可以直接和你聊天。"
    assert result["draft"] == "你好，我可以直接和你聊天。"


async def test_researcher_node_ignores_parent_tool_history_when_collecting_current_round_results():
    repository = _PaperRepository()
    rag_tool = build_retrieval_tools(_ResearchRagEngine())["rag_retrieve"]
    related_tool = build_paper_tools(repository)["related_papers"]
    llm = _ToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "rag_retrieve",
                        "args": {"query": "What does hybrid retrieval improve?"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Collected enough evidence."),
        ]
    )
    node = make_research_node(repository, llm, [rag_tool, related_tool], _PromptCatalog())

    result = await node(
        {
            "query": "What does hybrid retrieval improve?",
            "query_type": "qa",
            "request_payload": {"rag_strategy": "hybrid", "top_k": 4},
            "agent_trace": [],
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "rag_retrieve",
                            "args": {"query": "old query"},
                            "id": "old_call",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    content='{"chunks": [{"chunk_id": "old_chunk", "paper_id": "old", "title": "Old", "section": "method", "content": "old", "score": 0.1}], "latency_ms": 999}',
                    tool_call_id="old_call",
                    name="rag_retrieve",
                ),
                AIMessage(content="old grounded answer"),
            ],
        }
    )

    assert result["retrieved_chunks"][0]["chunk_id"] == "c1"
    assert result["rag_latency_ms"] == 8
    assert all(chunk["chunk_id"] != "old_chunk" for chunk in result["retrieved_chunks"])


async def test_writer_node_runs_tool_loop_inside_private_subgraph_for_idea_novelty():
    repository = _ToolRepository()
    citation_tool = build_paper_tools(repository)["citation_lookup"]
    llm = _ToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "citation_lookup",
                        "args": {"paper_ids": ["p1"]},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Novelty summary grounded in the retrieved evidence."),
        ]
    )
    node = make_writer_node(llm, [citation_tool], _PromptCatalog())
    retrieved_chunks = [
        {
            "chunk_id": "c1",
            "paper_id": "p1",
            "title": "Paper 1",
            "section": "Method",
            "content": "Hybrid retrieval combines sparse and dense evidence.",
            "score": 0.84,
        }
    ]

    result = await node(
        {
            "query": "将检索增强规划策略迁移到多智能体代码修复场景",
            "query_type": "idea_novelty",
            "retrieved_chunks": retrieved_chunks,
            "agent_trace": [],
            "messages": [AIMessage(content="previous turn context")],
        }
    )

    assert llm.bound_tools == [citation_tool]
    assert len(llm.invocations) == 2
    assert llm.invocations[0][-1].content == "previous turn context"
    assert result["active_agent"] is None
    assert result["draft"] == "Novelty summary grounded in the retrieved evidence."
    assert result["report_payload"]["references"][0]["year"] == 2024
    assert isinstance(llm.invocations[1][-2], AIMessage)
    assert llm.invocations[1][-2].tool_calls[0]["name"] == "citation_lookup"
    assert isinstance(llm.invocations[1][-1], ToolMessage)


class _CrossDomainRepository:
    def __init__(self):
        self._papers = {
            "2405.00005": SimpleNamespace(
                paper_id="2405.00005",
                title="Cross-Domain Transfer of Planning Algorithms from Reinforcement Learning to NLP",
                abstract="Planning algorithms can transfer from RL to NLP.",
                categories=["cs.AI", "cs.LG"],
            ),
            "x1": SimpleNamespace(
                paper_id="x1",
                title="Planning with retrieval for robotics control",
                abstract="Robotics control uses retrieval-guided planning loops.",
                categories=["cs.RO"],
            ),
        }

    def resolve_paper_queries(self, queries):
        return [
            {
                "requested_paper": queries[0],
                "resolved": True,
                "paper_id": "2405.00005",
                "title": (
                    "Cross-Domain Transfer of Planning Algorithms "
                    "from Reinforcement Learning to NLP"
                ),
                "categories": ["cs.AI", "cs.LG"],
                "summary": "Planning algorithms can transfer from RL to NLP.",
                "methodology_summary": (
                    "The method stages planning, rollout evaluation "
                    "and backtracking."
                ),
                "sources": [{"kind": "paper_repository", "paper_id": "2405.00005"}],
            }
        ]

    def get_paper(self, paper_id):
        return self._papers.get(paper_id)


class _CrossDomainRagEngine:
    def __init__(self, paper_repository):
        self.paper_repository = paper_repository
        self.calls = []

    def retrieve_sync(self, **_kwargs):
        self.calls.append(_kwargs)
        return [
            SimpleNamespace(
                paper_id="x1",
                chunk_id="c1",
                section="Method",
                content="Robotics control uses retrieval-guided planning loops.",
                score=0.81,
            )
        ], 12


async def test_top10_similar_papers_uses_final_candidate_count_as_rag_top_k():
    repository = _CrossDomainRepository()
    rag_engine = _CrossDomainRagEngine(repository)

    payload = retrieve_top10_similar_papers_payload(
        rag_engine,
        source_summary="Planning with rollout evaluation",
        target_domains=["机器人控制"],
    )

    assert rag_engine.calls[0]["top_k"] == 10
    assert [item["paper_id"] for item in payload["items"]] == ["x1"]


async def test_trend_node_runs_tool_loop_inside_private_subgraph():
    repository = _ToolRepository()
    analytics_tools = build_analytics_tools(repository)
    paper_tools = build_paper_tools(repository)
    llm = _ToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "paper_count_stats",
                        "args": {
                            "topic": "hybrid retrieval",
                            "categories": [],
                            "date_from": None,
                            "date_to": None,
                            "granularity": "quarterly",
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Trend summary grounded in the gathered statistics."),
        ]
    )
    node = make_trend_primary_node(
        repository,
        llm,
        [
            analytics_tools["paper_count_stats"],
            analytics_tools["keyword_trend_stats"],
            paper_tools["paper_search"],
        ],
        _PromptCatalog(),
    )

    result = await node(
        {
            "query": "hybrid retrieval",
            "query_type": "trend",
            "request_payload": {"categories": [], "granularity": "quarterly"},
            "agent_trace": [],
            "messages": [AIMessage(content="previous turn context")],
        }
    )

    assert len(llm.invocations) == 2
    assert llm.invocations[0][-1].content == "previous turn context"
    assert result["active_agent"] is None
    assert result["trend_data"]["paper_count_by_period"][0]["count"] == 3
    assert result["trend_data"]["summary"] == "Trend summary grounded in the gathered statistics."
    assert isinstance(llm.invocations[1][-2], AIMessage)
    assert llm.invocations[1][-2].tool_calls[0]["name"] == "paper_count_stats"
    assert isinstance(llm.invocations[1][-1], ToolMessage)


async def test_crossdomain_primary_requires_llm_before_candidates_exist():
    repository = _CrossDomainRepository()
    node = make_crossdomain_primary_node(
        repository,
        None,
        [build_retrieval_tools(_CrossDomainRagEngine(repository))["rag_top10_similar_papers"]],
        _PromptCatalog(),
    )

    with pytest.raises(RuntimeError):
        await node(
            {
                "query": (
                    "把 Cross-Domain Transfer of Planning Algorithms from "
                    "Reinforcement Learning to NLP 尝试应用到机器人控制"
                ),
                "query_type": "cross_domain",
                "request_payload": {"rag_strategy": "hybrid", "max_hypotheses": 3},
                "cross_domain_intent": {
                    "source_papers": [
                        "Cross-Domain Transfer of Planning Algorithms "
                        "from Reinforcement Learning to NLP"
                    ],
                    "target_domains": ["机器人控制"],
                    "sources": [{"kind": "planner_llm"}],
                },
                "messages": [],
            }
        )


async def test_crossdomain_primary_runs_tool_loop_inside_private_subgraph():
    repository = _CrossDomainRepository()
    rag_tool = build_retrieval_tools(_CrossDomainRagEngine(repository))[
        "rag_top10_similar_papers"
    ]
    llm = _ToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "rag_top10_similar_papers",
                        "args": {
                            "source_summary": (
                                "The method stages planning, rollout evaluation and backtracking."
                            ),
                            "target_domains": ["机器人控制"],
                            "exclude_paper_ids": ["2405.00005"],
                            "exclude_primary_categories": ["cs.AI"],
                            "strategy": "hybrid",
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content=(
                    '{"source_method_summary":"The method stages planning, rollout evaluation and backtracking.",'
                    '"candidates":[{"paper_id":"x1","methodology_similarity":0.78,'
                    '"transfer_rationale":"Both methods rely on iterative planning and correction."}]}'
                )
            ),
        ]
    )
    node = make_crossdomain_primary_node(
        repository,
        llm,
        [rag_tool],
        _PromptCatalog(),
    )

    result = await node(
        {
            "query": (
                "把 Cross-Domain Transfer of Planning Algorithms from "
                "Reinforcement Learning to NLP 尝试应用到机器人控制"
            ),
            "query_type": "cross_domain",
            "request_payload": {"rag_strategy": "hybrid", "max_hypotheses": 3},
            "cross_domain_intent": {
                "source_papers": [
                    "Cross-Domain Transfer of Planning Algorithms "
                    "from Reinforcement Learning to NLP"
                ],
                "target_domains": ["机器人控制"],
                "sources": [{"kind": "planner_llm"}],
            },
            "agent_trace": [],
            "messages": [AIMessage(content="previous turn context")],
        }
    )

    assert llm.bound_tools == [rag_tool]
    assert len(llm.invocations) == 2
    assert llm.invocations[0][-1].content == "previous turn context"
    assert result["active_agent"] is None
    assert result["source_methodology"]["source_papers"][0]["paper_id"] == "2405.00005"
    assert result["cross_domain_candidates"][0]["paper_id"] == "x1"
    assert result["rag_latency_ms"] == 12
    assert result["transfer_analysis"][0]["paper_id"] == "x1"
    assert result["transfer_analysis"][0]["methodology_similarity"] == 0.78
    assert result["transfer_analysis"][0]["sources"][0]["chunk_id"] == "c1"
    assert '"candidates"' in result["messages"][0].content
    assert isinstance(llm.invocations[1][-2], AIMessage)
    assert llm.invocations[1][-2].tool_calls[0]["name"] == "rag_top10_similar_papers"
    assert isinstance(llm.invocations[1][-1], ToolMessage)


async def test_crossdomain_primary_forwards_parent_messages_into_subgraph():
    repository = _CrossDomainRepository()
    rag_tool = build_retrieval_tools(_CrossDomainRagEngine(repository))[
        "rag_top10_similar_papers"
    ]
    llm = _ToolCallingLLM(
        [
            AIMessage(
                content=(
                    '{"source_method_summary":"The method stages planning, rollout evaluation and backtracking.",'
                    '"candidates":[]}'
                )
            )
        ]
    )
    node = make_crossdomain_primary_node(
        repository,
        llm,
        [rag_tool],
        _PromptCatalog(),
    )

    await node(
        {
            "query": (
                "把 Cross-Domain Transfer of Planning Algorithms from "
                "Reinforcement Learning to NLP 尝试应用到机器人控制"
            ),
            "query_type": "cross_domain",
            "request_payload": {"rag_strategy": "hybrid", "max_hypotheses": 3},
            "cross_domain_intent": {
                "source_papers": [
                    "Cross-Domain Transfer of Planning Algorithms "
                    "from Reinforcement Learning to NLP"
                ],
                "target_domains": ["机器人控制"],
                "sources": [{"kind": "planner_llm"}],
            },
            "agent_trace": [],
            "messages": [AIMessage(content="planner/global noise")],
        }
    )

    assert any(
        getattr(message, "content", None) == "planner/global noise"
        for message in llm.invocations[0]
    )


async def test_crossdomain_fallback_retrieves_and_scores_without_llm():
    repository = _CrossDomainRepository()
    node = make_crossdomain_fallback_node(
        repository,
        _CrossDomainRagEngine(repository),
    )

    result = await node(
        {
            "query": (
                "把 Cross-Domain Transfer of Planning Algorithms from "
                "Reinforcement Learning to NLP 尝试应用到机器人控制"
            ),
            "query_type": "cross_domain",
            "request_payload": {"rag_strategy": "hybrid", "max_hypotheses": 3},
            "cross_domain_intent": {
                "source_papers": [
                    "Cross-Domain Transfer of Planning Algorithms "
                    "from Reinforcement Learning to NLP"
                ],
                "target_domains": ["机器人控制"],
                "sources": [{"kind": "planner_llm"}],
            },
            "agent_trace": [],
            "messages": [],
        }
    )

    assert result["active_agent"] is None
    assert result["cross_domain_candidates"][0]["paper_id"] == "x1"
    assert result["transfer_analysis"][0]["paper_id"] == "x1"
    assert result["transfer_analysis"][0]["sources"][0]["chunk_id"] == "c1"


async def test_hypothesis_primary_raises_when_structured_output_has_no_hypotheses():
    repository = _ToolRepository()
    methodology_tool = build_paper_tools(repository)["paper_methodology_lookup"]
    llm = _ToolCallingLLM(
        [
            AIMessage(content='{"hypotheses": []}'),
        ]
    )
    node = make_hypothesis_primary_node(
        llm,
        [methodology_tool],
        _PromptCatalog(),
    )

    with pytest.raises(RuntimeError, match="no hypothesis drafts"):
        await node(
            {
                "query": "将规划迁移方法用于机器人控制",
                "request_payload": {"max_hypotheses": 2},
                "source_methodology": {
                    "summary": "The source method stages planning, rollout evaluation and backtracking.",
                    "requested_target_domains": ["机器人控制"],
                    "source_papers": [
                        {
                            "paper_id": "2405.00005",
                            "title": "Source Paper",
                            "methodology_summary": "Planning with iterative correction.",
                            "sources": [{"kind": "paper_repository", "paper_id": "2405.00005"}],
                        }
                    ],
                },
                "transfer_analysis": [
                    {
                        "paper_id": "x1",
                        "title": "Candidate Paper 1",
                        "categories": ["cs.RO"],
                        "summary": "Robotics control uses retrieval-guided planning loops.",
                        "methodology_similarity": 0.81,
                        "sources": [{"kind": "rag_top10_similar_papers", "chunk_id": "c1"}],
                    },
                    {
                        "paper_id": "x2",
                        "title": "Candidate Paper 2",
                        "categories": ["cs.RO"],
                        "summary": "A second robotics planning method.",
                        "methodology_similarity": 0.73,
                        "sources": [{"kind": "rag_top10_similar_papers", "chunk_id": "c2"}],
                    },
                ],
                "messages": [],
            }
        )


async def test_hypothesis_primary_runs_tool_loop_inside_private_subgraph():
    repository = _ToolRepository()
    methodology_tool = build_paper_tools(repository)["paper_methodology_lookup"]
    llm = _ToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "paper_methodology_lookup",
                        "args": {"paper_id": "x1"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content=(
                    '{"hypotheses":[{"hypothesis":"Combine planning with robotics control loops.",'
                    '"candidate_paper_ids":["x1","x2"],"novelty_is_novel":true,'
                    '"novelty_confidence":0.81,"novelty_rationale":"New combination.",'
                    '"target_domain":"机器人控制","core_intervention":"Inject planning module",'
                    '"datasets_or_tasks":["robotics control"],"baselines":["Candidate Paper 1"],'
                    '"metrics":["task success"],"ablations":["remove planning"]}]}'
                )
            ),
        ]
    )
    node = make_hypothesis_primary_node(
        llm,
        [methodology_tool],
        _PromptCatalog(),
    )

    result = await node(
        {
            "query": "将规划迁移方法用于机器人控制",
            "request_payload": {"max_hypotheses": 2},
            "source_methodology": {
                "summary": "The source method stages planning, rollout evaluation and backtracking.",
                "requested_target_domains": ["机器人控制"],
                "source_papers": [
                    {
                        "paper_id": "2405.00005",
                        "title": "Source Paper",
                        "methodology_summary": "Planning with iterative correction.",
                        "sources": [{"kind": "paper_repository", "paper_id": "2405.00005"}],
                    }
                ],
            },
            "transfer_analysis": [
                {
                    "paper_id": "x1",
                    "title": "Candidate Paper 1",
                    "categories": ["cs.RO"],
                    "summary": "Robotics control uses retrieval-guided planning loops.",
                    "methodology_similarity": 0.81,
                    "sources": [{"kind": "rag_top10_similar_papers", "chunk_id": "c1"}],
                },
                {
                    "paper_id": "x2",
                    "title": "Candidate Paper 2",
                    "categories": ["cs.RO"],
                    "summary": "A second robotics planning method.",
                    "methodology_similarity": 0.73,
                    "sources": [{"kind": "rag_top10_similar_papers", "chunk_id": "c2"}],
                },
            ],
            "agent_trace": [],
            "messages": [AIMessage(content="previous turn context")],
        }
    )

    assert len(llm.invocations) == 2
    assert llm.invocations[0][-1].content == "previous turn context"
    assert result["active_agent"] is None
    assert len(result["hypotheses"]) == 1
    assert result["hypotheses"][0]["candidate_paper_ids"] == ["x1", "x2"]
    assert isinstance(llm.invocations[1][-2], AIMessage)
    assert llm.invocations[1][-2].tool_calls[0]["name"] == "paper_methodology_lookup"
    assert isinstance(llm.invocations[1][-1], ToolMessage)


async def test_hypothesis_fallback_builds_heuristic_hypothesis_when_candidates_exist():
    node = make_hypothesis_fallback_node(_PromptCatalog())

    result = await node(
        {
            "query": "将规划迁移方法用于机器人控制",
            "request_payload": {"max_hypotheses": 2},
            "source_methodology": {
                "summary": "The source method stages planning, rollout evaluation and backtracking.",
                "requested_target_domains": ["机器人控制"],
                "source_papers": [
                    {
                        "paper_id": "2405.00005",
                        "title": "Source Paper",
                        "methodology_summary": "Planning with iterative correction.",
                        "sources": [{"kind": "paper_repository", "paper_id": "2405.00005"}],
                    }
                ],
            },
            "transfer_analysis": [
                {
                    "paper_id": "x1",
                    "title": "Candidate Paper 1",
                    "categories": ["cs.RO"],
                    "summary": "Robotics control uses retrieval-guided planning loops.",
                    "methodology_similarity": 0.81,
                    "sources": [{"kind": "rag_top10_similar_papers", "chunk_id": "c1"}],
                },
                {
                    "paper_id": "x2",
                    "title": "Candidate Paper 2",
                    "categories": ["cs.RO"],
                    "summary": "A second robotics planning method.",
                    "methodology_similarity": 0.73,
                    "sources": [{"kind": "rag_top10_similar_papers", "chunk_id": "c2"}],
                },
            ],
            "messages": [],
            "agent_trace": [],
        }
    )

    assert result["active_agent"] is None
    assert len(result["hypotheses"]) == 1
    assert result["hypotheses"][0]["candidate_paper_ids"] == ["x1", "x2"]
    assert result["hypotheses"][0]["novelty_check"]["is_novel"] is True
