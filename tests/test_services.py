from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from scholar_mind.agents.state import request_value
from scholar_mind.app import get_container
from scholar_mind.memory.compressor import MessageCompressor
from scholar_mind.models.domain import (
    AskRequest,
    CrossDomainRequest,
    IdeaNoveltyRequest,
    PaperReadingRequest,
    QueryType,
    StudyPlanRequest,
)


@pytest.mark.asyncio
async def test_memory_is_not_written_when_extraction_llm_unavailable():
    container = get_container()
    await container.research_service.ask(
        AskRequest(
            query="I prefer concise retrieval comparisons with citations", user_id="memory-user"
        )
    )

    context, hits = await container.memory_manager.get_context(
        "memory-user", "retrieval comparisons"
    )
    assert hits == 0
    assert context == ""


@pytest.mark.asyncio
async def test_idea_novelty_returns_structured_report():
    container = get_container()
    result = await container.research_service.idea_novelty(
        IdeaNoveltyRequest(
            idea="将检索增强规划策略迁移到多智能体代码修复场景",
            user_id="novelty-user",
        )
    )

    assert result["idea_novelty"]["overlapping_papers"]
    assert result["idea_novelty"]["novelty_report"]["overall_judgement"]
    assert result["papers_analyzed"] >= 1


@pytest.mark.asyncio
async def test_cross_domain_returns_candidates():
    container = get_container()
    result = await container.research_service.cross_domain(
        CrossDomainRequest(
            request=(
                "把 Cross-Domain Transfer of Planning Algorithms from Reinforcement Learning "
                "to NLP 尝试应用到代码生成"
            ),
            user_id="cross-user",
        )
    )

    assert result["cross_domain"]["source_methodology"]["source_papers"]
    assert result["cross_domain"]["candidate_papers"]
    assert result["cross_domain"]["hypotheses"]


@pytest.mark.asyncio
async def test_study_plan_generates_when_input_is_sparse():
    container = get_container()
    result = await container.research_service.study_plan(
        StudyPlanRequest(user_id="study-user")
    )

    assert result["study_plan"]["phases"]
    assert result["study_plan"]["plan_basis"] in {
        "memory_grounded",
        "input_grounded",
        "exploratory",
    }


@pytest.mark.asyncio
async def test_paper_reading_returns_passage_and_next_step():
    container = get_container()
    result = await container.research_service.paper_reading(
        PaperReadingRequest(
            paper_id="2401.00001",
            user_id="reader-user",
            instruction="开始精读，先讲摘要和引言",
        )
    )

    assert result["paper_reading"]["current_passage"]["text"]
    assert result["paper_reading"]["next_step"]["suggestion"]


@pytest.mark.asyncio
async def test_stream_service_emits_plan_and_done():
    container = get_container()
    events = []
    async for event, _ in container.research_service.stream(
        query="What does hybrid retrieval improve?",
        user_id="stream-user",
        session_id="stream-session",
        query_type=QueryType.QA,
        request_payload={"paper_ids": [], "rag_strategy": "hybrid", "top_k": 8},
    ):
        events.append(event)

    assert events[0] == "plan"
    assert events[-1] == "done"
    state = await container.orchestrator.get_state("stream-session")
    assert state is not None
    assert sum(1 for message in state["messages"] if message.type == "human") == 1


@pytest.mark.asyncio
async def test_stream_service_can_run_without_query_type_hint():
    container = get_container()
    events = []
    async for event, _ in container.research_service.stream(
        query="帮我精读 2604.20779 这篇论文",
        user_id="stream-auto-user",
        session_id="stream-auto-session",
        query_type=None,
        request_payload={
            "paper_id": "2604.20779",
            "instruction": "帮我精读 2604.20779 这篇论文",
            "section": None,
            "paragraph_index": None,
            "depth": "standard",
        },
    ):
        events.append(event)

    assert events[0] == "plan"
    assert events[-1] == "done"
    state = await container.orchestrator.get_state("stream-auto-session")
    assert state is not None
    assert request_value(state, "query_type_hint") is None
    assert request_value(state, "query_type") == QueryType.PAPER_READING.value

    request = container.online_eval_repository.get_session_evals("stream-auto-session")[0]
    assert request["query_type"] == QueryType.PAPER_READING.value


@pytest.mark.asyncio
async def test_session_messages_resume_from_langgraph_checkpoint():
    container = get_container()
    await container.research_service.ask(
        AskRequest(
            query="What does hybrid retrieval improve?",
            user_id="checkpoint-user",
            session_id="checkpoint-session",
        )
    )
    await container.research_service.ask(
        AskRequest(
            query="How does sparse retrieval differ?",
            user_id="checkpoint-user",
            session_id="checkpoint-session",
        )
    )

    state = await container.orchestrator.get_state("checkpoint-session")
    assert state is not None
    human_messages = [message for message in state["messages"] if message.type == "human"]
    assert len(human_messages) == 2
    session = container.session_repository.get("checkpoint-session")
    assert session is not None
    assert session.message_count == len(state["messages"])


@pytest.mark.asyncio
async def test_tool_messages_are_logged_after_successful_round():
    container = get_container()
    await container.research_service.ask(
        AskRequest(
            query="What does hybrid retrieval improve?",
            user_id="tool-log-user",
            session_id="tool-log-session",
        )
    )

    log_file = container.memory_manager._log_file_path("tool-log-user")
    content = log_file.read_text(encoding="utf-8")
    assert '"type":"tool"' in content
    assert '"name":"rag_retrieve"' in content


def test_message_compressor_keeps_recent_rounds():
    compressor = MessageCompressor(
        context_window_tokens=1000,
        compact_threshold_ratio=0.5,
    )
    messages = [
        HumanMessage(content="round 1 " + ("x" * 1000)),
        AIMessage(content="answer 1 " + ("y" * 1000)),
        HumanMessage(content="round 2"),
        AIMessage(content="answer 2"),
        HumanMessage(content="round 3"),
        AIMessage(content="answer 3"),
    ]

    compressed = compressor.compress(messages)

    assert compressed[0].type == "system"
    assert [message.content for message in compressed if message.type == "human"] == [
        "round 2",
        "round 3",
    ]


def test_message_compressor_skips_compaction_below_threshold():
    compressor = MessageCompressor(
        context_window_tokens=32768,
        compact_threshold_ratio=0.75,
    )
    messages = [
        HumanMessage(content="round 1"),
        AIMessage(content="answer 1"),
        HumanMessage(content="round 2"),
        AIMessage(content="answer 2"),
    ]

    compressed = compressor.compress(messages)

    assert compressed == messages
