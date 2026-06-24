from __future__ import annotations

import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient

from scholar_mind.api.routes.research import ask_stream, extract_transcript_memory
from scholar_mind.app import get_container
from scholar_mind.asgi import create_app
from scholar_mind.models.domain import (
    AskRequest,
    ChatRequest,
    TranscriptMemoryExtractionRequest,
)

REQUEST_TIMEOUT_SECONDS = 15


class _RecordingResearchService:
    def __init__(self):
        self.stream_calls: list[dict] = []
        self.wait_calls: list[dict] = []
        self.transcript_calls: list[dict] = []

    async def stream(self, *, query, user_id, session_id, query_type, request_payload):
        self.stream_calls.append(
            {
                "query": query,
                "user_id": user_id,
                "session_id": session_id,
                "query_type": query_type,
                "request_payload": request_payload,
            }
        )
        yield "answer", {"answer": "ok", "citations": []}

    def wait_for_pending_extractions(self, *, timeout: float = 300.0):
        self.wait_calls.append({"timeout": timeout})
        return {"total": 1, "succeeded": 1, "failed": 0}

    def extract_transcript_memories(
        self,
        *,
        user_id: str,
        request_id: str,
        session_id: str,
        round_messages: list[dict],
    ):
        self.transcript_calls.append(
            {
                "user_id": user_id,
                "request_id": request_id,
                "session_id": session_id,
                "round_messages": round_messages,
            }
        )
        return {"request_id": request_id, "dispatched": True}


class _RecordingContainer:
    def __init__(self):
        self.settings = type(
            "S",
            (),
            {
                "final_citation_top_k": 4,
                "conditional_memory_injection": False,
            },
        )()
        self.research_service = _RecordingResearchService()


def test_ask_request_accepts_single_character_query():
    request = AskRequest(query="你", user_id="tester")

    assert request.query == "你"


def test_memory_injection_condition_flag_defaults_to_false():
    ask_request = AskRequest(query="你", user_id="tester")
    chat_request = ChatRequest(query="你", user_id="tester")

    assert ask_request.conditional_memory_injection is False
    assert chat_request.conditional_memory_injection is False


def test_ask_request_accepts_conditional_memory_injection_override():
    request = AskRequest(
        query="你",
        user_id="tester",
        conditional_memory_injection=True,
    )

    assert request.conditional_memory_injection is True


def test_memory_injection_condition_flag_is_exposed_in_api_contract():
    schema = create_app().openapi()

    ask_properties = schema["components"]["schemas"]["AskRequest"]["properties"]
    chat_properties = schema["components"]["schemas"]["ChatRequest"]["properties"]
    assert ask_properties["conditional_memory_injection"]["default"] is False
    assert chat_properties["conditional_memory_injection"]["default"] is False

    stream_params = {
        param["name"]: param
        for param in schema["paths"]["/api/v1/research/stream"]["get"]["parameters"]
    }
    stream_schema = stream_params["conditional_memory_injection"]["schema"]
    stream_types = {item.get("type") for item in stream_schema.get("anyOf", [stream_schema])}
    assert "boolean" in stream_types


@pytest.mark.asyncio
async def test_ask_stream_post_passes_memory_flags_and_waits_for_extraction():
    container = _RecordingContainer()
    response = await ask_stream(
        AskRequest(
            query="remember this",
            user_id="tester",
            session_id="s1",
            paper_ids=[],
            rag_strategy="hybrid",
            memory_extraction_enabled=True,
            request_memory_extraction_enabled=True,
            wait_for_pending_extractions=True,
        ),
        container=container,
    )

    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    assert "event: answer" in "".join(chunks)
    request_payload = container.research_service.stream_calls[0]["request_payload"]
    assert request_payload["memory_extraction_enabled"] is True
    assert request_payload["request_memory_extraction_enabled"] is True
    assert request_payload["wait_for_pending_extractions"] is True
    assert container.research_service.wait_calls == [{"timeout": 300.0}]


@pytest.mark.asyncio
async def test_extract_transcript_memory_endpoint_dispatches_and_waits():
    container = _RecordingContainer()
    response = await extract_transcript_memory(
        TranscriptMemoryExtractionRequest(
            user_id="tester",
            request_id="req1",
            session_id="locomo-replay",
            round_messages=[
                {
                    "message_id": "s1:1",
                    "message": {"type": "human", "data": {"content": "我偏好短答案"}},
                    "thread_id": "locomo-replay",
                    "round_index": 1,
                    "metadata": {"speaker": "user"},
                },
                {
                    "message_id": "s1:2",
                    "message": {"type": "ai", "data": {"content": "我会保持简短。"}},
                    "thread_id": "locomo-replay",
                    "round_index": 1,
                    "metadata": {"speaker": "assistant"},
                },
            ],
            wait_for_pending_extractions=True,
        ),
        container=container,
    )

    assert response["success"] is True
    assert response["data"] == {"request_id": "req1", "dispatched": True}
    assert container.research_service.transcript_calls == [
        {
            "user_id": "tester",
            "request_id": "req1",
            "session_id": "locomo-replay",
            "round_messages": [
                {
                    "message": {"type": "human", "data": {"content": "我偏好短答案"}},
                    "message_id": "s1:1",
                    "thread_id": "locomo-replay",
                    "round_index": 1,
                    "metadata": {"speaker": "user"},
                },
                {
                    "message": {"type": "ai", "data": {"content": "我会保持简短。"}},
                    "message_id": "s1:2",
                    "thread_id": "locomo-replay",
                    "round_index": 1,
                    "metadata": {"speaker": "assistant"},
                },
            ],
        }
    ]
    assert container.research_service.wait_calls == [{"timeout": 300.0}]


@pytest.mark.asyncio
async def test_ask_endpoint_rejects_empty_query():
    app = create_app()
    get_container()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await asyncio.wait_for(
            client.post(
                "/api/v1/research/ask",
                json={
                    "query": "",
                    "user_id": "tester",
                    "mode": "qa",
                    "paper_ids": [],
                    "rag_strategy": "hybrid",
                },
            ),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_health_endpoint():
    app = create_app()
    get_container()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await asyncio.wait_for(
            client.get("/api/v1/health"), timeout=REQUEST_TIMEOUT_SECONDS
        )

    assert response.status_code == 200
    assert response.json()["data"]["status"] == "healthy"


@pytest.mark.asyncio
async def test_ask_stream_get_endpoint_returns_sse():
    app = create_app()
    get_container()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await asyncio.wait_for(
            client.get(
                "/api/v1/research/ask/stream",
                params={
                    "query": "What does hybrid retrieval improve?",
                    "user_id": "tester",
                    "session_id": "stream-get",
                },
            ),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: plan" in response.text


@pytest.mark.asyncio
async def test_stream_get_endpoint_returns_sse_without_fixed_type():
    app = create_app()
    get_container()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await asyncio.wait_for(
            client.get(
                "/api/v1/research/stream",
                params={
                    "query": "帮我阅读 2604.20779 这篇论文",
                    "user_id": "tester",
                    "session_id": "stream-default",
                },
            ),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: plan" in response.text


@pytest.mark.asyncio
async def test_ingest_local_stream_endpoint_ingests_from_category_directory(tmp_path):
    app = create_app()
    source_dir = tmp_path / "raw" / "arxiv" / "source" / "cs.AI"
    metadata_dir = tmp_path / "raw" / "arxiv" / "metadata" / "cs.AI"
    source_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    (source_dir / "2405.00009.json").write_text(
        json.dumps(
            {
                "paper_id": "2405.00009",
                "title": "Structured Local Paper",
                "authors": ["Ada Lovelace"],
                "abstract": "Local paper for API ingest test.",
                "categories": ["cs.AI"],
                "publish_date": "2024-05-09",
                "sections": [
                    {
                        "section_id": "section-1",
                        "title": "Intro",
                        "content": "This is a locally ingested paper.",
                        "level": 1,
                        "formulas": [],
                        "has_algorithm": False,
                    }
                ],
                "references": [],
                "metadata": {"source_format": "latex"},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (metadata_dir / "2405.00009.json").write_text(
        json.dumps(
            {
                "paper_id": "2405.00009",
                "title": "Metadata Title",
                "abstract": "Metadata abstract",
                "authors": ["Ada Lovelace"],
                "categories": ["cs.AI"],
                "created": "2024-05-09",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    get_container()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await asyncio.wait_for(
            client.post(
                "/api/v1/ingest/local/stream",
                json={"categories": ["cs.AI"]},
            ),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: paper_ingested" in response.text
    assert "2405.00009" in response.text
