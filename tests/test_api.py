from __future__ import annotations

import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient

from scholar_mind.app import get_container
from scholar_mind.asgi import create_app
from scholar_mind.models.domain import AskRequest, ChatRequest

REQUEST_TIMEOUT_SECONDS = 15


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
