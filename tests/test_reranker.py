from __future__ import annotations

import httpx

from scholar_mind.models.domain import RetrievalStrategyName, RetrievedChunk
from scholar_mind.rag.reranker import RemoteReranker


def _candidate(chunk_id: str, content: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        paper_id="paper-1",
        title="Test Paper",
        section="method",
        content=content,
        score=score,
        strategy=RetrievalStrategyName.HYBRID,
    )


def test_remote_reranker_uses_http_scores(monkeypatch):
    def fake_post(url, json, headers, timeout):
        assert url == "http://rerank.local/rerank"
        assert json["model"] == "bge-reranker-v2-m3"
        assert json["pairs"] == [
            {"query": "hybrid retrieval", "passage": "dense retrieval baseline"},
            {"query": "hybrid retrieval", "passage": "hybrid retrieval improves recall"},
        ]
        assert headers["Authorization"] == "Bearer secret"
        assert timeout == 6.0
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"query": "hybrid retrieval", "passage": "dense retrieval baseline", "score": 0.42},
                    {
                        "query": "hybrid retrieval",
                        "passage": "hybrid retrieval improves recall",
                        "score": 0.98,
                    },
                ]
            },
            request=request,
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    reranker = RemoteReranker(
        model_name="bge-reranker-v2-m3",
        base_url="http://rerank.local",
        api_key="secret",
        timeout_seconds=6.0,
    )

    ranked = reranker.rerank(
        "hybrid retrieval",
        [
            _candidate("c1", "dense retrieval baseline", 0.1),
            _candidate("c2", "hybrid retrieval improves recall", 0.2),
        ],
        top_k=2,
    )

    assert [item.chunk_id for item in ranked] == ["c2", "c1"]
    assert ranked[0].score == 0.98


def test_remote_reranker_falls_back_to_lexical_on_http_error(monkeypatch):
    def fake_post(url, json, headers, timeout):
        raise httpx.ConnectError("unreachable")

    monkeypatch.setattr(httpx, "post", fake_post)
    reranker = RemoteReranker(
        model_name="bge-reranker-v2-m3",
        base_url="http://rerank.local",
    )

    ranked = reranker.rerank(
        "hybrid retrieval",
        [
            _candidate("c1", "hybrid retrieval improves recall", 0.1),
            _candidate("c2", "completely unrelated content", 0.9),
        ],
        top_k=1,
    )

    assert [item.chunk_id for item in ranked] == ["c1"]
