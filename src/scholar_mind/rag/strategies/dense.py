from __future__ import annotations

from typing import TYPE_CHECKING

from scholar_mind.models.domain import RetrievalStrategyName, RetrievedChunk

if TYPE_CHECKING:
    from scholar_mind.rag.engine import RAGEngine


class DenseRetrieval:
    def __init__(self, engine: RAGEngine):
        self.engine = engine

    async def retrieve(
        self, query: str, top_k: int = 10, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        return self.retrieve_sync(query, top_k=top_k, filters=filters)

    def retrieve_sync(
        self, query: str, top_k: int = 10, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        filters = filters or {}
        vector = self.engine.embedder.embed_query(query)
        scored = self.engine.index.search_chunks_dense(
            vector,
            limit=top_k,
            filters=filters,
        )
        results: list[RetrievedChunk] = []
        for point in scored:
            payload = point.payload or {}
            if not self.engine.payload_matches(payload, filters):
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=payload["chunk_id"],
                    paper_id=payload["paper_id"],
                    title=payload["title"],
                    section=payload["section"],
                    content=payload["content"],
                    score=float(point.score),
                    strategy=RetrievalStrategyName.DENSE,
                    categories=list(payload.get("categories", [])),
                    publish_date=self.engine.parse_date(payload.get("publish_date")),
                )
            )
            if len(results) >= top_k:
                break
        return results
