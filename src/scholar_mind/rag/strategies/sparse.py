from __future__ import annotations

from typing import TYPE_CHECKING

from scholar_mind.models.domain import RetrievalStrategyName, RetrievedChunk

if TYPE_CHECKING:
    from scholar_mind.rag.engine import RAGEngine


class SparseRetrieval:
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
        indices, values = self.engine.sparse_query_vector(query)
        points = self.engine.index.search_chunks_sparse(
            indices,
            values,
            limit=top_k,
            filters=filters,
        )
        scored: list[RetrievedChunk] = []
        for point in points:
            payload = point.payload or {}
            if not self.engine.payload_matches(payload, filters):
                continue
            score = float(point.score)
            scored.append(
                RetrievedChunk(
                    chunk_id=payload["chunk_id"],
                    paper_id=payload["paper_id"],
                    title=payload["title"],
                    section=payload["section"],
                    content=payload["content"],
                    score=score,
                    strategy=RetrievalStrategyName.SPARSE,
                    categories=list(payload.get("categories", [])),
                    publish_date=self.engine.parse_date(payload.get("publish_date")),
                )
            )
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]
