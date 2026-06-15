from __future__ import annotations

from typing import TYPE_CHECKING

from scholar_mind.models.domain import RetrievalStrategyName, RetrievedChunk
from scholar_mind.rag.top_k import hybrid_candidate_limit

if TYPE_CHECKING:
    from scholar_mind.rag.engine import RAGEngine


class HybridRetrieval:
    def __init__(self, engine: RAGEngine, dense_weight: float = 0.7, sparse_weight: float = 0.3):
        self.engine = engine
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    async def retrieve(
        self, query: str, top_k: int = 10, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        return self.retrieve_sync(query, top_k=top_k, filters=filters)

    def retrieve_sync(
        self, query: str, top_k: int = 10, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        return self._retrieve_sync(
            query,
            top_k=top_k,
            candidate_limit=hybrid_candidate_limit(top_k),
            filters=filters,
        )

    def retrieve_candidates_sync(
        self, query: str, candidate_top_k: int, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        return self._retrieve_sync(
            query,
            top_k=candidate_top_k,
            candidate_limit=candidate_top_k,
            filters=filters,
        )

    def _retrieve_sync(
        self,
        query: str,
        top_k: int,
        candidate_limit: int,
        filters: dict | None,
    ) -> list[RetrievedChunk]:
        dense_vector = self.engine.embedder.embed_query(query)
        sparse_indices, sparse_values = self.engine.sparse_query_vector(query)
        dense_points = self.engine.index.search_chunks_dense(
            dense_vector,
            limit=candidate_limit,
            filters=filters,
        )
        sparse_points = self.engine.index.search_chunks_sparse(
            sparse_indices,
            sparse_values,
            limit=candidate_limit,
            filters=filters,
        )
        scores: dict[str, float] = {}
        payloads: dict[str, dict] = {}
        for rank, point in enumerate(dense_points, start=1):
            payload = point.payload or {}
            point_id = str(point.id)
            payloads[point_id] = payload
            scores[point_id] = scores.get(point_id, 0.0) + (self.dense_weight / (60 + rank))
        for rank, point in enumerate(sparse_points, start=1):
            payload = point.payload or {}
            point_id = str(point.id)
            payloads[point_id] = payload
            scores[point_id] = scores.get(point_id, 0.0) + (self.sparse_weight / (60 + rank))

        ranked: list[RetrievedChunk] = []
        for point_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            payload = payloads[point_id]
            if not self.engine.payload_matches(payload, filters or {}):
                continue
            ranked.append(
                RetrievedChunk(
                    chunk_id=payload["chunk_id"],
                    paper_id=payload["paper_id"],
                    title=payload["title"],
                    section=payload["section"],
                    content=payload["content"],
                    score=round(float(score), 6),
                    strategy=RetrievalStrategyName.HYBRID,
                    categories=list(payload.get("categories", [])),
                    publish_date=self.engine.parse_date(payload.get("publish_date")),
                )
            )
            if len(ranked) >= top_k:
                break
        return ranked
