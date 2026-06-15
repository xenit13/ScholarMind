from __future__ import annotations

from typing import TYPE_CHECKING

from scholar_mind.models.domain import RetrievalStrategyName, RetrievedChunk
from scholar_mind.rag.top_k import hybrid_candidate_limit

if TYPE_CHECKING:
    from scholar_mind.rag.engine import RAGEngine


class RerankedHybridRetrieval:
    def __init__(self, engine: RAGEngine, rerank_top_n: int = 20):
        self.engine = engine
        self.rerank_top_n = rerank_top_n

    async def retrieve(
        self, query: str, top_k: int = 10, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        return self.retrieve_sync(query, top_k=top_k, filters=filters)

    def retrieve_sync(
        self, query: str, top_k: int = 10, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        candidate_top_k = hybrid_candidate_limit(top_k)
        if hasattr(self.engine.hybrid, "retrieve_candidates_sync"):
            candidates = self.engine.hybrid.retrieve_candidates_sync(
                query, candidate_top_k=candidate_top_k, filters=filters
            )
        else:
            candidates = self.engine.hybrid.retrieve_sync(
                query, top_k=candidate_top_k, filters=filters
            )
        reranked = self.engine.reranker_service.rerank(query, candidates, top_k=top_k)
        return [
            item.model_copy(update={"strategy": RetrievalStrategyName.RERANKED_HYBRID})
            for item in reranked[:top_k]
        ]
