from __future__ import annotations

from datetime import date
from time import perf_counter
from typing import Any

from scholar_mind.models.domain import PaperChunk, RetrievalStrategyName, RetrievedChunk, StructuredPaper
from scholar_mind.rag.embeddings import EmbeddingService
from scholar_mind.rag.index import QdrantIndex
from scholar_mind.rag.query_transform import QueryTransformer
from scholar_mind.rag.reranker import LexicalReranker
from scholar_mind.rag.strategies import (
    DenseRetrieval,
    HybridRetrieval,
    RerankedHybridRetrieval,
    SparseRetrieval,
)
from scholar_mind.utils.text import SparseCorpusStats, encode_sparse_text

EMBEDDING_BATCH_LIMIT = 64


class RAGEngine:
    def __init__(self, paper_repository, index: QdrantIndex, embedder: EmbeddingService):
        self.paper_repository = paper_repository
        self.index = index
        self.embedder = embedder
        self.reranker_service = LexicalReranker()
        self.query_transformer = QueryTransformer()
        self.sparse_stats: SparseCorpusStats | None = None
        self.dense = DenseRetrieval(self)
        self.sparse = SparseRetrieval(self)
        self.hybrid = HybridRetrieval(self)
        self.reranked = RerankedHybridRetrieval(self)

    def ensure_sparse_stats(self) -> SparseCorpusStats:
        if self.sparse_stats is None:
            self.sparse_stats = self.paper_repository.build_sparse_stats()
        return self.sparse_stats

    def ensure_index(self) -> None:
        if not self.index.is_paper_collection_empty():
            return
        self.ensure_sparse_stats()
        batch = []
        batch_paper_ids: set[str] = set()
        for chunk in self.paper_repository.iter_chunk_models():
            batch.append(chunk)
            batch_paper_ids.add(chunk.paper_id)
            if len(batch) >= EMBEDDING_BATCH_LIMIT:
                self._upsert_chunk_batch(batch, batch_paper_ids)
                batch = []
                batch_paper_ids = set()
        if not batch:
            return
        self._upsert_chunk_batch(batch, batch_paper_ids)

    async def retrieve(
        self,
        query: str,
        strategy: RetrievalStrategyName = RetrievalStrategyName.HYBRID,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> tuple[list[RetrievedChunk], int]:
        return self.retrieve_sync(query=query, strategy=strategy, top_k=top_k, filters=filters)

    def retrieve_sync(
        self,
        query: str,
        strategy: RetrievalStrategyName = RetrievalStrategyName.HYBRID,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> tuple[list[RetrievedChunk], int]:
        started = perf_counter()
        query = self.query_transformer.normalize(query)
        if strategy == RetrievalStrategyName.DENSE:
            chunks = self.dense.retrieve_sync(query, top_k=top_k, filters=filters)
        elif strategy == RetrievalStrategyName.SPARSE:
            chunks = self.sparse.retrieve_sync(query, top_k=top_k, filters=filters)
        elif strategy == RetrievalStrategyName.RERANKED_HYBRID:
            chunks = self.reranked.retrieve_sync(query, top_k=top_k, filters=filters)
        else:
            chunks = self.hybrid.retrieve_sync(query, top_k=top_k, filters=filters)
        latency = int((perf_counter() - started) * 1000)
        return chunks, latency

    def payload_matches(self, payload: dict[str, Any], filters: dict[str, Any]) -> bool:
        paper_ids = set(filters.get("paper_ids", []) or [])
        categories = set(filters.get("categories", []) or [])
        start = filters.get("date_from")
        end = filters.get("date_to")
        if paper_ids and payload.get("paper_id") not in paper_ids:
            return False
        if categories and not categories.intersection(set(payload.get("categories", []))):
            return False
        return self.index.date_in_range(payload.get("publish_date"), start, end)

    @staticmethod
    def parse_date(value: str | None) -> date | None:
        if not value:
            return None
        return date.fromisoformat(value)

    def sparse_query_vector(self, query: str) -> tuple[list[int], list[float]]:
        return encode_sparse_text(query, self.ensure_sparse_stats(), is_query=True)

    def upsert_paper(
        self, paper: StructuredPaper, chunks: list[PaperChunk] | None = None
    ) -> None:
        chunk_batch = chunks or []
        if not chunk_batch:
            return
        self.index.delete_paper_chunks(paper.paper_id)
        self.sparse_stats = self.paper_repository.build_sparse_stats()
        embeddings = self._embed_chunk_contents([chunk.content for chunk in chunk_batch])
        sparse_vectors = [encode_sparse_text(chunk.content, self.sparse_stats) for chunk in chunk_batch]
        self.index.upsert_chunks(
            chunk_batch,
            embeddings,
            sparse_vectors,
            {
                paper.paper_id: {
                    "title": paper.title,
                    "categories": list(paper.categories),
                    "publish_date": paper.publish_date.isoformat(),
                }
            },
        )

    def _upsert_chunk_batch(self, chunks: list, paper_ids: set[str]) -> None:
        if not chunks:
            return
        embeddings = self._embed_chunk_contents([chunk.content for chunk in chunks])
        sparse_vectors = [encode_sparse_text(chunk.content, self.sparse_stats) for chunk in chunks]
        payloads = self.paper_repository.paper_payloads(sorted(paper_ids))
        self.index.upsert_chunks(chunks, embeddings, sparse_vectors, payloads)

    def _embed_chunk_contents(self, contents: list[str]) -> list[list[float]]:
        try:
            return self.embedder.embed_documents(contents)
        except Exception:
            return [self._embed_single_content(content) for content in contents]

    def _embed_single_content(self, content: str) -> list[float]:
        try:
            return self.embedder.embed_query(content)
        except Exception:
            return self.embedder.embed_query(self._fallback_embedding_text(content))

    @staticmethod
    def _fallback_embedding_text(content: str) -> str:
        collapsed = " ".join(content.split())
        if len(collapsed) <= 4000:
            return collapsed
        return collapsed[:4000]
