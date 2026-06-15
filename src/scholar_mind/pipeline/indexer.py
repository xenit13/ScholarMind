from __future__ import annotations

from scholar_mind.rag.engine import RAGEngine


class EmbeddingIndexer:
    def __init__(self, rag_engine: RAGEngine):
        self.rag_engine = rag_engine

    def build(self) -> None:
        self.rag_engine.ensure_index()
