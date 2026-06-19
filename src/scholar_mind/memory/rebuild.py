from __future__ import annotations

from scholar_mind.memory.repository import MemoryRepository
from scholar_mind.vector.embeddings import EmbeddingService
from scholar_mind.vector.index import QdrantIndex


def rebuild_memory_index(
    repository: MemoryRepository,
    index: QdrantIndex,
    embedder: EmbeddingService,
    *,
    user_id: str | None = None,
) -> int:
    rebuilt = 0
    for record in repository.list_active_records(user_id=user_id):
        embedding = embedder.embed_query(record.content)
        index.upsert_memory(record, embedding)
        rebuilt += 1
    return rebuilt
