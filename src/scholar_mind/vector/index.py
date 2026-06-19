from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from scholar_mind.config.settings import Settings
from scholar_mind.models.domain import MemoryRecord


class QdrantIndex:
    MEMORY_COLLECTION = "user_memory_dense"

    def __init__(self, settings: Settings, dimension: int = 1536):
        self.settings = settings
        self.dimension = dimension
        self.client = self._build_client(settings)
        self._ensure_collection(self.MEMORY_COLLECTION)

    def _build_client(self, settings: Settings) -> QdrantClient:
        if settings.qdrant_url:
            return QdrantClient(url=settings.qdrant_url)
        if settings.qdrant_location == ":memory:":
            return QdrantClient(":memory:")
        path = settings.resolve_path(settings.qdrant_location)
        path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(path))

    def _ensure_collection(self, name: str) -> None:
        if self.client.collection_exists(name):
            if self._collection_matches_dimension(name):
                return
            self.client.delete_collection(name)
        vectors_config = models.VectorParams(size=self.dimension, distance=models.Distance.COSINE)
        self.client.create_collection(collection_name=name, vectors_config=vectors_config)

    def _collection_matches_dimension(self, name: str) -> bool:
        config = self.client.get_collection(name).config.params.vectors
        if hasattr(config, "size"):
            return config.size == self.dimension
        return False

    def upsert_memory(self, record: MemoryRecord, embedding: list[float]) -> None:
        record_id = getattr(record, "record_id", getattr(record, "memory_id", ""))
        payload = {
            "record_id": record_id,
            "memory_id": record_id,
            "user_id": record.user_id,
            "source": record.source,
            "created_at": record.created_at.isoformat(),
            "content": record.content,
            "status": _payload_value(getattr(record, "status", "active")),
            "memory_type": _payload_value(
                getattr(record, "memory_type", "interaction_summary")
            ),
            "importance": float(getattr(record, "importance", 0.6)),
            "confidence": float(getattr(record, "confidence", 0.7)),
            "updated_at": getattr(record, "updated_at", record.created_at).isoformat(),
            "last_accessed_at": (
                getattr(record, "last_accessed_at", None).isoformat()
                if getattr(record, "last_accessed_at", None)
                else None
            ),
            "decay_rate": float(getattr(record, "decay_rate", 0.03)),
            "decay_floor": float(getattr(record, "decay_floor", 0.3)),
            "keywords": list(getattr(record, "keywords", [])),
        }
        self.client.upsert(
            self.MEMORY_COLLECTION,
            points=[
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, f"memory:{record_id}")),
                    vector=embedding,
                    payload=payload,
                )
            ],
        )

    def search_memory(
        self, user_id: str, vector: list[float], limit: int = 3
    ) -> list[models.ScoredPoint]:
        flt = models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id),
                ),
                models.FieldCondition(
                    key="status",
                    match=models.MatchValue(value="active"),
                ),
            ]
        )
        response = self.client.query_points(
            collection_name=self.MEMORY_COLLECTION,
            query=vector,
            limit=limit,
            query_filter=flt,
            with_payload=True,
        )
        return response.points


def _payload_value(value):
    return value.value if hasattr(value, "value") else value
