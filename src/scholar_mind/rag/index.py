from __future__ import annotations

from datetime import date
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from scholar_mind.config.settings import Settings
from scholar_mind.models.domain import MemoryRecord, PaperChunk


class QdrantIndex:
    PAPER_COLLECTION = "paper_chunks"
    MEMORY_COLLECTION = "user_memory_dense"
    DENSE_VECTOR = "dense"
    SPARSE_VECTOR = "sparse"

    def __init__(self, settings: Settings, dimension: int = 1536):
        self.settings = settings
        self.dimension = dimension
        self.client = self._build_client(settings)
        self._ensure_collection(self.PAPER_COLLECTION)
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
        if name == self.PAPER_COLLECTION:
            self.client.create_collection(
                collection_name=name,
                vectors_config={self.DENSE_VECTOR: vectors_config},
                sparse_vectors_config={
                    self.SPARSE_VECTOR: models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )
            return
        self.client.create_collection(collection_name=name, vectors_config=vectors_config)

    def _collection_matches_dimension(self, name: str) -> bool:
        config = self.client.get_collection(name).config.params.vectors
        if name == self.PAPER_COLLECTION:
            dense_config = config.get(self.DENSE_VECTOR) if isinstance(config, dict) else None
            return bool(dense_config and dense_config.size == self.dimension)
        if hasattr(config, "size"):
            return config.size == self.dimension
        return False

    def is_paper_collection_empty(self) -> bool:
        return self.client.count(self.PAPER_COLLECTION, exact=True).count == 0

    def upsert_chunks(
        self,
        chunks: list[PaperChunk],
        embeddings: list[list[float]],
        sparse_vectors: list[tuple[list[int], list[float]]],
        paper_payloads: dict[str, dict],
    ) -> None:
        points = []
        for chunk, embedding, sparse_vector in zip(chunks, embeddings, sparse_vectors, strict=True):
            paper_payload = paper_payloads[chunk.paper_id]
            points.append(
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, f"chunk:{chunk.chunk_id}")),
                    vector={
                        self.DENSE_VECTOR: embedding,
                        self.SPARSE_VECTOR: models.SparseVector(
                            indices=sparse_vector[0],
                            values=sparse_vector[1],
                        ),
                    },
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "paper_id": chunk.paper_id,
                        "title": paper_payload["title"],
                        "section": chunk.section,
                        "content": chunk.content,
                        "categories": paper_payload["categories"],
                        "publish_date": paper_payload["publish_date"],
                    },
                )
            )
        if points:
            self.client.upsert(self.PAPER_COLLECTION, points=points)

    def delete_paper_chunks(self, paper_id: str) -> None:
        self.client.delete(
            collection_name=self.PAPER_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="paper_id",
                            match=models.MatchValue(value=paper_id),
                        )
                    ]
                )
            ),
        )

    def search_chunks_dense(
        self,
        vector: list[float],
        limit: int = 10,
        filters: dict | None = None,
    ) -> list[models.ScoredPoint]:
        response = self.client.query_points(
            collection_name=self.PAPER_COLLECTION,
            query=vector,
            using=self.DENSE_VECTOR,
            limit=limit,
            query_filter=self._build_filter(filters),
            with_payload=True,
        )
        return response.points

    def search_chunks_sparse(
        self,
        indices: list[int],
        values: list[float],
        limit: int = 10,
        filters: dict | None = None,
    ) -> list[models.ScoredPoint]:
        response = self.client.query_points(
            collection_name=self.PAPER_COLLECTION,
            query=models.SparseVector(indices=indices, values=values),
            using=self.SPARSE_VECTOR,
            limit=limit,
            query_filter=self._build_filter(filters),
            with_payload=True,
        )
        return response.points

    def search_chunks_hybrid(
        self,
        dense_vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        limit: int = 10,
        filters: dict | None = None,
    ) -> list[models.ScoredPoint]:
        response = self.client.query_points(
            collection_name=self.PAPER_COLLECTION,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using=self.DENSE_VECTOR,
                    limit=max(limit * 2, 20),
                    filter=self._build_filter(filters),
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                    using=self.SPARSE_VECTOR,
                    limit=max(limit * 2, 20),
                    filter=self._build_filter(filters),
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return response.points

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

    @staticmethod
    def date_in_range(value: str | None, start: date | None, end: date | None) -> bool:
        if not value:
            return True
        published = date.fromisoformat(value)
        if start and published < start:
            return False
        if end and published > end:
            return False
        return True

    def _build_filter(self, filters: dict | None) -> models.Filter | None:
        if not filters:
            return None
        conditions: list[models.FieldCondition] = []
        paper_ids = list(filters.get("paper_ids") or [])
        if paper_ids:
            conditions.append(
                models.FieldCondition(
                    key="paper_id",
                    match=models.MatchAny(any=paper_ids),
                )
            )
        categories = list(filters.get("categories") or [])
        if categories:
            conditions.append(
                models.FieldCondition(
                    key="categories",
                    match=models.MatchAny(any=categories),
                )
            )
        date_from = filters.get("date_from")
        date_to = filters.get("date_to")
        if date_from or date_to:
            conditions.append(
                models.FieldCondition(
                    key="publish_date",
                    range=models.DatetimeRange(
                        gte=date_from.isoformat() if date_from else None,
                        lte=date_to.isoformat() if date_to else None,
                    ),
                )
            )
        return models.Filter(must=conditions) if conditions else None


def _payload_value(value):
    return value.value if hasattr(value, "value") else value
