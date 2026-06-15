from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import pytest

import scholar_mind.app as app_module
from scholar_mind.app import get_container
from scholar_mind.db.models import PaperModel
from scholar_mind.models.domain import PaperChunk, PaperSection, StructuredPaper
from scholar_mind.rag.engine import RAGEngine
from scholar_mind.rag.strategies.dense import DenseRetrieval
from scholar_mind.rag.strategies.hybrid import HybridRetrieval
from scholar_mind.rag.strategies.reranked import RerankedHybridRetrieval
from scholar_mind.rag.strategies.sparse import SparseRetrieval


def test_search_papers_relevance_paginates_across_pages():
    container = get_container()

    first_page, total = container.paper_repository.search_papers(
        "",
        sort_by="relevance",
        page=1,
        page_size=2,
    )
    second_page, second_total = container.paper_repository.search_papers(
        "",
        sort_by="relevance",
        page=2,
        page_size=2,
    )

    assert total == second_total
    assert total >= 4
    assert first_page
    assert second_page
    assert {item["paper_id"] for item in first_page}.isdisjoint(
        {item["paper_id"] for item in second_page}
    )


def test_search_papers_category_filter_matches_exact_json_value():
    container = get_container()
    with container.paper_repository.session_factory() as session:
        session.add_all(
            [
                PaperModel(
                    paper_id="exact-category",
                    title="Exact Category Match",
                    authors_json=json.dumps(["Test Author"]),
                    abstract="Study on exact category filtering for retrieval systems.",
                    categories_json=json.dumps(["cs.CL"]),
                    publish_date=date(2025, 1, 1),
                    citation_count=1,
                    has_source=False,
                ),
                PaperModel(
                    paper_id="prefix-category",
                    title="Prefix Category Match",
                    authors_json=json.dumps(["Test Author"]),
                    abstract="Study on prefix-like category values for retrieval systems.",
                    categories_json=json.dumps(["cs.CLF"]),
                    publish_date=date(2025, 1, 2),
                    citation_count=1,
                    has_source=False,
                ),
            ]
        )
        session.commit()

    papers, _ = container.paper_repository.search_papers(
        "",
        categories=["cs.CL"],
        sort_by="date",
        page=1,
        page_size=20,
    )
    returned_ids = {item["paper_id"] for item in papers}

    assert "exact-category" in returned_ids
    assert "prefix-category" not in returned_ids


def test_hybrid_retrieval_applies_configured_dense_and_sparse_weights():
    payload_a = {
        "chunk_id": "a",
        "paper_id": "p-a",
        "title": "A",
        "section": "Intro",
        "content": "dense first",
        "categories": ["cs.CL"],
        "publish_date": "2025-01-01",
    }
    payload_b = {
        "chunk_id": "b",
        "paper_id": "p-b",
        "title": "B",
        "section": "Intro",
        "content": "sparse first",
        "categories": ["cs.CL"],
        "publish_date": "2025-01-02",
    }

    class DummyIndex:
        def search_chunks_dense(self, *_args, **_kwargs):
            return [
                SimpleNamespace(id="a", payload=payload_a),
                SimpleNamespace(id="b", payload=payload_b),
            ]

        def search_chunks_sparse(self, *_args, **_kwargs):
            return [
                SimpleNamespace(id="b", payload=payload_b),
                SimpleNamespace(id="a", payload=payload_a),
            ]

    class DummyEmbedder:
        def embed_query(self, _query):
            return [0.1, 0.2]

    class DummyEngine:
        def __init__(self):
            self.index = DummyIndex()
            self.embedder = DummyEmbedder()

        def sparse_query_vector(self, _query):
            return [1], [0.5]

        def payload_matches(self, _payload, _filters):
            return True

        def parse_date(self, value):
            return date.fromisoformat(value)

    dense_favoring = HybridRetrieval(DummyEngine(), dense_weight=0.9, sparse_weight=0.1)
    sparse_favoring = HybridRetrieval(DummyEngine(), dense_weight=0.1, sparse_weight=0.9)

    assert dense_favoring.retrieve_sync("retrieval", top_k=2)[0].chunk_id == "a"
    assert sparse_favoring.retrieve_sync("retrieval", top_k=2)[0].chunk_id == "b"


def test_non_hybrid_retrieval_uses_final_top_k_as_search_limit():
    calls = {}
    payload = {
        "chunk_id": "a",
        "paper_id": "p-a",
        "title": "A",
        "section": "Intro",
        "content": "retrieval content",
        "categories": ["cs.CL"],
        "publish_date": "2025-01-01",
    }

    class DummyIndex:
        def search_chunks_dense(self, _vector, *, limit, filters):
            calls["dense_limit"] = limit
            return [SimpleNamespace(id="a", payload=payload, score=0.7)]

        def search_chunks_sparse(self, _indices, _values, *, limit, filters):
            calls["sparse_limit"] = limit
            return [SimpleNamespace(id="a", payload=payload, score=0.6)]

    class DummyEmbedder:
        def embed_query(self, _query):
            return [0.1, 0.2]

    class DummyEngine:
        def __init__(self):
            self.index = DummyIndex()
            self.embedder = DummyEmbedder()

        def sparse_query_vector(self, _query):
            return [1], [0.5]

        def payload_matches(self, _payload, _filters):
            return True

        def parse_date(self, value):
            return date.fromisoformat(value)

    engine = DummyEngine()

    assert len(DenseRetrieval(engine).retrieve_sync("retrieval", top_k=4)) == 1
    assert len(SparseRetrieval(engine).retrieve_sync("retrieval", top_k=4)) == 1
    assert calls["dense_limit"] == 4
    assert calls["sparse_limit"] == 4


def test_hybrid_retrieval_uses_four_times_final_top_k_for_candidates():
    calls = {}
    payload = {
        "chunk_id": "a",
        "paper_id": "p-a",
        "title": "A",
        "section": "Intro",
        "content": "retrieval content",
        "categories": ["cs.CL"],
        "publish_date": "2025-01-01",
    }

    class DummyIndex:
        def search_chunks_dense(self, _vector, *, limit, filters):
            calls["dense_limit"] = limit
            return [SimpleNamespace(id="a", payload=payload)]

        def search_chunks_sparse(self, _indices, _values, *, limit, filters):
            calls["sparse_limit"] = limit
            return [SimpleNamespace(id="a", payload=payload)]

    class DummyEmbedder:
        def embed_query(self, _query):
            return [0.1, 0.2]

    class DummyEngine:
        def __init__(self):
            self.index = DummyIndex()
            self.embedder = DummyEmbedder()

        def sparse_query_vector(self, _query):
            return [1], [0.5]

        def payload_matches(self, _payload, _filters):
            return True

        def parse_date(self, value):
            return date.fromisoformat(value)

    assert len(HybridRetrieval(DummyEngine()).retrieve_sync("retrieval", top_k=4)) == 1
    assert calls["dense_limit"] == 16
    assert calls["sparse_limit"] == 16


def test_reranked_hybrid_uses_four_times_final_top_k_for_candidates():
    calls = {}

    class DummyHybrid:
        def retrieve_sync(self, _query, *, top_k, filters):
            calls["candidate_top_k"] = top_k
            return [SimpleNamespace(model_copy=lambda update: update)]

    class DummyReranker:
        def rerank(self, _query, candidates, *, top_k):
            calls["rerank_top_k"] = top_k
            return candidates

    class DummyEngine:
        def __init__(self):
            self.hybrid = DummyHybrid()
            self.reranker_service = DummyReranker()

    RerankedHybridRetrieval(DummyEngine()).retrieve_sync("retrieval", top_k=3)

    assert calls["candidate_top_k"] == 12
    assert calls["rerank_top_k"] == 3


def test_rag_engine_upsert_paper_replaces_existing_chunks_for_single_paper():
    calls = {}

    class DummyRepository:
        def build_sparse_stats(self):
            calls["built_sparse_stats"] = True
            return SimpleNamespace(
                document_count=2,
                average_length=10.0,
                document_frequencies={"retrieval": 1, "planning": 1},
            )

    class DummyIndex:
        def delete_paper_chunks(self, paper_id):
            calls["deleted_paper_id"] = paper_id

        def upsert_chunks(self, chunks, embeddings, sparse_vectors, paper_payloads):
            calls["chunks"] = chunks
            calls["embeddings"] = embeddings
            calls["sparse_vectors"] = sparse_vectors
            calls["paper_payloads"] = paper_payloads

    class DummyEmbedder:
        def embed_documents(self, docs):
            calls["docs"] = docs
            return [[0.1, 0.2] for _ in docs]

    paper = StructuredPaper(
        paper_id="p-upsert",
        title="Planning with Retrieval",
        authors=["Ada Lovelace"],
        abstract="A paper about planning with retrieval.",
        categories=["cs.AI"],
        publish_date=date(2025, 1, 1),
        sections=[
            PaperSection(
                section_id="section-1",
                title="Method",
                content="We plan, retrieve, and rerank.",
            )
        ],
    )
    chunks = [
        PaperChunk(
            chunk_id="p-upsert::metadata",
            paper_id="p-upsert",
            section="metadata",
            content="[Paper: Planning with Retrieval]",
            token_count=4,
        ),
        PaperChunk(
            chunk_id="p-upsert::method::1",
            paper_id="p-upsert",
            section="Method",
            content="[Section: Method] We plan, retrieve, and rerank.",
            token_count=8,
        ),
    ]

    engine = RAGEngine(DummyRepository(), DummyIndex(), DummyEmbedder())

    engine.upsert_paper(paper, chunks=chunks)

    assert calls["built_sparse_stats"] is True
    assert calls["deleted_paper_id"] == "p-upsert"
    assert calls["docs"] == [chunk.content for chunk in chunks]
    assert calls["chunks"] == chunks
    assert len(calls["embeddings"]) == 2
    assert len(calls["sparse_vectors"]) == 2
    assert calls["paper_payloads"] == {
        "p-upsert": {
            "title": "Planning with Retrieval",
            "categories": ["cs.AI"],
            "publish_date": "2025-01-01",
        }
    }


def test_rag_engine_upsert_paper_falls_back_to_single_embeddings_on_batch_failure():
    calls = {"embed_documents": 0, "embed_query": []}

    class DummyRepository:
        def build_sparse_stats(self):
            return SimpleNamespace(
                document_count=2,
                average_length=10.0,
                document_frequencies={"retrieval": 1, "planning": 1},
            )

    class DummyIndex:
        def delete_paper_chunks(self, paper_id):
            calls["deleted_paper_id"] = paper_id

        def upsert_chunks(self, chunks, embeddings, sparse_vectors, paper_payloads):
            calls["chunks"] = chunks
            calls["embeddings"] = embeddings
            calls["sparse_vectors"] = sparse_vectors
            calls["paper_payloads"] = paper_payloads

    class DummyEmbedder:
        def embed_documents(self, docs):
            calls["embed_documents"] += 1
            raise ValueError("batch rejected")

        def embed_query(self, text):
            calls["embed_query"].append(text)
            return [float(len(text))]

    paper = StructuredPaper(
        paper_id="p-fallback",
        title="Planning with Retrieval",
        authors=["Ada Lovelace"],
        abstract="A paper about planning with retrieval.",
        categories=["cs.AI"],
        publish_date=date(2025, 1, 1),
        sections=[
            PaperSection(
                section_id="section-1",
                title="Method",
                content="We plan, retrieve, and rerank.",
            )
        ],
    )
    chunks = [
        PaperChunk(
            chunk_id="p-fallback::metadata",
            paper_id="p-fallback",
            section="metadata",
            content="[Paper: Planning with Retrieval]",
            token_count=4,
        ),
        PaperChunk(
            chunk_id="p-fallback::method::1",
            paper_id="p-fallback",
            section="Method",
            content="[Section: Method] We plan, retrieve, and rerank.",
            token_count=8,
        ),
    ]

    engine = RAGEngine(DummyRepository(), DummyIndex(), DummyEmbedder())

    engine.upsert_paper(paper, chunks=chunks)

    assert calls["embed_documents"] == 1
    assert calls["embed_query"] == [chunk.content for chunk in chunks]
    assert calls["chunks"] == chunks
    assert len(calls["embeddings"]) == 2
    assert calls["embeddings"] == [[float(len(chunk.content))] for chunk in chunks]


def test_rag_engine_upsert_paper_raises_when_single_embedding_rejects_content():
    calls = {"embed_documents": 0, "embed_query": []}

    class DummyRepository:
        def build_sparse_stats(self):
            return SimpleNamespace(
                document_count=1,
                average_length=10.0,
                document_frequencies={"retrieval": 1},
            )

    class DummyIndex:
        def delete_paper_chunks(self, paper_id):
            calls["deleted_paper_id"] = paper_id

        def upsert_chunks(self, chunks, embeddings, sparse_vectors, paper_payloads):
            calls["embeddings"] = embeddings

    class DummyEmbedder:
        dimension = 3

        def embed_documents(self, docs):
            calls["embed_documents"] += 1
            raise ValueError("batch rejected")

        def embed_query(self, text):
            calls["embed_query"].append(text)
            raise ValueError("single rejected")

    paper = StructuredPaper(
        paper_id="p-zero",
        title="Planning with Retrieval",
        authors=["Ada Lovelace"],
        abstract="A paper about planning with retrieval.",
        categories=["cs.AI"],
        publish_date=date(2025, 1, 1),
        sections=[
            PaperSection(
                section_id="section-1",
                title="Method",
                content="We plan, retrieve, and rerank.",
            )
        ],
    )
    chunks = [
        PaperChunk(
            chunk_id="p-zero::method::1",
            paper_id="p-zero",
            section="Method",
            content="[Section: Method] We plan, retrieve, and rerank.",
            token_count=8,
        )
    ]

    engine = RAGEngine(DummyRepository(), DummyIndex(), DummyEmbedder())

    with pytest.raises(ValueError, match="single rejected"):
        engine.upsert_paper(paper, chunks=chunks)

    assert calls["embed_documents"] == 1
    assert calls["embed_query"] == [chunks[0].content, "[Section: Method] We plan, retrieve, and rerank."]


def test_rag_engine_ensure_index_batches_embeddings_to_model_limit():
    calls = {"embed_batch_sizes": [], "upsert_batch_sizes": []}

    chunks = [
        PaperChunk(
            chunk_id=f"paper-batch::section::{idx}",
            paper_id="paper-batch",
            section="Method",
            content=f"[Section: Method] chunk {idx}",
            token_count=4,
        )
        for idx in range(129)
    ]

    class DummyRepository:
        def build_sparse_stats(self):
            calls["built_sparse_stats"] = True
            return SimpleNamespace(
                document_count=len(chunks),
                average_length=10.0,
                document_frequencies={"chunk": len(chunks)},
            )

        def iter_chunk_models(self):
            return iter(chunks)

        def paper_payloads(self, paper_ids):
            calls["paper_payload_ids"] = paper_ids
            return {
                paper_id: {
                    "title": "Batch Sized Paper",
                    "categories": ["cs.AI"],
                    "publish_date": "2025-01-01",
                }
                for paper_id in paper_ids
            }

    class DummyIndex:
        def is_paper_collection_empty(self):
            return True

        def upsert_chunks(self, chunk_batch, embeddings, sparse_vectors, paper_payloads):
            calls["upsert_batch_sizes"].append(len(chunk_batch))
            assert len(chunk_batch) == len(embeddings) == len(sparse_vectors)
            assert list(paper_payloads) == ["paper-batch"]

    class DummyEmbedder:
        def embed_documents(self, docs):
            calls["embed_batch_sizes"].append(len(docs))
            return [[0.1, 0.2] for _ in docs]

    engine = RAGEngine(DummyRepository(), DummyIndex(), DummyEmbedder())

    engine.ensure_index()

    assert calls["built_sparse_stats"] is True
    assert calls["embed_batch_sizes"] == [64, 64, 1]
    assert calls["upsert_batch_sizes"] == [64, 64, 1]
    assert calls["paper_payload_ids"] == ["paper-batch"]


def test_rag_engine_ensure_sparse_stats_builds_once_and_reuses_cache():
    calls = {"build_sparse_stats": 0}

    class DummyRepository:
        def build_sparse_stats(self):
            calls["build_sparse_stats"] += 1
            return SimpleNamespace(
                document_count=1,
                average_length=1.0,
                document_frequencies={"retrieval": 1},
            )

    engine = RAGEngine(DummyRepository(), SimpleNamespace(), SimpleNamespace())

    first = engine.ensure_sparse_stats()
    second = engine.ensure_sparse_stats()

    assert first is second
    assert calls["build_sparse_stats"] == 1


def test_build_container_warms_sparse_stats_on_startup(monkeypatch):
    calls = {"build_sparse_stats": 0}

    def fake_is_paper_collection_empty(self):
        return False

    def fake_build_sparse_stats(self, batch_size: int = 128):
        calls["build_sparse_stats"] += 1
        return SimpleNamespace(
            document_count=4,
            average_length=12.0,
            document_frequencies={"retrieval": 2},
        )

    monkeypatch.setattr(
        app_module.QdrantIndex,
        "is_paper_collection_empty",
        fake_is_paper_collection_empty,
    )
    monkeypatch.setattr(app_module.PaperRepository, "build_sparse_stats", fake_build_sparse_stats)

    container = app_module.build_container()

    assert calls["build_sparse_stats"] == 1
    assert container.rag_engine.sparse_stats.document_count == 4
