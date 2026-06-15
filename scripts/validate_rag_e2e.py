#!/usr/bin/env python3
"""
End-to-end RAG pipeline validation using real remote models.

Tests the complete flow:
  1. Load sample papers -> chunk
  2. Call remote bge-m3 embedding API -> index into Qdrant (in-memory)
  3. Dense retrieval
  4. Sparse retrieval
  5. Hybrid retrieval
  6. Remote bge-reranker-v2-m3 reranking -> Reranked Hybrid retrieval

Usage:
  PYTHONPATH=src python scripts/validate_rag_e2e.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# ── Bootstrap: use in-memory Qdrant + SQLite, point to real remote APIs ──

os.environ.setdefault("SCHOLARMIND_QDRANT_LOCATION", ":memory:")
os.environ.setdefault("SCHOLARMIND_DATABASE_URL", "sqlite:///data/sqlite/validate_e2e.db")
os.environ.setdefault("SCHOLARMIND_CHECKPOINT_DATABASE_URL", "sqlite:///data/sqlite/validate_checkpoints.db")
os.environ.setdefault("SCHOLARMIND_EMBEDDING_PROVIDER", "remote")
os.environ.setdefault("SCHOLARMIND_EMBEDDING_MODEL", "bge-m3")
os.environ.setdefault("SCHOLARMIND_EMBEDDING_BASE_URL", "https://exempt-journalists-programmer-well.trycloudflare.com/v1")
os.environ.setdefault("SCHOLARMIND_RERANKER_ENABLED", "true")
os.environ.setdefault("SCHOLARMIND_RERANKER_PROVIDER", "remote")
os.environ.setdefault("SCHOLARMIND_RERANKER_MODEL", "bge-reranker-v2-m3")
os.environ.setdefault("SCHOLARMIND_RERANKER_BASE_URL", "https://exempt-journalists-programmer-well.trycloudflare.com/v1")

from scholar_mind.config.settings import get_settings
from scholar_mind.models.domain import PaperChunk, RetrievalStrategyName, StructuredPaper
from scholar_mind.pipeline.chunker import StructureAwareChunker
from scholar_mind.rag.embeddings import RemoteEmbeddingService
from scholar_mind.rag.engine import RAGEngine
from scholar_mind.rag.index import QdrantIndex
from scholar_mind.rag.reranker import RemoteReranker
from scholar_mind.utils.text import SparseCorpusStats, build_sparse_corpus_stats, encode_sparse_text

# ── Helpers ──

EMBEDDING_BASE_URL = "https://exempt-journalists-programmer-well.trycloudflare.com/v1"
RERANK_BASE_URL = "https://exempt-journalists-programmer-well.trycloudflare.com/v1"
SEP = "=" * 70


def header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def info(msg: str) -> None:
    print(f"  [INFO] {msg}")


# ── Step 1: Verify remote APIs are reachable ──

def step1_verify_remote_apis() -> bool:
    import httpx

    header("Step 1: Verify remote API connectivity")
    all_ok = True

    # Embedding API
    try:
        info("Calling remote bge-m3 embedding API ...")
        resp = httpx.post(
            f"{EMBEDDING_BASE_URL}/embeddings",
            json={"input": ["远端服务器调用 bge-m3"]},
            headers={"Content-Type": "application/json"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        model = data.get("model", "?")
        embeddings = data.get("embeddings") or data.get("data")
        dim = 0
        if isinstance(embeddings, list) and embeddings:
            first = embeddings[0]
            if isinstance(first, list):
                dim = len(first)
            elif isinstance(first, dict):
                dim = len(first.get("embedding", []))
        ok(f"Embedding API alive  model={model}  dimension={dim}")
        if dim != 1024:
            fail(f"Expected bge-m3 dimension=1024, got {dim}")
            all_ok = False
    except Exception as exc:
        fail(f"Embedding API error: {exc}")
        all_ok = False

    # Rerank API
    try:
        info("Calling remote bge-reranker-v2-m3 rerank API ...")
        resp = httpx.post(
            f"{RERANK_BASE_URL}/rerank",
            json={
                "pairs": [
                    {"query": "什么是 LangGraph", "passage": "LangGraph 是一个 agent 工作流框架。"},
                    {"query": "什么是 LangGraph", "passage": "苹果是一种水果。"},
                ]
            },
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        model = data.get("model", "?")
        results = data.get("results", [])
        ok(f"Rerank API alive  model={model}  results_count={len(results)}")
        if results:
            for i, r in enumerate(results):
                info(f"  pair[{i}]: score={r.get('score', 'N/A'):.4f}")
    except Exception as exc:
        fail(f"Rerank API error: {exc}")
        all_ok = False

    return all_ok


# ── Step 2: Load & chunk sample papers ──

def step2_load_and_chunk() -> tuple[list[StructuredPaper], list[PaperChunk]]:
    header("Step 2: Load sample papers and chunk")

    papers_path = Path("data/processed/sample_papers.json")
    raw = json.loads(papers_path.read_text(encoding="utf-8"))
    papers = [StructuredPaper(**p) for p in raw]
    ok(f"Loaded {len(papers)} papers")

    chunker = StructureAwareChunker()
    all_chunks: list[PaperChunk] = []
    for paper in papers:
        chunks = chunker.chunk(paper)
        all_chunks.extend(chunks)
        info(f"  {paper.paper_id}: {len(chunks)} chunks")

    ok(f"Total chunks: {len(all_chunks)}")
    return papers, all_chunks


# ── Step 3: Embed with remote bge-m3 and index into Qdrant ──

def step3_embed_and_index(
    papers: list[StructuredPaper],
    chunks: list[PaperChunk],
) -> tuple[RAGEngine, SparseCorpusStats]:
    header("Step 3: Embed chunks with remote bge-m3 & index into Qdrant")

    settings = get_settings()
    dimension = settings.resolved_embedding_dimension
    info(f"Embedding dimension: {dimension}")

    # Build embedder
    embedder = RemoteEmbeddingService(
        model="bge-m3",
        base_url=EMBEDDING_BASE_URL,
        dimension=dimension,
    )

    # Build sparse stats
    sparse_stats = build_sparse_corpus_stats([c.content for c in chunks])
    info(f"Sparse stats: docs={sparse_stats.document_count}, avg_len={sparse_stats.average_length:.1f}")

    # Build Qdrant index (in-memory)
    index = QdrantIndex(settings, dimension=dimension)

    # Paper payloads for the index
    paper_map = {p.paper_id: p for p in papers}
    paper_payloads = {}
    for pid in sorted(paper_map):
        p = paper_map[pid]
        paper_payloads[pid] = {
            "title": p.title,
            "categories": p.categories,
            "publish_date": p.publish_date.isoformat(),
        }

    # Embed in batches of 16 (to avoid overwhelming the API)
    batch_size = 16
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(chunks))
        batch = chunks[start:end]
        info(f"Embedding batch {batch_idx + 1}/{total_batches} ({len(batch)} chunks) ...")

        texts = [c.content for c in batch]
        t0 = time.time()
        embeddings = embedder.embed_documents(texts)
        elapsed = time.time() - t0
        info(f"  -> {len(embeddings)} embeddings in {elapsed:.2f}s")

        sparse_vectors = [encode_sparse_text(c.content, sparse_stats) for c in batch]
        index.upsert_chunks(batch, embeddings, sparse_vectors, paper_payloads)

    # Check indexed count
    count = index.client.count(index.PAPER_COLLECTION, exact=True).count
    ok(f"Indexed {count} chunks in Qdrant")

    # Build a minimal RAGEngine for retrieval
    # We need a paper_repository-like object for sparse stats
    class MinimalPaperRepo:
        def build_sparse_stats(self):
            return sparse_stats
        def iter_chunk_models(self):
            return iter([])
        def paper_payloads(self, ids):
            return {k: v for k, v in paper_payloads.items() if k in ids}

    rag_engine = RAGEngine(MinimalPaperRepo(), index, embedder)
    rag_engine.sparse_stats = sparse_stats

    return rag_engine, sparse_stats


# ── Step 4: Dense retrieval ──

def step4_dense_retrieval(rag_engine: RAGEngine) -> None:
    header("Step 4: Dense retrieval")

    query = "How does hybrid retrieval improve recall?"
    info(f"Query: {query}")
    t0 = time.time()
    results, latency = rag_engine.retrieve_sync(query, strategy=RetrievalStrategyName.DENSE, top_k=5)
    elapsed = time.time() - t0

    ok(f"Dense retrieval returned {len(results)} results in {elapsed:.3f}s (engine latency={latency}ms)")
    for i, r in enumerate(results):
        info(f"  [{i+1}] score={r.score:.6f}  paper={r.paper_id}  section={r.section}")
        info(f"       {r.content[:100]}...")


# ── Step 5: Sparse retrieval ──

def step5_sparse_retrieval(rag_engine: RAGEngine) -> None:
    header("Step 5: Sparse (BM25) retrieval")

    query = "memory augmented scientific QA"
    info(f"Query: {query}")
    t0 = time.time()
    results, latency = rag_engine.retrieve_sync(query, strategy=RetrievalStrategyName.SPARSE, top_k=5)
    elapsed = time.time() - t0

    ok(f"Sparse retrieval returned {len(results)} results in {elapsed:.3f}s (engine latency={latency}ms)")
    for i, r in enumerate(results):
        info(f"  [{i+1}] score={r.score:.6f}  paper={r.paper_id}  section={r.section}")
        info(f"       {r.content[:100]}...")


# ── Step 6: Hybrid retrieval ──

def step6_hybrid_retrieval(rag_engine: RAGEngine) -> None:
    header("Step 6: Hybrid retrieval (dense + sparse fusion)")

    query = "retrieval augmented generation for scientific question answering"
    info(f"Query: {query}")
    t0 = time.time()
    results, latency = rag_engine.retrieve_sync(query, strategy=RetrievalStrategyName.HYBRID, top_k=5)
    elapsed = time.time() - t0

    ok(f"Hybrid retrieval returned {len(results)} results in {elapsed:.3f}s (engine latency={latency}ms)")
    for i, r in enumerate(results):
        info(f"  [{i+1}] score={r.score:.6f}  paper={r.paper_id}  section={r.section}")
        info(f"       {r.content[:100]}...")


# ── Step 7: Reranked hybrid with remote bge-reranker-v2-m3 ──

def step7_reranked_hybrid(rag_engine: RAGEngine) -> None:
    header("Step 7: Reranked hybrid (hybrid + remote bge-reranker-v2-m3)")

    # Attach remote reranker
    reranker = RemoteReranker(
        model_name="bge-reranker-v2-m3",
        base_url=RERANK_BASE_URL,
        timeout_seconds=15.0,
    )
    rag_engine.reranker_service = reranker

    query = "cross-domain transfer of planning algorithms"
    info(f"Query: {query}")
    t0 = time.time()
    results, latency = rag_engine.retrieve_sync(
        query, strategy=RetrievalStrategyName.RERANKED_HYBRID, top_k=5,
    )
    elapsed = time.time() - t0

    ok(f"Reranked hybrid returned {len(results)} results in {elapsed:.3f}s (engine latency={latency}ms)")
    for i, r in enumerate(results):
        info(f"  [{i+1}] score={r.score:.6f}  paper={r.paper_id}  section={r.section}")
        info(f"       {r.content[:100]}...")


# ── Step 8: Chinese query validation ──

def step8_chinese_query(rag_engine: RAGEngine) -> None:
    header("Step 8: Chinese query (bge-m3 multilingual)")

    query = "检索增强生成在科研问答中的应用"
    info(f"Query: {query}")
    t0 = time.time()
    results, latency = rag_engine.retrieve_sync(query, strategy=RetrievalStrategyName.DENSE, top_k=5)
    elapsed = time.time() - t0

    ok(f"Dense retrieval (Chinese) returned {len(results)} results in {elapsed:.3f}s")
    for i, r in enumerate(results):
        info(f"  [{i+1}] score={r.score:.6f}  paper={r.paper_id}  section={r.section}")
        info(f"       {r.content[:100]}...")


# ── Step 9: Compare strategies side-by-side ──

def step9_compare_strategies(rag_engine: RAGEngine) -> None:
    header("Step 9: Strategy comparison for 'LLM agent planning and memory'")

    query = "LLM agent planning and memory"
    strategies = [
        RetrievalStrategyName.DENSE,
        RetrievalStrategyName.SPARSE,
        RetrievalStrategyName.HYBRID,
        RetrievalStrategyName.RERANKED_HYBRID,
    ]

    print(f"\n  Query: {query}\n")
    print(f"  {'Strategy':<20} {'Top1 Paper':<15} {'Top1 Score':<12} {'Top1 Section':<15}")
    print(f"  {'-'*20} {'-'*15} {'-'*12} {'-'*15}")

    for strategy in strategies:
        results, latency = rag_engine.retrieve_sync(query, strategy=strategy, top_k=3)
        if results:
            top = results[0]
            print(f"  {strategy.value:<20} {top.paper_id:<15} {top.score:<12.6f} {top.section:<15}")
        else:
            print(f"  {strategy.value:<20} {'(no results)':<15}")


# ── Step 10: Validate full app container bootstraps ──

def step10_full_container_bootstrap() -> None:
    header("Step 10: Full AppContainer bootstrap with remote models")

    # Reset caches to pick up env vars
    from scholar_mind.app import get_container
    get_settings.cache_clear()
    get_container.cache_clear()

    info("Building AppContainer (this may take a moment for embedding index build) ...")
    t0 = time.time()
    container = get_container()
    elapsed = time.time() - t0

    ok(f"AppContainer built in {elapsed:.2f}s")
    info(f"  embedder type: {type(container.embedder).__name__}")
    info(f"  reranker type: {type(container.rag_engine.reranker_service).__name__}")
    info(f"  embedding dimension: {container.embedder.dimension}")

    # Quick smoke test through the container's RAG engine
    query = "What does hybrid retrieval improve?"
    info(f"Smoke test query: {query}")
    results, latency = container.rag_engine.retrieve_sync(query, strategy=RetrievalStrategyName.HYBRID, top_k=3)
    ok(f"Container RAG engine returned {len(results)} results, latency={latency}ms")
    for i, r in enumerate(results):
        info(f"  [{i+1}] {r.paper_id} | {r.section} | score={r.score:.6f}")

    # Verify the remote reranker is wired up
    assert isinstance(container.rag_engine.reranker_service, RemoteReranker), \
        f"Expected RemoteReranker, got {type(container.rag_engine.reranker_service)}"
    ok("Reranker is RemoteReranker (remote bge-reranker-v2-m3)")

    assert isinstance(container.embedder, RemoteEmbeddingService), \
        f"Expected RemoteEmbeddingService, got {type(container.embedder)}"
    ok("Embedder is RemoteEmbeddingService (remote bge-m3)")


# ── Main ──

def main() -> None:
    print(f"\n{'#' * 70}")
    print("  ScholarMind RAG Pipeline - End-to-End Validation")
    print("  Remote Models: bge-m3 (embedding) + bge-reranker-v2-m3 (reranking)")
    print(f"{'#' * 70}")

    overall_start = time.time()

    # Step 1: API connectivity
    if not step1_verify_remote_apis():
        fail("Remote APIs unreachable, aborting.")
        sys.exit(1)

    # Step 2: Load & chunk
    papers, chunks = step2_load_and_chunk()

    # Step 3: Embed & index
    rag_engine, sparse_stats = step3_embed_and_index(papers, chunks)

    # Steps 4-9: Retrieval strategies
    step4_dense_retrieval(rag_engine)
    step5_sparse_retrieval(rag_engine)
    step6_hybrid_retrieval(rag_engine)
    step7_reranked_hybrid(rag_engine)
    step8_chinese_query(rag_engine)
    step9_compare_strategies(rag_engine)

    # Step 10: Full container bootstrap
    step10_full_container_bootstrap()

    total_elapsed = time.time() - overall_start
    header("Summary")
    ok(f"All steps completed in {total_elapsed:.2f}s")
    print(f"\n  Remote embedding:  bge-m3 @ {EMBEDDING_BASE_URL}")
    print(f"  Remote reranking:  bge-reranker-v2-m3 @ {RERANK_BASE_URL}")
    print("  Qdrant:            in-memory")
    print(f"  Sample papers:     {len(papers)}")
    print(f"  Total chunks:      {len(chunks)}")
    print()


if __name__ == "__main__":
    main()
