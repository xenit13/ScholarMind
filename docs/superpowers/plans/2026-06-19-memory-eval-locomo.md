# Memory Eval Official LoCoMo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn ScholarMind into a memory-system-only branch named `memory_eval`, with prompts and evaluation surfaces prepared to run the official LoCoMo benchmark. Business-specific LoCoMo-style datasets, question generation, and extra adaptation layers are out of scope.

**Architecture:** Keep memory extraction, retrieval, structured storage, decay, consistency audit, memory evaluation, database/session infrastructure, model providers, embeddings, and vector index adapters. Remove paper/RAG/research-assistant business features from public CLI/API/container wiring, then delete orphaned modules once memory tests prove the retained surface is independent.

**Tech Stack:** Python 3.11, FastAPI, Typer, SQLAlchemy, Qdrant client, LangChain model interfaces, pytest, ruff.

---

## Scope

This branch keeps code that is directly needed to run, inspect, or evaluate long-term memory:

- `scholar_mind.memory.*`
- `MemoryManager`, `MemoryRepository`, `MemoryManagementService`
- `MemoryEvalServiceV2` and memory eval persistence
- memory request audit tables needed by memory eval
- model factory, embedding service, Qdrant index, settings, DB/session setup
- health/API dependency shell and memory-only routes
- CLI commands for memory ingestion, retrieval, export, evaluation, audit, and consistency checks

This branch removes or hides paper/research business capabilities:

- paper search/get/ingest user flows
- research assistant commands: ask, idea novelty, trend, cross-domain, study plan, paper reading
- arXiv pipeline and sample paper bootstrap
- RAG evaluation commands and RAG-only metrics
- research/paper prompts
- tests that only validate removed business behavior

LoCoMo support in this plan targets the official benchmark contract: official conversations are ingested as long-term memory inputs, official benchmark questions are asked unchanged, and evaluation uses official answers/evidence where available. This branch must not add a separate business dataset format or generate business-specific benchmark questions.

## Phase 0: Branch and Plan Commit

**Files:**
- Create: `docs/superpowers/plans/2026-06-19-memory-eval-locomo.md`

- [ ] **Step 1: Confirm branch**

Run: `git status --short --branch`

Expected: branch is `memory_eval`; unrelated untracked files are left untouched.

- [ ] **Step 2: Commit the plan**

Run:

```bash
git add docs/superpowers/plans/2026-06-19-memory-eval-locomo.md
git commit -m "docs: plan memory-only eval branch"
```

Expected: one commit on `memory_eval`.

## Phase 1: Protect Memory-Only Public Surface

**Files:**
- Modify: `tests/test_memory_eval_surface.py`
- Modify: `src/scholar_mind/main.py`
- Modify: `src/scholar_mind/app.py`
- Modify: `README.md`

- [ ] **Step 1: Write CLI surface tests**

Add tests that assert the Typer CLI exposes only memory-oriented top-level commands and memory eval commands. The failing assertion should identify at least one removed business command currently present, such as `paper-reading`, `trend`, `cross-domain`, `paper`, `rag-run`, or `rag-report`.

Run: `uv run pytest tests/test_memory_eval_surface.py -q`

Expected before implementation: fail because business commands are still registered.

- [ ] **Step 2: Trim CLI**

Remove public CLI commands for paper and research business workflows. Keep `eval memory-export`, `eval memory`, `eval memory-report`, `eval memory-library-export`, `eval memory-library`, `eval memory-library-report`, and `eval memory-consistency-audit`.

Run: `uv run pytest tests/test_memory_eval_surface.py tests/test_memory_eval_v2.py -q`

Expected after implementation: pass.

- [ ] **Step 3: Trim app container wiring**

Remove construction of `AgentOrchestrator`, `ResearchService`, `RagEvalService`, arXiv ingestion, paper repository public fields, sample paper bootstrap, and RAG eval repository from `AppContainer`. Keep session/database repositories only where memory eval or memory audit still needs them.

Run: `uv run pytest tests/test_asgi_import.py tests/test_memory_eval_v2.py -q`

Expected after implementation: pass.

- [ ] **Step 4: Update README public contract**

Rewrite README overview, CLI, and API sections to describe the memory-only branch. Do not mention paper search, arXiv ingest, RAG evaluation, or research assistants as supported functionality.

Run: `uv run pytest tests/test_memory_eval_surface.py -q`

Expected after implementation: pass.

- [ ] **Step 5: Commit Phase 1**

Run:

```bash
git add README.md src/scholar_mind/main.py src/scholar_mind/app.py tests/test_memory_eval_surface.py
git commit -m "refactor: expose memory-only runtime surface"
```

Expected: second commit on `memory_eval`.

## Phase 2: Remove Business Modules and Tests

**Files:**
- Delete: `src/scholar_mind/agents/`
- Delete: `src/scholar_mind/pipeline/`
- Delete: `src/scholar_mind/eval/rag_*`
- Delete: `src/scholar_mind/eval/ragas_official.py`
- Delete: `src/scholar_mind/models/rag_eval_models.py`
- Delete: paper/research-only tests under `tests/`
- Modify: memory modules that import `scholar_mind.agents.common`
- Modify: memory modules that import `scholar_mind.eval.context`

- [ ] **Step 1: Write import boundary tests**

Extend `tests/test_memory_eval_surface.py` to assert memory modules no longer import `scholar_mind.agents`, `scholar_mind.pipeline`, paper repositories, RAG eval services, or research services.

Run: `uv run pytest tests/test_memory_eval_surface.py -q`

Expected before implementation: fail on current imports from `memory.compressor` and `memory.extraction`.

- [ ] **Step 2: Move structured-output helpers**

Create a memory-local structured-output helper or shared non-business utility so memory extraction, admission, consistency audit, operation matching, and compression do not depend on `agents.common`.

Run: `uv run pytest tests/test_memory_extraction.py tests/test_memory_operations.py tests/test_memory_consistency_audit.py tests/test_memory_manager.py -q`

Expected after implementation: pass.

- [ ] **Step 3: Replace eval context dependency**

Keep only memory event recording needed by `MemoryManager` and `MemoryCompressor`, or move that event context into a memory-local module. Remove RAG event types and RAG-specific context fields.

Run: `uv run pytest tests/test_memory_eval_v2.py tests/test_memory_manager.py -q`

Expected after implementation: pass.

- [ ] **Step 4: Delete business modules and orphan tests**

Delete modules and tests that cannot be imported without paper/research/RAG business logic. Retain tests for memory repository, memory manager, memory extraction, memory operations, memory decay, memory management, pending memory, consistency audit, settings, token estimator, ASGI import, and memory eval.

Run: `uv run pytest tests -q`

Expected after implementation: pass.

- [ ] **Step 5: Commit Phase 2**

Run:

```bash
git add -A src tests
git commit -m "refactor: remove research and paper business modules"
```

Expected: third commit on `memory_eval`.

## Phase 3: Adapt Prompts and Settings for Official LoCoMo

**Files:**
- Create: `config/prompts/memory_extraction.txt`
- Create: `config/prompts/memory_answering.txt`
- Create: official LoCoMo runner files as needed
- Modify: `src/scholar_mind/memory/extraction.py`
- Modify: `src/scholar_mind/config/settings.py`
- Modify: `config/default.yaml`
- Modify: `config/memory.yaml`
- Delete: paper/research prompt files under `config/prompts/`

- [ ] **Step 1: Write official LoCoMo surface tests**

Add tests that check the configured prompt directory only contains memory prompts, memory extraction prompts mention long conversation provenance, temporal validity, evidence/source ids, and stale/conflicting facts, and the LoCoMo entrypoint expects official conversation/question records rather than business-specific generated questions.

Run: `uv run pytest tests/test_memory_eval_surface.py tests/test_memory_extraction.py -q`

Expected before implementation: fail because paper/research prompts remain, extraction prompt is embedded in code, or no official LoCoMo entrypoint exists.

- [ ] **Step 2: Externalize memory extraction prompt**

Make memory extraction load a memory-specific prompt template when configured, while preserving existing structured output schema and fallback behavior.

Run: `uv run pytest tests/test_memory_extraction.py -q`

Expected after implementation: pass.

- [ ] **Step 3: Add official LoCoMo ingest/query runner**

Add the minimal runner needed to load official LoCoMo conversations and questions, feed conversations into the memory system, answer official questions unchanged, and export predictions/trace data for official scoring. Do not add business-specific question generation or custom business labels.

Run: `uv run pytest tests/test_memory_eval_surface.py tests/test_memory_eval_v2.py -q`

Expected after implementation: pass.

- [ ] **Step 4: Delete business prompts**

Remove planner, researcher, reviewer, writer, trend, hypothesis, crossdomain, paper reader, and paper reading planner prompts. Keep only memory prompts.

Run: `uv run pytest tests/test_memory_eval_surface.py -q`

Expected after implementation: pass.

- [ ] **Step 5: Commit Phase 3**

Run:

```bash
git add -A config src tests
git commit -m "feat: support official locomo memory eval"
```

Expected: fourth commit on `memory_eval`.

## Phase 4: Final Verification and Audit

**Files:**
- Modify as needed based on verification failures.

- [ ] **Step 1: Run formatting and lint**

Run:

```bash
uv run ruff check src tests
```

Expected: no lint failures.

- [ ] **Step 2: Run full retained test suite**

Run:

```bash
uv run pytest tests -q
```

Expected: all retained tests pass.

- [ ] **Step 3: Audit removed business strings**

Run:

```bash
rg -n "paper|arxiv|rag-run|rag-report|trend|cross-domain|study-plan|paper-reading|idea-novelty" README.md src tests config
```

Expected: no public business surface remains. Allowed hits must be in historical comments only if removing them would harm memory behavior.

- [ ] **Step 4: Commit final cleanup if needed**

Run:

```bash
git add -A
git commit -m "chore: verify memory-only eval branch"
```

Expected: final cleanup commit only if Phase 4 changed files.
