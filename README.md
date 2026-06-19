# ScholarMind Memory Eval

This `memory_eval` branch keeps ScholarMind's long-term memory system and removes the paper-centered research assistant surface. It is intended as a smaller runtime for memory extraction, retrieval, management, and offline memory evaluation, with later LoCoMo-style benchmark adapters built on top.

## Features

- Session memory extraction, retrieval, decay, structured storage, and conflict handling.
- Pending-memory buffering and memory admission controls.
- Memory consistency audit and memory library duplicate/conflict audit.
- Memory V2 request-level evaluation from recorded retrieval/extraction traces.
- SQLite persistence and Qdrant vector storage for memory records.
- Configurable OpenAI-compatible chat and embedding providers.

## Project Layout

```text
src/scholar_mind/
  api/          FastAPI shell, health, sessions, and memory evaluation routes
  config/       Settings loading from YAML, .env, and environment variables
  db/           SQLAlchemy models, sessions, and database initialization
  memory/       Memory extraction, retrieval, operations, decay, and persistence
  models/       Domain models and model provider factories
  rag/          Embedding and Qdrant index infrastructure retained for memory vectors
  services/     Memory evaluation, memory management, and repository helpers
  utils/        Message, streaming, sample data, and token helpers retained by memory code
config/         Runtime configuration and memory prompt templates
data/           Local SQLite, Qdrant, Redis, logs, memory, and evaluation artifacts
tests/          Memory, settings, API shell, and evaluation tests
```

## Requirements

- Python 3.11+
- `uv` for dependency management
- Qdrant for persistent vector storage, or `SCHOLARMIND_QDRANT_LOCATION=:memory:` for tests
- Model provider credentials for LLM and embedding calls when running real extraction/retrieval

Environment variables use the `SCHOLARMIND_` prefix. Supported aliases:

- `ZAI_API_KEY` -> `llm_api_key`
- `ZAI_BASE_URL` -> `llm_base_url`
- `EMBEDDING_API_KEY` -> `embedding_api_key`

## Setup

Install dependencies:

```bash
uv sync --extra dev
```

Create a local `.env` as needed:

```bash
SCHOLARMIND_ENVIRONMENT=development
ZAI_API_KEY=your-llm-api-key
ZAI_BASE_URL=https://your-openai-compatible-endpoint/v1
EMBEDDING_API_KEY=your-embedding-api-key
SCHOLARMIND_EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4
SCHOLARMIND_EMBEDDING_MODEL=embedding-3
```

## Run Locally

Start the FastAPI app directly:

```bash
PYTHONPATH=src uv run uvicorn scholar_mind.asgi:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl -fsS http://127.0.0.1:8000/api/v1/health
```

## CLI

The package installs a `scholar` command. This branch exposes memory evaluation commands only:

```bash
uv run scholar eval memory-export --from-request-id <request_id> --limit 1
uv run scholar eval memory --batch-id <batch_id>
uv run scholar eval memory-report --report-id <report_id>
uv run scholar eval memory-library-export
uv run scholar eval memory-library --batch-id <batch_id>
uv run scholar eval memory-library-report --report-id <report_id>
uv run scholar eval memory-consistency-audit --user-id <user_id>
```

## Memory Evaluation Flow

Memory V2 is an offline evaluation flow:

1. Run or import memory-backed requests so `request_runs`, `memory_retrieval_events_v2`, and `memory_extraction_events_v2` exist.
2. Export a batch:

   ```bash
   uv run scholar eval memory-export --from-request-id <request_id> --limit 20
   ```

3. Fill `data/eval/memory_batches/<batch_id>/annotations.jsonl`.
4. Evaluate:

   ```bash
   uv run scholar eval memory --batch-id <batch_id>
   ```

The generated report includes memory hit@k, relevant recall/precision, first relevant rank, stale retrieval rate, answer relevance, extraction precision, and the combined memory score.

## LoCoMo-Style Adaptation

This branch is prepared for LoCoMo-style long conversational memory testing. A benchmark adapter should:

- import each benchmark case as a distinct `user_id`;
- preserve source provenance such as `sample_id`, `session_id`, and dialog/evidence ids in memory metadata;
- issue benchmark questions through a memory-backed answering layer;
- export predictions and retrieved provenance ids for official or business-specific answer and evidence scoring.

The existing Memory V2 score should be treated as a diagnostic metric, not as a replacement for official LoCoMo answer scoring.

## Configuration

Common settings:

- `SCHOLARMIND_ENVIRONMENT`: selects `config/<environment>.yaml`
- `SCHOLARMIND_DATABASE_URL`: application SQLite URL
- `SCHOLARMIND_QDRANT_URL`: remote Qdrant URL
- `SCHOLARMIND_QDRANT_LOCATION`: local Qdrant storage path or `:memory:`
- `SCHOLARMIND_LLM_BASE_URL` / `ZAI_BASE_URL`: OpenAI-compatible chat endpoint
- `SCHOLARMIND_LLM_API_KEY` / `ZAI_API_KEY`: chat provider API key
- `SCHOLARMIND_EMBEDDING_BASE_URL`: embedding endpoint
- `SCHOLARMIND_EMBEDDING_API_KEY` / `EMBEDDING_API_KEY`: embedding API key

Configuration precedence is:

1. `config/default.yaml`
2. `config/memory.yaml`
3. `config/<SCHOLARMIND_ENVIRONMENT>.yaml`
4. `.env`
5. process environment variables

## Testing and Linting

Run the retained test suite:

```bash
uv run pytest
```

Run lint checks:

```bash
uv run ruff check src tests
```
