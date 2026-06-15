# ScholarMind

ScholarMind is a Python 3.11 research assistant MVP for paper-centered question answering, research planning, idea novelty checks, trend analysis, cross-domain hypothesis generation, paper reading, memory management, and RAG evaluation.

The project exposes both a FastAPI web service and a Typer CLI. It uses SQLite for application data and checkpoints, Qdrant for vector retrieval, Redis/Celery for background ingestion work, and configurable OpenAI-compatible model providers.

## Features

- Multi-agent research workflows built around LangGraph.
- RAG over paper metadata and chunks with dense, sparse, hybrid, and reranked strategies.
- arXiv ingestion from recent papers, explicit paper IDs, or local files.
- Session memory extraction, retrieval, decay, and structured storage.
- Online request auditing plus RAG and memory evaluation endpoints.
- Static web UI served from `static/`.
- Local development through `uv`, direct `uvicorn`, helper scripts, or Docker Compose.

## Project Layout

```text
src/scholar_mind/
  agents/       LangGraph agents and prompts orchestration
  api/          FastAPI routes and response helpers
  config/       Settings loading from YAML, .env, and environment variables
  db/           SQLAlchemy models, sessions, and database initialization
  eval/         RAG and answer-quality evaluation services
  memory/       Memory extraction, retrieval, decay, and persistence
  models/       Domain models and model provider factories
  pipeline/     Paper download, parse, chunk, index, and Celery tasks
  rag/          Embeddings, vector index, retrieval engine, and strategies
  services/     Application service layer and repositories
  utils/        Shared message, streaming, sample data, and token helpers
config/         Runtime configuration and prompt templates
data/           Local sample, SQLite, Qdrant, Redis, logs, raw, and processed data
scripts/        Local deployment, stop, ingestion, and evaluation helpers
static/         Browser UI assets
tests/          Unit, integration, API, RAG, memory, and script tests
```

## Requirements

- Python 3.11+
- `uv` for dependency management
- Docker or Podman for Redis/Qdrant when running the full local stack
- Model provider credentials for LLM and embedding calls

The default config reads `config/default.yaml`, then `config/<environment>.yaml`, then `.env` and process environment variables. Environment variables use the `SCHOLARMIND_` prefix, with these supported aliases:

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

Initialize the database and sample data:

```bash
bash scripts/init.sh
```

## Run Locally

Start supporting services and the application with the project helper:

```bash
bash scripts/deploy.sh
```

The API and static UI are served at:

```text
http://127.0.0.1:8000
```

Stop the local services:

```bash
bash scripts/stop.sh
```

To run only the FastAPI app directly:

```bash
PYTHONPATH=src uv run uvicorn scholar_mind.asgi:app --host 127.0.0.1 --port 8000
```

## Docker Compose

Run the complete stack:

```bash
docker compose up -d
```

Services include:

- `app`: FastAPI service on port `8000`
- `worker`: Celery worker
- `scheduler`: Celery beat scheduler
- `qdrant`: vector database on ports `6333` and `6334`
- `redis`: broker/backend on port `6379`
- optional Cloudflare tunnel profiles: `tunnel`, `named-tunnel`

Stop the stack:

```bash
docker compose down
```

## API Overview

Health check:

```bash
curl -fsS http://127.0.0.1:8000/api/v1/health
```

Research endpoints:

- `POST /api/v1/research/ask`
- `POST /api/v1/research/ask/stream`
- `POST /api/v1/research/stream`
- `POST /api/v1/research/idea-novelty`
- `POST /api/v1/research/trend`
- `POST /api/v1/research/cross-domain`
- `POST /api/v1/research/study-plan`
- `POST /api/v1/research/paper-reading`

Paper and ingestion endpoints:

- `GET /api/v1/papers/search`
- `GET /api/v1/papers/{paper_id}`
- `POST /api/v1/ingest/recent`
- `POST /api/v1/ingest/recent/stream`
- `POST /api/v1/ingest/papers/stream`
- `POST /api/v1/ingest/local/stream`

Session and evaluation endpoints:

- `POST /api/v1/sessions`
- `GET /api/v1/sessions/{session_id}`
- `DELETE /api/v1/sessions/{session_id}`
- `POST /api/v1/eval/rag/runs`
- `GET /api/v1/eval/rag/runs`
- `GET /api/v1/eval/dashboard/online`
- `GET /api/v1/eval/dashboard/requests`
- `GET /api/v1/eval/memory/batches/{batch_id}`

Example ask request:

```bash
curl -fsS -X POST http://127.0.0.1:8000/api/v1/research/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What are the main ideas in the indexed papers?",
    "user_id": "local-user",
    "paper_ids": [],
    "rag_strategy": "hybrid"
  }'
```

## CLI

The package installs a `scholar` command:

```bash
uv run scholar ask "What does the paper collection say about RAG evaluation?"
uv run scholar idea-novelty "Use memory-aware RAG for long-term research planning"
uv run scholar trend "retrieval augmented generation"
uv run scholar cross-domain "Connect graph neural networks with recommender systems"
uv run scholar study-plan "帮我制定一个学习计划" --goal "掌握 RAG 评估"
uv run scholar paper-reading <paper_id> "开始精读"
```

Paper and evaluation commands:

```bash
uv run scholar paper search "RAG"
uv run scholar paper get <paper_id>
uv run scholar paper ingest-arxiv 2401.00001
uv run scholar eval rag-run
uv run scholar eval rag-report --run-id <run_id>
uv run scholar eval memory-export --from-request-id <request_id>
uv run scholar eval memory --batch-id <batch_id>
```

## Data Ingestion

Ingest recent arXiv papers through the API:

```bash
curl -fsS -X POST 'http://127.0.0.1:8000/api/v1/ingest/recent?count=5&category=cs.AI'
```

Use helper scripts for local workflows:

```bash
PYTHONPATH=src uv run python scripts/download_recent_arxiv.py
PYTHONPATH=src uv run python scripts/ingest_recent_arxiv.py
PYTHONPATH=src uv run python scripts/ingest_local_arxiv.py
```

Local data is stored under `data/` by default. Important paths are configurable in `config/default.yaml`, including SQLite databases, Qdrant storage, raw papers, processed papers, memory logs, and evaluation artifacts.

## Configuration

Common settings:

- `SCHOLARMIND_ENVIRONMENT`: selects `config/<environment>.yaml`
- `SCHOLARMIND_DATABASE_URL`: application SQLite URL
- `SCHOLARMIND_CHECKPOINT_DATABASE_URL`: LangGraph checkpoint SQLite URL
- `SCHOLARMIND_QDRANT_URL`: remote Qdrant URL
- `SCHOLARMIND_QDRANT_LOCATION`: local Qdrant storage path or `:memory:`
- `SCHOLARMIND_REDIS_URL`: Redis URL for Celery
- `SCHOLARMIND_LLM_BASE_URL` / `ZAI_BASE_URL`: OpenAI-compatible chat endpoint
- `SCHOLARMIND_LLM_API_KEY` / `ZAI_API_KEY`: chat provider API key
- `SCHOLARMIND_EMBEDDING_BASE_URL`: embedding endpoint
- `SCHOLARMIND_EMBEDDING_API_KEY` / `EMBEDDING_API_KEY`: embedding API key
- `SCHOLARMIND_RERANKER_ENABLED`: enable reranking
- `SCHOLARMIND_RERANKER_BASE_URL`: reranker endpoint

Configuration precedence is:

1. `config/default.yaml`
2. `config/memory.yaml`
3. `config/<SCHOLARMIND_ENVIRONMENT>.yaml`
4. `.env`
5. process environment variables

## Testing and Linting

Run the test suite:

```bash
uv run pytest
```

Run lint checks:

```bash
uv run ruff check .
```

The tests cover API behavior, settings, services, RAG retrieval, memory operations, orchestration, evaluation, scripts, and frontend request details.

## Development Notes

- The project root can be overridden with `SCHOLARMIND_ROOT_DIR`.
- The app bootstraps sample papers when `bootstrap_sample_data` is true.
- `development` config uses local Qdrant storage and eager Celery task execution.
- `production` config expects Qdrant and Redis services and disables sample-data bootstrap by default.
- Docker Compose overrides production service settings for the container stack and mounts `./data` into `/app/data`.
