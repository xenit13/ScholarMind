#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

if [[ -x "$ROOT_DIR/.venv/bin/python3" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python3"
else
  PYTHON="$(command -v python3)"
fi
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:${PORT}/api/v1/health}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/data/logs/scholar_mind.log}"
PID_FILE="${PID_FILE:-$ROOT_DIR/data/run/scholar_mind.pid}"
CLOUDFLARED_ORIGIN_URL="${CLOUDFLARED_ORIGIN_URL:-http://127.0.0.1:${PORT}}"
CLOUDFLARED_LOG_FILE="${CLOUDFLARED_LOG_FILE:-$ROOT_DIR/data/logs/cloudflared.log}"
CLOUDFLARED_PID_FILE="${CLOUDFLARED_PID_FILE:-$ROOT_DIR/data/run/cloudflared.pid}"
CLOUDFLARED_URL_FILE="${CLOUDFLARED_URL_FILE:-$ROOT_DIR/data/run/cloudflared.url}"
REDIS_MARKER_FILE="${REDIS_MARKER_FILE:-$ROOT_DIR/data/run/redis.started_by_deploy}"
QDRANT_MARKER_FILE="${QDRANT_MARKER_FILE:-$ROOT_DIR/data/run/qdrant.started_by_deploy}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-60}"
CLOUDFLARED_STARTUP_TIMEOUT_SECONDS="${CLOUDFLARED_STARTUP_TIMEOUT_SECONDS:-30}"

ENABLE_WEB_TUNNEL=0

usage() {
  cat <<EOF
Usage: bash scripts/deploy.sh [--web]

Options:
  --web   Start a temporary cloudflared Quick Tunnel for the local API service.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --web)
      ENABLE_WEB_TUNNEL=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      usage >&2
      exit 1
      ;;
  esac
done

export SCHOLARMIND_ENVIRONMENT="${SCHOLARMIND_ENVIRONMENT:-production}"

mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$CLOUDFLARED_LOG_FILE")" \
  "$(dirname "$PID_FILE")" "$(dirname "$CLOUDFLARED_PID_FILE")" \
  "$(dirname "$CLOUDFLARED_URL_FILE")" "$ROOT_DIR/data/sqlite" \
  "$ROOT_DIR/data/qdrant" "$ROOT_DIR/data/redis"

# ── helpers ──────────────────────────────────────────────────────────────────

docker_cmd() {
  if command -v podman-compose >/dev/null 2>&1; then
    podman-compose "$@"
  elif command -v docker >/dev/null 2>&1; then
    docker compose "$@"
  else
    echo "Neither podman-compose nor docker is installed" >&2
    exit 1
  fi
}

verify_python_runtime() {
  if PYTHONPATH=src "$PYTHON" -c 'from scholar_mind.config.settings import get_settings; get_settings()' >/dev/null 2>&1; then
    return
  fi

  echo "Python environment check failed for $PYTHON" >&2
  echo "If this virtualenv was copied from another machine, rebuild it with: uv sync --frozen --reinstall --extra dev" >&2
  exit 1
}

redis_ping() {
  PYTHONPATH=src "$PYTHON" -c 'from redis import Redis; from scholar_mind.config.settings import get_settings; client = Redis.from_url(get_settings().redis_url, socket_connect_timeout=1, socket_timeout=1); raise SystemExit(0 if client.ping() else 1)' >/dev/null 2>&1
}

qdrant_ping() {
  curl -fsS http://127.0.0.1:6333/collections >/dev/null 2>&1
}

# ── ensure infrastructure services ───────────────────────────────────────────

ensure_redis() {
  if redis_ping; then
    echo "[redis] already available"
    return
  fi

  echo "[redis] starting via docker..."
  docker_cmd up -d redis >/dev/null
  echo "1" > "$REDIS_MARKER_FILE"

  for _ in $(seq 1 "$STARTUP_TIMEOUT_SECONDS"); do
    if redis_ping; then
      echo "[redis] started successfully"
      return
    fi
    sleep 1
  done

  echo "[redis] failed to become available within ${STARTUP_TIMEOUT_SECONDS}s" >&2
  exit 1
}

ensure_qdrant() {
  if qdrant_ping; then
    echo "[qdrant] already available"
    return
  fi

  echo "[qdrant] starting via docker..."
  docker_cmd up -d qdrant >/dev/null
  echo "1" > "$QDRANT_MARKER_FILE"

  for _ in $(seq 1 "$STARTUP_TIMEOUT_SECONDS"); do
    if qdrant_ping; then
      echo "[qdrant] started successfully"
      return
    fi
    sleep 1
  done

  echo "[qdrant] failed to become available within ${STARTUP_TIMEOUT_SECONDS}s" >&2
  exit 1
}

# ── start application services ───────────────────────────────────────────────

start_app() {
  if [[ -f "$PID_FILE" ]]; then
    existing_pid="$(cat "$PID_FILE")"
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "[app] already running with pid $existing_pid"
      return
    fi
    rm -f "$PID_FILE"
  fi

  echo "[app] starting..."
  if command -v setsid >/dev/null 2>&1; then
    setsid env PYTHONPATH=src "$PYTHON" -m uvicorn scholar_mind.asgi:app \
      --host "$HOST" \
      --port "$PORT" \
      </dev/null \
      >"$LOG_FILE" 2>&1 &
  else
    nohup env PYTHONPATH=src "$PYTHON" -m uvicorn scholar_mind.asgi:app \
      --host "$HOST" \
      --port "$PORT" \
      </dev/null \
      >"$LOG_FILE" 2>&1 &
  fi
  app_pid="$!"
  echo "$app_pid" > "$PID_FILE"

  for _ in $(seq 1 "$STARTUP_TIMEOUT_SECONDS"); do
    if ! kill -0 "$app_pid" 2>/dev/null; then
      echo "[app] exited before becoming healthy" >&2
      echo "Last 50 log lines:" >&2
      tail -n 50 "$LOG_FILE" >&2 || true
      rm -f "$PID_FILE"
      exit 1
    fi

    if response="$(curl -fsS "$HEALTH_URL" 2>/dev/null)"; then
      if printf '%s' "$response" | "$PYTHON" -c 'import json, sys; payload=json.load(sys.stdin); raise SystemExit(0 if payload["data"]["status"]=="healthy" else 1)'; then
        echo "[app] started successfully (pid $app_pid)"
        return
      fi
    fi
    sleep 1
  done

  echo "[app] failed to become healthy within ${STARTUP_TIMEOUT_SECONDS}s" >&2
  if kill -0 "$app_pid" 2>/dev/null; then
    echo "Last 50 log lines:" >&2
    tail -n 50 "$LOG_FILE" >&2 || true
    kill "$app_pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  exit 1
}

extract_cloudflared_url() {
  awk '
    match($0, /https:\/\/[-A-Za-z0-9.]+\.trycloudflare\.com/) {
      url = substr($0, RSTART, RLENGTH)
    }
    END {
      if (url != "") {
        print url
        exit 0
      }
      exit 1
    }
  ' "$CLOUDFLARED_LOG_FILE"
}

start_web_tunnel() {
  if [[ -f "$CLOUDFLARED_PID_FILE" ]]; then
    existing_pid="$(cat "$CLOUDFLARED_PID_FILE")"
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "[web] tunnel already running with pid $existing_pid"
      if [[ -f "$CLOUDFLARED_URL_FILE" ]]; then
        echo "[web] public url: $(cat "$CLOUDFLARED_URL_FILE")"
      fi
      return
    fi
    rm -f "$CLOUDFLARED_PID_FILE" "$CLOUDFLARED_URL_FILE"
  fi

  if ! command -v cloudflared >/dev/null 2>&1; then
    echo "[web] cloudflared is not installed or not on PATH" >&2
    exit 1
  fi

  : > "$CLOUDFLARED_LOG_FILE"
  echo "[web] starting cloudflared Quick Tunnel for $CLOUDFLARED_ORIGIN_URL..."
  if command -v setsid >/dev/null 2>&1; then
    setsid cloudflared tunnel \
      --no-autoupdate \
      --url "$CLOUDFLARED_ORIGIN_URL" \
      </dev/null \
      >"$CLOUDFLARED_LOG_FILE" 2>&1 &
  else
    nohup cloudflared tunnel \
      --no-autoupdate \
      --url "$CLOUDFLARED_ORIGIN_URL" \
      </dev/null \
      >"$CLOUDFLARED_LOG_FILE" 2>&1 &
  fi
  cloudflared_pid="$!"
  echo "$cloudflared_pid" > "$CLOUDFLARED_PID_FILE"

  for _ in $(seq 1 "$CLOUDFLARED_STARTUP_TIMEOUT_SECONDS"); do
    if ! kill -0 "$cloudflared_pid" 2>/dev/null; then
      echo "[web] cloudflared exited before publishing a public URL" >&2
      echo "Last 50 cloudflared log lines:" >&2
      tail -n 50 "$CLOUDFLARED_LOG_FILE" >&2 || true
      rm -f "$CLOUDFLARED_PID_FILE" "$CLOUDFLARED_URL_FILE"
      exit 1
    fi

    if public_url="$(extract_cloudflared_url 2>/dev/null)"; then
      echo "$public_url" > "$CLOUDFLARED_URL_FILE"
      echo "[web] tunnel started successfully (pid $cloudflared_pid)"
      echo "[web] public url: $public_url"
      return
    fi
    sleep 1
  done

  echo "[web] failed to receive a public URL within ${CLOUDFLARED_STARTUP_TIMEOUT_SECONDS}s" >&2
  echo "Last 50 cloudflared log lines:" >&2
  tail -n 50 "$CLOUDFLARED_LOG_FILE" >&2 || true
  kill "$cloudflared_pid" 2>/dev/null || true
  rm -f "$CLOUDFLARED_PID_FILE" "$CLOUDFLARED_URL_FILE"
  exit 1
}

# ── main ─────────────────────────────────────────────────────────────────────

verify_python_runtime
ensure_redis
ensure_qdrant

echo "[db] initializing..."
PYTHONPATH=src "$PYTHON" -m scholar_mind.db.init_db

start_app

if [[ "$ENABLE_WEB_TUNNEL" -eq 1 ]]; then
  start_web_tunnel
fi

echo ""
echo "========================================="
echo " ScholarMind is running"
echo "========================================="
echo " App:     $HEALTH_URL  (pid $(cat "$PID_FILE"))"
if [[ "$ENABLE_WEB_TUNNEL" -eq 1 && -f "$CLOUDFLARED_URL_FILE" ]]; then
  echo " Web:     $(cat "$CLOUDFLARED_URL_FILE")  (pid $(cat "$CLOUDFLARED_PID_FILE"))"
fi
echo " Log:     $LOG_FILE"
if [[ "$ENABLE_WEB_TUNNEL" -eq 1 ]]; then
  echo " Web log: $CLOUDFLARED_LOG_FILE"
fi
echo "========================================="
