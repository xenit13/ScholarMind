#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PID_FILE="${PID_FILE:-$ROOT_DIR/data/run/scholar_mind.pid}"
WORKER_PID_FILE="${WORKER_PID_FILE:-$ROOT_DIR/data/run/scholar_mind_worker.pid}"
SCHEDULER_PID_FILE="${SCHEDULER_PID_FILE:-$ROOT_DIR/data/run/scholar_mind_scheduler.pid}"
CLOUDFLARED_PID_FILE="${CLOUDFLARED_PID_FILE:-$ROOT_DIR/data/run/cloudflared.pid}"
CLOUDFLARED_URL_FILE="${CLOUDFLARED_URL_FILE:-$ROOT_DIR/data/run/cloudflared.url}"
REDIS_MARKER_FILE="${REDIS_MARKER_FILE:-$ROOT_DIR/data/run/redis.started_by_deploy}"
QDRANT_MARKER_FILE="${QDRANT_MARKER_FILE:-$ROOT_DIR/data/run/qdrant.started_by_deploy}"
STOP_TIMEOUT_SECONDS="${STOP_TIMEOUT_SECONDS:-10}"

STOP_WEB_TUNNEL=0

usage() {
  cat <<EOF
Usage: bash scripts/stop.sh [--web]

Options:
  --web   Stop the cloudflared Quick Tunnel along with the local services.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --web)
      STOP_WEB_TUNNEL=1
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

# ── helpers ──────────────────────────────────────────────────────────────────

docker_cmd() {
  if command -v podman-compose >/dev/null 2>&1; then
    podman-compose "$@"
  elif command -v docker >/dev/null 2>&1; then
    docker compose "$@"
  else
    return 1
  fi
}

stop_process() {
  local name="$1" pid_file="$2"

  if [[ ! -f "$pid_file" ]]; then
    echo "[$name] not running (no pid file)"
    return
  fi

  local pid
  pid="$(cat "$pid_file")"
  if [[ -z "$pid" ]]; then
    rm -f "$pid_file"
    echo "[$name] removed empty pid file"
    return
  fi

  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$pid_file"
    echo "[$name] removed stale pid file for $pid"
    return
  fi

  echo "[$name] stopping (pid $pid)..."
  kill "$pid"

  for _ in $(seq 1 "$STOP_TIMEOUT_SECONDS"); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$pid_file"
      echo "[$name] stopped"
      return
    fi
    sleep 1
  done

  kill -9 "$pid" 2>/dev/null || true
  rm -f "$pid_file"
  echo "[$name] force stopped"
}

stop_infra_if_started() {
  local name="$1" marker_file="$2"

  if [[ ! -f "$marker_file" ]]; then
    return
  fi

  echo "[$name] stopping..."
  docker_cmd stop "$name" >/dev/null 2>&1 || true
  rm -f "$marker_file"
  echo "[$name] stopped"
}

stop_web_tunnel() {
  if [[ -f "$CLOUDFLARED_PID_FILE" ]]; then
    stop_process "web" "$CLOUDFLARED_PID_FILE"
    rm -f "$CLOUDFLARED_URL_FILE"
    return
  fi

  rm -f "$CLOUDFLARED_URL_FILE"
  if [[ "$STOP_WEB_TUNNEL" -eq 1 ]]; then
    echo "[web] not running (no pid file)"
  fi
}

# ── stop in reverse order ────────────────────────────────────────────────────

echo "Stopping ScholarMind..."
if [[ "$STOP_WEB_TUNNEL" -eq 1 || -f "$CLOUDFLARED_PID_FILE" || -f "$CLOUDFLARED_URL_FILE" ]]; then
  stop_web_tunnel
fi
stop_process "scheduler" "$SCHEDULER_PID_FILE"
stop_process "worker" "$WORKER_PID_FILE"
stop_process "app" "$PID_FILE"
stop_infra_if_started "qdrant" "$QDRANT_MARKER_FILE"
stop_infra_if_started "redis" "$REDIS_MARKER_FILE"

echo ""
echo "ScholarMind fully stopped."
