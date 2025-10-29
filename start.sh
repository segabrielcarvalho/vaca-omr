#!/bin/sh

set -euo pipefail

APP_MODULE="${APP_MODULE:-src.main:app}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-11001}"
LOG_LEVEL="${LOG_LEVEL:-info}"
GRACEFUL_TIMEOUT="${GRACEFUL_TIMEOUT:-30}"
TIMEOUT="${TIMEOUT:-120}"

if command -v nproc >/dev/null 2>&1; then
  DEFAULT_WORKERS="$(nproc)"
else
  DEFAULT_WORKERS="1"
fi

WORKERS="${WORKERS:-$DEFAULT_WORKERS}"

echo "Starting VACA OMR (${APP_ENV:-production})..."

if [ "${RELOAD:-false}" = "true" ] || [ "${APP_ENV:-}" = "development" ]; then
  exec uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT" --reload --log-level "$LOG_LEVEL"
fi

exec gunicorn "$APP_MODULE" \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind "${HOST}:${PORT}" \
  --workers "$WORKERS" \
  --graceful-timeout "$GRACEFUL_TIMEOUT" \
  --timeout "$TIMEOUT" \
  --log-level "$LOG_LEVEL" \
  "$@"
