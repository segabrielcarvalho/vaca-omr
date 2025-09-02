#!/bin/sh

echo "Starting application..."
set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting Gunicorn server (ASGI/FastAPI)..."
exec gunicorn --reload -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:${PORT:-11001} src.main:app