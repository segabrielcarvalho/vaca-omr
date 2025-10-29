FROM python:3.12.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENCV_OPENCL_RUNTIME=disabled \
    PYTHONPATH="/app/src" \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS deps

COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip wheel \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

FROM deps AS build

COPY src ./src
RUN python -m compileall src

FROM base AS runtime

COPY --from=deps /opt/venv /opt/venv
COPY --from=build /app/src ./src
COPY start.sh ./start.sh

RUN chmod +x /app/start.sh

EXPOSE 11001

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 CMD \
    python -c "import sys, urllib.request; \
try: \
    sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:11001/api/v1/health', timeout=2).status == 200 else 1) \
except Exception: \
    sys.exit(1)"

CMD ["/bin/sh", "/app/start.sh"]

FROM runtime AS staging

ENV APP_ENV=staging

FROM runtime AS production

ENV APP_ENV=production

FROM base AS development

COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip wheel \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x /app/start.sh

ENV APP_ENV=development

EXPOSE 11001

CMD ["/bin/sh", "/app/start.sh"]
