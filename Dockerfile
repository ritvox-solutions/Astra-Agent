# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.13

FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1

# ==========================
# Build Stage
# ==========================
FROM base AS build

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

RUN mkdir -p src

# Install dependencies
RUN uv sync --locked

# Copy application
COPY . .

# Download models/assets during build (optional)
RUN uv run src/agent.py download-files

# ==========================
# Runtime Stage
# ==========================
FROM base

ARG UID=10001

RUN adduser \
    --disabled-password \
    --gecos "" \
    --home /app \
    --shell /usr/sbin/nologin \
    --uid ${UID} \
    appuser

WORKDIR /app

COPY --from=build --chown=appuser:appuser /app /app

USER appuser

CMD ["uv", "run", "src/agent.py", "start"]