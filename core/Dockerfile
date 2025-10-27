# Optimized Multi-stage Dockerfile for Core Service
# syntax=docker/dockerfile:1.4

# Stage 1: Base with system dependencies
FROM harbor.in.lazarev.cloud/proxy/python:3.12-slim AS base

# Configure apt repositories and install system dependencies
RUN echo 'deb https://nexus.in.lazarev.cloud/repository/debian-proxy/ trixie main' > /etc/apt/sources.list && \
    echo 'deb https://nexus.in.lazarev.cloud/repository/debian-proxy/ trixie-updates main' >> /etc/apt/sources.list && \
    echo 'deb https://nexus.in.lazarev.cloud/repository/debian-security/ trixie-security main' >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (single layer, cached)
COPY --from=harbor.in.lazarev.cloud/gh-proxy/astral-sh/uv:latest /uv /usr/local/bin/uv

# Configure UV (environment only, very fast)
ENV UV_INDEX_URL=https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CREATE_VENV=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Stage 2: Dependencies only (heavily cached)
FROM base AS deps

WORKDIR /app

# Copy requirements files (include base for -r reference)
COPY requirements-base.txt requirements-core.txt ./

# Install dependencies with cache mount
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements-core.txt

# Stage 3: Application code
FROM deps AS app

WORKDIR /app

# Copy shared utilities first (changes less frequently)
COPY shared/ ./shared/

# Copy core service code (changes more frequently)
COPY core/ ./core/

# Set runtime environment
ENV PYTHONPATH=/app \
    TORCH_HOME=/app/data/models/torch_hub \
    TRANSFORMERS_CACHE=/app/data/models/transformers \
    HF_HOME=/app/data/models/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/data/models/sentence_transformers

# Create necessary directories
RUN mkdir -p logs data/conversations data/models/{torch_hub,transformers,huggingface,sentence_transformers}

WORKDIR /app/core

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "main.py"]
