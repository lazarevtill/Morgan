# Optimized Dockerfile for LLM Service (No CUDA needed - API client only)
# syntax=docker/dockerfile:1.4

# Stage 1: Base with system dependencies
FROM harbor.in.lazarev.cloud/proxy/python:3.12-slim AS base

# Configure apt repositories and install system dependencies
RUN echo 'deb https://nexus.in.lazarev.cloud/repository/debian-proxy/ trixie main' > /etc/apt/sources.list && \
    echo 'deb https://nexus.in.lazarev.cloud/repository/debian-proxy/ trixie-updates main' >> /etc/apt/sources.list && \
    echo 'deb https://nexus.in.lazarev.cloud/repository/debian-security/ trixie-security main' >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (single layer, cached)
COPY --from=harbor.in.lazarev.cloud/gh-proxy/astral-sh/uv:latest /uv /usr/local/bin/uv

# Configure UV
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
COPY requirements-base.txt requirements-llm.txt ./

# Install dependencies with cache mount
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements-llm.txt

# Stage 3: Application code
FROM deps AS app

WORKDIR /app

# Copy shared utilities first
COPY shared/ ./shared/

# Copy service code
COPY services/llm/ ./services/llm/

# Set runtime environment
ENV PYTHONPATH=/app \
    TORCH_HOME=/app/data/models/torch_hub \
    TRANSFORMERS_CACHE=/app/data/models/transformers \
    HF_HOME=/app/data/models/huggingface

# Create directories
RUN mkdir -p logs data/models/{torch_hub,transformers,huggingface}

WORKDIR /app/services/llm

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run application
CMD ["python", "main.py"]
