# Multi-stage Dockerfile for LLM Service (OpenAI-compatible Ollama client)
# This service acts as a client to external Ollama, no CUDA needed

# Use shared base image
FROM harbor.in.lazarev.cloud/proxy/python:3.12-slim AS base

# Configure apt to use Nexus proxy repositories (Debian-based)
RUN echo 'deb https://nexus.in.lazarev.cloud/repository/debian-proxy/ trixie main' > /etc/apt/sources.list && \
    echo 'deb https://nexus.in.lazarev.cloud/repository/debian-proxy/ trixie-updates main' >> /etc/apt/sources.list && \
    echo 'deb https://nexus.in.lazarev.cloud/repository/debian-security/ trixie-security main' >> /etc/apt/sources.list

# Install common system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (ultra-fast Python package manager)
COPY --from=harbor.in.lazarev.cloud/gh-proxy/astral-sh/uv:latest /uv /usr/local/bin/uv

# Configure UV to use system Python and Nexus PyPI proxy
ENV UV_INDEX_URL=https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CREATE_VENV=1

# Python dependencies stage
FROM base AS python-deps

WORKDIR /app

# Copy pyproject.toml and install LLM-specific dependencies
COPY pyproject.toml .
RUN uv pip install --system \
    fastapi \
    uvicorn[standard] \
    pydantic \
    aiohttp \
    pyyaml \
    python-dotenv \
    structlog \
    psutil \
    redis \
    python-multipart \
    openai

# Copy shared utilities
COPY shared/ ./shared/

# Application build
FROM python-deps AS build

WORKDIR /app

# Copy LLM service code
COPY services/llm/ ./

# Create necessary directories
RUN mkdir -p logs data/models data/models/torch_hub data/models/transformers data/models/huggingface

# Set common environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set model cache directories
ENV TORCH_HOME=/app/data/models/torch_hub
ENV TRANSFORMERS_CACHE=/app/data/models/transformers
ENV HF_HOME=/app/data/models/huggingface

# Expose ports
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the LLM API service
CMD ["python", "main.py"]
