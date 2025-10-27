# Multi-stage Dockerfile for Core Service (Orchestrator)

# Stage 1: Base Python environment with uv
FROM harbor.in.lazarev.cloud/proxy/python:3.12-slim AS base

# Configure apt to use Nexus proxy repositories (Debian-based)
RUN echo 'deb https://nexus.in.lazarev.cloud/repository/debian-proxy/ trixie main' > /etc/apt/sources.list && \
    echo 'deb https://nexus.in.lazarev.cloud/repository/debian-proxy/ trixie-updates main' >> /etc/apt/sources.list && \
    echo 'deb https://nexus.in.lazarev.cloud/repository/debian-security/ trixie-security main' >> /etc/apt/sources.list

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (ultra-fast Python package manager)
COPY --from=harbor.in.lazarev.cloud/gh-proxy/astral-sh/uv:latest /uv /usr/local/bin/uv

# Configure UV to use system Python and Nexus PyPI proxy
ENV UV_INDEX_URL=https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CREATE_VENV=1

# Stage 2: Python dependencies
FROM base AS python-deps

WORKDIR /app

# Copy pyproject.toml and install dependencies directly to system Python
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
    websockets \
    numpy \
    asyncpg \
    qdrant-client \
    pytz \
    sentence-transformers

# Stage 3: Application build
FROM python-deps AS build

WORKDIR /app

# Copy shared utilities first
COPY shared/ ./shared/

# Copy core service code
COPY core/ ./

# Create necessary directories
RUN mkdir -p logs data/conversations data/models data/models/torch_hub data/models/transformers data/models/huggingface data/models/sentence_transformers

# Stage 4: Runtime image
FROM build AS runtime

WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set model cache directories
ENV TORCH_HOME=/app/data/models/torch_hub
ENV TRANSFORMERS_CACHE=/app/data/models/transformers
ENV HF_HOME=/app/data/models/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/data/models/sentence_transformers

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "main.py"]
