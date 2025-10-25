# Multi-stage Dockerfile for VAD Service (CPU optimized with Silero VAD)

# Stage 1: Base Python environment with uv
FROM harbor.in.lazarev.cloud/proxy/python:3.12-slim AS base

# Configure apt to use Nexus proxy repositories (Debian-based)
RUN echo 'deb https://nexus.in.lazarev.cloud/repository/debian-proxy/ trixie main' > /etc/apt/sources.list && \
    echo 'deb https://nexus.in.lazarev.cloud/repository/debian-proxy/ trixie-updates main' >> /etc/apt/sources.list && \
    echo 'deb https://nexus.in.lazarev.cloud/repository/debian-security/ trixie-security main' >> /etc/apt/sources.list

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (ultra-fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

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
RUN uv pip install silero-vad numpy torch --system

# Stage 3: Application build
FROM python-deps AS build

WORKDIR /app

# Copy shared utilities first
COPY shared/ ./shared/

# Copy VAD service code
COPY services/vad/ ./

# Create necessary directories
RUN mkdir -p logs

# Stage 4: Runtime image
FROM build AS runtime

WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose ports
EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# Run the application
CMD ["python", "main.py"]
