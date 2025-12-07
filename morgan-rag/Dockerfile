# Multi-stage build for optimal image size
FROM harbor.in.lazarev.cloud/proxy/python:3.11-slim as builder

# Configure apt sources - use standard Debian repos for faster builds
RUN echo "deb http://deb.debian.org/debian bookworm main contrib non-free" > /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian-security bookworm-security main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian bookworm-updates main contrib non-free" >> /etc/apt/sources.list

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager (pinned version for deterministic builds)
COPY --from=harbor.in.lazarev.cloud/gh-proxy/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

# Configure uv to use standard PyPI repositories
ENV UV_INDEX_URL=https://pypi.org/simple \
    UV_NO_CACHE=1

# Upgrade pip to fix CVE-2025-8869
RUN uv pip install --system "pip>=25.2"

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Use CPU-only PyTorch wheels to keep image smaller and avoid CUDA downloads
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM harbor.in.lazarev.cloud/proxy/python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app user for security
RUN groupadd -r morgan && useradd -r -g morgan morgan

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/config /app/knowledge && \
    chown -R morgan:morgan /app

# Switch to non-root user
USER morgan

# Set Python path
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -m morgan health || exit 1

# Expose ports
EXPOSE 8080 8000 9000

# Default command
CMD ["python", "-m", "morgan", "serve", "--host", "0.0.0.0", "--port", "8080"]
