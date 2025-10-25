# Multi-stage Dockerfile for STT Service with CUDA 13 support

# Stage 1: Base CUDA development environment
FROM harbor.in.lazarev.cloud/proxy/nvidia/cuda:13.0.1-devel-ubuntu22.04 AS cuda-base

# Configure apt to use Nexus proxy repositories (Ubuntu 22.04 - Jammy)
RUN echo 'deb https://nexus.in.lazarev.cloud/repository/ubuntu-group/ jammy main restricted universe multiverse' > /etc/apt/sources.list && \
    echo 'deb https://nexus.in.lazarev.cloud/repository/ubuntu-group/ jammy-updates main restricted universe multiverse' >> /etc/apt/sources.list && \
    echo 'deb https://nexus.in.lazarev.cloud/repository/ubuntu-group/ jammy-backports main restricted universe multiverse' >> /etc/apt/sources.list && \
    echo 'deb https://nexus.in.lazarev.cloud/repository/ubuntu-group/ jammy-security main restricted universe multiverse' >> /etc/apt/sources.list

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libasound2-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install uv (ultra-fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Configure UV to use system Python and Nexus PyPI proxy
ENV UV_INDEX_URL=https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CREATE_VENV=1

# Install Rust (required for faster-whisper)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Stage 2: Python dependencies
FROM cuda-base AS python-deps

WORKDIR /app

# Copy pyproject.toml and install dependencies directly to system Python
COPY pyproject.toml .
RUN uv pip install faster-whisper soundfile librosa pydub torch torchaudio silero-vad --system

# Stage 3: Application build
FROM python-deps AS build

WORKDIR /app

# Copy shared utilities first
COPY shared/ ./shared/

# Copy STT service code
COPY services/stt/ ./

# Create necessary directories
RUN mkdir -p logs data/models

# Stage 4: Runtime image
FROM build AS runtime

WORKDIR /app

# Set environment variables for CUDA and audio
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV RUST_BACKTRACE=1

# Audio environment variables
ENV AUDIODEV=null

# Expose ports
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Run the application
CMD ["python", "main.py"]
