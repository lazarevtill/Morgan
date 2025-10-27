# Multi-stage Dockerfile for STT Service with CUDA 13 support
FROM harbor.in.lazarev.cloud/proxy/nvidia/cuda:13.0.1-devel-ubuntu22.04

# Configure apt to use Nexus proxy repositories (Ubuntu 22.04 - Jammy)
RUN echo 'deb https://nexus.in.lazarev.cloud/repository/ubuntu-group/ jammy main restricted universe multiverse' > /etc/apt/sources.list && \
    echo 'deb https://nexus.in.lazarev.cloud/repository/ubuntu-group/ jammy-updates main restricted universe multiverse' >> /etc/apt/sources.list && \
    echo 'deb https://nexus.in.lazarev.cloud/repository/ubuntu-group/ jammy-backports main restricted universe multiverse' >> /etc/apt/sources.list && \
    echo 'deb https://nexus.in.lazarev.cloud/repository/ubuntu-group/ jammy-security main restricted universe multiverse' >> /etc/apt/sources.list

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libasound2-dev \
    espeak \
    espeak-ng \
    pkg-config \
    libgomp1 \
    alsa-utils \
    libasound2 \
    software-properties-common \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install cuDNN (required for PyTorch CUDA operations)
RUN apt-get update && apt-get install -y \
    libcudnn9-dev-cuda-13 \
    libcudnn9-cuda-13 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (ultra-fast Python package manager)
COPY --from=harbor.in.lazarev.cloud/gh-proxy/astral-sh/uv:latest /uv /usr/local/bin/uv

# Configure UV to use system Python and Nexus PyPI proxy
ENV UV_INDEX_URL=https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CREATE_VENV=1

# Copy pyproject.toml and install common dependencies
WORKDIR /app
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
    torch \
    torchaudio \
    python-multipart

# Copy shared utilities
COPY shared/ ./shared/

# Set environment variables for CUDA and audio
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV AUDIODEV=null
ENV ESPEAK_DATA_PATH=/usr/share/espeak-ng-data

# Install STT-specific dependencies
RUN uv pip install --system \
    faster-whisper \
    soundfile \
    librosa \
    pydub \
    silero-vad \
    websockets \
    aiofiles \
    python-multipart \
    numpy \
    torch \
    torchaudio

# Copy STT service code
COPY services/stt/ ./

# Create necessary directories
RUN mkdir -p logs data/models data/models/torch_hub data/models/transformers data/models/huggingface

# Set environment variables for STT
ENV RUST_BACKTRACE=1

# Set model cache directories
ENV TORCH_HOME=/app/data/models/torch_hub
ENV TRANSFORMERS_CACHE=/app/data/models/transformers
ENV HF_HOME=/app/data/models/huggingface

# Expose ports
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Run the application
CMD ["python", "main.py"]
