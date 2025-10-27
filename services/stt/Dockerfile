# Optimized Dockerfile for STT Service with CUDA 12.4
# Using CUDA 12.4.1 (verified compatible with faster-whisper)
# Note: Silero VAD is integrated into faster-whisper, not a separate service

# Stage 1: Base CUDA image with system dependencies
FROM harbor.in.lazarev.cloud/proxy/nvidia/cuda:12.4.0-devel-ubuntu22.04 AS base

# Configure apt repositories and install system dependencies
RUN echo 'deb https://nexus.in.lazarev.cloud/repository/ubuntu-group/ jammy main restricted universe multiverse' > /etc/apt/sources.list && \
    echo 'deb https://nexus.in.lazarev.cloud/repository/ubuntu-group/ jammy-updates main restricted universe multiverse' >> /etc/apt/sources.list && \
    echo 'deb https://nexus.in.lazarev.cloud/repository/ubuntu-group/ jammy-backports main restricted universe multiverse' >> /etc/apt/sources.list && \
    echo 'deb https://nexus.in.lazarev.cloud/repository/ubuntu-group/ jammy-security main restricted universe multiverse' >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        curl \
        build-essential \
        ffmpeg \
        libsndfile1 \
        libasound2-dev \
        espeak-ng \
        pkg-config \
        libgomp1 \
        libasound2 \
        wget \
        libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install uv (cached)
COPY --from=harbor.in.lazarev.cloud/gh-proxy/astral-sh/uv:latest /uv /usr/local/bin/uv

# Configure UV and CUDA environment
ENV UV_INDEX_URL=https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CREATE_VENV=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES=0 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH} \
    AUDIODEV=null \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data \
    RUST_BACKTRACE=1

# Stage 2: CUDA dependencies (PyTorch - heavily cached)
FROM base AS cuda-deps

WORKDIR /app

# Copy requirements for CUDA packages (include base for -r reference)
COPY requirements-base.txt requirements-cuda.txt ./

# Install PyTorch and CUDA packages with cache
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements-cuda.txt

# Stage 3: STT-specific dependencies (moderately cached)
FROM cuda-deps AS stt-deps

# Copy STT requirements (include all referenced files)
COPY requirements-base.txt requirements-cuda.txt requirements-stt.txt ./

# Install STT packages with cache (excluding already installed)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-deps faster-whisper soundfile librosa pydub silero-vad websockets aiofiles

# Stage 4: Application code
FROM stt-deps AS app

WORKDIR /app

# Copy shared utilities first
COPY shared/ ./shared/

# Copy service code
COPY services/stt/ ./services/stt/

# Set runtime environment
ENV PYTHONPATH=/app \
    TORCH_HOME=/app/data/models/torch_hub \
    TRANSFORMERS_CACHE=/app/data/models/transformers \
    HF_HOME=/app/data/models/huggingface

# Create directories
RUN mkdir -p logs data/models/{torch_hub,transformers,huggingface}

WORKDIR /app/services/stt

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Run application
CMD ["python", "main.py"]
