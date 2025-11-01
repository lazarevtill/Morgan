# Optimized Dockerfile for TTS Service with CUDA 13
# Using CUDA 13.0.1 with cuDNN on Ubuntu 22.04 for csm-streaming with PyTorch 2.5.1
# Reference: https://github.com/davidbrowne17/csm-streaming

# Stage 1: Base CUDA image with system dependencies
FROM harbor.in.lazarev.cloud/proxy/nvidia/cuda:13.0.1-cudnn-devel-ubuntu22.04 AS base

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
        git \
        build-essential \
        ffmpeg \
        libsndfile1 \
        libasound2-dev \
        libportaudio2 \
        portaudio19-dev \
        espeak-ng \
        pkg-config \
        libgomp1 \
        libasound2 \
        wget \
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
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data

# Stage 2: CUDA dependencies (PyTorch - heavily cached)
FROM base AS cuda-deps

WORKDIR /app

# Copy requirements for CUDA packages (include base for -r reference)
COPY requirements-cuda.txt requirements-base.txt ./

# Install PyTorch and CUDA packages with cache
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --index-strategy unsafe-best-match -r requirements-cuda.txt

# Stage 3: TTS-specific dependencies (moderately cached)
FROM cuda-deps AS tts-deps

# Copy TTS requirements (include all referenced files)
COPY requirements-tts.txt requirements-cuda.txt requirements-base.txt ./

# Install TTS packages with cache
# Step 1: Install PyTorch (already done in cuda-deps stage)
# cuda-deps already installs requirements-cuda.txt

# Step 2: Install all TTS requirements (base + cuda + tts-specific)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --index-strategy unsafe-best-match -r requirements-base.txt -r requirements-cuda.txt -r requirements-tts.txt

# CSM streaming models are loaded directly from HuggingFace Hub
# No manual installation needed - using official transformers implementation

# Stage 4: Application code
FROM tts-deps AS app

WORKDIR /app

# Copy shared utilities first
COPY shared/ ./shared/

# Copy service code
COPY services/tts/ ./services/tts/

# Set runtime environment
ENV PYTHONPATH=/app \
    TORCH_HOME=/app/data/models/torch_hub \
    TTS_CACHE_DIR=/app/data/models/tts \
    TRANSFORMERS_CACHE=/app/data/models/transformers \
    HF_HOME=/app/data/models/huggingface

# Create directories
RUN mkdir -p logs data/models/{torch_hub,tts,transformers,huggingface} data/voices

WORKDIR /app/services/tts

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run application
CMD ["python", "main.py"]
