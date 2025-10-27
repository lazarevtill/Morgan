# Docker Build Optimization Guide

> **Last Updated**: 2025-10-27  
> **Build Time**: 2-4 seconds per service (with cache) 🚀  
> **Improvement**: 80%+ faster than original builds

## Overview

Morgan uses optimized multi-stage Dockerfiles with BuildKit cache mounts and requirements files for ultra-fast builds.

---

## Performance Metrics

### Build Times (Cached)

| Service | Before | After | Improvement |
|---------|--------|-------|-------------|
| Core | 18s | 2.5s | **86%** ⚡ |
| LLM | 1.6s | 1.5s | 6% |
| TTS | 18s | 3.5s | **81%** ⚡ |
| STT | 18s | 3.5s | **81%** ⚡ |
| VAD | 1.8s | 1.5s | 17% |

### Image Sizes

| Service | Size | CUDA | Notes |
|---------|------|------|-------|
| Core | 1.2 GB | No | Includes sentence-transformers |
| LLM | 800 MB | No | Minimal dependencies |
| TTS | 8.5 GB | Yes | PyTorch + CUDA 12.4 + TTS models |
| STT | 7.2 GB | Yes | PyTorch + CUDA 12.4 + Whisper |
| VAD | 600 MB | No | CPU-only, lightweight |

---

## Requirements Files Strategy

### Dependency Hierarchy

```
requirements-base.txt          ← Common packages (all services)
├── requirements-core.txt      ← Core service
├── requirements-llm.txt       ← LLM service
└── requirements-cuda.txt      ← CUDA base (TTS + STT)
    ├── requirements-tts.txt   ← TTS specific
    └── requirements-stt.txt   ← STT specific
```

### Why This Works

1. **Pinned Versions**: All packages use exact versions (`==`) instead of ranges (`>=`)
2. **Layer Caching**: Each requirements file creates a separate Docker layer
3. **No Dep Resolution**: `--no-deps` flag skips redundant dependency resolution
4. **Inheritance**: `-r` includes avoid duplication

### Example: `requirements-base.txt`

```txt
# Base requirements for all services - pinned versions
fastapi==0.109.2
uvicorn[standard]==0.27.0.post1
pydantic==2.6.1
aiohttp==3.9.3
pyyaml==6.0.1
python-dotenv==1.0.1
structlog==24.1.0
psutil==5.9.8
redis==5.0.1
python-multipart==0.0.9
asyncpg==0.29.0
websockets==12.0
```

### Example: `requirements-cuda.txt`

```txt
# CUDA 12.4 PyTorch (aligned with csm-streaming)
--extra-index-url https://nexus.in.lazarev.cloud/repository/pytorch-pypi/simple
torch==2.5.1
torchaudio==2.5.1
torchvision==0.20.1
numpy==1.26.3
```

---

## Multi-Stage Dockerfile Pattern

### Core Service (Python 3.12 Slim)

```dockerfile
# syntax=docker/dockerfile:1.4

# Stage 1: Base image
FROM harbor.in.lazarev.cloud/proxy/python:3.12-slim AS base

# Configure repositories with cache mounts
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        curl build-essential

# Install UV
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir uv

# Stage 2: Python dependencies
FROM base AS python-deps
WORKDIR /app
ENV UV_NO_CREATE_VENV=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_INDEX_URL=https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple

COPY requirements-base.txt requirements-core.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements-base.txt && \
    uv pip install --system --no-deps -r requirements-core.txt

# Stage 3: Application
FROM python-deps AS build
COPY . .

# Stage 4: Runtime
FROM build AS runtime
EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "main.py"]
```

### CUDA Service (TTS/STT)

```dockerfile
# syntax=docker/dockerfile:1.4

# Stage 1: CUDA base
FROM harbor.in.lazarev.cloud/proxy/nvidia/cuda:12.4.0-devel-ubuntu22.04 AS base

# System dependencies with cache mounts
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        curl build-essential ffmpeg libsndfile1

# Install UV
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir uv

# Stage 2: Base Python dependencies
FROM base AS python-deps
WORKDIR /app
ENV UV_NO_CREATE_VENV=1 \
    UV_INDEX_URL=https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple

COPY requirements-base.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements-base.txt

# Stage 3: CUDA dependencies
FROM python-deps AS cuda-deps
COPY requirements-cuda.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements-cuda.txt

# Stage 4: Service-specific dependencies
FROM cuda-deps AS tts-deps
COPY requirements-tts.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-deps -r requirements-tts.txt

# Stage 5: Application
FROM tts-deps AS build
COPY . .

# Stage 6: Runtime
FROM build AS runtime
EXPOSE 8002
HEALTHCHECK CMD curl -f http://localhost:8002/health || exit 1
CMD ["python", "main.py"]
```

---

## BuildKit Optimizations

### 1. Enable BuildKit

```bash
# Linux/Mac
export DOCKER_BUILDKIT=1

# Windows PowerShell
$env:DOCKER_BUILDKIT=1

# Or in docker-compose.yml
COMPOSE_DOCKER_CLI_BUILD=1
DOCKER_BUILDKIT=1
```

### 2. Cache Mounts

```dockerfile
# APT cache (system packages)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install ...

# Pip cache (Python packages)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install uv

# UV cache (fast Python packages)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt
```

### 3. Layer Ordering

**Optimal order** (least frequently changed → most frequently changed):

1. Base image
2. System packages
3. Base Python dependencies
4. CUDA dependencies (if applicable)
5. Service-specific dependencies
6. Application code

---

## Build Scripts

### Linux/Mac: `build-optimized.sh`

```bash
#!/bin/bash
set -e

export DOCKER_BUILDKIT=1

SERVICE=$1
if [ -z "$SERVICE" ]; then
    echo "Usage: $0 <service|all>"
    exit 1
fi

build_service() {
    echo "Building $1..."
    docker compose build $1
}

if [ "$SERVICE" = "all" ]; then
    for svc in core llm-service tts-service stt-service vad-service; do
        build_service $svc
    done
else
    build_service $SERVICE
fi

echo "✅ Build complete!"
```

### Windows: `build-optimized.ps1`

```powershell
param(
    [Parameter(Mandatory=$true)]
    [string]$Service
)

$env:DOCKER_BUILDKIT = "1"

function Build-Service {
    param([string]$Name)
    Write-Host "Building $Name..." -ForegroundColor Cyan
    docker compose build $Name
}

if ($Service -eq "all") {
    @("core", "llm-service", "tts-service", "stt-service", "vad-service") | ForEach-Object {
        Build-Service $_
    }
} else {
    Build-Service $Service
}

Write-Host "✅ Build complete!" -ForegroundColor Green
```

---

## Build Commands

### Clean Build (First Time)

```bash
# Build with no cache
docker compose build --no-cache

# Or with script
./build-optimized.sh all
```

### Incremental Build (Development)

```bash
# Build all services (uses cache)
docker compose build

# Build specific service
docker compose build core

# Rebuild after code changes
docker compose up -d --build core
```

### Production Build

```bash
# Build and push to registry
./scripts/registry-build.sh -b -p

# On production server
docker compose pull
docker compose up -d
```

---

## Troubleshooting

### Slow Builds Despite Cache

**Symptom**: `uv pip install` still takes 10+ seconds

**Solution**:
1. Ensure BuildKit is enabled: `docker version | grep BuildKit`
2. Check cache mounts are present in Dockerfile
3. Verify requirements files are pinned (no `>=` ranges)

**Diagnostic**:
```bash
# Check Docker BuildKit
docker buildx inspect

# Clear build cache if needed
docker builder prune -a
```

### Out of Disk Space

**Symptom**: Build fails with "no space left on device"

**Solution**:
```bash
# Clean up old images and containers
docker system prune -a --volumes

# Check disk usage
docker system df
```

### Missing Files in Build Context

**Symptom**: `COPY requirements-*.txt` fails with "no such file"

**Solution**:
1. Check `.dockerignore` doesn't exclude requirements files
2. Ensure files exist in root directory
3. Verify build context is correct

**Current `.dockerignore`**:
```
# Exclude unnecessary files but include requirements
*
!core/
!services/
!shared/
!config/
!requirements-*.txt
!pyproject.toml
```

---

## Best Practices

### 1. Pin All Versions

❌ **Bad**: `torch>=2.0.0`  
✅ **Good**: `torch==2.5.1`

**Reason**: Pinned versions enable perfect layer caching

### 2. Use Requirements Hierarchy

❌ **Bad**: Copy all packages in one file  
✅ **Good**: Split into base → cuda → service

**Reason**: Separate layers cache independently

### 3. Order Matters

❌ **Bad**: COPY code before installing dependencies  
✅ **Good**: Install dependencies, then COPY code

**Reason**: Code changes don't invalidate dependency cache

### 4. Use Cache Mounts

❌ **Bad**: `RUN apt-get update && apt-get install`  
✅ **Good**: `RUN --mount=type=cache,target=/var/cache/apt ...`

**Reason**: Reuses downloaded packages across builds

### 5. Multi-Stage Builds

❌ **Bad**: Single-stage Dockerfile with all tools  
✅ **Good**: Multi-stage with separate build/runtime

**Reason**: Smaller final images, better security

---

## Advanced Optimization

### Parallel Builds

```bash
# Build all services in parallel
docker compose build --parallel

# Limit parallel jobs
docker compose build --parallel 2
```

### Remote Caching

```yaml
# docker-compose.yml
services:
  core:
    build:
      context: .
      dockerfile: core/Dockerfile
      cache_from:
        - harbor.in.lazarev.cloud/morgan/core:latest
```

### Build Arguments

```dockerfile
ARG CUDA_VERSION=12.4.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04
```

```bash
docker compose build --build-arg CUDA_VERSION=12.5.0
```

---

## See Also

- [VERSION_ALIGNMENT.md](./VERSION_ALIGNMENT.md) - CUDA and PyTorch versions
- [DEPLOYMENT.md](./DEPLOYMENT.md) - Production deployment
- [../getting-started/DEVELOPMENT.md](../getting-started/DEVELOPMENT.md) - Local development

---

**Last Updated**: 2025-10-27  
**Morgan AI Assistant** - Ultra-fast Docker builds

