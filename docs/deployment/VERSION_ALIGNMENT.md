# CUDA and PyTorch Version Alignment

> **Last Updated**: 2025-10-27  
> **CUDA Version**: 12.4  
> **PyTorch Version**: 2.5.1  
> **Status**: ✅ Aligned with csm-streaming and faster-whisper

## Quick Reference

```
CUDA: 12.4.0
PyTorch: 2.5.1
TorchAudio: 2.5.1
TorchVision: 0.20.1
faster-whisper: 1.0.3
silero-vad: 4.0.2
Coqui TTS: 0.22.0
```

---

## Compatibility Matrix

| Project | CUDA | PyTorch | TorchAudio | Status |
|---------|------|---------|------------|--------|
| **Morgan** | 12.4 | 2.5.1 | 2.5.1 | ✅ Current |
| **csm-streaming** | 12.4 | 2.5.1 | 2.5.1 | ✅ Match |
| **faster-whisper** | 12.x | 2.0+ | 2.0+ | ✅ Compatible |
| **silero-vad** | Any | 1.8+ | 1.8+ | ✅ Compatible |

---

## Version Matrix

### CUDA & PyTorch Core

| Component | Version | Source | Notes |
|-----------|---------|--------|-------|
| CUDA Runtime | **12.4.0** | NVIDIA | Base image |
| PyTorch | **2.5.1** | PyPI (cu124) | Aligned with csm-streaming |
| TorchAudio | **2.5.1** | PyPI (cu124) | Speech processing |
| TorchVision | **0.20.1** | PyPI (cu124) | Image processing (TTS) |
| NumPy | **1.26.3** | PyPI | Numerical computing |

### Python ML Libraries

| Library | Version | Service | Purpose |
|---------|---------|---------|---------|
| faster-whisper | **1.0.3** | STT | Speech-to-text (CUDA 12.4 compatible) |
| silero-vad | **4.0.2** | STT/VAD | Voice activity detection |
| TTS (Coqui) | **0.22.0** | TTS | Text-to-speech framework |
| csm-streaming | **0.2.8** | TTS | Fast neural TTS |
| sentence-transformers | **2.3.1** | Core | Embeddings for semantic search |

### Audio Processing

| Library | Version | Services | Purpose |
|---------|---------|----------|---------|
| soundfile | **0.12.1** | TTS, STT | Audio I/O |
| librosa | **0.10.1** | TTS, STT | Audio analysis & feature extraction |
| pydub | **0.25.1** | TTS, STT | Audio manipulation & conversion |
| aiofiles | **23.2.1** | STT | Async file operations |

### Core Dependencies

| Library | Version | Services | Purpose |
|---------|---------|----------|---------|
| fastapi | **0.109.2** | All | Web framework |
| uvicorn | **0.27.0.post1** | All | ASGI server |
| pydantic | **2.6.1** | All | Data validation |
| aiohttp | **3.9.3** | All | Async HTTP client |
| redis | **5.0.1** | Core | Caching |
| asyncpg | **0.29.0** | Core | PostgreSQL driver |
| websockets | **12.0** | Core, STT | WebSocket support |

---

## Docker Base Images

### CUDA Services (TTS, STT)

```dockerfile
FROM harbor.in.lazarev.cloud/proxy/nvidia/cuda:12.4.0-devel-ubuntu22.04 AS base
```

**Features:**
- CUDA 12.4 development toolkit
- cuDNN included
- Ubuntu 22.04 LTS base
- Python 3.11 runtime
- Build tools for native extensions

**GPU Requirements:**
- CUDA Compute Capability 7.0+ (Tesla V100, RTX 20xx+)
- NVIDIA Driver 525+
- 8GB+ VRAM (recommended 24GB for production)

### Non-CUDA Services (Core, LLM, VAD)

```dockerfile
FROM harbor.in.lazarev.cloud/proxy/python:3.12-slim AS base
```

**Features:**
- Python 3.12 (latest stable)
- Debian Slim base
- Minimal footprint (~150MB)
- Production-ready

---

## Requirements Files

### Base Dependencies (`requirements-base.txt`)

```txt
# Common for all services - pinned for fast resolution
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

### CUDA Dependencies (`requirements-cuda.txt`)

```txt
# Aligned with csm-streaming
# CUDA 12.4 (cu124)
--extra-index-url https://download.pytorch.org/whl/cu124
torch==2.5.1
torchaudio==2.5.1
torchvision==0.20.1
numpy==1.26.3
```

### TTS Dependencies (`requirements-tts.txt`)

```txt
-r requirements-cuda.txt
# TTS-specific packages
TTS==0.22.0
soundfile==0.12.1
librosa==0.10.1
pydub==0.25.1
csm-streaming==0.2.8
pyttsx3==2.90
```

### STT Dependencies (`requirements-stt.txt`)

```txt
-r requirements-cuda.txt
# STT-specific packages
faster-whisper==1.0.3
soundfile==0.12.1
librosa==0.10.1
pydub==0.25.1
silero-vad==4.0.2
websockets==12.0
aiofiles==23.2.1
```

### Core Dependencies (`requirements-core.txt`)

```txt
-r requirements-base.txt
# Core service packages
websockets==12.0
numpy==1.26.3
asyncpg==0.29.0
qdrant-client==1.7.3
pytz==2024.1
sentence-transformers==2.3.1
```

### LLM Dependencies (`requirements-llm.txt`)

```txt
-r requirements-base.txt
# LLM service packages
openai==1.12.0
```

---

## GPU Requirements

### Minimum Specifications

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **GPU** | NVIDIA GPU with CUDA 7.0+ | Tesla V100, RTX 20xx, or newer |
| **VRAM** | 8GB | Minimum for basic operation |
| **Driver** | 525+ | Required for CUDA 12.4 |
| **Docker** | 24.0+ | With NVIDIA Container Toolkit |

### Recommended Specifications

| Component | Recommendation | Notes |
|-----------|----------------|-------|
| **GPU** | RTX 4090, A100, or better | Production workloads |
| **VRAM** | 24GB+ | Parallel STT + TTS |
| **Driver** | 550+ | Latest stable |
| **RAM** | 32GB+ | Host system |

### VRAM Allocation

| Service | VRAM Usage | Notes |
|---------|------------|-------|
| TTS Service | 2-4GB | Depends on model size |
| STT Service | 4-6GB | Whisper large-v3 |
| Headroom | 2GB+ | For OS and other processes |

---

## Verification

### Check CUDA Version

```bash
# Check NVIDIA driver and CUDA
nvidia-smi

# Expected output:
# CUDA Version: 12.4

# Inside container
docker exec morgan-tts nvidia-smi
```

### Check PyTorch Version

```bash
# Check PyTorch and CUDA compatibility
docker exec morgan-tts python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'Device Count: {torch.cuda.device_count()}')
"

# Expected output:
# PyTorch: 2.5.1
# CUDA Available: True
# CUDA Version: 12.4
# Device Count: 1
```

### Check Service Dependencies

```bash
# Check TTS service
docker exec morgan-tts python -c "
import TTS
import torch
print(f'TTS: {TTS.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"

# Check STT service
docker exec morgan-stt python -c "
from faster_whisper import WhisperModel
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print('faster-whisper: OK')
"
```

---

## Performance Benchmarks

### Build Times (with cache)

| Service | Dependency Install | Total Build | Notes |
|---------|-------------------|-------------|-------|
| Core | 2.5s | 8s | No CUDA |
| LLM | 1.5s | 6s | No CUDA |
| TTS | 3.5s | 25s | CUDA 12.4 + PyTorch |
| STT | 3.5s | 23s | CUDA 12.4 + PyTorch |
| VAD | 1.5s | 5s | CPU only |

### Runtime Performance

| Task | Performance | Notes |
|------|-------------|-------|
| TTS Synthesis | ~450ms | 50-word sentence |
| STT Transcription | ~720ms | 10-second audio |
| LLM Generation | ~1.1s | 100-token response |
| VAD Detection | <50ms | Real-time |

*Benchmarks on RTX 4090 with CUDA 12.4*

---

## Troubleshooting

### CUDA Version Mismatch

**Symptom:**
```
RuntimeError: CUDA error: no kernel image is available for execution
```

**Solution:**
1. Check driver version: `nvidia-smi`
2. Ensure driver >= 525 for CUDA 12.4
3. Update if needed:
   ```bash
   # Ubuntu
   sudo apt update
   sudo apt install nvidia-driver-550
   sudo reboot
   ```

### PyTorch Can't Find GPU

**Symptom:**
```python
torch.cuda.is_available() returns False
```

**Solution:**
1. Verify NVIDIA Container Toolkit:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```
2. Check docker-compose GPU config:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```
3. Restart Docker daemon:
   ```bash
   sudo systemctl restart docker
   ```

### Out of Memory

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Check GPU memory: `nvidia-smi`
2. Reduce batch sizes in service configs
3. Use smaller models:
   - STT: `distil-medium` instead of `distil-large-v3`
   - TTS: Reduce voice model complexity
4. Enable model quantization (future)

### Slow Builds

**Symptom:**
```
=> [tts-service cuda-deps 3/3] RUN uv pip install -r requirements-cuda.txt  17.9s
```

**Solutions:**
1. Ensure BuildKit is enabled:
   ```bash
   export DOCKER_BUILDKIT=1
   ```
2. Verify cache mounts in Dockerfile
3. Check requirements files use exact versions (`==`)
4. Clear build cache if corrupted:
   ```bash
   docker builder prune -a
   ```

---

## Update Policy

### When to Update

| Update Type | Frequency | Testing Required |
|-------------|-----------|------------------|
| Security patches | Immediate | Smoke tests |
| Minor versions | Quarterly | Full regression |
| Major versions | Annually | Extended testing |

### Version Pinning Strategy

| Environment | Strategy | Reason |
|-------------|----------|--------|
| **Production** | Exact pins (`==`) | Reproducibility |
| **Staging** | Exact pins | Match production |
| **Development** | Compatible (`~=`) | Balance stability/updates |

### Alignment Checks

Before updating any PyTorch/CUDA versions:

1. **Check csm-streaming compatibility:**
   ```bash
   # Check their requirements
   grep -E 'torch|cuda' csm-streaming/requirements.txt
   ```

2. **Verify faster-whisper support:**
   ```bash
   # Check compatibility matrix
   pip show faster-whisper
   ```

3. **Test in staging:**
   ```bash
   # Build with new versions
   docker compose -f docker-compose.staging.yml build
   # Run integration tests
   ./scripts/test-integration.sh
   ```

---

## Migration Guide

### Upgrading from CUDA 13.0 to 12.4

**Reason**: Align with csm-streaming and PyTorch ecosystem

**Steps:**

1. **Update Dockerfiles:**
   ```dockerfile
   # Before
   FROM nvidia/cuda:13.0.1-devel-ubuntu22.04
   
   # After
   FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
   ```

2. **Update requirements-cuda.txt:**
   ```txt
   # Before
   torch==2.1.2+cu121
   torchaudio==2.1.2+cu121
   
   # After
   --extra-index-url https://download.pytorch.org/whl/cu124
   torch==2.5.1
   torchaudio==2.5.1
   torchvision==0.20.1
   ```

3. **Update faster-whisper:**
   ```txt
   # Before
   faster-whisper==1.0.0
   
   # After
   faster-whisper==1.0.3
   ```

4. **Rebuild containers:**
   ```bash
   docker compose down -v
   docker compose build --no-cache
   docker compose up -d
   ```

5. **Verify:**
   ```bash
   ./scripts/test-integration.sh
   ```

---

## References

- [PyTorch CUDA Compatibility](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA Containers](https://hub.docker.com/r/nvidia/cuda)
- [faster-whisper GitHub](https://github.com/guillaumekln/faster-whisper)
- [csm-streaming Documentation](https://github.com/facebookresearch/csm)
- [Silero VAD](https://github.com/snakers4/silero-vad)

---

**Last Updated**: 2025-10-27  
**CUDA**: 12.4.0  
**PyTorch**: 2.5.1  
**Morgan AI Assistant** - Optimized for modern GPUs

