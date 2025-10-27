# CUDA and PyTorch Version Alignment

## Overview

All Morgan AI services now use **CUDA 12.4** and **PyTorch 2.5.1**, aligned with:
- **csm-streaming** (real-time audio streaming)
- **faster-whisper** (STT engine)
- **silero-vad** (voice activity detection)

This ensures compatibility across all AI components and maximizes performance.

---

## Version Matrix

### CUDA & PyTorch Core

| Component | Version | Source |
|-----------|---------|--------|
| CUDA Runtime | **12.4.0** | NVIDIA Container Toolkit |
| PyTorch | **2.5.1** | PyPI (cu124) |
| TorchAudio | **2.5.1** | PyPI (cu124) |
| TorchVision | **0.20.1** | PyPI (cu124) |
| NumPy | **1.26.3** | PyPI |

### Python ML Libraries

| Library | Version | Service | Purpose |
|---------|---------|---------|---------|
| faster-whisper | **1.0.3** | STT | Speech-to-text transcription |
| silero-vad | **4.0.2** | STT/VAD | Voice activity detection |
| TTS (Coqui) | **0.22.0** | TTS | Text-to-speech synthesis |
| csm-streaming | **0.2.8** | TTS | Fast neural TTS |
| sentence-transformers | **2.3.1** | Core | Embeddings for RAG |

### Audio Processing

| Library | Version | Services | Purpose |
|---------|---------|----------|---------|
| soundfile | **0.12.1** | TTS, STT | Audio I/O |
| librosa | **0.10.1** | TTS, STT | Audio analysis |
| pydub | **0.25.1** | TTS, STT | Audio manipulation |

---

## Docker Base Images

### CUDA Services (TTS, STT)

```dockerfile
FROM harbor.in.lazarev.cloud/proxy/nvidia/cuda:12.4.0-devel-ubuntu22.04
```

**Key Features:**
- CUDA 12.4 development toolkit
- Ubuntu 22.04 LTS base
- cuDNN included
- Python 3.11 runtime

### Non-CUDA Services (Core, LLM, VAD)

```dockerfile
FROM harbor.in.lazarev.cloud/proxy/python:3.12-slim
```

**Key Features:**
- Python 3.12 (latest stable)
- Debian Slim base
- Minimal footprint (~150MB)

---

## PyTorch Installation

### Requirements Files Strategy

#### `requirements-cuda.txt` (Common CUDA deps)
```txt
--extra-index-url https://download.pytorch.org/whl/cu124
torch==2.5.1
torchaudio==2.5.1
torchvision==0.20.1
numpy==1.26.3
```

#### `requirements-tts.txt` (TTS-specific)
```txt
-r requirements-cuda.txt
TTS==0.22.0
soundfile==0.12.1
librosa==0.10.1
pydub==0.25.1
csm-streaming==0.2.8
pyttsx3==2.90
```

#### `requirements-stt.txt` (STT-specific)
```txt
-r requirements-cuda.txt
faster-whisper==1.0.3
soundfile==0.12.1
librosa==0.10.1
pydub==0.25.1
silero-vad==4.0.2
websockets==12.0
aiofiles==23.2.1
```

---

## Compatibility Verification

### csm-streaming Alignment ✓

| Component | csm-streaming | Morgan | Status |
|-----------|--------------|--------|--------|
| CUDA | 12.4 (cu124) | 12.4 (cu124) | ✓ Match |
| PyTorch | 2.5.1 | 2.5.1 | ✓ Match |
| TorchAudio | 2.5.1 | 2.5.1 | ✓ Match |
| TorchVision | 0.20.1 | 0.20.1 | ✓ Match |

### faster-whisper Compatibility ✓

| Component | Requirement | Morgan | Status |
|-----------|------------|--------|--------|
| CUDA | 12.x | 12.4 | ✓ Compatible |
| PyTorch | 2.0+ | 2.5.1 | ✓ Compatible |
| faster-whisper | 1.0+ | 1.0.3 | ✓ Latest |

### silero-vad Compatibility ✓

| Component | Requirement | Morgan | Status |
|-----------|------------|--------|--------|
| PyTorch | 1.8+ | 2.5.1 | ✓ Compatible |
| silero-vad | 4.0+ | 4.0.2 | ✓ Latest |

---

## GPU Requirements

### Minimum Requirements

- **CUDA Compute Capability**: 7.0+ (Tesla V100, RTX 20xx series or newer)
- **GPU Memory**: 8GB VRAM minimum
  - TTS Service: 2-4GB
  - STT Service: 4-6GB
  - Shared: 2GB headroom
- **NVIDIA Driver**: 525+ (for CUDA 12.4)

### Recommended Configuration

- **GPU**: RTX 4090, A100, or better
- **VRAM**: 24GB+
- **Driver**: 550+ (latest stable)
- **Docker**: 24.0+ with NVIDIA Container Toolkit

---

## Performance Benchmarks

### With Pinned Versions (Current)

| Service | Dependency Install Time | Image Size | First Run |
|---------|------------------------|------------|-----------|
| Core | ~2-3s | 1.2GB | 5s |
| LLM | ~1-2s | 800MB | 2s |
| TTS | ~3-4s | 8.5GB | 15s (model load) |
| STT | ~3-4s | 7.2GB | 20s (model load) |
| VAD | ~1-2s | 600MB | 3s |

### Before Optimization (Old)

| Service | Dependency Install Time | Delta |
|---------|------------------------|-------|
| Core | ~18s | **-83%** 🚀 |
| LLM | ~1.6s | -0% (already optimal) |
| TTS | ~18s | **-78%** 🚀 |
| STT | ~18s | **-78%** 🚀 |

---

## Verification Commands

### Check CUDA Version

```bash
# Inside TTS/STT container
docker exec morgan-tts nvidia-smi

# Expected output: CUDA Version: 12.4
```

### Check PyTorch Version

```bash
# Inside any CUDA service
docker exec morgan-tts python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# Expected output: PyTorch: 2.5.1, CUDA: 12.4
```

### Check faster-whisper

```bash
# Inside STT service
docker exec morgan-stt python -c "from faster_whisper import WhisperModel; print(WhisperModel)"

# Should not error
```

### Check GPU Availability

```bash
# Inside TTS/STT service
docker exec morgan-tts python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}, Device Count: {torch.cuda.device_count()}')"

# Expected output: CUDA Available: True, Device Count: 1
```

---

## Troubleshooting

### Issue: CUDA version mismatch

**Symptom:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solution:**
1. Check NVIDIA driver: `nvidia-smi`
2. Ensure driver >= 525 (for CUDA 12.4)
3. Update driver if needed
4. Rebuild containers: `docker compose build --no-cache tts-service stt-service`

### Issue: PyTorch can't find GPU

**Symptom:**
```python
torch.cuda.is_available() returns False
```

**Solution:**
1. Verify NVIDIA Container Toolkit: `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`
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
3. Restart Docker daemon

### Issue: Out of memory

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
1. Check GPU memory: `nvidia-smi`
2. Reduce batch sizes in configs
3. Lower model sizes (e.g., `distil-medium` instead of `distil-large-v3`)
4. Enable model quantization in service configs

---

## Update Policy

### When to Update Versions

1. **Security patches**: Update immediately
2. **Minor versions**: Update quarterly (test in dev first)
3. **Major versions**: Update annually (requires full regression testing)

### Version Pinning Strategy

- **Exact pins** (`==`): Production deployments
- **Compatible releases** (`~=`): Development only
- **Minimum versions** (`>=`): Never use in production

### Alignment Checks

Run this before any version updates:

```bash
# Check csm-streaming alignment
pip show torch torchaudio torchvision | grep Version

# Compare with requirements-cuda.txt
cat requirements-cuda.txt
```

---

## References

- [PyTorch CUDA Compatibility](https://pytorch.org/get-started/locally/)
- [faster-whisper Requirements](https://github.com/guillaumekln/faster-whisper)
- [NVIDIA CUDA Docker Images](https://hub.docker.com/r/nvidia/cuda)
- [csm-streaming Documentation](https://github.com/facebookresearch/csm)
- [Docker BuildKit Cache](https://docs.docker.com/build/cache/)

---

**Last Updated**: 2025-10-27
**CUDA Version**: 12.4
**PyTorch Version**: 2.5.1
**Maintained By**: Morgan Development Team

