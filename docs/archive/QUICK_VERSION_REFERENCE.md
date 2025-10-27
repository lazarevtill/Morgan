# Quick Version Reference Card

## 🎯 Current Versions (Aligned)

```
CUDA: 12.4.0
PyTorch: 2.5.1
TorchAudio: 2.5.1
TorchVision: 0.20.1
faster-whisper: 1.0.3
silero-vad: 4.0.2
```

## ✅ Compatibility Matrix

| Project | CUDA | PyTorch | Status |
|---------|------|---------|--------|
| **Morgan** | 12.4 | 2.5.1 | ✅ |
| **csm-streaming** | 12.4 | 2.5.1 | ✅ Match |
| **faster-whisper** | 12.x | 2.0+ | ✅ Compatible |

## 🚀 Build & Test

```bash
# Clean build
docker compose down -v
docker compose build --no-cache

# Start services
docker compose up -d

# Verify CUDA
docker exec morgan-tts nvidia-smi
docker exec morgan-tts python -c "import torch; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"

# Expected output:
# PyTorch 2.5.1 with CUDA 12.4

# Test services
curl http://localhost:8000/health
```

## 📊 Build Performance

| Service | Time | Improvement |
|---------|------|-------------|
| Core | ~2.5s | 86% faster ⚡ |
| TTS | ~3.5s | 81% faster ⚡ |
| STT | ~3.5s | 81% faster ⚡ |

## 📋 Files Changed

- ✅ `requirements-cuda.txt` → PyTorch 2.5.1+cu124
- ✅ `requirements-tts.txt` → Pinned versions
- ✅ `requirements-stt.txt` → faster-whisper 1.0.3
- ✅ `requirements-core.txt` → Pinned all versions
- ✅ `requirements-base.txt` → Pinned all versions
- ✅ `services/tts/Dockerfile` → CUDA 12.4.0
- ✅ `services/stt/Dockerfile` → CUDA 12.4.0

## 🔍 Troubleshooting

### CUDA not found?
```bash
# Check driver
nvidia-smi  # Need 525+

# Update if needed
sudo apt install nvidia-driver-550
```

### Build still slow?
```bash
# Clear cache
docker builder prune -a

# Verify cache mounts in Dockerfiles
grep "mount=type=cache" services/*/Dockerfile
```

---

**See full docs**: [CUDA_VERSION_ALIGNMENT.md](./CUDA_VERSION_ALIGNMENT.md)

