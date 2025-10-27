# Morgan AI Assistant - Quick Start Guide

> **Get Morgan running in 5 minutes** 🚀

## Prerequisites

- Docker 24.0+ & Docker Compose 2.20+
- NVIDIA GPU with CUDA 12.4+ support (optional but recommended)
- NVIDIA Container Toolkit (for GPU support)
- 8GB+ RAM, 50GB+ disk space

---

## 1. Clone Repository

```bash
git clone <repository-url>
cd Morgan
```

---

## 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and set:
# - OLLAMA_BASE_URL (external Ollama service)
# - Database credentials (if needed)
# - Service ports
```

**Minimum required settings:**
```bash
# External Ollama service
OLLAMA_BASE_URL=http://192.168.101.3:11434

# Service configuration
MORGAN_CONFIG_DIR=./config
MORGAN_LOG_LEVEL=INFO
```

---

## 3. Build & Start Services

### Option A: Quick Start (Recommended)

```bash
# Build with optimizations
export DOCKER_BUILDKIT=1  # Linux/Mac
# or
$env:DOCKER_BUILDKIT=1  # Windows PowerShell

# Start all services
docker compose up -d --build

# Check status
docker compose ps
```

### Option B: Optimized Build Script

```bash
# Linux/Mac
chmod +x build-optimized.sh
./build-optimized.sh all

# Windows PowerShell
.\build-optimized.ps1 all
```

---

## 4. Verify Installation

### Check Service Health

```bash
# Core service
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","version":"0.2.0","services":{...}}

# Check all services
curl http://localhost:8001/health  # LLM
curl http://localhost:8002/health  # TTS
curl http://localhost:8003/health  # STT
```

### Check GPU (if applicable)

```bash
# Check GPU in TTS service
docker exec morgan-tts nvidia-smi

# Check PyTorch CUDA
docker exec morgan-tts python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# Expected: CUDA: True
```

---

## 5. Test the System

### Test Text Processing

```bash
curl -X POST http://localhost:8000/api/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello Morgan, how are you?",
    "user_id": "test_user"
  }'
```

### Test Voice Interface

Open browser and navigate to:
```
http://localhost:8000/voice
```

Click "Start Recording" and speak. Morgan will transcribe, process, and respond with audio.

### Test TTS (Text-to-Speech)

```bash
curl -X POST http://localhost:8002/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is Morgan speaking", "voice": "default"}'
```

### Test STT (Speech-to-Text)

```bash
# Using a test audio file
curl -X POST http://localhost:8003/transcribe \
  -F "audio=@test_audio.wav"
```

---

## 6. Access API Documentation

- **Core Service**: http://localhost:8000/docs
- **LLM Service**: http://localhost:8001/docs
- **TTS Service**: http://localhost:8002/docs
- **STT Service**: http://localhost:8003/docs

---

## Common Commands

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f core
docker compose logs -f tts-service

# Last 100 lines
docker compose logs --tail=100 core
```

### Restart Services

```bash
# Restart all services
docker compose restart

# Restart specific service
docker compose restart core
```

### Stop Services

```bash
# Stop all services
docker compose down

# Stop and remove volumes (clean slate)
docker compose down -v
```

### Update Services

```bash
# Pull latest images
docker compose pull

# Rebuild and restart
docker compose up -d --build
```

---

## Troubleshooting

### Services Won't Start

**Check logs:**
```bash
docker compose logs core
```

**Common issues:**
- Missing external Ollama service (check OLLAMA_BASE_URL)
- Port conflicts (check ports 8000-8004 are available)
- GPU not detected (check NVIDIA driver and Container Toolkit)

### GPU Not Detected

```bash
# Verify NVIDIA driver
nvidia-smi

# Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# If fails, install NVIDIA Container Toolkit:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Out of Memory

**Reduce model sizes in configs:**

`config/stt.yaml`:
```yaml
model: "distil-medium"  # Instead of "distil-large-v3"
```

`config/llm.yaml`:
```yaml
model: "llama3.2:1b"  # Smaller model
```

### Slow Response Times

**Check resource usage:**
```bash
docker stats
```

**If CPU/Memory maxed out:**
- Reduce concurrent requests
- Use smaller models
- Add more RAM/GPU memory

---

## Next Steps

### Development

Continue with:
- [DEVELOPMENT.md](./DEVELOPMENT.md) - Full development setup
- [API.md](../architecture/API.md) - API reference
- [ARCHITECTURE.md](../architecture/ARCHITECTURE.md) - System architecture

### Production Deployment

Continue with:
- [DEPLOYMENT.md](../deployment/DEPLOYMENT.md) - Production deployment guide
- [DOCKER_BUILD_GUIDE.md](../deployment/DOCKER_BUILD_GUIDE.md) - Optimized Docker builds
- [VERSION_ALIGNMENT.md](../deployment/VERSION_ALIGNMENT.md) - CUDA/PyTorch versions

### Features

- **Memory**: Morgan can remember information across conversations
- **MCP Tools**: Calculator, datetime, custom API integrations
- **Voice Interface**: Real-time voice interaction
- **Streaming**: Real-time audio/text streaming (WebSocket)

---

## Configuration Overview

### Core Service (`config/core.yaml`)

```yaml
host: "0.0.0.0"
port: 8000
llm_service_url: "http://llm-service:8001"
tts_service_url: "http://tts-service:8002"
stt_service_url: "http://stt-service:8003"
enable_memory: true
enable_tools: true
```

### LLM Service (`config/llm.yaml`)

```yaml
model: "llama3.2:latest"
ollama_url: "http://192.168.101.3:11434"
max_tokens: 2048
temperature: 0.7
```

### TTS Service (`config/tts.yaml`)

```yaml
model: "csm-streaming"
device: "cuda"  # or "cpu"
voice: "default"
speed: 1.0
```

### STT Service (`config/stt.yaml`)

```yaml
model: "distil-large-v3"
device: "cuda"  # or "cpu"
language: "auto"
```

---

## Support

- **Documentation**: Check `/docs` directory
- **API Docs**: http://localhost:8000/docs
- **Issues**: Report bugs in issue tracker
- **Logs**: `docker compose logs -f`

---

**Ready to develop?** → [DEVELOPMENT.md](./DEVELOPMENT.md)  
**Ready to deploy?** → [DEPLOYMENT.md](../deployment/DEPLOYMENT.md)  
**Need help?** → Check [Troubleshooting](#troubleshooting)

---

**Morgan AI Assistant v0.2.0** - Your intelligent voice assistant

