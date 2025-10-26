# Morgan AI Assistant - Deployment Guide

## Quick Start

### Prerequisites

- Docker 24.0+ with Docker Compose
- NVIDIA GPU with CUDA 13 support (for TTS/STT services)
- Access to Ollama service (external LLM backend)
- 16GB+ RAM recommended
- 20GB+ disk space for Docker images

### 1. Build All Services

```bash
# Build all service images
./scripts/registry-build.sh -b

# Or build specific service
./scripts/registry-build.sh -b -s core
./scripts/registry-build.sh -b -s llm
./scripts/registry-build.sh -b -s tts
./scripts/registry-build.sh -b -s stt
```

### 2. Configure Environment

Edit `.env` file (create from .env.example if needed):

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://host.docker.internal:11434

# Service Ports (defaults)
MORGAN_CORE_PORT=8000
MORGAN_LLM_PORT=8001
MORGAN_TTS_PORT=8002
MORGAN_STT_PORT=8003

# Logging
MORGAN_LOG_LEVEL=INFO

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
```

### 3. Update Configuration Files

**config/llm.yaml**:
```yaml
model: "llama3.2:latest"  # Or your preferred model
ollama_url: "http://host.docker.internal:11434"
max_tokens: 2048
temperature: 0.7
```

**config/tts.yaml**:
```yaml
model: "kokoro"  # or "tts-1", "pyttsx3"
device: "cuda"
voice: "af_heart"  # or am_michael, bf_emma, bm_george
speed: 1.0
```

**config/stt.yaml**:
```yaml
model: "large-v3"  # or "distil-large-v3", "medium"
device: "cuda"
language: "auto"  # or specific language code
vad_enabled: true
vad_threshold: 0.5
```

### 4. Start Services

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f core

# Check service health
curl http://localhost:8000/health
```

### 5. Verify Deployment

```bash
# Check all services are healthy
docker-compose ps

# Test core service
curl http://localhost:8000/

# Test LLM service
curl http://localhost:8001/health

# Test TTS service
curl http://localhost:8002/health

# Test STT service
curl http://localhost:8003/health
```

---

## Service Architecture

### Running Services

| Service | Port | Purpose | GPU Required |
|---------|------|---------|--------------|
| Core | 8000 | API Gateway & Orchestration | No |
| LLM | 8001 | Ollama Client (OpenAI compatible) | No |
| TTS | 8002 | Text-to-Speech (Kokoro/Coqui) | Yes |
| STT | 8003 | Speech-to-Text (Whisper + VAD) | Yes |

### Removed Services

- ❌ **VAD Service** (Port 8004): Redundant - now integrated into STT
- ❌ **Redis** (Port 6379): Not implemented yet
- ❌ **PostgreSQL** (Port 5432): Not implemented yet

---

## API Endpoints

### Core Service (Port 8000)

**Health Check**:
```bash
curl http://localhost:8000/health
```

**Process Text**:
```bash
curl -X POST http://localhost:8000/api/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "user_id": "user123"
  }'
```

**Process Audio**:
```bash
curl -X POST http://localhost:8000/api/audio \
  -F "file=@audio.wav" \
  -F "user_id=user123"
```

**Reset Conversation**:
```bash
curl -X POST http://localhost:8000/api/conversation/reset \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123"}'
```

### LLM Service (Port 8001)

**Generate Text**:
```bash
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Tell me a joke",
    "max_tokens": 100
  }'
```

**OpenAI Compatible**:
```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:latest",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### TTS Service (Port 8002)

**Synthesize Speech**:
```bash
curl -X POST http://localhost:8002/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "voice": "af_heart"
  }' \
  --output speech.wav
```

### STT Service (Port 8003)

**Transcribe Audio**:
```bash
curl -X POST http://localhost:8003/transcribe \
  -F "file=@audio.wav" \
  -F "language=auto"
```

---

## Docker Commands

### Managing Services

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d core

# Stop all services
docker-compose down

# Restart specific service
docker-compose restart core

# View logs
docker-compose logs -f core
docker-compose logs --tail=100 llm-service

# Exec into container
docker-compose exec core bash
```

### Maintenance

```bash
# View resource usage
docker stats

# Clean up unused images
docker system prune -a

# Rebuild specific service
docker-compose build core
docker-compose up -d core

# View service health
docker-compose ps
```

---

## Troubleshooting

### Service Won't Start

1. **Check logs**:
   ```bash
   docker-compose logs core
   ```

2. **Check configuration**:
   ```bash
   docker-compose config
   ```

3. **Verify environment variables**:
   ```bash
   docker-compose exec core env | grep MORGAN
   ```

### GPU Not Detected

1. **Verify NVIDIA Docker runtime**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:13.0.1-runtime-ubuntu22.04 nvidia-smi
   ```

2. **Check docker-compose GPU config**:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```

### Ollama Connection Failed

1. **Check Ollama is running**:
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. **Verify network connectivity from container**:
   ```bash
   docker-compose exec llm-service curl http://host.docker.internal:11434/api/tags
   ```

3. **Check LLM service logs**:
   ```bash
   docker-compose logs llm-service
   ```

### Service Health Check Failing

1. **Check if service is running**:
   ```bash
   docker-compose ps
   ```

2. **Check service logs**:
   ```bash
   docker-compose logs [service-name]
   ```

3. **Test endpoint manually**:
   ```bash
   curl http://localhost:8000/health
   ```

---

## Performance Tuning

### GPU Memory Optimization

**For TTS Service** (config/tts.yaml):
```yaml
# Use smaller model if needed
model: "tts-1"  # instead of kokoro
```

**For STT Service** (config/stt.yaml):
```yaml
# Use distilled model
model: "distil-large-v3"  # instead of large-v3

# Disable VAD if not needed
vad_enabled: false
```

### Memory Limits

Add to docker-compose.yml:
```yaml
services:
  core:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

### CPU Limits

```yaml
services:
  llm-service:
    deploy:
      resources:
        limits:
          cpus: '2.0'
```

---

## Monitoring

### Health Checks

All services expose `/health` endpoints:

```bash
# Core service
curl http://localhost:8000/health | jq

# LLM service
curl http://localhost:8001/health | jq

# TTS service
curl http://localhost:8002/health | jq

# STT service
curl http://localhost:8003/health | jq
```

### Request Tracing

All requests include:
- `X-Request-ID`: Unique request identifier
- `X-Process-Time`: Request processing time in seconds

```bash
curl -v http://localhost:8000/health 2>&1 | grep X-Request-ID
curl -v http://localhost:8000/health 2>&1 | grep X-Process-Time
```

### Logs

Logs are stored in:
- **Container**: `/app/logs/`
- **Host**: `./logs/`

```bash
# View logs
tail -f logs/core/core.log
tail -f logs/llm/llm_service.log
tail -f logs/tts/tts_service.log
tail -f logs/stt/stt_service.log
```

---

## Production Deployment

### Security

1. **Update CORS settings** (core/api/server.py):
   ```python
   allow_origins=["https://yourdomain.com"]  # Instead of ["*"]
   ```

2. **Add API authentication**:
   ```python
   # Add API key middleware
   ```

3. **Enable HTTPS**:
   ```bash
   # Use reverse proxy (nginx/traefik)
   ```

### Scaling

1. **Horizontal scaling**:
   ```yaml
   services:
     core:
       deploy:
         replicas: 3
   ```

2. **Load balancing**:
   ```bash
   # Use nginx or traefik for load balancing
   ```

### Backup

1. **Configuration**:
   ```bash
   tar -czf config-backup.tar.gz config/
   ```

2. **Data** (when persistence is implemented):
   ```bash
   docker-compose exec postgres pg_dump morgan > backup.sql
   ```

---

## Upgrade Guide

### From Previous Version

1. **Backup current configuration**:
   ```bash
   cp -r config config.backup
   ```

2. **Pull new images**:
   ```bash
   ./scripts/registry-build.sh -l
   ```

3. **Update configuration**:
   - Remove `vad_service_url` from core.yaml
   - Update stt.yaml with new VAD parameters

4. **Restart services**:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

5. **Verify health**:
   ```bash
   curl http://localhost:8000/health
   ```

---

## Support

### Logs Location

- Core: `logs/core/core.log`
- LLM: `logs/llm/llm_service.log`
- TTS: `logs/tts/tts_service.log`
- STT: `logs/stt/stt_service.log`

### Debug Mode

Enable debug logging in config files:
```yaml
log_level: "DEBUG"
```

### Common Issues

1. **Port already in use**:
   ```bash
   # Change port in docker-compose.yml
   ports:
     - "9000:8000"  # Use 9000 instead of 8000
   ```

2. **Out of memory**:
   ```bash
   # Reduce model sizes or add memory limits
   ```

3. **GPU out of memory**:
   ```bash
   # Use smaller models or reduce batch sizes
   ```

---

## Next Steps

- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Implement API authentication
- [ ] Add rate limiting
- [ ] Enable conversation persistence (Redis/PostgreSQL)
- [ ] Set up CI/CD pipeline
- [ ] Add comprehensive logging
- [ ] Implement metrics collection

---

**Last Updated**: 2025-10-25
**Version**: 0.2.1
