# Troubleshooting Guide

> **Common issues and solutions for Morgan AI Assistant**

---

## Table of Contents

- [Installation Issues](#installation-issues)
- [GPU & CUDA Problems](#gpu--cuda-problems)
- [Service Issues](#service-issues)
- [Performance Problems](#performance-problems)
- [Build Issues](#build-issues)
- [Network & Connectivity](#network--connectivity)
- [Database Issues](#database-issues)

---

## Installation Issues

### Docker Build Fails

**Symptom:**
```
ERROR [internal] load metadata for...
failed to solve with frontend dockerfile.v0
```

**Solutions:**

1. **Enable BuildKit:**
   ```bash
   # Linux/Mac
   export DOCKER_BUILDKIT=1
   
   # Windows PowerShell
   $env:DOCKER_BUILDKIT=1
   
   # Make permanent (add to ~/.bashrc or PowerShell profile)
   ```

2. **Check Docker version:**
   ```bash
   docker version
   # Ensure Docker >= 24.0
   ```

3. **Update Docker:**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install docker-ce docker-ce-cli containerd.io
   
   # Or use Docker Desktop for Windows/Mac
   ```

### Missing Dependencies

**Symptom:**
```
ModuleNotFoundError: No module named 'fastapi'
```

**Solutions:**

1. **Rebuild containers:**
   ```bash
   docker compose down -v
   docker compose build --no-cache
   docker compose up -d
   ```

2. **Check requirements files exist:**
   ```bash
   ls requirements-*.txt
   # Should show: base, core, llm, cuda, tts, stt
   ```

3. **Verify Dockerfile COPY statements:**
   ```dockerfile
   COPY requirements-base.txt requirements-core.txt ./
   ```

---

## GPU & CUDA Problems

### GPU Not Detected

**Symptom:**
```python
torch.cuda.is_available() returns False
```

**Diagnostics:**

```bash
# 1. Check NVIDIA driver on host
nvidia-smi
# Should show driver version and GPU info

# 2. Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
# Should show same GPU info

# 3. Check service GPU access
docker exec morgan-tts nvidia-smi
docker exec morgan-stt nvidia-smi
```

**Solutions:**

1. **Install/Update NVIDIA Driver:**
   ```bash
   # Ubuntu
   sudo apt update
   sudo apt install nvidia-driver-550
   sudo reboot
   
   # After reboot
   nvidia-smi  # Verify driver
   ```

2. **Install NVIDIA Container Toolkit:**
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt update
   sudo apt install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

3. **Verify docker-compose GPU config:**
   ```yaml
   # docker-compose.yml
   tts-service:
     deploy:
       resources:
         reservations:
           devices:
             - driver: nvidia
               count: 1
               capabilities: [gpu]
   ```

### CUDA Version Mismatch

**Symptom:**
```
RuntimeError: CUDA error: no kernel image is available for execution
```

**Solutions:**

1. **Check CUDA version compatibility:**
   ```bash
   nvidia-smi  # Check "CUDA Version" in output
   # Should be 12.4 or higher
   ```

2. **Update NVIDIA driver for CUDA 12.4:**
   ```bash
   # Minimum driver: 525
   # Recommended: 550+
   sudo apt install nvidia-driver-550
   sudo reboot
   ```

3. **Verify PyTorch CUDA version:**
   ```bash
   docker exec morgan-tts python -c "import torch; print(torch.version.cuda)"
   # Should output: 12.4
   ```

### Out of GPU Memory

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**

1. **Check GPU memory usage:**
   ```bash
   nvidia-smi
   # Look at "Memory-Usage" column
   ```

2. **Use smaller models:**
   
   `config/stt.yaml`:
   ```yaml
   model: "distil-medium"  # Instead of "distil-distil-large-v3.5 "
   ```
   
   `config/tts.yaml`:
   ```yaml
   model: "csm-streaming"  # Lightest option
   ```

3. **Reduce batch sizes** (if applicable in future updates)

4. **Clear GPU memory:**
   ```bash
   docker compose restart tts-service stt-service
   ```

---

## Service Issues

### Service Won't Start

**Symptom:**
```bash
docker compose ps
# Shows service as "Exited (1)"
```

**Diagnostics:**

```bash
# Check logs
docker compose logs core
docker compose logs llm-service
docker compose logs tts-service
docker compose logs stt-service

# Check last 50 lines
docker compose logs --tail=50 core
```

**Common Causes:**

1. **Missing configuration:**
   ```bash
   ls config/*.yaml
   # Should show: core.yaml, llm.yaml, tts.yaml, stt.yaml
   ```

2. **Port already in use:**
   ```bash
   # Linux/Mac
   sudo lsof -i :8000
   sudo lsof -i :8001
   
   # Windows
   netstat -ano | findstr :8000
   ```
   
   **Fix**: Stop conflicting service or change port in `docker-compose.yml`

3. **External Ollama not reachable:**
   ```bash
   curl http://192.168.101.3:11434/v1/models
   # Should return list of models
   ```
   
   **Fix**: Update `OLLAMA_BASE_URL` in `.env` or `config/llm.yaml`

### Health Check Fails

**Symptom:**
```bash
curl http://localhost:8000/health
# Connection refused or 503 error
```

**Solutions:**

1. **Wait for service initialization:**
   ```bash
   docker compose logs -f core
   # Wait for "Application startup complete"
   ```

2. **Check service dependencies:**
   ```bash
   # Ensure all services are running
   docker compose ps
   
   # Core depends on: LLM, TTS, STT
   # Check each service health
   curl http://localhost:8001/health
   curl http://localhost:8002/health
   curl http://localhost:8003/health
   ```

3. **Restart in correct order:**
   ```bash
   # Stop all
   docker compose down
   
   # Start dependencies first
   docker compose up -d postgres qdrant redis
   sleep 10
   
   # Start AI services
   docker compose up -d llm-service tts-service stt-service
   sleep 30
   
   # Start core
   docker compose up -d core
   ```

---

## Performance Problems

### Slow Response Times

**Symptom:** API requests take > 5 seconds

**Diagnostics:**

```bash
# Check resource usage
docker stats

# Check GPU utilization
nvidia-smi -l 1  # Updates every second
```

**Solutions:**

1. **CPU bottleneck:**
   - Check if `docker stats` shows 100% CPU
   - Reduce concurrent requests
   - Use GPU for TTS/STT if available

2. **Memory bottleneck:**
   - Check if `docker stats` shows high memory usage
   - Increase Docker memory limit
   - Use smaller models

3. **GPU bottleneck:**
   - Check `nvidia-smi` shows 100% GPU utilization
   - Use model quantization (future feature)
   - Reduce batch sizes

4. **Network latency:**
   - Check external Ollama latency:
     ```bash
     time curl http://192.168.101.3:11434/v1/models
     ```
   - Move Ollama closer to Morgan
   - Use internal Docker network if possible

### High Memory Usage

**Symptom:** System running out of RAM

**Solutions:**

1. **Limit Docker memory:**
   ```yaml
   # docker-compose.yml
   services:
     core:
       deploy:
         resources:
           limits:
             memory: 2G
   ```

2. **Clear cache:**
   ```bash
   docker system prune -a
   docker volume prune
   ```

3. **Use smaller models:**
   - See [Out of GPU Memory](#out-of-gpu-memory) solutions

---

## Build Issues

### Slow Docker Builds

**Symptom:** `docker compose build` takes > 5 minutes

**Solutions:**

1. **Enable BuildKit** (see [Installation Issues](#docker-build-fails))

2. **Check cache is working:**
   ```bash
   # Build should show "CACHED" for most steps
   docker compose build core 2>&1 | grep CACHED
   ```

3. **Use requirements files** (already implemented):
   - Ensure `requirements-*.txt` files exist
   - Should see 2-4 second dependency installs

4. **Clear corrupted cache:**
   ```bash
   docker builder prune -a
   docker compose build --no-cache
   ```

### Missing Files During Build

**Symptom:**
```
COPY failed: file not found in build context
```

**Solutions:**

1. **Check .dockerignore:**
   ```bash
   cat .dockerignore
   # Should include: !requirements-*.txt
   ```

2. **Verify files exist:**
   ```bash
   ls -la requirements-*.txt
   ls -la core/
   ls -la services/*/
   ```

3. **Check build context:**
   ```yaml
   # docker-compose.yml
   services:
     core:
       build:
         context: .  # Should be "." (root)
         dockerfile: core/Dockerfile
   ```

---

## Network & Connectivity

### Service Can't Connect to Another Service

**Symptom:**
```
aiohttp.client_exceptions.ClientConnectorError: Cannot connect to host llm-service:8001
```

**Solutions:**

1. **Check Docker network:**
   ```bash
   docker network ls
   # Should show "morgan_morgan-net"
   
   docker network inspect morgan_morgan-net
   # Should show all services
   ```

2. **Use internal DNS names:**
   ```yaml
   # Use: http://llm-service:8001
   # NOT: http://localhost:8001 (from inside container)
   ```

3. **Verify service is running:**
   ```bash
   docker compose ps llm-service
   # Should show "Up"
   ```

4. **Test connectivity:**
   ```bash
   # From core container
   docker exec morgan-core curl http://llm-service:8001/health
   ```

### External Service Not Reachable

**Symptom:**
```
Cannot connect to Ollama at http://192.168.101.3:11434
```

**Solutions:**

1. **Check Ollama is running:**
   ```bash
   curl http://192.168.101.3:11434/v1/models
   ```

2. **Check firewall:**
   ```bash
   # Linux
   sudo ufw status
   sudo ufw allow from <docker-bridge-ip> to any port 11434
   ```

3. **Use host.docker.internal** (if on same machine):
   ```yaml
   # config/llm.yaml or .env
   ollama_url: "http://host.docker.internal:11434"
   ```

---

## Database Issues

### PostgreSQL Won't Start

**Symptom:**
```
postgres exited with code 1
```

**Solutions:**

1. **Check logs:**
   ```bash
   docker compose logs postgres
   ```

2. **Permission issues:**
   ```bash
   sudo chown -R 999:999 ./data/postgres
   ```

3. **Port conflict:**
   ```bash
   # Check if port 5432 is in use
   sudo lsof -i :5432
   ```

4. **Reset database:**
   ```bash
   docker compose down -v
   docker volume rm morgan_postgres-data
   docker compose up -d postgres
   ```

### Qdrant Connection Issues

**Symptom:**
```
Cannot connect to Qdrant at http://qdrant:6333
```

**Solutions:**

1. **Check Qdrant is running:**
   ```bash
   docker compose ps qdrant
   curl http://localhost:6333/health
   ```

2. **Check logs:**
   ```bash
   docker compose logs qdrant
   ```

3. **Reset Qdrant:**
   ```bash
   docker compose down
   docker volume rm morgan_qdrant-data
   docker compose up -d qdrant
   ```

---

## Advanced Troubleshooting

### Enable Debug Logging

**In `.env` or config files:**
```bash
MORGAN_LOG_LEVEL=DEBUG
```

**Restart services:**
```bash
docker compose restart
docker compose logs -f core
```

### Interactive Debugging

**Enter container:**
```bash
docker exec -it morgan-core bash
# or
docker exec -it morgan-tts bash
```

**Test imports:**
```bash
python3
>>> import torch
>>> print(torch.cuda.is_available())
>>> import faster_whisper
>>> import TTS
```

### Clean Slate

**Complete reset:**
```bash
# Stop everything
docker compose down -v

# Remove all containers, images, volumes
docker system prune -a --volumes

# Remove logs
rm -rf logs/*

# Rebuild
docker compose build --no-cache
docker compose up -d
```

---

## Getting Help

### Before Asking for Help

1. **Check logs:**
   ```bash
   docker compose logs -f > debug.log
   # Share debug.log
   ```

2. **Check versions:**
   ```bash
   docker --version
   docker compose version
   nvidia-smi
   python --version
   ```

3. **Check configuration:**
   ```bash
   cat docker-compose.yml
   ls -la config/
   env | grep MORGAN
   ```

### Include in Bug Reports

- Output of `docker compose logs`
- Output of `docker compose ps`
- Output of `nvidia-smi` (if GPU issue)
- Your `docker-compose.yml` (sanitized)
- Steps to reproduce
- Expected vs actual behavior

---

## See Also

- [Quick Start](../getting-started/QUICK_START.md) - Initial setup
- [Docker Build Guide](../deployment/DOCKER_BUILD_GUIDE.md) - Build optimization
- [Version Alignment](../deployment/VERSION_ALIGNMENT.md) - CUDA/PyTorch compatibility
- [Architecture](../architecture/ARCHITECTURE.md) - System design

---

**Last Updated**: 2025-10-27  
**Morgan AI Assistant** - Troubleshooting Guide

