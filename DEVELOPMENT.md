# Morgan AI Assistant - Development Guide

## 🎯 Project Vision

Morgan is a modern, distributed AI assistant that leverages the latest in machine learning and distributed systems architecture. The project emphasizes:

- **Microservices Architecture**: Modular, scalable services
- **GPU Optimization**: CUDA 13+ support with proper resource management
- **Developer Experience**: Fast development cycles with UV and Docker
- **Production Ready**: Comprehensive monitoring, logging, and error handling

## 🏗️ Architecture Deep Dive

### Service Responsibilities

#### Core Service (`core/`)
- **Primary API Gateway**: Handles all external requests
- **Orchestration**: Coordinates between AI services
- **Conversation Management**: Maintains context and history
- **Integration Hub**: Connects to external services (Home Assistant, etc.)

#### LLM Service (`services/llm/`)
- **OpenAI Compatibility**: Provides OpenAI API-compatible endpoints
- **Ollama Integration**: Acts as client to external Ollama service
- **Model Management**: Handles model switching and configuration
- **Streaming Support**: Real-time text generation streaming

#### TTS Service (`services/tts/`)
- **Multi-Model Support**: Coqui TTS with various voice models
- **GPU Acceleration**: CUDA-optimized for fast synthesis
- **Audio Processing**: WAV/MP3 output with configurable quality
- **Voice Management**: Multiple voice support with easy switching

#### STT Service (`services/stt/`)
- **Faster Whisper**: Latest Whisper models with GPU acceleration
- **Silero VAD Integration**: Real-time voice activity detection
- **Multi-language**: Automatic language detection
- **Audio Preprocessing**: Noise reduction and normalization

#### VAD Service (`services/vad/`)
- **Real-time Processing**: Ultra-low latency voice detection
- **Silero VAD**: State-of-the-art voice activity detection
- **CPU Optimized**: Efficient processing for edge deployment
- **Configurable Thresholds**: Adjustable sensitivity

## 🛠️ Development Environment Setup

### Prerequisites

#### System Requirements
- **Python 3.11+**: For local development
- **Docker 24.0+**: For containerized development
- **Docker Compose 2.20+**: For service orchestration
- **NVIDIA Container Toolkit**: For CUDA services (optional)
- **Git**: For version control

#### Development Tools
- **UV**: Ultra-fast Python package manager
- **Black**: Code formatter
- **isort**: Import sorter
- **flake8**: Linter
- **pytest**: Testing framework
- **VS Code/Cursor**: IDE with Python support

### Initial Setup

#### 1. Clone and Configure
```bash
git clone <repository-url>
cd morgan

# Copy configuration templates
cp config/*.yaml.example config/*.yaml

# Edit configuration files
nano config/core.yaml  # Main service URLs
nano config/llm.yaml   # Ollama connection
nano config/tts.yaml   # Voice settings
nano config/stt.yaml   # Model settings
nano config/vad.yaml   # Detection parameters
```

#### 2. Local Development Setup
```bash
# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate.ps1  # Windows

# Install dependencies
uv pip install -e .
uv pip install -e .[dev]  # Development dependencies

# Run tests
pytest
```

#### 3. Docker Development Setup
```bash
# Build all services
docker-compose build

# Start services
docker-compose up -d

# Check service health
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
curl http://localhost:8004/health
```

## 🔧 Service Development

### Adding New Services

#### 1. Create Service Structure
```bash
mkdir services/new-service
cd services/new-service

# Create basic files
touch __init__.py main.py service.py
mkdir api
touch api/__init__.py api/server.py
```

#### 2. Implement Core Service Logic
```python
# services/new-service/service.py
from shared.models.base import BaseModel, ProcessingResult
from shared.utils.logging import setup_logging

class NewServiceConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8005
    # Add service-specific configuration

class NewService:
    def __init__(self, config: ServiceConfig = None):
        self.config = config or ServiceConfig("new-service")
        self.logger = setup_logging("new_service", self.config.get("log_level", "INFO"))

    async def process(self, request: RequestModel) -> ProcessingResult:
        # Implement service logic
        pass

    async def health_check(self) -> dict:
        return {
            "status": "healthy",
            "service": "new-service",
            "version": "1.0.0"
        }
```

#### 3. Create FastAPI Routes
```python
# services/new-service/api/server.py
from fastapi import FastAPI, HTTPException
from new_service.service import NewService

app = FastAPI(title="New Service API", version="1.0.0")

@app.post("/process")
async def process_request(request: RequestModel):
    service = NewService()
    result = await service.process(request)
    return result

@app.get("/health")
async def health_check():
    service = NewService()
    return await service.health_check()
```

#### 4. Add Dockerfile
```dockerfile
# Multi-stage Dockerfile for New Service
FROM python:3.12-slim AS base

# Configure apt repositories (use appropriate Debian/Ubuntu)
RUN echo 'deb https://nexus.in.lazarev.cloud/repository/debian-proxy/ trixie main' > /etc/apt/sources.list

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Configure UV
ENV UV_INDEX_URL=https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CREATE_VENV=1

FROM base AS python-deps
WORKDIR /app
COPY pyproject.toml .
RUN uv pip install <service-dependencies> --system

FROM python-deps AS runtime
WORKDIR /app
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

CMD ["python", "main.py"]
```

#### 5. Update Configuration
```yaml
# config/new-service.yaml
host: "0.0.0.0"
port: 8005
# Add service-specific configuration
```

#### 6. Update docker-compose.yml
```yaml
new-service:
  build:
    context: .
    dockerfile: services/new-service/Dockerfile
  container_name: morgan-new-service
  restart: unless-stopped
  ports:
    - "8005:8005"
  environment:
    - MORGAN_CONFIG_DIR=/app/config
    - PYTHONPATH=/app
  networks:
    - morgan-net
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8005/health"]
    interval: 30s
    timeout: 10s
    retries: 3
```

### Service Communication Patterns

#### HTTP Client Pattern
```python
# shared/utils/http_client.py
import aiohttp
from typing import Dict, Any, Optional

class HTTPClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(f"{self.base_url}{endpoint}", json=data) as response:
                response.raise_for_status()
                return await response.json()

    async def get(self, endpoint: str) -> Dict[str, Any]:
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.get(f"{self.base_url}{endpoint}") as response:
                response.raise_for_status()
                return await response.json()
```

#### Service Integration Pattern
```python
# core/services/orchestrator.py
from shared.utils.http_client import HTTPClient
from shared.models.base import LLMRequest, TTSRequest, STTRequest

class ServiceOrchestrator:
    def __init__(self):
        self.llm_client = HTTPClient("http://llm-service:8001/v1")
        self.tts_client = HTTPClient("http://tts-service:8002")
        self.stt_client = HTTPClient("http://stt-service:8003")

    async def process_text(self, text: str, user_id: str) -> str:
        # Generate LLM response
        llm_response = await self.llm_client.post("/chat/completions", {
            "model": "llama3.2:latest",
            "messages": [{"role": "user", "content": text}]
        })

        # Generate speech if requested
        audio_data = await self.tts_client.post("/generate", {
            "text": llm_response["choices"][0]["message"]["content"],
            "voice": "af_heart"
        })

        return audio_data
```

## 🧪 Testing Strategy

### Unit Tests
```python
# tests/test_services/test_llm.py
import pytest
from services.llm.service import LLMService

@pytest.fixture
async def llm_service():
    service = LLMService()
    await service.start()
    yield service
    await service.stop()

@pytest.mark.asyncio
async def test_llm_generation(llm_service):
    request = LLMRequest(prompt="Hello, world!")
    response = await llm_service.generate(request)

    assert response.text is not None
    assert response.model == "llama3.2:latest"
    assert response.usage is not None
```

### Integration Tests
```python
# tests/test_integration.py
import pytest
import aiohttp

@pytest.mark.asyncio
async def test_end_to_end_conversation():
    # Test complete conversation flow
    async with aiohttp.ClientSession() as session:
        # 1. Send text to core service
        async with session.post("http://localhost:8000/api/text",
                              json={"text": "Hello Morgan"}) as response:
            assert response.status == 200
            result = await response.json()

        # 2. Verify LLM service response
        async with session.post("http://localhost:8001/v1/chat/completions",
                              json={"model": "llama3.2:latest", "messages": [{"role": "user", "content": "Hello"}]}) as response:
            assert response.status == 200
```

### Performance Tests
```python
# tests/test_performance.py
import time
import pytest

@pytest.mark.asyncio
async def test_llm_response_time():
    start_time = time.time()

    # Make request to LLM service
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8001/v1/chat/completions",
                              json={"model": "llama3.2:latest", "messages": [{"role": "user", "content": "Test"}]}) as response:
            await response.json()

    response_time = time.time() - start_time
    assert response_time < 5.0  # Should respond within 5 seconds
```

## 📊 Monitoring and Debugging

### Service Health Monitoring
```bash
# Check all services
docker-compose ps

# View logs for specific service
docker-compose logs -f llm-service

# Check health endpoints
curl http://localhost:8000/health  # Core
curl http://localhost:8001/health  # LLM
curl http://localhost:8002/health  # TTS
curl http://localhost:8003/health  # STT
curl http://localhost:8004/health  # VAD
```

### GPU Monitoring
```bash
# Check GPU usage
nvidia-smi

# Monitor GPU memory per container
docker stats

# Check CUDA availability in containers
docker-compose exec tts-service nvidia-smi
docker-compose exec stt-service nvidia-smi
```

### Debugging Techniques

#### Local Development Debugging
```python
# Add debug breakpoints
import pdb; pdb.set_trace()

# Enable debug logging
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Profile performance
import cProfile
cProfile.run('main()')
```

#### Docker Container Debugging
```bash
# Attach to running container
docker-compose exec llm-service bash

# View container resource usage
docker-compose exec llm-service top

# Check environment variables
docker-compose exec llm-service env

# Run Python debugger in container
docker-compose exec llm-service python -m pdb main.py
```

## 🚀 Performance Optimization

### GPU Memory Management
```python
# services/tts/service.py
import torch

class TTSService:
    def __init__(self):
        # Clear GPU cache on initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def optimize_gpu_memory(self):
        # Use half precision for faster inference
        self.model = self.model.half()

        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()

        # Set memory fraction
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.8)
```

### Connection Pooling
```python
# shared/utils/http_client.py
import aiohttp

class HTTPClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=10),
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
```

### Batch Processing
```python
# core/services/batch_processor.py
from typing import List, Dict, Any

class BatchProcessor:
    async def process_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Process multiple requests in parallel
        tasks = [self.process_single(request) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def process_single(self, request: Dict[str, Any]) -> Dict[str, Any]:
        # Process individual request
        pass
```

## 🔒 Production Deployment

### Environment Variables
```bash
# Production environment
export CUDA_VISIBLE_DEVICES=0,1  # Multiple GPUs
export MORGAN_CONFIG_DIR=/etc/morgan
export LOG_LEVEL=WARNING
export REDIS_URL=redis://production-redis:6379
export POSTGRES_URL=postgresql://morgan:password@production-postgres:5432/morgan
```

### Docker Compose Production
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  core:
    deploy:
      replicas: 3
      resources:
        reservations:
          memory: 1G
          cpus: '0.5'
        limits:
          memory: 2G
          cpus: '1.0'
    environment:
      - LOG_LEVEL=WARNING
      - REDIS_URL=redis://redis:6379
```

### Load Balancing
```yaml
# nginx.conf
upstream morgan_core {
    server core_1:8000;
    server core_2:8000;
    server core_3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://morgan_core;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🤝 Contributing Workflow

### 1. Feature Development
```bash
# Create feature branch
git checkout -b feature/add-whisper-large-model

# Make changes
# Add tests
# Update documentation

# Commit with conventional format
git commit -m "feat: add Whisper large model support"

# Push and create PR
git push origin feature/add-whisper-large-model
```

### 2. Code Review Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Configuration examples provided
- [ ] Performance impact considered
- [ ] Security implications reviewed
- [ ] Backward compatibility maintained

### 3. Testing Requirements
- Unit tests for all new functions
- Integration tests for service interactions
- Performance tests for optimization changes
- Docker tests for container modifications

## 📚 API Reference

### Core Service API
```bash
# Process text input
POST /api/text
{
  "text": "Turn on the lights",
  "user_id": "user123",
  "metadata": {"generate_audio": true}
}

# Process audio input
POST /api/audio
Content-Type: multipart/form-data
file: audio.wav
user_id: user123
```

### LLM Service API (OpenAI Compatible)
```bash
# Chat completions
POST /v1/chat/completions
{
  "model": "llama3.2:latest",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
}

# Embeddings
POST /v1/embeddings
{
  "input": "Hello world",
  "model": "llama3.2:latest"
}
```

## 🔧 Troubleshooting

### Common Issues

#### 1. GPU Not Detected
```bash
# Check GPU availability
nvidia-smi

# Verify NVIDIA container toolkit
docker run --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi

# Check container GPU access
docker-compose exec tts-service nvidia-smi
```

#### 2. Service Connection Issues
```bash
# Check service status
docker-compose ps

# View service logs
docker-compose logs llm-service

# Test connectivity
curl -f http://localhost:8001/health
```

#### 3. Memory Issues
```bash
# Check container memory usage
docker stats

# Monitor GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Reduce model size or batch size in configuration
```

#### 4. Network Issues
```bash
# Check Docker network
docker network ls
docker network inspect morgan-net

# Test inter-service communication
docker-compose exec core curl http://llm-service:8001/health
```

## 📈 Performance Metrics

### Key Performance Indicators
- **Response Time**: < 2 seconds for text processing
- **GPU Utilization**: > 80% during inference
- **Memory Usage**: < 2GB per CUDA service
- **Container Startup**: < 10 seconds
- **Error Rate**: < 1%

### Monitoring Commands
```bash
# Service performance
docker-compose exec core curl http://localhost:8000/metrics

# GPU performance
docker-compose exec tts-service nvidia-ml-py

# Memory usage
docker stats --no-stream

# Network I/O
docker-compose exec core netstat -tuln
```

## 🚨 Emergency Procedures

### Service Recovery
```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart llm-service

# Emergency stop
docker-compose down

# Clean restart
docker-compose down -v --rmi all
docker-compose up -d --build
```

### Data Recovery
```bash
# Backup data volumes
docker run --rm -v morgan_postgres_data:/backup alpine tar czf /backup/postgres.tar.gz /var/lib/postgresql/data

# Restore data
docker run --rm -v morgan_postgres_data:/data alpine tar xzf /backup/postgres.tar.gz -C /
```

## 📝 Change Log

### v0.2.0 (Current)
- Complete microservices architecture redesign
- UV package manager integration
- Nexus repository proxy configuration
- CUDA 13 optimization
- OpenAI-compatible LLM service
- Comprehensive testing framework

### v0.1.0 (Legacy)
- Monolithic architecture
- Basic LLM integration
- Simple Docker setup
- Limited service separation

---

**Morgan AI Assistant Development Guide** - Comprehensive documentation for building, testing, and deploying the distributed AI assistant system.
