# Morgan AI Assistant - Architecture Documentation

## 🎯 Design Philosophy

Morgan AI Assistant is built on the principle of **"right tool for the right job"** - each service is optimized for its specific function while maintaining loose coupling and high cohesion.

### Core Principles

1. **Microservices Architecture**: Modular, independently deployable services
2. **GPU Optimization**: Maximum utilization of CUDA hardware
3. **Developer Experience**: Fast development cycles with modern tools
4. **Production Ready**: Comprehensive monitoring and error handling
5. **External Integration**: Seamless connection to existing systems

## 🏗️ System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    External Systems                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   Web UI    │    │ Home        │    │ Smart Home Systems  │  │
│  │   Clients   │◄──►│ Assistant   │◄──►│ (Home Assistant)    │  │
│  └─────────────┘    │ Integration  │    └─────────────────────┘  │
│         │           └─────────────┘              │              │
│         ▼                  │                     ▼              │
│  ┌─────────────┐           │            ┌─────────────────────┐ │
│  │ Audio Input │           │            │     External APIs   │ │
│  │ Processing  │           │            │     (Weather, etc)  │ │
│  └─────────────┘           │            └─────────────────────┘ │
│                            │                                    │
│  ┌─────────────┐ ┌─────────┴─────────┐ ┌─────────────┐          │
│  │ LLM Service │ │   TTS Service     │ │ STT Service │          │
│  │ (Ollama)    │ │    (Coqui TTS)    │ │ (Whisper)   │          │
│  │             │ │                   │ │             │          │
│  └─────────────┘ └───────────────────┘ └─────────────┘          │
│                                                                 │
│  ┌─────────────┐ ┌───────────────────┐ ┌─────────────────────┐  │
│  │ VAD Service │ │   Redis Cache     │ │   PostgreSQL DB     │  │
│  │ (Silero)    │ │   (Optional)      │ │   (Optional)        │  │
│  │             │ │                   │ │                     │  │
│  └─────────────┘ └───────────────────┘ └─────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                 External Ollama Service                     │  │
│  │              (192.168.101.3:11434)                         │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Service Design Patterns

### 1. Core Service (Orchestrator Pattern)

**Purpose**: Central coordination and API gateway
**Technology**: FastAPI, AsyncIO, Pydantic
**Key Features**:
- Request routing and load balancing
- Conversation context management
- Integration with external services
- Health monitoring and metrics

**Implementation**:
```python
class CoreService:
    def __init__(self):
        self.llm_client = HTTPClient("http://llm-service:8001")
        self.tts_client = HTTPClient("http://tts-service:8002")
        self.stt_client = HTTPClient("http://stt-service:8003")

    async def process_request(self, request: UserRequest) -> AIResponse:
        # Route to appropriate service
        # Manage conversation context
        # Handle errors gracefully
        pass
```

### 2. LLM Service (Adapter Pattern)

**Purpose**: OpenAI-compatible interface to external Ollama
**Technology**: OpenAI Python client, FastAPI
**Key Features**:
- OpenAI API compatibility
- Model management and switching
- Streaming response support
- Connection pooling

**Implementation**:
```python
class LLMService:
    def __init__(self, ollama_url: str):
        self.client = AsyncOpenAI(base_url=f"{ollama_url}/v1", api_key="ollama")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            stream=request.stream
        )
        return LLMResponse.from_openai(response)
```

### 3. TTS Service (Factory Pattern)

**Purpose**: Multi-model text-to-speech synthesis
**Technology**: Coqui TTS, PyTorch, CUDA
**Key Features**:
- Multiple voice model support
- GPU-accelerated synthesis
- Configurable audio quality
- Real-time processing capability

**Implementation**:
```python
class TTSService:
    def __init__(self, model_name: str = "kokoro"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS(model_name, progress_bar=False).to(self.device)

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        # Generate audio using appropriate model
        # Handle different voice configurations
        # Return audio data with metadata
        pass
```

### 4. STT Service (Pipeline Pattern)

**Purpose**: Speech-to-text with voice activity detection
**Technology**: Faster Whisper, Silero VAD, CUDA
**Key Features**:
- Real-time audio processing
- Multi-language support
- Noise reduction and preprocessing
- VAD integration for efficient processing

**Implementation**:
```python
class STTService:
    def __init__(self, model_size: str = "base"):
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
        self.vad = VADProcessor()

    async def transcribe(self, request: STTRequest) -> STTResponse:
        # Preprocess audio with VAD
        # Transcribe with Whisper
        # Return text with confidence scores
        pass
```

### 5. VAD Service (Strategy Pattern)

**Purpose**: Real-time voice activity detection
**Technology**: Silero VAD, NumPy
**Key Features**:
- Ultra-low latency processing
- Configurable sensitivity thresholds
- CPU-optimized for edge deployment
- Multiple detection strategies

**Implementation**:
```python
class VADService:
    def __init__(self, threshold: float = 0.5):
        self.model = silero_vad.load_model()
        self.threshold = threshold

    async def detect_speech(self, request: VADRequest) -> VADResponse:
        # Process audio in real-time
        # Detect speech segments
        # Return detection results
        pass
```

## 🔄 Data Flow Architecture

### Request Processing Pipeline

```
User Request
     ↓
┌─────────────┐
│   Core      │ ← Routing & Context Management
│  Service    │
└─────────────┘
     ↓
┌─────────────────────────────────────┐
│         Service Selection           │
│  ┌─────────┐ ┌─────────┐ ┌────────┐ │
│  │   LLM   │ │   TTS   │ │  STT   │ │
│  │ Service │ │ Service │ │Service │ │
│  └─────────┘ └─────────┘ └────────┘ │
└─────────────────────────────────────┘
     ↓
Response Assembly & Delivery
```

### Conversation Context Flow

```
1. User Input → Core Service
2. Context Retrieval → Database/Redis
3. LLM Processing → Context + New Input
4. Response Generation → LLM Service
5. Audio Synthesis → TTS Service (if requested)
6. Context Update → Database/Redis
7. Response Delivery → User
```

## 🐳 Docker Strategy

### Multi-Stage Builds

Each service uses optimized multi-stage Docker builds:

```dockerfile
# Stage 1: Base system setup
FROM nvidia/cuda:13.0.1-devel-ubuntu22.04 AS cuda-base
# System dependencies and CUDA setup

# Stage 2: Python dependencies
FROM cuda-base AS python-deps
# UV installation and Python package setup

# Stage 3: Application build
FROM python-deps AS build
# Application code and configuration

# Stage 4: Runtime
FROM cuda-base AS runtime
# Copy installed packages and run application
```

### GPU Resource Management

```yaml
# docker-compose.yml GPU configuration
services:
  tts-service:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
```

### Network Architecture

```yaml
# Internal Docker network
networks:
  morgan-net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

# Service discovery via container names
services:
  core:
    networks:
      - morgan-net
    depends_on:
      - llm-service
      - tts-service
      - stt-service
```

## 📦 Dependency Management

### UV Configuration Strategy

#### Local Development
```bash
# Virtual environment (system Python protected)
export UV_INDEX_URL="https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple"
uv venv
source .venv/bin/activate
uv pip install -e .
```

#### Docker Containers
```bash
# System Python (no venv in containers)
export UV_INDEX_URL="https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple"
export UV_NO_CREATE_VENV=1
uv pip install <packages> --system
```

### Service-Specific Dependencies

#### LLM Service
```toml
# pyproject.toml
llm = [
    "openai>=1.3.0",  # OpenAI API compatibility
]
```

#### TTS Service
```toml
tts = [
    "TTS>=0.13.0",           # Coqui TTS
    "torch>=2.1.0",          # PyTorch with CUDA
    "torchaudio>=2.1.0",
    "torchvision>=0.16.0",
]
```

#### STT Service
```toml
stt = [
    "faster-whisper>=1.0.0",  # Whisper implementation
    "torch>=2.1.0",           # CUDA support
    "torchaudio>=2.1.0",
    "silero-vad>=0.1.0",      # Voice activity detection
]
```

## 🔐 Configuration Architecture

### Hierarchical Configuration

```
Environment Variables (highest priority)
    ↓
YAML Configuration Files
    ↓
Default Values (lowest priority)
```

### Configuration Files Structure

```yaml
# config/core.yaml
service:
  host: "0.0.0.0"
  port: 8000

integrations:
  llm_service_url: "http://llm-service:8001"
  tts_service_url: "http://tts-service:8002"
  stt_service_url: "http://stt-service:8003"

conversation:
  timeout: 1800  # 30 minutes
  max_history: 50

logging:
  level: "INFO"
  format: "json"
```

### Environment Variable Override

```python
# shared/config/base.py
class ServiceConfig(BaseConfig):
    def get_service_url(self, default_port: int = 8000) -> str:
        host = self.get("host", "localhost")
        port = self.get("port", default_port)
        return f"http://{host}:{port}"

    def get_database_url(self) -> str:
        # Environment variable: MORGAN_DATABASE_URL
        # Falls back to config file or defaults
        pass
```

## 🚀 Performance Architecture

### GPU Memory Management

#### Memory Pool Strategy
```python
# services/tts/service.py
class TTSService:
    def __init__(self):
        self.gpu_memory_pool = GPUMemoryPool()
        self.model_cache = ModelCache()

    async def process_batch(self, requests: List[TTSRequest]) -> List[TTSResponse]:
        # Allocate GPU memory for batch
        memory_context = self.gpu_memory_pool.allocate_batch(len(requests))

        try:
            # Process all requests in batch
            results = await self._batch_synthesis(requests)
            return results
        finally:
            # Always release GPU memory
            self.gpu_memory_pool.release(memory_context)
```

#### Model Optimization
```python
# Model quantization and optimization
model = TTS("tts_models/en/ljspeech/tacotron2-DDC_ph")

# Half precision for faster inference
model = model.half()

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Dynamic memory allocation
torch.cuda.set_per_process_memory_fraction(0.8)
```

### Async Processing Architecture

#### Event Loop Optimization
```python
# core/main.py
async def main():
    # Configure event loop for high performance
    import asyncio
    loop = asyncio.get_event_loop()

    # Set optimal thread pool size
    loop.set_default_executor(
        ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2)
    )

    # Start server with optimized settings
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for async
        loop="asyncio",
        access_log=True
    )
```

#### Connection Pooling
```python
# shared/utils/http_client.py
class HTTPClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self._connector = None

    async def _get_connector(self) -> aiohttp.TCPConnector:
        if not self._connector:
            self._connector = aiohttp.TCPConnector(
                limit=100,           # Total connection limit
                limit_per_host=10,   # Per-host limit
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
        return self._connector

    async def post(self, endpoint: str, data: dict) -> dict:
        connector = await self._get_connector()
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        ) as session:
            async with session.post(f"{self.base_url}{endpoint}", json=data) as response:
                return await response.json()
```

## 📊 Monitoring Architecture

### Health Check System

#### Service Health Endpoints
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "llm_service",
        "model": current_model,
        "gpu_available": torch.cuda.is_available(),
        "memory_usage": get_memory_usage(),
        "uptime": get_uptime()
    }
```

#### Dependency Health Checks
```python
async def check_dependencies():
    checks = {
        "ollama": await check_ollama_connection(),
        "redis": await check_redis_connection(),
        "database": await check_database_connection()
    }
    return all(checks.values())
```

### Metrics Collection

#### Prometheus Metrics
```python
# core/metrics.py
from prometheus_client import Counter, Histogram, Gauge

request_count = Counter('morgan_requests_total', 'Total requests', ['service', 'endpoint'])
request_duration = Histogram('morgan_request_duration_seconds', 'Request duration', ['service'])
gpu_memory_usage = Gauge('morgan_gpu_memory_mb', 'GPU memory usage', ['service'])
active_connections = Gauge('morgan_active_connections', 'Active connections', ['service'])
```

#### Custom Metrics
```python
# Service-specific metrics
llm_tokens_generated = Counter('llm_tokens_generated_total', 'Tokens generated', ['model'])
tts_audio_duration = Histogram('tts_audio_duration_seconds', 'Audio duration')
stt_transcription_confidence = Histogram('stt_transcription_confidence', 'Transcription confidence')
vad_detection_accuracy = Gauge('vad_detection_accuracy', 'VAD detection accuracy')
```

## 🔒 Security Architecture

### Input Validation
```python
# shared/models/security.py
from pydantic import BaseModel, validator, Field
from typing import Optional

class SecureRequest(BaseModel):
    text: str = Field(..., max_length=10000, min_length=1)
    user_id: str = Field(..., regex=r'^[a-zA-Z0-9_-]{1,50}$')

    @validator('text')
    def validate_text(cls, v):
        # Remove potentially harmful content
        # Check for injection attempts
        # Validate character encoding
        pass
```

### API Security
```python
# core/api/security.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Verify API token
    # Check rate limits
    # Validate permissions
    pass
```

### Container Security
```dockerfile
# Security hardening
FROM python:3.12-slim AS base

# Create non-root user
RUN useradd --create-home --shell /bin/bash morgan

# Set proper permissions
RUN chown -R morgan:morgan /app
USER morgan

# Minimal attack surface
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*
```

## 📈 Scalability Architecture

### Horizontal Scaling

#### Service Replication
```yaml
# docker-compose.scale.yml
services:
  core:
    deploy:
      replicas: 3
  llm-service:
    deploy:
      replicas: 2
  tts-service:
    deploy:
      replicas: 2
```

#### Load Balancing
```yaml
# nginx load balancer
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

### Vertical Scaling

#### GPU Resource Allocation
```python
# GPU memory management
class GPUResourceManager:
    def __init__(self, total_memory: int = 8*1024*1024*1024):  # 8GB
        self.total_memory = total_memory
        self.allocated_memory = 0
        self.memory_pools = {}

    def allocate_model_memory(self, model_size: int) -> bool:
        if self.allocated_memory + model_size <= self.total_memory * 0.9:
            self.allocated_memory += model_size
            return True
        return False

    def optimize_memory_layout(self):
        # Defragment memory
        # Reallocate for better performance
        pass
```

## 🔄 Error Handling Architecture

### Global Error Handler
```python
# core/api/error_handlers.py
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log error with context
    logger.error(f"Unhandled exception: {exc}", extra={
        "request_id": getattr(request.state, "request_id", "unknown"),
        "user_id": getattr(request.state, "user_id", "unknown"),
        "endpoint": str(request.url)
    })

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )
```

### Service-Specific Error Handling
```python
# services/llm/service.py
class LLMService:
    async def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            response = await self._call_ollama(request)
            return self._format_response(response)
        except aiohttp.ClientError as e:
            logger.error(f"Ollama connection failed: {e}")
            raise ModelError("LLM service unavailable", ErrorCode.SERVICE_UNAVAILABLE)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise ModelError("Text generation failed", ErrorCode.MODEL_INFERENCE_ERROR)
```

## 🚀 Deployment Architecture

### Development Environment
```bash
# Local development with hot reload
uvicorn core.app:app --reload --host 0.0.0.0 --port 8000

# Docker development
docker-compose up -d --build
```

### Staging Environment
```yaml
# docker-compose.staging.yml
services:
  core:
    environment:
      - LOG_LEVEL=DEBUG
      - DEBUG=True
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
```

### Production Environment
```yaml
# docker-compose.prod.yml
services:
  core:
    environment:
      - LOG_LEVEL=WARNING
      - DEBUG=False
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 📚 Technology Stack

### Core Technologies
- **Python 3.12**: Modern Python with latest features
- **FastAPI**: High-performance async web framework
- **Pydantic**: Data validation and serialization
- **AsyncIO**: Asynchronous programming
- **aiohttp**: Async HTTP client

### Machine Learning
- **PyTorch 2.1+**: Deep learning framework with CUDA support
- **Coqui TTS**: Text-to-speech synthesis
- **Faster Whisper**: Speech-to-text with GPU acceleration
- **Silero VAD**: Voice activity detection
- **Ollama**: LLM serving and management

### Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Service orchestration
- **NVIDIA Container Toolkit**: GPU support in containers
- **Nexus Repository**: Package proxy and caching
- **Redis**: Caching and session management
- **PostgreSQL**: Persistent data storage

### Development Tools
- **UV**: Ultra-fast Python package manager
- **Black**: Code formatter
- **isort**: Import sorter
- **flake8**: Linter
- **pytest**: Testing framework
- **mypy**: Type checker

## 🎯 Design Decisions

### 1. Microservices vs Monolith
**Decision**: Microservices architecture
**Rationale**:
- Independent scaling of AI services
- Technology stack flexibility per service
- Fault isolation between services
- Easier development and testing

### 2. External Ollama vs Embedded
**Decision**: External Ollama service
**Rationale**:
- Dedicated GPU resources for LLM
- Model management flexibility
- Existing infrastructure utilization
- Separation of concerns

### 3. UV vs pip
**Decision**: UV for all Python package management
**Rationale**:
- Ultra-fast installation (10-100x faster)
- Better dependency resolution
- Lockfile management
- Consistent experience across environments

### 4. AsyncIO vs Threading
**Decision**: AsyncIO for all I/O operations
**Rationale**:
- Better performance for I/O bound services
- Native support in FastAPI
- Easier error handling and cancellation
- Better resource utilization

### 5. YAML vs JSON Configuration
**Decision**: YAML configuration files
**Rationale**:
- Human-readable format
- Complex nested structures support
- Environment variable override support
- Industry standard for configuration

---

**Morgan AI Assistant Architecture** - A comprehensive, production-ready distributed AI system designed for performance, scalability, and maintainability.
