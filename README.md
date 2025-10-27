# Morgan AI Assistant - v0.2.0

A modern, distributed AI assistant with real-time voice interaction, persistent memory, and GPU-accelerated AI services.

> **Quick Start**: Get Morgan running in 5 minutes → [docs/getting-started/QUICK_START.md](docs/getting-started/QUICK_START.md)  
> **Documentation**: Complete docs → [docs/README.md](docs/README.md)

---

## ✨ Key Features

- 🎙️ **Real-time Voice Interface** - WebSocket streaming for audio
- 🤖 **Ollama Integration** - OpenAI-compatible API for LLMs
- 💾 **Persistent Memory** - PostgreSQL + Qdrant vector database
- ⚡ **GPU Accelerated** - CUDA 12.4 optimization
- 🔧 **MCP Tools** - Calculator, datetime, custom integrations
- 🚀 **Fast Builds** - 80%+ faster Docker builds with optimizations

---

## 🏗️ Architecture Overview

Morgan v0.2.0 features a completely redesigned microservices architecture optimized for performance and scalability:

```
┌────────────────────────────────────────────────────────────────┐
│                    Docker Host Environment                     │
│                                                                │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────────┐ │
│  │   Web UI    │    │                  │    │ External APIs  │ │
│  │   Voice UI  │◄──►│   Morgan Core    │◄──►│ MCP Tools      │ │
│  │   Clients   │    │  (Orchestrator)  │    │ Integrations   │ │
│  └─────────────┘    └──────────────────┘    └────────────────┘ │
│                              │                                 │
│         ┌────────────────────┼────────────────────┐            │
│         ▼                    ▼                    ▼            │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ LLM Service │    │ TTS Service  │    │ STT Service  │       │
│  │  (Ollama)   │    │(csm-streaming│    │  (Whisper +  │       │
│  │   (CPU/GPU) │    │  Real-time)  │    │ Silero VAD)  │       │
│  │             │    │  CUDA 12.4   │    │  CUDA 12.4   │       │
│  └─────────────┘    └──────────────┘    └──────────────┘       │
│                                                                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │   PostgreSQL     │  │     Qdrant       │  │    Redis     │  │
│  │ (Structured DB)  │  │  (Vector Store)  │  │   (Cache)    │  │
│  │   Memories       │  │   Embeddings     │  │              │  │
│  │   Tools Logs     │  │   Semantic       │  │              │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            External Ollama Service                       │  │
│  │            (192.168.101.3:11434)                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

## ✨ Key Features

### 🚀 Modern Architecture
- **Ollama Integration**: Uses Ollama with OpenAI-compatible API for LLM services
- **Persistent Memory**: PostgreSQL for structured data + Qdrant for semantic search
- **MCP Tools**: Model Context Protocol tools integration (calculator, datetime, remember, custom APIs)
- **Distributed Services**: Separate TTS and STT services with integrated VAD for optimal performance
- **CUDA 12.4 Optimization**: Aligned with csm-streaming and latest PyTorch
- **Async/Await**: Full async support for high performance and concurrency

### 🛠️ Technical Highlights
- **Python 3.11+**: Modern Python with latest features
- **FastAPI**: High-performance async web framework
- **Pydantic Models**: Type-safe data validation
- **Structured Logging**: JSON logging with metadata
- **Health Monitoring**: Comprehensive health checks and metrics
- **Docker Multi-stage**: Optimized container builds

## 📋 System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA Compute 7.0+ (Tesla V100, RTX 20xx+, recommended)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 50GB+ for models and data
- **Network**: Stable network connection for inter-service communication

### Software Requirements
- **Docker**: 24.0+
- **Docker Compose**: 2.20+
- **NVIDIA Container Toolkit**: For GPU support (CUDA 12.4)
- **NVIDIA Driver**: 525+ (550+ recommended)
- **OS**: Linux (Ubuntu 22.04+ recommended), Windows 10/11, macOS

## 📦 Development Setup

### Using uv (Recommended)

Morgan uses [uv](https://github.com/astral-sh/uv) for fast Python dependency management with Nexus proxy repositories:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up development environment (Linux/macOS)
cd morgan
./scripts/dev-setup.sh

# Or for Windows PowerShell
.\scripts\dev-setup.ps1 -Build -Up

# Or install dependencies manually (system Python for containers)
export UV_INDEX_URL="https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple"
export UV_NO_CREATE_VENV=1  # No venv creation in containers
uv pip install fastapi uvicorn[standard] pydantic aiohttp pyyaml python-dotenv structlog psutil redis --system
```

### Docker Optimizations

The Docker setup has been optimized to avoid unnecessary virtual environments:

- **No Virtual Environments in Containers**: `UV_NO_CREATE_VENV=1` + `uv pip install --system` prevents venv creation
- **Direct System Installation**: UV installs packages directly to system Python
- **Faster Startup**: Each container starts ~2-3 seconds faster
- **Memory Efficient**: ~100-200MB less RAM usage per container
- **Proper Repository Separation**: Debian vs Ubuntu repositories correctly configured

**Note**: Local development uses virtual environments since system Python is protected on Ubuntu/Debian systems, but Docker containers use system Python for optimal performance.

### Troubleshooting

If you encounter build issues, use the diagnostic tools:

```bash
# For Windows PowerShell
.\scripts\docker-diagnose.ps1 -CheckFiles    # Check file structure
.\scripts\docker-diagnose.ps1 -FixImports    # Fix missing imports
.\scripts\docker-diagnose.ps1 -ValidateBuild # Test Docker builds
.\scripts\docker-diagnose.ps1 -CleanRebuild  # Fresh start

# For Linux/macOS
./scripts/docker-diagnose.sh -c  # Check files
./scripts/docker-diagnose.sh -f  # Fix imports
./scripts/docker-diagnose.sh -v  # Validate build
./scripts/docker-diagnose.sh -r  # Fresh start
```

These tools will help identify and fix common issues like missing files, import errors, and Docker build problems.

### Local Development

```bash
# Run services locally (requires virtual environment)
source .venv/bin/activate
python core/main.py                    # Core service
python services/llm/main.py            # LLM service
python services/tts/main.py            # TTS service
python services/stt/main.py            # STT service

# Run tests
pytest

# Or using Docker (recommended for production)
docker-compose up -d --build
```

### Nexus Repository Configuration

The project is configured to use your Nexus proxy repositories:

- **Ubuntu/Debian**: `https://nexus.in.lazarev.cloud/repository/ubuntu-group/`
- **Security Updates**: `https://nexus.in.lazarev.cloud/repository/debian-security/`
- **PyPI Proxy**: `https://nexus.in.lazarev.cloud/repository/pypi-proxy/`

See [DEVELOPMENT.md](./DEVELOPMENT.md) for detailed development instructions.

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd morgan
```

### 2. Configure UV and Dependencies
```bash
# For Windows PowerShell
.\scripts\setup-uv.ps1 -Install -Configure
.\scripts\docker-build-helper.ps1 -GenerateLockfile

# For Linux/macOS
./scripts/dev-setup.sh
./scripts/docker-build-helper.sh -g  # Generate lockfile
```

### 3. Diagnose and Fix Issues (if needed)
```bash
# For Windows PowerShell
.\scripts\docker-diagnose.ps1 -CheckFiles
.\scripts\docker-diagnose.ps1 -FixImports
.\scripts\docker-diagnose.ps1 -ValidateBuild

# For Linux/macOS
./scripts/docker-diagnose.sh -c  # Check files
./scripts/docker-diagnose.sh -f  # Fix imports
./scripts/docker-diagnose.sh -v  # Validate build
```

### 4. Build and Start Services
```bash
# Build with optimizations (recommended)
.\scripts\docker-build-helper.ps1 -BuildNoCache

# Start all services
docker compose up -d

# Or use the helper script for clean start
.\scripts\docker-build-helper.ps1 -CleanStart
```

### 5. Verify Installation
```bash
# Test all services
.\scripts\test-integration.ps1 -Quick

# Check individual service health
curl http://localhost:8000/health  # Core
curl http://localhost:8001/health  # LLM
curl http://localhost:8002/health  # TTS
curl http://localhost:8003/health  # STT
```

### 5. Access the System
- **API Documentation**: http://localhost:8000/docs
- **Voice Interface**: http://localhost:8000/voice
- **Core Service**: http://localhost:8000
- **LLM Service**: http://localhost:8001
- **TTS Service**: http://localhost:8002
- **STT Service**: http://localhost:8003

## 🔧 Configuration

### Core Service Configuration
```yaml
# config/core.yaml
host: "0.0.0.0"
port: 8000
llm_service_url: "http://llm-service:8001"
tts_service_url: "http://tts-service:8002"
stt_service_url: "http://stt-service:8003"
conversation_timeout: 1800
max_history: 50
log_level: "INFO"
enable_memory: true
enable_tools: true
postgres_host: "postgres"
qdrant_host: "qdrant"
```

### LLM Service Configuration
```yaml
# config/llm.yaml
host: "0.0.0.0"
port: 8001
model: "llama3.2:latest"
ollama_url: "http://ollama:11434"
max_tokens: 2048
temperature: 0.7
timeout: 30.0
gpu_layers: -1
context_window: 4096
system_prompt: "You are Morgan, a helpful AI assistant."
log_level: "INFO"
```

### TTS Service Configuration
```yaml
# config/tts.yaml
host: "0.0.0.0"
port: 8002
model: "csm-streaming"  # Real-time TTS
device: "cuda"
language: "en"
voice: "default"
speed: 1.0
output_format: "wav"
sample_rate: 24000  # csm-streaming default
log_level: "INFO"
streaming_enabled: true  # Enable real-time streaming
```

### STT Service Configuration
```yaml
# config/stt.yaml
host: "0.0.0.0"
port: 8003
model: "whisper-large-v3"
device: "cuda"
language: "auto"
sample_rate: 16000
chunk_size: 1024
threshold: 0.5
min_silence_duration: 0.5
log_level: "INFO"
```

### Memory & Tools Configuration
Memory and tools are configured in core.yaml (shown above). Additional database settings:
- `postgres_port`: 5432
- `postgres_db`: "morgan"
- `qdrant_port`: 6333
- `embedding_dimension`: 384 (for semantic search)
- `memory_search_limit`: 5 (memories per query)
- `memory_min_importance`: 3 (importance filter)

## 📡 API Usage

### Core Service Endpoints

#### Process Text Input
```bash
curl -X POST http://localhost:8000/api/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Turn on the living room lights",
    "user_id": "user123",
    "metadata": {"generate_audio": true}
  }'
```

#### Process Audio Input
```bash
curl -X POST http://localhost:8000/api/audio \
  -F "file=@audio.wav" \
  -F "user_id=user123"
```

#### Health Check
```bash
curl http://localhost:8000/health
```

### LLM Service Endpoints (OpenAI Compatible)

#### Chat Completions
```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:latest",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

#### Embeddings
```bash
curl -X POST http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world",
    "model": "llama3.2:latest"
  }'
```

### TTS Service Endpoints

#### Generate Speech
```bash
curl -X POST http://localhost:8002/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test message.",
    "voice": "af_heart",
    "speed": 1.0
  }'
```

### STT Service Endpoints

#### Transcribe Audio
```bash
curl -X POST http://localhost:8003/transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "audio_data": "base64_encoded_audio_data",
    "language": "en"
  }'
```

### Memory Commands

#### Remember Command
```bash
# Store information in memory
curl -X POST http://localhost:8000/api/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Remember that I like coffee",
    "user_id": "user123"
  }'
```

Supported remember patterns:
- "Remember that I like coffee"
- "Remember: my birthday is June 15th"
- "Please remember my favorite color is blue"

## 🐳 Docker Services

### Service Architecture
- **External Ollama**: LLM backend service (running at 192.168.101.3:11434)
- **LLM Service**: OpenAI-compatible API wrapper for external Ollama (CPU only)
- **TTS Service**: Real-time text-to-speech with **csm-streaming** (GPU optimized, CUDA 12.4)
  - **Note**: Using Facebook Research's csm-streaming for real-time TTS
- **STT Service**: Speech-to-text with Faster Whisper + **integrated** Silero VAD (GPU optimized, CUDA 12.4)
  - **Note**: VAD is built into faster-whisper, not a separate service
- **Core Service**: Main orchestration, memory management, and API service (CPU only)
- **PostgreSQL**: Structured memory storage and tools execution logging
- **Qdrant**: Vector database for semantic memory search
- **Redis**: Caching and session state

### GPU Configuration
All GPU-enabled services use NVIDIA container toolkit:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## 🔍 Monitoring and Logging

### Health Monitoring
Each service provides comprehensive health checks:
- Service availability
- Model status
- GPU memory usage
- Request/response times
- Error rates

### Structured Logging
All services use structured JSON logging:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "service": "llm_service",
  "message": "Request processed",
  "request_id": "req_12345",
  "processing_time": 0.123,
  "model": "llama3.2:latest"
}
```

### Metrics
Prometheus metrics are available at `/metrics` endpoints for monitoring dashboards.

## 🛠️ Development

### Project Structure
```
morgan/
├── core/                    # Core orchestration service
│   ├── api/                # FastAPI server
│   ├── conversation/       # Conversation management
│   ├── handlers/          # Command handlers (including remember)
│   ├── integrations/      # External integrations
│   ├── memory/            # Memory manager (PostgreSQL + Qdrant)
│   ├── services/          # Service orchestration
│   ├── tools/             # MCP tools manager
│   └── static/            # Voice UI (voice_simple.html)
├── services/               # Microservices
│   ├── llm/               # LLM service (Ollama wrapper)
│   │   └── api/           # FastAPI endpoints
│   ├── tts/               # TTS service (Kokoro)
│   │   └── api/           # FastAPI endpoints
│   └── stt/               # STT service (Whisper + integrated VAD)
│       └── api/           # FastAPI endpoints
├── shared/                # Shared components
│   ├── config/            # Configuration management
│   ├── models/            # Data models
│   └── utils/             # Shared utilities (HTTP client, logging, errors)
├── database/              # Database initialization
│   └── init/              # SQL schema files
├── config/                # YAML configuration files
├── data/                  # Data and models (gitignored)
├── logs/                  # Service logs (gitignored)
├── docs/                  # Documentation
│   └── archive/           # Old documentation
└── tests/                 # Test suite
    └── manual/            # Manual test files
```

### Adding New Features

#### Custom Command Handler
```python
from core.handlers.registry import BaseHandler

class CustomHandler(BaseHandler):
    async def handle(self, command, context):
        # Your custom logic here
        return {
            "success": True,
            "response": "Custom command executed",
            "data": result
        }
```

#### New Integration
```python
class CustomIntegration:
    async def execute(self, command, parameters):
        # Integration logic here
        return {"success": True, "data": result}
```

## 🚨 Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check GPU availability
nvidia-smi

# Verify NVIDIA container toolkit
docker run --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

#### Service Connection Issues
```bash
# Check service logs
docker-compose logs llm-service
docker-compose logs core

# Test service connectivity
curl -f http://localhost:8001/health
```

#### Model Loading Issues
```bash
# Check available disk space
df -h

# Verify model downloads
docker-compose exec llm-service ls -la data/models/
```

### Performance Optimization

#### GPU Memory Management
- Monitor GPU usage: `nvidia-smi`
- Adjust batch sizes in configuration
- Use appropriate model sizes for your hardware

#### Network Optimization
- Use internal Docker network for service communication
- Adjust timeouts based on network latency
- Enable connection pooling

## 📚 Documentation

### Getting Started

- **[Quick Start Guide](docs/getting-started/QUICK_START.md)** - Get running in 5 minutes
- **[Development Guide](docs/getting-started/DEVELOPMENT.md)** - Local development setup

### Architecture

- **[System Architecture](docs/architecture/ARCHITECTURE.md)** - Overall design
- **[Streaming Architecture](docs/architecture/STREAMING_ARCHITECTURE.md)** - Real-time streaming
- **[API Reference](docs/architecture/API.md)** - Complete API docs

### Deployment

- **[Deployment Guide](docs/deployment/DEPLOYMENT.md)** - Production deployment
- **[Docker Build Guide](docs/deployment/DOCKER_BUILD_GUIDE.md)** - Build optimization (80%+ faster!)
- **[Version Alignment](docs/deployment/VERSION_ALIGNMENT.md)** - CUDA/PyTorch compatibility

### Guides

- **[Voice Interface](docs/guides/VOICE_INTERFACE.md)** - Voice setup
- **[Troubleshooting](docs/guides/TROUBLESHOOTING.md)** - Common issues

### Full Documentation

📖 **[Complete Documentation Index](docs/README.md)** - All documentation in one place

---

## 🚨 Quick Troubleshooting

### Services won't start?
```bash
docker compose logs -f core
# Check for errors, verify Ollama URL in config
```

### GPU not detected?
```bash
nvidia-smi  # Check driver (need 525+)
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### Slow builds?
```bash
export DOCKER_BUILDKIT=1
docker compose build
# Should take 2-4 seconds per service (with cache)
```

**More solutions** → [Troubleshooting Guide](docs/guides/TROUBLESHOOTING.md)

---

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Read [Development Guide](docs/getting-started/DEVELOPMENT.md)
2. Fork the repository and create a feature branch
3. Follow code standards in [CLAUDE.md](CLAUDE.md)
4. Add tests and update documentation
5. Submit a pull request

## 📞 Support

- **Documentation**: [docs/README.md](docs/README.md)
- **API Docs**: http://localhost:8000/docs (when running)
- **Issues**: Report bugs in issue tracker

---

**Morgan AI Assistant v0.2.0** - Your intelligent voice assistant  
**Last Updated**: 2025-10-27