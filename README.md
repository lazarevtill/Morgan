# Morgan AI Assistant - v0.2.0

A modern, distributed AI assistant built with Ollama, OpenAI-compatible APIs, and optimized for CUDA 13 with NVIDIA container toolkit support. Uses Nexus proxy repositories for faster builds and development.

## 🏗️ Architecture Overview

Morgan v0.2.0 features a completely redesigned microservices architecture optimized for performance and scalability:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Host Environment                      │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   Web UI    │    │             │    │ Home Assistant      │  │
│  │   Clients   │◄──►│ Morgan Core │◄──►│ Integration Service │  │
│  └─────────────┘    │             │    └─────────────────────┘  │
│         │           └─────────────┘              │              │
│         ▼                  │                     ▼              │
│  ┌─────────────┐           │            ┌─────────────────────┐ │
│  │ Audio Input │           │            │     External APIs   │ │
│  │ Processing  │           │            │     (Weather, etc)  │ │
│  └─────────────┘           │            └─────────────────────┘ │
│                            │                                    │
│  ┌─────────────┐ ┌─────────┴─────────┐ ┌─────────────┐          │
│  │             │ │                   │ │             │          │
│  │ LLM Service │ │   TTS Service     │ │ STT Service │          │
│  │ (Ollama)    │ │    (Coqui TTS)    │ │ (Whisper)   │          │
│  │             │ │                   │ │             │          │
│  └─────────────┘ └───────────────────┘ └─────────────┘          │
│                                                                 │
│  ┌─────────────┐ ┌───────────────────┐ ┌─────────────────────┐  │
│  │             │ │                   │ │                     │  │
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

## ✨ Key Features

### 🚀 Modern Architecture
- **Ollama Integration**: Uses Ollama with OpenAI-compatible API for LLM services
- **Distributed Services**: Separate TTS, STT, and VAD services for optimal performance
- **Silero VAD**: Advanced voice activity detection for improved audio processing
- **CUDA 13 Optimization**: Optimized for latest NVIDIA GPUs with container toolkit
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
- **GPU**: NVIDIA GPU with CUDA 13+ support (recommended)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 50GB+ for models and data
- **Network**: Stable network connection for inter-service communication

### Software Requirements
- **Docker**: 24.0+
- **Docker Compose**: 2.20+
- **NVIDIA Container Toolkit**: For GPU support
- **Linux**: Ubuntu 22.04+ recommended

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
python core/app.py                     # Core service
python services/llm/main.py            # LLM service
python services/tts/main.py            # TTS service
python services/stt/main.py            # STT service
python services/vad/main.py            # VAD service

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
curl http://localhost:8004/health  # VAD
```

### 5. Access the System
- **API Documentation**: http://localhost:8000/docs
- **Core Service**: http://localhost:8000
- **LLM Service**: http://localhost:8001
- **TTS Service**: http://localhost:8002
- **STT Service**: http://localhost:8003
- **VAD Service**: http://localhost:8004

## 🔧 Configuration

### Core Service Configuration
```yaml
# config/core.yaml
host: "0.0.0.0"
port: 8000
llm_service_url: "http://llm-service:8001"
tts_service_url: "http://tts-service:8002"
stt_service_url: "http://stt-service:8003"
vad_service_url: "http://vad-service:8004"
conversation_timeout: 1800
max_history: 50
log_level: "INFO"
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
model: "kokoro"
device: "cuda"
language: "en-us"
voice: "af_heart"
speed: 1.0
output_format: "wav"
sample_rate: 22050
log_level: "INFO"
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

### VAD Service Configuration
```yaml
# config/vad.yaml
host: "0.0.0.0"
port: 8004
model: "silero_vad"
threshold: 0.5
min_speech_duration: 0.25
max_speech_duration: 30.0
window_size: 512
sample_rate: 16000
device: "cpu"
log_level: "INFO"
```

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

### VAD Service Endpoints

#### Detect Speech
```bash
curl -X POST http://localhost:8004/detect \
  -H "Content-Type: application/json" \
  -d '{
    "audio_data": "base64_encoded_audio_data",
    "threshold": 0.5
  }'
```

## 🐳 Docker Services

### Service Architecture
- **External Ollama**: LLM backend service (running at 192.168.101.3:11434)
- **LLM Service**: OpenAI-compatible API wrapper for external Ollama (CPU only)
- **TTS Service**: Text-to-speech synthesis with Coqui TTS (GPU optimized)
- **STT Service**: Speech-to-text with Faster Whisper + Silero VAD (GPU optimized)
- **VAD Service**: Voice activity detection with Silero VAD (CPU optimized)
- **Core Service**: Main orchestration and API service (CPU only)
- **Redis**: Optional caching and message queuing
- **PostgreSQL**: Optional persistent storage

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
│   ├── handlers/          # Command handlers
│   ├── integrations/      # External integrations
│   ├── services/          # Service orchestration
│   └── utils/             # Utilities
├── services/               # Microservices
│   ├── llm/               # LLM service (Ollama)
│   ├── tts/               # TTS service (Kokoro)
│   ├── stt/               # STT service (Whisper + VAD)
│   └── vad/               # VAD service (Silero)
├── shared/                # Shared components
│   ├── config/            # Configuration management
│   ├── models/            # Data models
│   └── utils/             # Shared utilities
├── config/                # Configuration files
├── data/                  # Data and models
└── logs/                  # Service logs
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

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

- **Documentation**: Check the `/docs` endpoints
- **Issues**: Report bugs and request features
- **Discussions**: Community discussions and support

## 🔄 Migration from v0.1.0

### Breaking Changes
- Complete architecture redesign
- New service endpoints and APIs
- Configuration format changes
- Docker compose structure updated

### Migration Steps
1. Backup existing data and configurations
2. Update docker-compose.yml
3. Update configuration files
4. Pull new Docker images
5. Start services in new architecture

---

**Morgan AI Assistant v0.2.0** - Modern, distributed, and optimized for performance.