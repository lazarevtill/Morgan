# Morgan AI Assistant - Claude Code Documentation

> **Last Updated**: 2025-10-25
> **Version**: 0.2.0
> **Architecture**: Microservices with Docker Compose

This document provides comprehensive guidance for Claude Code when working with the Morgan AI Assistant codebase. It is the primary reference for understanding project structure, architecture patterns, development standards, and implementation guidelines.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Directory Structure](#directory-structure)
4. [Service Implementation Guide](#service-implementation-guide)
5. [Configuration System](#configuration-system)
6. [API Endpoints & Communication](#api-endpoints--communication)
7. [Data Models](#data-models)
8. [Development Workflow](#development-workflow)
9. [Docker & Deployment](#docker--deployment)
10. [Testing Standards](#testing-standards)
11. [Known Issues & Technical Debt](#known-issues--technical-debt)
12. [Quick Reference](#quick-reference)

---

## Project Overview

Morgan is a modular AI assistant platform built with a microservices architecture, designed for voice and text-based interactions. The system orchestrates multiple AI services (LLM, TTS, STT, VAD) through a central core service.

### Core Technologies

- **Language**: Python 3.12
- **Framework**: FastAPI (async)
- **Dependency Manager**: UV
- **Container Runtime**: Docker Compose
- **ML Framework**: PyTorch (CUDA 13)
- **Package Registry**: Nexus (harbor.in.lazarev.cloud)

### External Dependencies

- **Ollama** (192.168.101.3:11434): External LLM service
- **Redis** (optional): Caching layer
- **PostgreSQL** (optional): Persistent storage

---

## Architecture

### Service Topology

```
External Services:
├── Ollama (192.168.101.3:11434) ← LLM Backend
├── Redis (optional) ← Caching
└── PostgreSQL (optional) ← Persistence

Docker Services:
├── Core Service (8000) ← Main API & Orchestration
├── LLM Service (8001) ← OpenAI-compatible Ollama client
├── TTS Service (8002) ← Coqui TTS with CUDA
├── STT Service (8003) ← Faster Whisper + Silero VAD
└── VAD Service (8004) ← Silero VAD (CPU optimized)
```

### Communication Pattern

```
User Request → Core Service (8000)
                    ↓
            ConversationManager
                    ↓
            ServiceOrchestrator
          ┌─────────┼─────────┐
          ↓         ↓         ↓
    LLM (8001) TTS (8002) STT (8003) VAD (8004)
          ↓         ↓         ↓
    Response with text + optional audio
```

### Key Design Principles

1. **Async-First**: All I/O operations use async/await
2. **Service Isolation**: Each service is independently deployable
3. **Configuration-Driven**: YAML configs with environment overrides
4. **HTTP Communication**: Services communicate via HTTP/JSON
5. **Health Monitoring**: All services expose `/health` endpoints
6. **Error Propagation**: Custom exceptions with proper HTTP status codes

---

## Directory Structure

```text
Morgan/
├── core/                           # Main orchestration service (Port 8000)
│   ├── __init__.py
│   ├── main.py                    # Entry point → calls app.py
│   ├── app.py                     # MorganCore class & FastAPI init
│   ├── Dockerfile                 # Multi-stage build (Debian slim)
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── server.py              # APIServer with FastAPI routes
│   │
│   ├── conversation/
│   │   ├── __init__.py
│   │   └── manager.py             # ConversationManager (in-memory)
│   │
│   ├── handlers/
│   │   ├── __init__.py
│   │   └── registry.py            # HandlerRegistry for commands
│   │
│   ├── integrations/
│   │   ├── __init__.py
│   │   └── manager.py             # IntegrationManager (external APIs)
│   │
│   └── services/
│       ├── __init__.py
│       └── orchestrator.py        # ServiceOrchestrator
│
├── services/                       # Microservices
│   ├── llm/                       # LLM Service (Port 8001)
│   │   ├── __init__.py
│   │   ├── main.py               # Entry point
│   │   ├── service.py            # LLMService (OpenAI SDK)
│   │   ├── Dockerfile            # Debian slim
│   │   └── api/
│   │       ├── __init__.py
│   │       └── server.py         # FastAPI routes
│   │
│   ├── tts/                       # TTS Service (Port 8002)
│   │   ├── __init__.py
│   │   ├── main.py               # Entry point
│   │   ├── service.py            # TTSService (csm-streaming/Coqui/pyttsx3)
│   │   ├── Dockerfile            # CUDA 12.4 + PyTorch
│   │   └── api/
│   │       ├── __init__.py
│   │       └── server.py
│   │
│   ├── stt/                       # STT Service (Port 8003)
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── service.py            # STTService (Faster Whisper)
│   │   ├── Dockerfile            # CUDA 13 + PyTorch
│   │   └── api/
│   │       ├── __init__.py
│   │       └── server.py
│   │
│   └── vad/                       # VAD Service (Port 8004)
│       ├── __init__.py
│       ├── main.py
│       ├── service.py            # VADService (Silero VAD)
│       ├── Dockerfile            # CPU-only (Debian slim)
│       └── api/
│           ├── __init__.py
│           └── server.py
│
├── shared/                        # Common utilities & models
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── base.py               # BaseConfig, ServiceConfig
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── base.py               # Data models (dataclass-based)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── http_client.py        # MorganHTTPClient, ServiceRegistry
│       ├── logging.py            # Logger setup + Timer
│       ├── errors.py             # Custom exceptions
│       └── audio.py              # Audio processing utilities
│
├── config/                        # YAML configuration files
│   ├── core.yaml                 # Core service config
│   ├── llm.yaml                  # LLM service config
│   ├── tts.yaml                  # TTS service config
│   ├── stt.yaml                  # STT service config
│   └── vad.yaml                  # VAD service config
│
├── data/                         # Runtime data (volume mounts)
│   ├── conversations/            # Conversation history (future)
│   ├── models/                   # ML models cache
│   │   ├── llm/
│   │   ├── stt/
│   │   └── tts/
│   └── voices/                   # Voice models
│
├── logs/                         # Service logs (volume mounts)
│   ├── core/
│   ├── llm/
│   ├── stt/
│   ├── tts/
│   └── vad/
│
├── scripts/                      # Automation scripts
│   └── registry-build.sh         # Build & push to Harbor
│
├── docs/                         # Documentation
├── tests/                        # Test suite
│   └── test_ollama.py           # Basic Ollama tests
│
├── pyproject.toml               # UV dependencies & config
├── docker-compose.yml           # Service orchestration
├── .cursorrules                 # Project development rules
├── .dockerignore
├── .env                         # Environment variables
└── DEVELOPMENT.md              # Development setup guide
```

---

## Service Implementation Guide

### 1. Core Service (Port 8000)

**Purpose**: API gateway and service orchestration

**File**: [core/app.py](core/app.py)

**Main Class**: `MorganCore`

**Key Components**:
- `ConversationManager`: Maintains in-memory conversation history
- `HandlerRegistry`: Routes command execution to handlers
- `IntegrationManager`: Manages external integrations (Home Assistant, etc.)
- `ServiceOrchestrator`: Coordinates between AI services
- `APIServer`: FastAPI routes for text/audio processing

**Entry Point Flow**:
```python
# core/main.py
async def main():
    config = ServiceConfig("core", args.config)
    await core_main()  # calls core/app.py:main()

# core/app.py
async def main():
    morgan_core = MorganCore(config)
    await morgan_core.start()
    # Start background tasks
    # Run until stopped
```

**Key Routes** ([core/api/server.py](core/api/server.py)):
- `GET /health`: Comprehensive health check with all service status
- `GET /status`: Detailed system metrics
- `POST /api/text`: Process text requests
- `POST /api/audio`: Process audio files
- `POST /api/conversation/reset`: Reset user conversation
- `GET /`: Service information

**Background Tasks**:
- Conversation cleanup (60s interval): Remove expired conversations
- Health monitoring (30s interval): Check service health

**Service Registration Pattern**:
```python
# In MorganCore.__init__()
def _register_services(self):
    service_registry.register_service(
        "llm",
        self.config.get("llm_service_url"),
        timeout=30.0,
        max_retries=3
    )
    # Similar for tts, stt, vad
```

### 2. LLM Service (Port 8001)

**Purpose**: OpenAI-compatible Ollama client

**File**: [services/llm/service.py](services/llm/service.py)

**Main Class**: `LLMService`

**Implementation Details**:
- Uses `AsyncOpenAI` from OpenAI SDK
- Connects to **external Ollama service** (NOT bundled)
- Maintains conversation cache (in-memory)
- Supports up to 10 context messages per request

**Configuration** ([config/llm.yaml](config/llm.yaml)):
```yaml
model: "llama3.2:latest"
ollama_url: "http://192.168.101.3:11434"
max_tokens: 2048
temperature: 0.7
context_window: 4096
```

**Key Methods**:
```python
async def generate(self, prompt: str, context: List[Message]) -> str:
    """Generate completion with conversation context"""

async def chat(self, messages: List[dict]) -> str:
    """Direct chat completion"""

async def stream(self, messages: List[dict]) -> AsyncIterator[str]:
    """Streaming chat completion"""
```

**Health Check**:
- Validates Ollama connectivity
- Checks model availability
- Returns model list

### 3. TTS Service (Port 8002)

**Purpose**: Text-to-Speech synthesis with CUDA acceleration

**File**: [services/tts/service.py](services/tts/service.py)

**Main Class**: `TTSService`

**Supported Models** (priority order):
1. **csm-streaming**: Facebook Research real-time TTS (preferred)
2. **Coqui TTS**: Full TTS framework (Tacotron2, HiFiGAN)
3. **pyttsx3**: System TTS fallback

**csm-streaming Features**:
- Real-time streaming synthesis
- Low latency
- GPU-accelerated (CUDA 12.4)
- Aligned with PyTorch 2.5.1
- 24kHz sample rate
- Default voice optimized for clarity

**Configuration** ([config/tts.yaml](config/tts.yaml)):
```yaml
model: "csm"  # csm-streaming
device: "cuda"
voice: "default"
speed: 1.0
sample_rate: 24000  # csm-streaming default
output_format: "wav"
streaming_enabled: true
```

**Key Methods**:
```python
async def synthesize(self, text: str, voice: str = None) -> bytes:
    """Synthesize text to audio bytes"""

async def _synthesize_csm(self, text: str, voice: str) -> bytes:
    """csm-streaming synthesis"""

async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
    """Streaming synthesis for real-time"""

async def _synthesize_coqui(self, text: str, voice: str) -> bytes:
    """Coqui TTS synthesis"""
```

**GPU Management**:
- Uses CUDA 13
- Automatic GPU memory cleanup on stop
- Fallback to CPU if GPU unavailable

### 4. STT Service (Port 8003)

**Purpose**: Speech-to-Text with Voice Activity Detection

**File**: [services/stt/service.py](services/stt/service.py)

**Main Class**: `STTService`

**Models**:
- **Whisper**: distil-distil-large-v3.5  (default), distil-large-v3.5 , medium, small
- **Silero VAD**: Integrated for voice activity detection

**Configuration** ([config/stt.yaml](config/stt.yaml)):
```yaml
model: "distil-distil-large-v3.5 "
device: "cuda"
language: "auto"
sample_rate: 16000
threshold: 0.5
min_silence_duration: 0.5
```

**Supported Languages** (16 total):
- English, Spanish, French, German, Italian
- Portuguese, Russian, Chinese, Japanese, Korean
- Arabic, Hindi, Turkish, Polish, Dutch, Swedish

**Key Methods**:
```python
async def transcribe(self, audio_data: bytes) -> STTResponse:
    """Transcribe audio with VAD preprocessing"""

async def detect_speech(self, audio_data: bytes) -> bool:
    """Detect speech using Silero VAD"""

async def process_audio(self, audio_data: bytes) -> STTResponse:
    """Full pipeline: VAD + transcription"""
```

**Audio Processing Pipeline**:
1. Load audio data
2. Resample to 16kHz
3. Run VAD to detect speech segments
4. Transcribe speech segments
5. Return text + metadata (duration, language, segments)

### 5. VAD Service (Port 8004)

**Purpose**: Real-time Voice Activity Detection (CPU-optimized)

**File**: [services/vad/service.py](services/vad/service.py)

**Main Class**: `VADService`

**Model**: Silero VAD

**Configuration** ([config/vad.yaml](config/vad.yaml)):
```yaml
model: "silero_vad"
threshold: 0.5
min_speech_duration: 0.25
max_speech_duration: 30.0
window_size: 512
device: "cpu"
```

**Key Features**:
- Ultra-low latency detection
- Per-session state tracking
- Configurable sensitivity (threshold 0-1)
- Min/max speech duration enforcement

**Key Methods**:
```python
async def detect(self, audio_data: bytes, session_id: str) -> VADResponse:
    """Detect speech in audio chunk"""

async def process_stream(self, audio_chunk: bytes, session_id: str) -> bool:
    """Process real-time audio stream"""

def reset_session(self, session_id: str):
    """Reset VAD state for session"""
```

---

## Configuration System

### Configuration Hierarchy

Configuration values are loaded in the following priority order (lowest to highest):

1. **Default values** (hardcoded in `shared/config/base.py`)
2. **YAML files** (`config/*.yaml`)
3. **Environment variables** (`MORGAN_*`)
4. **Runtime overrides** (code)

### Configuration Classes

**File**: [shared/config/base.py](shared/config/base.py)

#### BaseConfig

Core configuration loader with environment override support.

```python
class BaseConfig:
    def __init__(self, config_path: str):
        """Load YAML config with env var overrides"""

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with dot notation"""

    def _get_env_override(self, key: str) -> Optional[Any]:
        """Check for MORGAN_<KEY> environment variable"""
```

**Environment Variable Pattern**: `MORGAN_<KEY_PATH_UPPERCASED>`

**Examples**:
```bash
# Override LLM service URL
MORGAN_LLM_SERVICE_URL=http://custom-llm:8001

# Override TTS model
MORGAN_TTS_MODEL=tts-1

# Override log level
MORGAN_LOG_LEVEL=DEBUG
```

#### ServiceConfig

Service-specific configuration wrapper with auto-detection of config directory.

```python
class ServiceConfig(BaseConfig):
    def __init__(self, service_name: str, config_dir: str = None):
        """Load service-specific config"""
```

**Config Directory Search Order**:
1. `MORGAN_CONFIG_DIR` environment variable
2. `./config` (current directory)
3. `~/.morgan/config` (user home)
4. `/etc/morgan` (system-wide)

### Configuration Files

All config files use YAML format and support environment variable overrides.

#### core.yaml

```yaml
# Service URLs (internal Docker network)
llm_service_url: "http://llm-service:8001"
tts_service_url: "http://tts-service:8002"
stt_service_url: "http://stt-service:8003"
vad_service_url: "http://vad-service:8004"

# Server settings
host: "0.0.0.0"
port: 8000
log_level: "INFO"

# Conversation settings
conversation_timeout: 3600  # 1 hour
max_history: 100

# Database (optional)
redis_url: "redis://redis:6379"
postgres_url: "postgresql://morgan:morgan_password@postgres:5432/morgan"
```

#### llm.yaml

```yaml
model: "llama3.2:latest"
ollama_url: "http://192.168.101.3:11434"
max_tokens: 2048
temperature: 0.7
context_window: 4096
system_prompt: "You are Morgan, a helpful AI assistant."
```

#### tts.yaml

```yaml
model: "csm"  # csm-streaming (real-time TTS)
device: "cuda"
voice: "default"  # csm-streaming default voice
speed: 1.0
sample_rate: 24000  # csm-streaming uses 24kHz
output_format: "wav"
streaming_enabled: true
```

#### stt.yaml

```yaml
model: "distil-distil-large-v3.5 "  # distil-distil-large-v3.5 , distil-large-v3.5 , medium, small
device: "cuda"
language: "auto"  # auto or specific language code
sample_rate: 16000
threshold: 0.5
min_silence_duration: 0.5
```

#### vad.yaml

```yaml
model: "silero_vad"
threshold: 0.5
min_speech_duration: 0.25
max_speech_duration: 30.0
window_size: 512
device: "cpu"
```

---

## API Endpoints & Communication

### Core Service API

All external requests go through the Core Service (Port 8000).

#### Health Check

**Endpoint**: `GET /health`

**Response**:
```json
{
    "status": "healthy|degraded|unhealthy",
    "version": "0.2.0",
    "uptime": "1d 5h 30m",
    "request_count": 1234,
    "services": {
        "llm": true,
        "tts": true,
        "stt": true,
        "vad": true
    },
    "orchestrator": {
        "llm_available": true,
        "tts_available": true,
        "stt_available": true,
        "vad_available": true
    },
    "conversations": {
        "total": 10,
        "active": 5,
        "total_messages": 150
    }
}
```

#### Text Processing

**Endpoint**: `POST /api/text`

**Request**:
```json
{
    "text": "What is the weather today?",
    "user_id": "user123",
    "metadata": {
        "source": "web",
        "timestamp": "2025-10-25T12:00:00Z"
    }
}
```

**Response**:
```json
{
    "text": "I can help you check the weather...",
    "metadata": {
        "conversation_id": "conv_abc123",
        "processing_time": 0.5,
        "llm_model": "llama3.2:latest"
    }
}
```

#### Audio Processing

**Endpoint**: `POST /api/audio`

**Request**: Multipart form data
- `file`: Audio file (WAV, MP3, etc.)
- `user_id`: User identifier
- `metadata`: Optional JSON metadata

**Response**:
```json
{
    "text": "Transcribed text from audio",
    "audio": "base64_encoded_audio_response",
    "metadata": {
        "conversation_id": "conv_abc123",
        "transcription_time": 0.8,
        "synthesis_time": 0.3,
        "language": "en"
    }
}
```

### Service Communication Pattern

**File**: [shared/utils/http_client.py](shared/utils/http_client.py)

#### MorganHTTPClient

Individual service client with retry logic and connection pooling.

```python
class MorganHTTPClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        """Initialize HTTP client for a service"""

    async def get(self, endpoint: str, **kwargs) -> dict:
        """GET request with retry"""

    async def post(self, endpoint: str, **kwargs) -> dict:
        """POST request with retry"""

    async def health_check(self) -> bool:
        """Check service health"""
```

**Retry Strategy**:
- Exponential backoff: `delay * (2 ** attempt)`
- Max retries: Configurable (default 3)
- Timeout: Configurable per service (default 30s)

#### ServiceRegistry

Global registry for all service clients.

```python
class ServiceRegistry:
    def register_service(
        self,
        name: str,
        base_url: str,
        **kwargs
    ) -> MorganHTTPClient:
        """Register a service client"""

    def get_service(self, name: str) -> MorganHTTPClient:
        """Get service client by name"""

    async def health_check_all(self) -> dict[str, bool]:
        """Check health of all services"""
```

**Global Instance**:
```python
from shared.utils.http_client import service_registry

# Register service
service_registry.register_service(
    "llm",
    "http://llm-service:8001",
    timeout=30.0,
    max_retries=3
)

# Use service
llm_client = service_registry.get_service("llm")
response = await llm_client.post("/api/generate", json={...})
```

---

## Data Models

**File**: [shared/models/base.py](shared/models/base.py)

### Current Implementation

**NOTE**: The codebase currently uses Python `@dataclass` instead of Pydantic `BaseModel` for shared models. This is inconsistent with `.cursorrules` requirements and should be standardized.

### Core Models

#### BaseModel

```python
@dataclass
class BaseModel:
    """Base model with serialization"""

    def to_dict(self) -> dict:
        """Convert to dictionary"""

    def to_json(self) -> str:
        """Convert to JSON string"""

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary"""
```

#### Message

```python
@dataclass
class Message(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str
    metadata: dict = field(default_factory=dict)
```

#### ConversationContext

```python
@dataclass
class ConversationContext(BaseModel):
    conversation_id: str
    user_id: str
    messages: List[Message]
    created_at: str
    updated_at: str
    metadata: dict = field(default_factory=dict)
```

### Service Request/Response Models

All service API models use Pydantic `BaseModel`:

#### LLM Service

```python
class LLMRequest(BaseModel):
    prompt: str
    context: Optional[List[dict]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

class LLMResponse(BaseModel):
    text: str
    model: str
    usage: dict
    metadata: dict = {}
```

#### TTS Service

```python
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: Optional[float] = None
    output_format: Optional[str] = None

class TTSResponse(BaseModel):
    audio: bytes  # Audio data
    format: str
    sample_rate: int
    duration: float
    metadata: dict = {}
```

#### STT Service

```python
class STTRequest(BaseModel):
    audio: bytes  # Audio data
    language: Optional[str] = None

class STTResponse(BaseModel):
    text: str
    language: str
    confidence: float
    duration: float
    segments: List[dict] = []
    metadata: dict = {}
```

#### VAD Service

```python
class VADRequest(BaseModel):
    audio: bytes  # Audio chunk
    session_id: str

class VADResponse(BaseModel):
    speech_detected: bool
    confidence: float
    timestamp: float
    metadata: dict = {}
```

---

## Development Workflow

### Local Development Setup

#### 1. Prerequisites

```bash
# Python 3.12 required
python --version  # Should be 3.12+

# UV package manager
pip install uv

# Docker & Docker Compose
docker --version
docker-compose --version
```

#### 2. Environment Setup

```bash
# Clone repository
git clone <repo_url>
cd Morgan

# Create virtual environment
uv venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Configure UV index
export UV_INDEX_URL="https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple"

# Install dependencies
uv pip install -e .
```

#### 3. Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit configuration
# Set OLLAMA_BASE_URL, service ports, etc.
```

#### 4. Running Services

**Option A: Docker Compose (Recommended)**

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f core

# Stop services
docker-compose down
```

**Option B: Local Development (Individual Services)**

```bash
# Terminal 1: Core service
cd core
python main.py --config ../config/core.yaml

# Terminal 2: LLM service
cd services/llm
python main.py --config ../../config/llm.yaml

# Terminal 3: TTS service
cd services/tts
python main.py --config ../../config/tts.yaml

# ... etc for STT and VAD
```

### Code Standards

#### Python Rules

1. **Async/Await**: All I/O operations MUST use async/await
   ```python
   # Good
   async def process_request(self, request):
       response = await self.llm_client.post("/api/generate", json=request)
       return response

   # Bad
   def process_request(self, request):
       response = requests.post(url, json=request)
       return response
   ```

2. **Type Hints**: All functions MUST have proper type annotations
   ```python
   # Good
   async def generate(self, prompt: str, context: List[Message]) -> str:
       ...

   # Bad
   async def generate(self, prompt, context):
       ...
   ```

3. **Pydantic Models**: All data structures SHOULD use Pydantic BaseModel (future standard)
   ```python
   # Target standard
   from pydantic import BaseModel, Field

   class Message(BaseModel):
       role: str = Field(..., description="Message role")
       content: str = Field(..., description="Message content")
       timestamp: str
   ```

4. **Error Handling**: Use custom exceptions with proper HTTP status codes
   ```python
   from shared.utils.errors import ServiceError, ErrorCode

   if not model_available:
       raise ServiceError(
           message="Model not available",
           code=ErrorCode.MODEL_NOT_FOUND,
           status_code=503
       )
   ```

5. **Structured Logging**: Include relevant metadata in log messages
   ```python
   from shared.utils.logging import Timer

   with Timer(self.logger, "llm_generation"):
       response = await self.generate(prompt)

   self.logger.info(
       "Generated response",
       extra={
           "user_id": user_id,
           "prompt_length": len(prompt),
           "response_length": len(response)
       }
   )
   ```

#### Configuration Rules

1. **YAML Format**: All config files MUST use YAML
2. **Environment Override**: All config values MUST support `MORGAN_*` env vars
3. **Validation**: Validate configuration on service startup
4. **Documentation**: Document all config options in YAML comments

#### Testing Rules

1. **Unit Tests**: Required for all new functions and classes
2. **Integration Tests**: Required for service interactions
3. **Coverage**: Minimum 80% test coverage
4. **Async Tests**: Use pytest-asyncio for async functions

```python
# Example test
import pytest
from services.llm.service import LLMService

@pytest.mark.asyncio
async def test_llm_generate():
    service = LLMService(config)
    await service.start()

    response = await service.generate("Hello", context=[])

    assert isinstance(response, str)
    assert len(response) > 0

    await service.stop()
```

### Development Tools

#### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov=services

# Run specific test file
pytest tests/test_ollama.py

# Run with verbose output
pytest -v
```

#### Linting & Formatting

```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8 .

# Type checking
mypy core/ services/
```

---

## Docker & Deployment

### Docker Compose Configuration

**File**: [docker-compose.yml](docker-compose.yml)

#### Service Definitions

**Core Service**:
```yaml
core:
  image: harbor.in.lazarev.cloud/morgan/core:latest
  container_name: morgan-core
  ports:
    - "8000:8000"
  depends_on:
    llm-service:
      condition: service_healthy
    tts-service:
      condition: service_healthy
    stt-service:
      condition: service_healthy
    vad-service:
      condition: service_healthy
  volumes:
    - ./config:/app/config:ro
    - ./data:/app/data
    - ./logs:/app/logs
  environment:
    - MORGAN_CONFIG_DIR=/app/config
    - MORGAN_LOG_LEVEL=INFO
  networks:
    - morgan-net
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
    interval: 30s
    timeout: 10s
    retries: 3
```

**LLM Service**:
```yaml
llm-service:
  image: harbor.in.lazarev.cloud/morgan/llm-service:latest
  container_name: morgan-llm
  ports:
    - "8001:8001"
  environment:
    - OLLAMA_BASE_URL=http://host.docker.internal:11434
    - MORGAN_CONFIG_DIR=/app/config
  volumes:
    - ./config:/app/config:ro
    - ./logs/llm:/app/logs
  networks:
    - morgan-net
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
    interval: 30s
    timeout: 10s
    retries: 3
```

**TTS Service** (CUDA 13):
```yaml
tts-service:
  image: harbor.in.lazarev.cloud/morgan/tts-service:latest
  container_name: morgan-tts
  ports:
    - "8002:8002"
  environment:
    - CUDA_VISIBLE_DEVICES=0
    - MORGAN_CONFIG_DIR=/app/config
  volumes:
    - ./config:/app/config:ro
    - ./data/models/tts:/app/models
    - ./data/voices:/app/voices
    - ./logs/tts:/app/logs
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  networks:
    - morgan-net
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
    interval: 30s
    timeout: 10s
    retries: 3
```

**STT Service** (CUDA 13):
```yaml
stt-service:
  image: harbor.in.lazarev.cloud/morgan/stt-service:latest
  container_name: morgan-stt
  ports:
    - "8003:8003"
  environment:
    - CUDA_VISIBLE_DEVICES=0
    - MORGAN_CONFIG_DIR=/app/config
  volumes:
    - ./config:/app/config:ro
    - ./data/models/stt:/app/models
    - ./logs/stt:/app/logs
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  networks:
    - morgan-net
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8003/health"]
    interval: 30s
    timeout: 10s
    retries: 3
```

**VAD Service** (CPU-only):
```yaml
vad-service:
  image: harbor.in.lazarev.cloud/morgan/vad-service:latest
  container_name: morgan-vad
  ports:
    - "8004:8004"
  environment:
    - MORGAN_CONFIG_DIR=/app/config
  volumes:
    - ./config:/app/config:ro
    - ./logs/vad:/app/logs
  networks:
    - morgan-net
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8004/health"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 15s
```

#### Networks

```yaml
networks:
  morgan-net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Dockerfile Patterns

All services follow multi-stage build pattern:

```dockerfile
# Stage 1: Base image
FROM harbor.in.lazarev.cloud/proxy/python:3.12-slim AS base

# Configure Nexus repositories
RUN echo 'deb https://nexus.in.lazarev.cloud/repository/debian-proxy/ ...' > /etc/apt/sources.list

# Install system dependencies
RUN apt-get update && apt-get install -y ...

# Stage 2: Python dependencies
FROM base AS python-deps

# Install UV
RUN pip install uv

# Install Python packages
ENV UV_NO_CREATE_VENV=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_INDEX_URL=https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple

RUN uv pip install --system <packages>

# Stage 3: Application code
FROM python-deps AS build

WORKDIR /app
COPY . .

# Stage 4: Runtime
FROM build AS runtime

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "main.py"]
```

### Build & Deployment

#### Local Build

```bash
# Build all services
docker-compose build

# Build specific service
docker-compose build core

# Build with no cache
docker-compose build --no-cache
```

#### Registry Build & Push

```bash
# Login to Harbor registry
docker login harbor.in.lazarev.cloud

# Build and push all services
./scripts/registry-build.sh -b -p

# Or manually:
docker build -t harbor.in.lazarev.cloud/morgan/core:latest ./core
docker push harbor.in.lazarev.cloud/morgan/core:latest
```

#### Deployment from Registry

```bash
# Pull latest images
docker-compose pull

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Restart specific service
docker-compose restart core
```

### Volume Mounts

| Path | Purpose | Mount Type |
|------|---------|------------|
| `./config` | Configuration files | Read-only |
| `./data` | Runtime data (models, conversations) | Read-write |
| `./logs` | Service logs | Read-write |

---

## Testing Standards

### Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest fixtures
├── unit/
│   ├── test_config.py
│   ├── test_models.py
│   └── test_http_client.py
├── integration/
│   ├── test_core_service.py
│   ├── test_llm_service.py
│   ├── test_tts_service.py
│   ├── test_stt_service.py
│   └── test_vad_service.py
└── e2e/
    └── test_full_conversation.py
```

### Test Requirements

1. **Unit Tests**: 100% coverage for business logic
   - Test individual functions and classes
   - Mock external dependencies
   - Test error conditions

2. **Integration Tests**: Test service interactions
   - Test HTTP communication between services
   - Mock external services (Ollama)
   - Test error propagation

3. **End-to-End Tests**: Test complete workflows
   - Test full conversation flow
   - Test audio processing pipeline
   - Test real-world scenarios

### Example Tests

#### Unit Test

```python
import pytest
from shared.config.base import ServiceConfig

def test_config_loading():
    """Test configuration loading from YAML"""
    config = ServiceConfig("core", "tests/fixtures/config")

    assert config.get("host") == "0.0.0.0"
    assert config.get("port") == 8000

def test_config_env_override(monkeypatch):
    """Test environment variable override"""
    monkeypatch.setenv("MORGAN_PORT", "9000")

    config = ServiceConfig("core")
    assert config.get("port") == 9000
```

#### Integration Test

```python
import pytest
from httpx import AsyncClient
from core.app import create_app

@pytest.mark.asyncio
async def test_text_endpoint():
    """Test text processing endpoint"""
    app = create_app()

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/text",
            json={
                "text": "Hello, Morgan",
                "user_id": "test_user"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "metadata" in data
```

#### E2E Test

```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_full_conversation_flow():
    """Test complete conversation with audio"""
    async with AsyncClient(base_url="http://localhost:8000") as client:
        # 1. Send text message
        text_response = await client.post(
            "/api/text",
            json={"text": "What is the weather?", "user_id": "user1"}
        )
        assert text_response.status_code == 200

        # 2. Process audio
        with open("tests/fixtures/test_audio.wav", "rb") as f:
            audio_response = await client.post(
                "/api/audio",
                files={"file": f},
                data={"user_id": "user1"}
            )

        assert audio_response.status_code == 200
        data = audio_response.json()
        assert "text" in data  # Transcription
        assert "audio" in data  # TTS response
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=core --cov=services --cov-report=html

# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with markers
pytest -m "not slow"
pytest -m integration

# Continuous testing (watch mode)
pytest-watch
```

---

## Recent Improvements & Security Enhancements

### ✅ Security & Reliability Fixes (2025-11-08)

The following critical issues have been resolved:

#### 1. Import Path Error ✅ FIXED
- **File**: `core/main.py` (Line 10)
- **Issue**: Relative import breaking when running from project root
- **Fix**: Changed `from app import main` to `from core.app import main`
- **Status**: ✅ **RESOLVED**

#### 2. Rate Limiting ✅ IMPLEMENTED
- **Coverage**: All API endpoints (Core, LLM, TTS, STT services)
- **Implementation**: Token bucket algorithm with per-IP tracking
- **Configuration**: Configurable via `core.yaml` (10 req/s default, 20 burst)
- **Features**:
  - Per-IP rate limiting
  - Configurable burst size
  - Exempt paths (/health, /docs, /redoc)
  - X-RateLimit headers in responses
- **Status**: ✅ **FULLY IMPLEMENTED**

#### 3. CORS Configuration ✅ SECURED
- **Issue**: Wildcard CORS allowing any origin with credentials
- **Fix**: Configurable whitelist via `core.yaml`
- **Features**:
  - Whitelisted origins (localhost by default)
  - Environment variable support
  - Comma-separated origin lists
  - Proper method restrictions
- **Status**: ✅ **SECURED**

#### 4. Input Validation ✅ IMPLEMENTED
- **Coverage**: All file uploads and base64 decoding
- **Features**:
  - Size validation (10MB max default)
  - Format validation (magic byte detection)
  - Base64 character validation
  - Audio format whitelisting
- **Functions**: `safe_base64_decode()`, `validate_audio_file()`
- **Status**: ✅ **FULLY IMPLEMENTED**

#### 5. Request Size Limits ✅ IMPLEMENTED
- **Limit**: 10MB maximum request body
- **Enforcement**: Middleware-level validation
- **Response**: HTTP 413 for oversized requests
- **Status**: ✅ **IMPLEMENTED**

#### 6. Database Connection Validation ✅ IMPLEMENTED
- **Features**:
  - PostgreSQL connection validation (SELECT 1 test)
  - Redis connection validation (PING test)
  - Graceful fallback to in-memory
  - Clear error logging
  - Optional database support
- **Status**: ✅ **IMPLEMENTED**

#### 7. Request ID Propagation ✅ IMPLEMENTED
- **Features**:
  - Request ID generation via middleware
  - Propagation through service calls
  - X-Request-ID header support
  - HTTP client integration
- **Status**: ✅ **IMPLEMENTED**

#### 8. JSON Validation in WebSockets ✅ IMPLEMENTED
- **Coverage**: All WebSocket handlers
- **Features**:
  - JSON parse error handling
  - Message structure validation
  - Type validation
  - Clear error messages to clients
- **Status**: ✅ **IMPLEMENTED**

### Remaining Technical Debt

#### 1. Data Model Inconsistency
**Status**: LOW PRIORITY

**Issue**: Mix of `@dataclass` and Pydantic `BaseModel` across codebase.

**Files Affected**:
- [shared/models/base.py](shared/models/base.py): Uses `@dataclass`
- Service request/response models: Use Pydantic `BaseModel`

**Impact**: Inconsistent serialization and validation.

**Future Enhancement**: Standardize on Pydantic `BaseModel` throughout.

#### 2. JSON Structured Logging
**Status**: LOW PRIORITY

**Current**: Standard Python logging
**Future**: Implement structured JSON logging with `structlog`

#### 3. Modern HTTP Client Migration
**Status**: LOW PRIORITY

**Current**: Using older `aiohttp` client
**Future**: Migrate to modern infrastructure HTTP client with circuit breakers

#### 4. Enhanced Monitoring
**Status**: LOW PRIORITY

**Future Enhancements**:
- Prometheus `/metrics` endpoint
- OpenAPI documentation improvements
- Connection pooling limits
- Request tracing enhancements

---

## Quick Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MORGAN_CONFIG_DIR` | Configuration directory | `./config` |
| `MORGAN_LOG_LEVEL` | Logging level | `INFO` |
| `MORGAN_LLM_SERVICE_URL` | LLM service URL | `http://llm-service:8001` |
| `MORGAN_TTS_SERVICE_URL` | TTS service URL | `http://tts-service:8002` |
| `MORGAN_STT_SERVICE_URL` | STT service URL | `http://stt-service:8003` |
| `MORGAN_VAD_SERVICE_URL` | VAD service URL | `http://vad-service:8004` |
| `OLLAMA_BASE_URL` | External Ollama URL | `http://192.168.101.3:11434` |

### Service Ports

| Service | Port | Protocol |
|---------|------|----------|
| Core | 8000 | HTTP |
| LLM | 8001 | HTTP |
| TTS | 8002 | HTTP |
| STT | 8003 | HTTP |
| VAD | 8004 | HTTP |
| Redis | 6379 | Redis |
| PostgreSQL | 5432 | PostgreSQL |

### Common Commands

```bash
# Development
uv venv && source .venv/bin/activate
uv pip install -e .
pytest --cov

# Docker
docker-compose up -d
docker-compose logs -f core
docker-compose restart <service>
docker-compose down

# Registry
docker login harbor.in.lazarev.cloud
./scripts/registry-build.sh -b -p
docker-compose pull && docker-compose up -d

# Testing
curl http://localhost:8000/health
curl http://localhost:8000/status
curl -X POST http://localhost:8000/api/text -H "Content-Type: application/json" -d '{"text":"Hello","user_id":"test"}'
```

### File References

- Configuration: [config/*.yaml](config/)
- Core Service: [core/app.py](core/app.py)
- Service Registry: [shared/utils/http_client.py](shared/utils/http_client.py)
- Data Models: [shared/models/base.py](shared/models/base.py)
- Docker Compose: [docker-compose.yml](docker-compose.yml)
- Development Guide: [DEVELOPMENT.md](DEVELOPMENT.md)

---

## Contributing

When making changes to the codebase, ensure:

1. Follow all code standards in this document
2. Add tests for new functionality (minimum 80% coverage)
3. Update configuration examples if adding new config options
4. Update this CLAUDE.md if making architectural changes
5. Use async/await for all I/O operations
6. Add proper type hints to all functions
7. Use structured logging with relevant metadata
8. Run tests locally before committing

---

**Document Version**: 1.0
**Last Review**: 2025-10-25
**Maintained By**: Morgan Development Team
