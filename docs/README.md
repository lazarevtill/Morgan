# Morgan AI Assistant - Documentation

> **Complete documentation for Morgan AI Assistant v0.2.0**

---

## 📖 Documentation Structure

### 🚀 Getting Started

Perfect for new users and quick setup:

- **[Quick Start](./getting-started/QUICK_START.md)** - Get Morgan running in 5 minutes
- **[Development Guide](./getting-started/DEVELOPMENT.md)** - Local development setup

### 🏗️ Architecture

Understanding Morgan's design and structure:

- **[System Architecture](./architecture/ARCHITECTURE.md)** - Overall system design
- **[Streaming Architecture](./architecture/STREAMING_ARCHITECTURE.md)** - Real-time audio/text streaming
- **[API Reference](./architecture/API.md)** - Complete API documentation

### 🚢 Deployment

Production deployment and optimization:

- **[Deployment Guide](./deployment/DEPLOYMENT.md)** - Production deployment
- **[Docker Build Guide](./deployment/DOCKER_BUILD_GUIDE.md)** - Optimized Docker builds (80%+ faster!)
- **[Version Alignment](./deployment/VERSION_ALIGNMENT.md)** - CUDA/PyTorch version compatibility

### 📚 Guides

Feature-specific guides:

- **[Voice Interface](./guides/VOICE_INTERFACE.md)** - Voice interaction setup
- **[Troubleshooting](./guides/TROUBLESHOOTING.md)** - Common issues and solutions

### 📦 Archive

Historical documentation and migration guides:

- **[Archived Docs](./archive/)** - Old documentation versions

---

## 🎯 Quick Navigation

### I want to...

#### Get Started
- [Install Morgan for the first time](./getting-started/QUICK_START.md)
- [Set up local development environment](./getting-started/DEVELOPMENT.md)
- [Deploy to production](./deployment/DEPLOYMENT.md)

#### Understand the System
- [Learn how Morgan works](./architecture/ARCHITECTURE.md)
- [Understand the API](./architecture/API.md)
- [Learn about streaming](./architecture/STREAMING_ARCHITECTURE.md)

#### Optimize & Deploy
- [Build Docker images faster](./deployment/DOCKER_BUILD_GUIDE.md)
- [Align CUDA/PyTorch versions](./deployment/VERSION_ALIGNMENT.md)
- [Deploy to production](./deployment/DEPLOYMENT.md)

#### Use Features
- [Set up voice interaction](./guides/VOICE_INTERFACE.md)
- [Troubleshoot issues](./guides/TROUBLESHOOTING.md)

---

## 📊 System Overview

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    Docker Host Environment                     │
│                                                                │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────────┐│
│  │   Web UI    │    │                  │    │ External APIs  ││
│  │   Voice UI  │◄──►│   Morgan Core    │◄──►│ MCP Tools      ││
│  │   Clients   │    │  (Orchestrator)  │    │ Integrations   ││
│  └─────────────┘    └──────────────────┘    └────────────────┘│
│                              │                                 │
│         ┌────────────────────┼────────────────────┐            │
│         ▼                    ▼                    ▼            │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ LLM Service │    │ TTS Service  │    │ STT Service  │      │
│  │  (Ollama)   │    │   (csm-streaming)   │    │  (Whisper)   │      │
│  │   OpenAI    │    │  CUDA 12.4   │    │  CUDA 12.4   │      │
│  └─────────────┘    └──────────────┘    └──────────────┘      │
│                                                                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │   PostgreSQL     │  │     Qdrant       │  │    Redis     │ │
│  │ (Structured DB)  │  │  (Vector Store)  │  │   (Cache)    │ │
│  └──────────────────┘  └──────────────────┘  └──────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

### Key Features

- **🎙️ Voice Interface**: Real-time speech-to-text and text-to-speech
- **🤖 LLM Integration**: Ollama with OpenAI-compatible API
- **💾 Persistent Memory**: PostgreSQL + Qdrant for conversation history
- **🔧 MCP Tools**: Calculator, datetime, custom integrations
- **⚡ GPU Acceleration**: CUDA 12.4 optimization
- **📡 Streaming**: Real-time WebSocket streaming

### Service Ports

| Service | Port | Purpose |
|---------|------|---------|
| Core | 8000 | Main API & orchestration |
| LLM | 8001 | Language model service |
| TTS | 8002 | Text-to-speech |
| STT | 8003 | Speech-to-text |
| PostgreSQL | 5432 | Structured database |
| Qdrant | 6333 | Vector database |
| Redis | 6379 | Cache layer |

---

## 🛠️ Development Resources

### Configuration Files

- `config/core.yaml` - Core service configuration
- `config/llm.yaml` - LLM service configuration
- `config/tts.yaml` - TTS service configuration
- `config/stt.yaml` - STT service configuration
- `docker-compose.yml` - Service orchestration
- `.env` - Environment variables

### Code Structure

```
morgan/
├── core/                    # Core orchestration service
│   ├── api/                # FastAPI routes
│   ├── conversation/       # Conversation management
│   ├── handlers/           # Command handlers
│   ├── integrations/       # External integrations
│   ├── memory/             # PostgreSQL + Qdrant
│   ├── services/           # Service orchestration
│   └── tools/              # MCP tools
├── services/               # Microservices
│   ├── llm/               # LLM service
│   ├── tts/               # TTS service
│   └── stt/               # STT service
├── shared/                 # Shared utilities
│   ├── config/            # Configuration management
│   ├── models/            # Data models
│   └── utils/             # HTTP client, logging, etc.
├── database/              # Database schemas
├── config/                # YAML configurations
└── docs/                  # This documentation
```

---

## 📈 Performance

### Build Times (with cache)

| Service | Time | Notes |
|---------|------|-------|
| Core | ~2.5s | 86% faster ⚡ |
| LLM | ~1.5s | Optimized |
| TTS | ~3.5s | 81% faster ⚡ |
| STT | ~3.5s | 81% faster ⚡ |

### Runtime Performance

| Operation | Time | Hardware |
|-----------|------|----------|
| TTS Synthesis | ~450ms | RTX 4090 |
| STT Transcription | ~720ms | RTX 4090 |
| LLM Generation | ~1.1s | External Ollama |

---

## 🔧 Technology Stack

### Core Technologies

- **Python** 3.11+ with async/await
- **FastAPI** for API services
- **Docker** with multi-stage builds
- **CUDA** 12.4 for GPU acceleration

### AI/ML Stack

- **PyTorch** 2.5.1 (CUDA 12.4)
- **Ollama** for LLM backend
- **Faster Whisper** 1.0.3 for STT
- **Coqui TTS** 0.22.0 for speech synthesis
- **Silero VAD** 4.0.2 for voice detection

### Data Storage

- **PostgreSQL** for structured data
- **Qdrant** for vector embeddings
- **Redis** for caching

---

## 🤝 Contributing

### Before You Start

1. Read [Development Guide](./getting-started/DEVELOPMENT.md)
2. Understand [System Architecture](./architecture/ARCHITECTURE.md)
3. Check [CLAUDE.md](../CLAUDE.md) for AI assistant rules

### Making Changes

1. Fork the repository
2. Create a feature branch
3. Follow code standards in CLAUDE.md
4. Add tests for new features
5. Update documentation
6. Submit pull request

---

## 📞 Support & Resources

### Documentation

- **Quick Start**: [getting-started/QUICK_START.md](./getting-started/QUICK_START.md)
- **API Docs**: http://localhost:8000/docs (when running)
- **Architecture**: [architecture/ARCHITECTURE.md](./architecture/ARCHITECTURE.md)

### Troubleshooting

- **Common Issues**: [guides/TROUBLESHOOTING.md](./guides/TROUBLESHOOTING.md)
- **Logs**: `docker compose logs -f`
- **Health Checks**: `curl http://localhost:8000/health`

### External Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)

---

## 📄 License

This project is licensed under the MIT License.

---

## 🗂️ Document Index

### Getting Started
- [Quick Start Guide](./getting-started/QUICK_START.md)
- [Development Guide](./getting-started/DEVELOPMENT.md)

### Architecture
- [System Architecture](./architecture/ARCHITECTURE.md)
- [Streaming Architecture](./architecture/STREAMING_ARCHITECTURE.md)
- [API Reference](./architecture/API.md)

### Deployment
- [Deployment Guide](./deployment/DEPLOYMENT.md)
- [Docker Build Optimization](./deployment/DOCKER_BUILD_GUIDE.md)
- [Version Alignment (CUDA/PyTorch)](./deployment/VERSION_ALIGNMENT.md)

### Guides
- [Voice Interface](./guides/VOICE_INTERFACE.md)
- [Troubleshooting](./guides/TROUBLESHOOTING.md)

### Archive
- [Archived Documentation](./archive/)

---

**Morgan AI Assistant v0.2.0** - Complete documentation for your intelligent voice assistant.

**Last Updated**: 2025-10-27

