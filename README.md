# Morgan AI Assistant

Morgan is a personal AI assistant with empathic intelligence, knowledge management, and natural conversation capabilities. Built with a clean client-server architecture for self-hosted deployment.

## Features

- **Empathic Engine** - Emotional intelligence, personality system, and relationship management
- **Knowledge Engine** - RAG system with vector database and semantic search
- **Personalization Layer** - User profiles, preferences, and conversation memory
- **Self-Hosted** - Run on your own hardware with Ollama or OpenAI-compatible LLMs
- **Clean Architecture** - Separate server and client for flexibility
- **Production Ready** - Docker support, health checks, metrics, and structured logging

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/your-repo/morgan.git
cd morgan

# Start services
cd docker
docker-compose up -d

# Pull LLM model
docker-compose exec ollama ollama pull gemma3

# Install client
pip install morgan-cli

# Start chatting
export MORGAN_SERVER_URL=http://localhost:8080
morgan chat
```

### Manual Installation

**1. Install Dependencies:**

```bash
# Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# Start Ollama
ollama serve &
ollama pull gemma3
```

**2. Install Server:**

```bash
cd morgan-server
pip install -e .
python -m morgan_server
```

**3. Install Client:**

```bash
cd morgan-cli
pip install -e .
morgan chat
```

## Documentation

### 📚 Complete Documentation

- **[Documentation Index](./DOCUMENTATION.md)** - Complete documentation index
- **[Migration Guide](./MIGRATION.md)** - Migrate from old Morgan system

### 🚀 Getting Started

- **[Server Quick Start](./morgan-server/README.md#quick-start)** - Get the server running
- **[Client Quick Start](./morgan-cli/README.md#quick-start)** - Start chatting
- **[Docker Quick Start](./docker/README.md#quick-start)** - Deploy with Docker

### ⚙️ Configuration

- **[Configuration Guide](./morgan-server/docs/CONFIGURATION.md)** - Complete configuration reference
- **[Embedding Configuration](./morgan-server/docs/EMBEDDING_CONFIGURATION.md)** - Embedding provider setup

### 🚢 Deployment

- **[Deployment Guide](./morgan-server/docs/DEPLOYMENT.md)** - Docker and bare metal deployment
- **[Docker README](./docker/README.md)** - Docker deployment guide

### 🔌 API

- **[API Documentation](./morgan-server/docs/API.md)** - REST and WebSocket API reference
- **[Client Library](./morgan-cli/README.md)** - Python client library

## Architecture

```
┌─────────────┐
│   Clients   │
│  (TUI, Web) │
└──────┬──────┘
       │ HTTP/WebSocket
       │
┌──────▼──────────────────────┐
│     Morgan Server           │
│  ┌────────────────────────┐ │
│  │   API Gateway          │ │
│  │   (FastAPI)            │ │
│  └───────┬────────────────┘ │
│          │                   │
│  ┌───────▼────────────────┐ │
│  │  Empathic Engine       │ │
│  │  - Emotional Intel     │ │
│  │  - Personality         │ │
│  │  - Relationships       │ │
│  └────────────────────────┘ │
│  ┌────────────────────────┐ │
│  │  Knowledge Engine      │ │
│  │  - RAG System          │ │
│  │  - Vector Search       │ │
│  │  - Doc Processing      │ │
│  └────────────────────────┘ │
│  ┌────────────────────────┐ │
│  │  Personalization       │ │
│  │  - User Profiles       │ │
│  │  - Preferences         │ │
│  │  - Memory              │ │
│  └────────────────────────┘ │
└─────────────────────────────┘
       │
       │
┌──────▼──────────────────────┐
│   External Services         │
│  - Ollama (LLM)             │
│  - Qdrant (Vector DB)       │
└─────────────────────────────┘
```

## Components

### Morgan Server

Standalone FastAPI server with all core functionality:

- **Empathic Engine** - Emotional intelligence and personality
- **Knowledge Engine** - RAG and semantic search
- **Personalization** - User profiles and preferences
- **API Gateway** - REST and WebSocket endpoints
- **Health Checks** - Monitoring and metrics

[Server Documentation →](./morgan-server/README.md)

### Morgan CLI

Lightweight terminal client:

- **Rich TUI** - Beautiful terminal interface
- **HTTP/WebSocket** - Server communication
- **Commands** - Chat, learn, memory, knowledge, health
- **Configuration** - Flexible server connection

[Client Documentation →](./morgan-cli/README.md)

### Docker

Containerized deployment:

- **Docker Compose** - Full stack with all dependencies
- **Standalone** - Server-only container
- **Monitoring** - Optional Prometheus integration

[Docker Documentation →](./docker/README.md)

## Requirements

### System Requirements

- **CPU:** 2+ cores (4+ recommended)
- **RAM:** 4+ GB (8+ GB recommended)
- **Disk:** 10+ GB free space (50+ GB recommended)
- **OS:** Linux, macOS, or Windows

### Software Requirements

- **Python:** 3.11+
- **Docker:** 20.10+ (for Docker deployment)
- **Ollama:** Latest (or other OpenAI-compatible LLM)
- **Qdrant:** Latest (vector database)

## Configuration

Morgan Server supports multiple configuration sources:

1. **Environment variables** (highest precedence)
2. **Configuration files** (YAML, JSON, .env)
3. **Default values** (lowest precedence)

**Example Configuration:**

```yaml
# Server settings
host: "0.0.0.0"
port: 8080

# LLM settings
llm_provider: "ollama"
llm_endpoint: "http://localhost:11434"
llm_model: "gemma3"

# Vector database
vector_db_url: "http://localhost:6333"

# Embedding settings
embedding_provider: "local"
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
```

[Configuration Guide →](./morgan-server/docs/CONFIGURATION.md)

## API

Morgan exposes a comprehensive REST and WebSocket API:

### Chat

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, Morgan!"}'
```

### Knowledge

```bash
curl -X POST http://localhost:8080/api/knowledge/learn \
  -H "Content-Type: application/json" \
  -d '{"source": "/path/to/document.pdf"}'
```

### WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/user123');
ws.send(JSON.stringify({
  type: 'message',
  message: 'Hello, Morgan!'
}));
```

[API Documentation →](./morgan-server/docs/API.md)

## Development

### Running Tests

```bash
# Server tests
cd morgan-server
pytest

# Client tests
cd morgan-cli
pytest

# Integration tests
cd morgan-server
pytest tests/test_integration_e2e.py
```

### Code Quality

```bash
# Format code
black morgan_server tests

# Lint code
ruff check morgan_server tests

# Type check
mypy morgan_server
```

## Deployment

### Docker Compose (Recommended)

```bash
cd docker
docker-compose up -d
```

### Bare Metal

```bash
# Install server
cd morgan-server
pip install -e .
python -m morgan_server

# Install client
cd morgan-cli
pip install -e .
morgan chat
```

### Production

For production deployments:

- Use Docker or systemd service
- Configure reverse proxy (nginx, traefik)
- Enable HTTPS
- Set up monitoring (Prometheus)
- Configure backups
- Review security settings

[Deployment Guide →](./morgan-server/docs/DEPLOYMENT.md)

## Troubleshooting

### Server Won't Start

```bash
# Check logs
docker-compose logs morgan-server

# Verify configuration
python -m morgan_server --check-config

# Test dependencies
curl http://localhost:11434/api/tags  # Ollama
curl http://localhost:6333/healthz    # Qdrant
```

### Connection Issues

```bash
# Test server health
curl http://localhost:8080/health

# Check server status
curl http://localhost:8080/api/status
```

[Troubleshooting Guide →](./morgan-server/docs/DEPLOYMENT.md#troubleshooting)

## Migration

Migrating from the old Morgan system? See the [Migration Guide](./MIGRATION.md).

## Contributing

Contributions are welcome! Please read the contributing guidelines (coming soon).

## License

MIT License - see LICENSE file for details.

## Support

- **Documentation:** [Complete Documentation Index](./DOCUMENTATION.md)
- **GitHub Issues:** Report bugs or request features
- **Discussions:** Ask questions and share ideas

## Version

Current version: 0.1.0

Last updated: December 8, 2024
