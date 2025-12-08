# Morgan Server

Morgan Server is a standalone FastAPI application that hosts Morgan's core functionality, including the Empathic Engine, Knowledge Engine, and Personalization Layer.

## Features

- **Empathic Engine**: Emotional intelligence, personality system, and relationship management
- **Knowledge Engine**: RAG system with vector database and semantic search
- **Personalization Layer**: User profiles, preferences, and conversation memory
- **Flexible Configuration**: Support for local and remote services
- **Health Monitoring**: Built-in health checks and metrics
- **Production Ready**: Docker support, structured logging, and graceful shutdown

## Installation

### From Source

```bash
cd morgan-server
pip install -e ".[dev]"
```

### Using Docker

```bash
docker build -t morgan-server -f docker/Dockerfile.server .
docker run -p 8080:8080 morgan-server
```

## Quick Start

### 1. Start Required Services

```bash
# Start Qdrant (vector database)
docker run -p 6333:6333 qdrant/qdrant

# Start Ollama (LLM)
docker run -p 11434:11434 ollama/ollama
ollama pull gemma3
```

### 2. Configure Morgan Server

Copy the example configuration:

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` as needed. See [Configuration Guide](docs/CONFIGURATION.md) for details.

### 3. Run Morgan Server

```bash
python -m morgan_server --config config.yaml
```

### 4. Test the Server

```bash
curl http://localhost:8080/health
```

## Configuration

Morgan Server supports multiple configuration sources with the following precedence:

1. **Environment variables** (highest precedence)
2. **Configuration files** (YAML, JSON, .env)
3. **Default values** (lowest precedence)

For detailed configuration options, see:
- **[Configuration Guide](docs/CONFIGURATION.md)** - Complete configuration reference
- **[Embedding Configuration](docs/EMBEDDING_CONFIGURATION.md)** - Embedding provider setup

### Quick Configuration Example

Create a `config.yaml` file:

```yaml
# Server settings
host: "0.0.0.0"
port: 8080
workers: 4

# LLM settings
llm_provider: "ollama"  # Options: "ollama", "openai-compatible"
llm_endpoint: "http://localhost:11434"
llm_model: "gemma3"
# llm_api_key: "your-api-key"  # Optional

# Vector database settings
vector_db_url: "http://localhost:6333"
# vector_db_api_key: "your-api-key"  # Optional

# Embedding settings
embedding_provider: "local"  # Options: "local", "ollama", "openai-compatible"
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
embedding_device: "cpu"  # For local only: "cpu", "cuda", "mps"
# embedding_endpoint: "http://localhost:11434"  # Required for remote providers
# embedding_api_key: "your-api-key"  # Optional for remote providers

# Cache settings
cache_dir: "./data/cache"
cache_size_mb: 1000

# Logging settings
log_level: "INFO"
log_format: "json"

# Performance settings
max_concurrent_requests: 100
request_timeout_seconds: 60
session_timeout_minutes: 60
```

### Environment Variables

All configuration can be set via environment variables with the `MORGAN_` prefix:

```bash
# Required
export MORGAN_LLM_ENDPOINT="http://localhost:11434"

# Optional (with defaults)
export MORGAN_HOST="0.0.0.0"
export MORGAN_PORT="8080"
export MORGAN_LLM_PROVIDER="ollama"
export MORGAN_LLM_MODEL="gemma3"
export MORGAN_VECTOR_DB_URL="http://localhost:6333"
export MORGAN_EMBEDDING_PROVIDER="local"
export MORGAN_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export MORGAN_EMBEDDING_DEVICE="cpu"
export MORGAN_LOG_LEVEL="INFO"
```

## Embedding Configuration

Morgan Server supports three embedding providers: **local**, **remote Ollama**, and **OpenAI-compatible**.

For detailed setup instructions, model recommendations, and troubleshooting, see the **[Embedding Configuration Guide](docs/EMBEDDING_CONFIGURATION.md)**.

### Quick Comparison

| Provider | Pros | Cons | Best For |
|----------|------|------|----------|
| **Local** | Free, private, offline | Requires compute | Development |
| **Ollama** | Self-hosted, GPU support | Infrastructure needed | Production (self-hosted) |
| **OpenAI** | Managed, scalable | Costs money, cloud | Production (cloud) |

### Quick Setup Examples

**Local (Default):**
```yaml
embedding_provider: "local"
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
embedding_device: "cpu"
```

**Remote Ollama:**
```yaml
embedding_provider: "ollama"
embedding_model: "nomic-embed-text"
embedding_endpoint: "http://ollama-server:11434"
```

**OpenAI:**
```yaml
embedding_provider: "openai-compatible"
embedding_model: "text-embedding-3-small"
embedding_endpoint: "https://api.openai.com/v1"
embedding_api_key: "your-api-key"
```

## Running the Server

### Development

```bash
cd morgan-server
python -m morgan_server
```

### Production

```bash
uvicorn morgan_server.app:create_app --factory --host 0.0.0.0 --port 8080 --workers 4
```

### Docker Compose

```bash
docker-compose up -d
```

This starts:
- Morgan Server
- Ollama (for LLM)
- Qdrant (vector database)
- Prometheus (metrics)

## API Endpoints

### Chat

- `POST /api/chat` - Send a message and get a response
- `WS /ws/{user_id}` - WebSocket for real-time chat

### Memory

- `GET /api/memory/stats` - Get memory statistics
- `GET /api/memory/search` - Search conversation history
- `DELETE /api/memory/cleanup` - Clean old conversations

### Knowledge

- `POST /api/knowledge/learn` - Add documents to knowledge base
- `GET /api/knowledge/search` - Search knowledge base
- `GET /api/knowledge/stats` - Get knowledge statistics

### Profile

- `GET /api/profile/{user_id}` - Get user profile
- `PUT /api/profile/{user_id}` - Update user preferences
- `GET /api/timeline/{user_id}` - Get relationship timeline

### System

- `GET /health` - Health check
- `GET /api/status` - Detailed status
- `GET /metrics` - Prometheus metrics
- `GET /docs` - OpenAPI documentation

## Health Checks

The server provides two health check endpoints:

### Simple Health Check

```bash
curl http://localhost:8080/health
```

Returns:
```json
{
  "status": "healthy",
  "timestamp": "2024-12-08T10:30:00Z",
  "version": "0.1.0",
  "uptime_seconds": 3600.0
}
```

### Detailed Status

```bash
curl http://localhost:8080/api/status
```

Returns detailed information about each component (vector DB, LLM, memory system).

## Testing

### Run All Tests

```bash
pytest
```

### Run Specific Test Suite

```bash
pytest tests/test_config.py -v
```

### Run Property-Based Tests

```bash
pytest tests/test_config.py::TestInvalidConfigurationRejection -v
```

### Run with Coverage

```bash
pytest --cov=morgan_server --cov-report=html
```

## Development

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### Code Formatting

```bash
black morgan_server tests
```

### Linting

```bash
ruff check morgan_server tests
```

### Type Checking

```bash
mypy morgan_server
```

## Architecture

Morgan Server follows a clean architecture with clear separation of concerns:

```
morgan_server/
├── api/              # API layer (routes, models, middleware)
├── empathic/         # Empathic Engine (emotional intelligence, personality)
├── knowledge/        # Knowledge Engine (RAG, vector DB, search)
├── personalization/  # Personalization Layer (profiles, preferences, memory)
├── llm/              # LLM Client (Ollama, OpenAI-compatible)
├── config.py         # Configuration management
├── health.py         # Health checks and monitoring
└── app.py            # FastAPI application factory
```

## License

MIT

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting PRs.
