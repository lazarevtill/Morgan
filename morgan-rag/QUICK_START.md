# Morgan RAG Quick Start

## Installation

```bash
cd morgan-rag
pip install -e ".[dev]"
```

## Starting Services

### Using Docker Compose (Recommended)

```bash
# Start all services
cd ../docker
cp env.example .env
docker-compose up -d

# Pull required models
docker-compose exec ollama ollama pull qwen2.5:7b
docker-compose exec ollama ollama pull qwen3-embedding:4b
```

### Manual Setup

```bash
# Start Qdrant (vector database)
docker run -d -p 6333:6333 qdrant/qdrant

# Start Ollama and pull models
ollama serve &
ollama pull qwen2.5:7b
ollama pull qwen3-embedding:4b
```

## Verify Services

- **Server**: http://localhost:8080/health
- **Qdrant**: http://localhost:6333/dashboard
- **Redis**: localhost:6379

## Check Logs

```bash
# All services
docker-compose logs -f

# Just the server
docker-compose logs -f morgan-server
```

## Stop Services

```bash
docker-compose down
```

## Configuration

Use environment variables with `MORGAN_` prefix:

```bash
MORGAN_LLM_ENDPOINT=http://localhost:11434/v1
MORGAN_LLM_MODEL=qwen2.5:7b
MORGAN_EMBEDDING_MODEL=qwen3-embedding:4b
MORGAN_QDRANT_URL=http://localhost:6333
MORGAN_REDIS_URL=redis://localhost:6379
```

See [CLAUDE.md](../CLAUDE.md) for full configuration reference.

## Troubleshooting

1. **Services won't start**: Check logs with `docker-compose logs`
2. **LLM connection fails**: Verify Ollama is running and model is pulled
3. **Embedding service fails**: Check Ollama host and embedding model availability
4. **Qdrant connection fails**: Ensure Qdrant container is running
