# Migration Guide: Old Morgan to Client-Server Architecture

This guide helps you migrate from the old monolithic Morgan implementation to the new client-server architecture.

## Overview

The new architecture separates Morgan into:
- **morgan-server**: Standalone server with all core functionality
- **morgan-cli**: Lightweight terminal client
- **docker**: Containerized deployment

## Key Changes

### Architecture
- **Old**: Monolithic CLI with embedded core components
- **New**: Client-server with REST/WebSocket APIs

### Deployment
- **Old**: Single Python application
- **New**: Server + Client, can run on different machines

### Configuration
- **Old**: Mixed configuration in code
- **New**: Environment-based configuration with clear precedence

## Migration Steps

### 1. Install New Packages

```bash
# Install server
cd morgan-server
pip install -e .

# Install client (in a separate environment or same)
cd ../morgan-cli
pip install -e .
```

### 2. Configure Server

Create `morgan-server/.env`:

```bash
# Copy from old configuration
MORGAN_LLM_PROVIDER=ollama
MORGAN_LLM_ENDPOINT=http://localhost:11434
MORGAN_LLM_MODEL=gemma3
MORGAN_VECTOR_DB_URL=http://localhost:6333
MORGAN_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 3. Start Services

**Option A: Docker (Recommended)**

```bash
cd docker
docker-compose up -d
docker exec -it morgan-ollama ollama pull gemma3
```

**Option B: Manual**

```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Start Ollama
ollama serve

# Start Morgan Server
cd morgan-server
morgan-server
```

### 4. Use New CLI

```bash
# Configure client
export MORGAN_SERVER_URL=http://localhost:8080

# Start chatting
morgan chat
```

## Feature Mapping

### Old CLI Commands → New CLI Commands

| Old | New | Notes |
|-----|-----|-------|
| `python cli.py` | `morgan chat` | Interactive chat |
| N/A | `morgan ask "question"` | Single question |
| N/A | `morgan learn --file doc.pdf` | Document ingestion |
| N/A | `morgan memory stats` | Memory management |
| N/A | `morgan knowledge search` | Knowledge search |
| N/A | `morgan health` | Server health |

### Old Code → New Architecture

| Old Component | New Location | Notes |
|---------------|--------------|-------|
| `cli.py` | `morgan-cli/morgan_cli/cli.py` | Rewritten as pure client |
| `morgan/cli/click_cli.py` | Deprecated | Replaced by new CLI |
| Core components | `morgan-server/morgan_server/` | Server-side only |
| Emotional intelligence | `morgan-server/morgan_server/empathic/` | Enhanced |
| RAG system | `morgan-server/morgan_server/knowledge/` | Enhanced |
| Memory | `morgan-server/morgan_server/personalization/` | Enhanced |

## Data Migration

### Vector Database

If you have existing Qdrant data:

```bash
# Backup old data
docker exec qdrant-old /qdrant/qdrant --backup /backup

# Restore to new instance
docker cp qdrant-old:/backup ./backup
docker cp ./backup morgan-qdrant:/backup
docker exec morgan-qdrant /qdrant/qdrant --restore /backup
```

### Conversation History

The new system uses a different schema. To migrate:

1. Export old conversations (if applicable)
2. Use the new `/api/knowledge/learn` endpoint to re-ingest

## Configuration Reference

### Old Configuration
```python
# Scattered in code
LLM_ENDPOINT = "http://localhost:11434"
VECTOR_DB_URL = "http://localhost:6333"
```

### New Configuration
```bash
# Centralized in .env
MORGAN_HOST=0.0.0.0
MORGAN_PORT=8080
MORGAN_LLM_PROVIDER=ollama
MORGAN_LLM_ENDPOINT=http://localhost:11434
MORGAN_LLM_MODEL=gemma3
MORGAN_VECTOR_DB_URL=http://localhost:6333
MORGAN_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
MORGAN_CACHE_DIR=./data/cache
MORGAN_LOG_LEVEL=INFO
```

## API Access

The new architecture exposes APIs for custom clients:

```python
import aiohttp

async with aiohttp.ClientSession() as session:
    # Chat
    async with session.post(
        "http://localhost:8080/api/chat",
        json={"message": "Hello!", "user_id": "user123"}
    ) as resp:
        data = await resp.json()
        print(data["answer"])
```

See API documentation at http://localhost:8080/docs

## Troubleshooting

### Server won't start
- Check configuration: `morgan-server --check-config`
- Verify Qdrant is running: `curl http://localhost:6333`
- Verify Ollama is running: `curl http://localhost:11434`

### Client can't connect
- Check server URL: `echo $MORGAN_SERVER_URL`
- Test server health: `curl http://localhost:8080/health`
- Check firewall settings

### Performance issues
- Monitor metrics: http://localhost:8080/metrics
- Check resource usage: `docker stats`
- Review logs: `docker-compose logs -f morgan-server`

## Rollback

To rollback to the old system:

1. Stop new services: `docker-compose down`
2. Use old CLI: `python cli.py`
3. Keep old data intact (separate directories)

## Support

For issues or questions:
- Check documentation in `morgan-server/README.md` and `morgan-cli/README.md`
- Review API docs at http://localhost:8080/docs
- Check logs for error messages

## Next Steps

1. Test the new system with your existing workflows
2. Gradually migrate data and configurations
3. Update any custom integrations to use the new APIs
4. Consider containerized deployment for production

## Documentation

### Server Documentation

- **[Server README](./morgan-server/README.md)** - Overview and quick start
- **[Configuration Guide](./morgan-server/docs/CONFIGURATION.md)** - Complete configuration reference
- **[Embedding Configuration](./morgan-server/docs/EMBEDDING_CONFIGURATION.md)** - Embedding provider setup
- **[Deployment Guide](./morgan-server/docs/DEPLOYMENT.md)** - Docker and bare metal deployment
- **[API Documentation](./morgan-server/docs/API.md)** - REST and WebSocket API reference

### Client Documentation

- **[Client README](./morgan-cli/README.md)** - Client overview and usage
- **[Client API Reference](./morgan-cli/README.md#api-methods)** - Python client library

### Docker Documentation

- **[Docker README](./docker/README.md)** - Docker deployment guide
