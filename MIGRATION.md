# Migration Guide

**Last Updated:** December 26, 2025

This guide helps you migrate from older Morgan implementations to the current architecture.

## Overview

Morgan now uses a clean, modular architecture:

- **morgan-rag**: Core intelligence (services, emotional intelligence, memory, search)
- **morgan-server**: FastAPI server with REST/WebSocket API
- **morgan-cli**: Terminal client
- **docker**: Containerized deployment

## Quick Start (New Installation)

If you're starting fresh, skip migration and just install:

```bash
# Start services with Docker
cd docker
cp env.example .env
docker-compose up -d

# Pull LLM model
docker-compose exec ollama ollama pull qwen2.5:7b

# Install CLI
pip install -e ../morgan-cli

# Start chatting
export MORGAN_SERVER_URL=http://localhost:8080
morgan chat
```

## Migration from Old System

### 1. Update Imports

If you have code that imports from old service locations, update to the new unified services:

**Old imports:**
```python
# These no longer work
from morgan.services.llm_service import LLMService
from morgan.services.distributed_llm_service import DistributedLLMService
from morgan.embeddings.service import EmbeddingService
from morgan.infrastructure.local_embeddings import LocalEmbeddingService
from morgan.infrastructure.local_reranking import LocalRerankingService
```

**New imports:**
```python
# Use these instead
from morgan.services import (
    get_llm_service,
    get_embedding_service,
    get_reranking_service,
)

# Or import classes directly
from morgan.services.llm import LLMService, LLMResponse, LLMMode
from morgan.services.embeddings import EmbeddingService
from morgan.services.reranking import RerankingService
```

### 2. Update Service Usage

**Old usage:**
```python
# Old LLM service
llm = LLMService(endpoint="http://localhost:11434/v1")
response = llm.generate("Hello")

# Old embedding service
embeddings = EmbeddingService()
vector = embeddings.embed("Text")
```

**New usage:**
```python
# New unified services
from morgan.services import get_llm_service, get_embedding_service

llm = get_llm_service()
response = llm.generate("Hello")
print(response.content)

embeddings = get_embedding_service()
vector = embeddings.encode("Text")
```

### 3. Update Configuration

**Old configuration (scattered):**
```python
# Hardcoded in various files
LLM_ENDPOINT = "http://localhost:11434/v1"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

**New configuration (centralized):**
```bash
# Environment variables
export MORGAN_LLM_ENDPOINT=http://localhost:11434/v1
export MORGAN_LLM_MODEL=qwen2.5:7b
export MORGAN_EMBEDDING_ENDPOINT=http://localhost:11434/v1
export MORGAN_EMBEDDING_MODEL=qwen3-embedding:4b
```

Or use the defaults module:
```python
from morgan.config.defaults import Defaults

endpoint = Defaults.LLM_ENDPOINT
model = Defaults.LLM_MODEL
```

### 4. Update Exception Handling

**Old exception handling:**
```python
try:
    response = llm.generate("Hello")
except Exception as e:
    print(f"Error: {e}")
```

**New exception handling:**
```python
from morgan.exceptions import LLMServiceError, MorganError

try:
    response = llm.generate("Hello")
except LLMServiceError as e:
    print(f"LLM error: {e.message}")
    print(f"Operation: {e.operation}")
except MorganError as e:
    print(f"Morgan error: {e}")
```

## Feature Mapping

### Services

| Old Location | New Location | Notes |
|-------------|--------------|-------|
| `services/llm_service.py` | `services/llm/` | Unified with distributed |
| `services/distributed_llm_service.py` | `services/llm/` | Merged |
| `embeddings/service.py` | `services/embeddings/` | Unified with fallback |
| `infrastructure/local_embeddings.py` | `services/embeddings/` | Merged |
| `infrastructure/local_reranking.py` | `services/reranking/` | Merged |
| `jina/reranking/service.py` | `services/reranking/` | Merged |

### Utilities

| Old Pattern | New Location | Notes |
|-------------|--------------|-------|
| Repeated singletons | `utils/singleton.py` | SingletonFactory |
| Repeated cache setup | `utils/model_cache.py` | Unified cache |
| Repeated deduplication | `utils/deduplication.py` | ResultDeduplicator |

### Configuration

| Old Pattern | New Location | Notes |
|-------------|--------------|-------|
| Hardcoded values | `config/defaults.py` | Centralized defaults |
| Scattered env vars | `config/settings.py` | Unified settings |

## Data Migration

### Vector Database

If you have existing Qdrant data, it should continue to work. The new services use the same vector database.

### Conversation History

Conversation history stored in the memory system should continue to work with the new architecture.

## Troubleshooting

### Import Errors

If you get import errors like:
```
ModuleNotFoundError: No module named 'morgan.services.llm_service'
```

Update your imports to use the new paths (see section 1 above).

### Service Not Found

If services aren't being found:
```python
# Make sure you're using the factory functions
from morgan.services import get_llm_service

llm = get_llm_service()  # Creates singleton instance
```

### Configuration Issues

If configuration isn't being loaded:
```bash
# Check environment variables
echo $MORGAN_LLM_ENDPOINT

# Or use defaults
python -c "from morgan.config.defaults import Defaults; print(Defaults.LLM_ENDPOINT)"
```

## Getting Help

1. **Check Documentation** - [DOCUMENTATION.md](./DOCUMENTATION.md)
2. **Check Project Context** - [claude.md](./claude.md)
3. **Check Architecture** - [morgan-rag/docs/ARCHITECTURE.md](./morgan-rag/docs/ARCHITECTURE.md)
4. **GitHub Issues** - Report problems or ask questions

---

**Morgan** - Your private, emotionally intelligent AI companion.
