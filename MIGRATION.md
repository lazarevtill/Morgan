# Migration Guide

**Last Updated**: December 26, 2025

This guide helps you migrate from older Morgan implementations to the current unified architecture.

---

## Overview

### Architecture Evolution

```mermaid
graph LR
    subgraph Old["Old Architecture"]
        direction TB
        O1[llm_service.py]
        O2[distributed_llm_service.py]
        O3[embeddings/service.py]
        O4[local_embeddings.py]
        O5[local_reranking.py]
    end
    
    subgraph New["New Architecture"]
        direction TB
        N1[services/llm/]
        N2[services/embeddings/]
        N3[services/reranking/]
    end
    
    O1 & O2 -->|Consolidated| N1
    O3 & O4 -->|Consolidated| N2
    O5 -->|Consolidated| N3
```

### Migration Flow

```mermaid
flowchart TD
    Start[Start Migration] --> Check{Using Old Imports?}
    
    Check -->|Yes| Update[Update Imports]
    Check -->|No| Skip[Skip to Usage]
    
    Update --> Config{Update Config?}
    Config -->|Yes| Env[Update Environment]
    Config -->|No| Usage[Update Usage Patterns]
    
    Env --> Usage
    Usage --> Test[Run Tests]
    Test --> Done[Migration Complete]
```

---

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

---

## Import Migration

### Import Path Changes

```mermaid
flowchart LR
    subgraph Old["Old Import Paths"]
        O1["morgan.services.llm_service"]
        O2["morgan.services.distributed_llm_service"]
        O3["morgan.embeddings.service"]
        O4["morgan.infrastructure.local_embeddings"]
        O5["morgan.infrastructure.local_reranking"]
    end
    
    subgraph New["New Import Paths"]
        N1["morgan.services.llm"]
        N2["morgan.services.embeddings"]
        N3["morgan.services.reranking"]
    end
    
    O1 & O2 -->|Replace with| N1
    O3 & O4 -->|Replace with| N2
    O5 -->|Replace with| N3
```

### LLM Service Migration

**Old imports (no longer work):**

```python
# ❌ These imports are removed
from morgan.services.llm_service import LLMService
from morgan.services.distributed_llm_service import DistributedLLMService
```

**New imports:**

```python
# ✅ Use these instead
from morgan.services import get_llm_service
from morgan.services.llm import LLMService, LLMResponse, LLMMode
```

### Embedding Service Migration

**Old imports (no longer work):**

```python
# ❌ These imports are removed
from morgan.embeddings.service import EmbeddingService
from morgan.infrastructure.local_embeddings import LocalEmbeddingService
from morgan.services.distributed_embedding_service import DistributedEmbeddingService
```

**New imports:**

```python
# ✅ Use these instead
from morgan.services import get_embedding_service
from morgan.services.embeddings import EmbeddingService
```

### Reranking Service Migration

**Old imports (no longer work):**

```python
# ❌ These imports are removed
from morgan.infrastructure.local_reranking import LocalRerankingService
from morgan.jina.reranking.service import JinaRerankingService
```

**New imports:**

```python
# ✅ Use these instead
from morgan.services import get_reranking_service
from morgan.services.reranking import RerankingService, RerankResult
```

---

## Usage Pattern Migration

### Service Initialization

```mermaid
sequenceDiagram
    participant Old as Old Pattern
    participant New as New Pattern
    
    Note over Old: Manual instantiation
    Old->>Old: service = LLMService(endpoint=...)
    
    Note over New: Factory pattern (singleton)
    New->>New: service = get_llm_service()
```

**Old pattern:**

```python
# ❌ Old: Manual instantiation
llm = LLMService(endpoint="http://localhost:11434/v1")
embeddings = EmbeddingService(model="all-MiniLM-L6-v2")
```

**New pattern:**

```python
# ✅ New: Factory functions (singletons)
from morgan.services import get_llm_service, get_embedding_service

llm = get_llm_service()  # Uses defaults or environment
embeddings = get_embedding_service()
```

### LLM Generation

**Old pattern:**

```python
# ❌ Old
response = llm.generate("Hello")
text = response["content"]
```

**New pattern:**

```python
# ✅ New: Typed response object
response = llm.generate("Hello")
text = response.content  # LLMResponse object
model = response.model
usage = response.usage
```

### Embedding Generation

**Old pattern:**

```python
# ❌ Old
vector = embeddings.embed("Text")
vectors = embeddings.embed_batch(["Text 1", "Text 2"])
```

**New pattern:**

```python
# ✅ New: Consistent naming
vector = embeddings.encode("Text")
vectors = embeddings.encode_batch(["Text 1", "Text 2"])

# Async versions
vector = await embeddings.aencode("Text")
vectors = await embeddings.aencode_batch(["Text 1", "Text 2"])
```

### Reranking

**Old pattern:**

```python
# ❌ Old
results = reranker.rerank(query, documents)
for doc, score in results:
    print(f"{score}: {doc}")
```

**New pattern:**

```python
# ✅ New: Typed result objects
results = await reranking.rerank(query, documents, top_k=10)
for result in results:
    print(f"{result.score}: {result.text}")
    print(f"  Original index: {result.index}")
```

---

## Configuration Migration

### Configuration Flow

```mermaid
flowchart TD
    subgraph Old["Old Configuration"]
        O1[Hardcoded in code]
        O2[Scattered .env files]
        O3[Multiple config locations]
    end
    
    subgraph New["New Configuration"]
        N1[config/defaults.py]
        N2[Environment variables]
        N3[Centralized settings]
    end
    
    O1 -->|Move to| N1
    O2 -->|Standardize to| N2
    O3 -->|Consolidate to| N3
```

### Environment Variables

**Old (scattered):**

```bash
# Various names in different places
LLM_ENDPOINT=http://localhost:11434/v1
EMBEDDING_MODEL=all-MiniLM-L6-v2
QDRANT_HOST=localhost
```

**New (standardized):**

```bash
# Consistent MORGAN_ prefix
MORGAN_LLM_ENDPOINT=http://localhost:11434/v1
MORGAN_LLM_MODEL=qwen2.5:7b
MORGAN_EMBEDDING_ENDPOINT=http://localhost:11434/v1
MORGAN_EMBEDDING_MODEL=qwen3-embedding:4b
MORGAN_QDRANT_URL=http://localhost:6333
MORGAN_REDIS_URL=redis://localhost:6379
```

### Using Defaults

```python
from morgan.config.defaults import Defaults

# Access default values
endpoint = Defaults.LLM_ENDPOINT
model = Defaults.LLM_MODEL
dimensions = Defaults.EMBEDDING_DIMENSIONS
```

---

## Exception Handling Migration

### Exception Hierarchy

```mermaid
classDiagram
    class MorganError {
        +message: str
        +service: str
        +operation: str
    }
    
    class LLMServiceError
    class EmbeddingServiceError
    class RerankingServiceError
    class ConfigurationError
    class ValidationError
    
    MorganError <|-- LLMServiceError
    MorganError <|-- EmbeddingServiceError
    MorganError <|-- RerankingServiceError
    MorganError <|-- ConfigurationError
    MorganError <|-- ValidationError
```

**Old pattern:**

```python
# ❌ Old: Generic exceptions
try:
    response = llm.generate("Hello")
except Exception as e:
    print(f"Error: {e}")
```

**New pattern:**

```python
# ✅ New: Typed exceptions
from morgan.exceptions import LLMServiceError, MorganError

try:
    response = llm.generate("Hello")
except LLMServiceError as e:
    print(f"LLM error: {e.message}")
    print(f"Service: {e.service}")
    print(f"Operation: {e.operation}")
except MorganError as e:
    print(f"Morgan error: {e}")
```

---

## Feature Mapping

### Services

```mermaid
graph LR
    subgraph Old["Old Services"]
        O1[llm_service.py]
        O2[distributed_llm_service.py]
        O3[embeddings/service.py]
        O4[distributed_embedding_service.py]
        O5[local_embeddings.py]
        O6[local_reranking.py]
        O7[jina/reranking/service.py]
    end
    
    subgraph New["New Services"]
        N1[services/llm/<br/>Unified LLM]
        N2[services/embeddings/<br/>Unified Embeddings]
        N3[services/reranking/<br/>Unified Reranking]
    end
    
    O1 & O2 --> N1
    O3 & O4 & O5 --> N2
    O6 & O7 --> N3
```

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

---

## Troubleshooting

### Common Issues

```mermaid
flowchart TD
    E1[Import Error] --> S1[Update import paths]
    E2[Service Not Found] --> S2[Use factory functions]
    E3[Config Not Loading] --> S3[Check env var names]
    E4[Type Error] --> S4[Use new response objects]
```

### Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'morgan.services.llm_service'
```

**Solution:**
```python
# Update to new import path
from morgan.services import get_llm_service
llm = get_llm_service()
```

### Service Not Found

**Error:**
```
AttributeError: 'NoneType' object has no attribute 'generate'
```

**Solution:**
```python
# Use factory function, not direct import
from morgan.services import get_llm_service
llm = get_llm_service()  # Creates singleton
```

### Configuration Issues

**Error:**
```
ConfigurationError: LLM_ENDPOINT not found
```

**Solution:**
```bash
# Use MORGAN_ prefix
export MORGAN_LLM_ENDPOINT=http://localhost:11434/v1
```

---

## Verification

### Test Migration

```bash
# Run verification script
cd morgan-rag
python -c "
from morgan.services import get_llm_service, get_embedding_service, get_reranking_service

llm = get_llm_service()
emb = get_embedding_service()
rrk = get_reranking_service()

print('✅ All services loaded successfully')
print(f'  LLM: {type(llm).__name__}')
print(f'  Embeddings: {type(emb).__name__}')
print(f'  Reranking: {type(rrk).__name__}')
"
```

### Run Tests

```bash
cd morgan-rag
pytest tests/ -v
```

---

## Getting Help

1. **Check Documentation** - [DOCUMENTATION.md](./DOCUMENTATION.md)
2. **Check Project Context** - [claude.md](./claude.md)
3. **Check Architecture** - [morgan-rag/docs/ARCHITECTURE.md](./morgan-rag/docs/ARCHITECTURE.md)
4. **GitHub Issues** - Report problems or ask questions

---

**Morgan** - Your private, emotionally intelligent AI companion.
