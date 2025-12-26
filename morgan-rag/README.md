# Morgan RAG - Core Intelligence Engine

**Last Updated:** December 26, 2025

Morgan RAG is the core intelligence engine of the Morgan AI Assistant. It provides unified services for LLM, embeddings, and reranking, along with emotional intelligence, memory management, and multi-stage search.

## Overview

Morgan RAG is a **human-first** AI assistant built with KISS principles (Keep It Simple, Stupid). It provides:

- **Unified Services Layer** - Clean access to LLM, embeddings, and reranking
- **Emotional Intelligence** - Emotion detection, empathy, and relationship management
- **Memory System** - Conversation memory with emotional context
- **Search Pipeline** - Multi-stage semantic search with reranking
- **Distributed Infrastructure** - Scale across multiple hosts

## Quick Start

### Installation

```bash
cd morgan-rag
pip install -e .
```

### Basic Usage

```python
from morgan.services import (
    get_llm_service,
    get_embedding_service,
    get_reranking_service,
)

# Get service instances (singletons)
llm = get_llm_service()
embeddings = get_embedding_service()
reranking = get_reranking_service()

# LLM Generation
response = llm.generate("What is Python?")
print(response.content)

# Async LLM Generation
response = await llm.agenerate("Explain Docker")

# Embeddings
embedding = embeddings.encode("Document text")
embeddings_batch = embeddings.encode_batch(["Doc 1", "Doc 2"])

# Reranking
results = await reranking.rerank(
    query="Python programming",
    documents=["Doc 1", "Doc 2", "Doc 3"],
    top_k=10
)
```

## Architecture

```
morgan-rag/morgan/
├── services/                    # Unified Services Layer
│   ├── llm/                     # LLM service (single + distributed)
│   ├── embeddings/              # Embedding service (remote + local)
│   ├── reranking/               # Reranking service (4-level fallback)
│   └── external_knowledge/      # External knowledge sources
│
├── intelligence/                # Emotional Intelligence
│   ├── emotions/                # Emotion detection
│   ├── empathy/                 # Empathic responses
│   └── core/                    # Intelligence engine
│
├── memory/                      # Memory System
├── search/                      # Search Pipeline
├── learning/                    # Pattern Learning
├── companion/                   # Relationship Management
├── reasoning/                   # Multi-step Reasoning
├── proactive/                   # Proactive Assistance
│
├── infrastructure/              # Distributed Infrastructure
│   ├── distributed_llm.py       # Load balancing
│   ├── distributed_gpu_manager.py
│   └── factory.py               # Infrastructure factory
│
├── config/                      # Configuration
│   ├── defaults.py              # Default values
│   ├── settings.py              # Application settings
│   └── distributed_config.py    # Distributed deployment
│
├── utils/                       # Utilities
│   ├── singleton.py             # Singleton factory
│   ├── model_cache.py           # Model caching
│   ├── deduplication.py         # Result deduplication
│   └── logger.py                # Logging
│
└── exceptions.py                # Exception hierarchy
```

## Services

### LLM Service

```python
from morgan.services import get_llm_service

# Single mode (default)
llm = get_llm_service()
response = llm.generate("Hello!")

# Distributed mode
llm = get_llm_service(
    mode="distributed",
    endpoints=["http://host1:11434/v1", "http://host2:11434/v1"]
)

# With fast model for simple queries
response = llm.generate("Hi!", use_fast_model=True)

# Streaming
async for chunk in llm.astream("Tell me a story"):
    print(chunk, end="")
```

### Embedding Service

```python
from morgan.services import get_embedding_service

embeddings = get_embedding_service()

# Single embedding
vector = embeddings.encode("Text to embed")

# Batch embeddings
vectors = embeddings.encode_batch(["Doc 1", "Doc 2", "Doc 3"])

# Async
vector = await embeddings.aencode("Async text")
```

### Reranking Service

```python
from morgan.services import get_reranking_service

reranking = get_reranking_service()

# Rerank documents
results = await reranking.rerank(
    query="Python programming",
    documents=["Doc 1", "Doc 2", "Doc 3"],
    top_k=10
)

for result in results:
    print(f"{result.score:.3f}: {result.text[:50]}...")
```

## Configuration

### Environment Variables

```bash
# LLM
MORGAN_LLM_ENDPOINT=http://localhost:11434/v1
MORGAN_LLM_MODEL=qwen2.5:32b-instruct-q4_K_M
MORGAN_LLM_FAST_MODEL=qwen2.5:7b-instruct-q5_K_M

# Embeddings
MORGAN_EMBEDDING_ENDPOINT=http://localhost:11434/v1
MORGAN_EMBEDDING_MODEL=qwen3-embedding:4b
MORGAN_EMBEDDING_DIMENSIONS=2048

# Reranking
MORGAN_RERANKING_ENDPOINT=http://localhost:8080/rerank
MORGAN_RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Vector Database
MORGAN_QDRANT_URL=http://localhost:6333

# Cache
MORGAN_REDIS_URL=redis://localhost:6379
MORGAN_MODEL_CACHE=~/.morgan/models
```

### Defaults Module

```python
from morgan.config.defaults import Defaults

# Access defaults
port = Defaults.MORGAN_PORT  # 8080
model = Defaults.LLM_MODEL   # "qwen2.5:32b-instruct-q4_K_M"
```

## Exception Handling

```python
from morgan.exceptions import (
    MorganError,           # Base exception
    LLMServiceError,       # LLM failures
    EmbeddingServiceError, # Embedding failures
    RerankingServiceError, # Reranking failures
)

try:
    response = llm.generate("Hello")
except LLMServiceError as e:
    print(f"LLM failed: {e.message}")
    print(f"Operation: {e.operation}")
except MorganError as e:
    print(f"Morgan error: {e}")
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=morgan tests/

# Run specific tests
pytest tests/test_llm_service.py
```

## Documentation

- [Architecture](./docs/ARCHITECTURE.md) - Detailed architecture documentation
- [Quick Start](./QUICK_START.md) - Quick start guide
- [Migration Guide](./docs/MIGRATION_GUIDE.md) - Migration guide

## Related Components

- **morgan-server** - FastAPI server with REST/WebSocket API
- **morgan-cli** - Terminal client
- **docker** - Docker deployment configurations

---

**Morgan RAG** - The intelligent core of Morgan AI Assistant.
