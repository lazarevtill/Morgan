# Morgan RAG - Architecture Documentation

**Last Updated**: December 26, 2025

## Overview

Morgan RAG is the core intelligence engine of the Morgan AI Assistant. It provides:

- **Unified Services Layer** - Clean access to LLM, embeddings, and reranking
- **Emotional Intelligence** - Emotion detection, empathy, and relationship management
- **Memory System** - Conversation memory with emotional context
- **Search Pipeline** - Multi-stage semantic search with reranking
- **Distributed Infrastructure** - Scale across multiple hosts

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Morgan RAG                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      Services Layer                              │    │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐    │    │
│  │  │     LLM     │  │  Embeddings  │  │     Reranking       │    │    │
│  │  │   Service   │  │   Service    │  │      Service        │    │    │
│  │  │             │  │              │  │                     │    │    │
│  │  │ - Single    │  │ - Remote     │  │ - Remote endpoint   │    │    │
│  │  │ - Distributed│ │ - Local      │  │ - CrossEncoder      │    │    │
│  │  │ - Streaming │  │ - Batch      │  │ - Embedding-based   │    │    │
│  │  │             │  │ - Caching    │  │ - BM25 fallback     │    │    │
│  │  └─────────────┘  └──────────────┘  └─────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Intelligence Layer                            │    │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐    │    │
│  │  │  Emotional  │  │    Memory    │  │      Search         │    │    │
│  │  │Intelligence │  │    System    │  │     Pipeline        │    │    │
│  │  │             │  │              │  │                     │    │    │
│  │  │ - Emotions  │  │ - Processor  │  │ - Multi-stage       │    │    │
│  │  │ - Empathy   │  │ - Storage    │  │ - Reranking         │    │    │
│  │  │ - Relations │  │ - Context    │  │ - Deduplication     │    │    │
│  │  └─────────────┘  └──────────────┘  └─────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   Infrastructure Layer                           │    │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐    │    │
│  │  │ Distributed │  │     GPU      │  │     Factory         │    │    │
│  │  │     LLM     │  │   Manager    │  │                     │    │    │
│  │  │             │  │              │  │ - Service creation  │    │    │
│  │  │ - Load bal  │  │ - Multi-GPU  │  │ - Configuration     │    │    │
│  │  │ - Failover  │  │ - Distributed│  │ - Health checks     │    │    │
│  │  │ - Health    │  │ - Monitoring │  │                     │    │    │
│  │  └─────────────┘  └──────────────┘  └─────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        External Services                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐            │
│  │   Ollama    │  │    Qdrant    │  │       Redis         │            │
│  │   (LLM)     │  │  (Vector DB) │  │      (Cache)        │            │
│  └─────────────┘  └──────────────┘  └─────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
morgan-rag/morgan/
├── services/                    # Unified Services Layer
│   ├── __init__.py              # Service exports and initialization
│   ├── llm/                     # LLM Service
│   │   ├── __init__.py
│   │   ├── models.py            # LLMResponse, LLMMode
│   │   └── service.py           # LLMService class
│   ├── embeddings/              # Embedding Service
│   │   ├── __init__.py
│   │   ├── models.py            # EmbeddingStats
│   │   └── service.py           # EmbeddingService class
│   ├── reranking/               # Reranking Service
│   │   ├── __init__.py
│   │   ├── models.py            # RerankResult, RerankStats
│   │   └── service.py           # RerankingService class
│   └── external_knowledge/      # External Knowledge Sources
│       ├── context7.py          # Context7 integration
│       ├── mcp_client.py        # MCP client
│       └── web_search.py        # Web search
│
├── infrastructure/              # Infrastructure Layer
│   ├── __init__.py              # Infrastructure exports
│   ├── distributed_llm.py       # Distributed LLM with load balancing
│   ├── distributed_gpu_manager.py # Distributed GPU management
│   ├── multi_gpu_manager.py     # Single-host GPU management
│   └── factory.py               # Infrastructure factory
│
├── intelligence/                # Emotional Intelligence
│   ├── emotions/                # Emotion detection
│   │   ├── detector.py          # EmotionDetector
│   │   └── memory.py            # Emotional memory
│   ├── empathy/                 # Empathic responses
│   │   ├── generator.py         # EmpathyGenerator
│   │   ├── mirror.py            # EmotionalMirror
│   │   ├── support.py           # EmotionalSupport
│   │   ├── tone.py              # ToneAnalyzer
│   │   └── validator.py         # ResponseValidator
│   └── core/                    # Intelligence engine
│       ├── intelligence_engine.py
│       └── models.py            # Core models
│
├── memory/                      # Memory System
│   ├── memory_processor.py      # Memory processing
│   └── ...
│
├── search/                      # Search Pipeline
│   ├── multi_stage_search.py    # Multi-stage search
│   ├── reranker.py              # Search reranker
│   └── ...
│
├── learning/                    # Pattern Learning
│   ├── engine.py                # Learning engine
│   ├── patterns.py              # Pattern analysis
│   └── ...
│
├── companion/                   # Relationship Management
│   ├── relationship_manager.py
│   └── ...
│
├── reasoning/                   # Multi-step Reasoning
│   ├── engine.py                # Reasoning engine
│   ├── planner.py               # Task planner
│   └── ...
│
├── proactive/                   # Proactive Assistance
│   ├── anticipator.py           # Need anticipation
│   ├── suggestions.py           # Contextual suggestions
│   └── ...
│
├── communication/               # Communication Preferences
│   ├── style.py                 # Communication style
│   ├── preferences.py           # User preferences
│   └── feedback.py              # Feedback handling
│
├── config/                      # Configuration
│   ├── __init__.py
│   ├── defaults.py              # Default values
│   ├── settings.py              # Application settings
│   └── distributed_config.py    # Distributed deployment
│
├── utils/                       # Utilities
│   ├── logger.py                # Logging
│   ├── singleton.py             # Singleton factory
│   ├── model_cache.py           # Model caching
│   ├── deduplication.py         # Result deduplication
│   ├── error_handling.py        # Error handling
│   └── ...
│
├── core/                        # Core Assistant Logic
│   ├── assistant.py             # MorganAssistant
│   ├── emotional_processor.py   # Emotional processing
│   ├── milestone_tracker.py     # Milestone tracking
│   └── application/             # Application orchestrators
│
├── exceptions.py                # Exception hierarchy
│
└── ...
```

## Services Layer

### LLM Service

The LLM service provides text generation with support for single and distributed modes.

**Features:**
- Single endpoint mode (OpenAI-compatible)
- Distributed mode with load balancing
- Automatic failover on errors
- Fast model support for simple queries
- Both sync and async interfaces
- Streaming support

**Usage:**
```python
from morgan.services import get_llm_service

llm = get_llm_service()

# Sync generation
response = llm.generate("What is Python?")
print(response.content)

# Async generation
response = await llm.agenerate("Explain Docker")

# With fast model
response = llm.generate("Hi!", use_fast_model=True)

# Distributed mode
llm = get_llm_service(
    mode="distributed",
    endpoints=["http://host1:11434/v1", "http://host2:11434/v1"]
)
```

### Embedding Service

The embedding service provides text embeddings with remote and local fallback.

**Features:**
- Remote Ollama/OpenAI-compatible endpoints (primary)
- Local sentence-transformers fallback
- Automatic failover between providers
- Batch processing with configurable size
- Content-based caching
- Both sync and async interfaces

**Usage:**
```python
from morgan.services import get_embedding_service

embeddings = get_embedding_service()

# Single embedding
vector = embeddings.encode("Document text")

# Batch embeddings
vectors = embeddings.encode_batch(["Doc 1", "Doc 2", "Doc 3"])

# Async
vector = await embeddings.aencode("Async text")
```

### Reranking Service

The reranking service provides document reranking with multiple fallback strategies.

**Features:**
- Remote reranking endpoint (primary)
- Local CrossEncoder fallback
- Embedding-based similarity fallback
- BM25-style lexical matching (last resort)
- Both sync and async interfaces

**Fallback Hierarchy:**
1. Remote FastAPI endpoint
2. Local CrossEncoder (sentence-transformers)
3. Embedding similarity (cosine)
4. BM25 lexical matching

**Usage:**
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

## Exception Hierarchy

Morgan provides a consistent exception hierarchy:

```python
MorganError                    # Base exception
├── LLMServiceError            # LLM failures
├── EmbeddingServiceError      # Embedding failures
├── RerankingServiceError      # Reranking failures
├── VectorDBError              # Vector database errors
├── MemoryServiceError         # Memory service errors
├── SearchServiceError         # Search pipeline errors
├── ConfigurationError         # Configuration issues
├── ValidationError            # Input validation errors
└── InfrastructureError        # Infrastructure errors
    ├── ConnectionError        # Connection failures
    └── TimeoutError           # Operation timeouts
```

**Usage:**
```python
from morgan.exceptions import LLMServiceError, MorganError

try:
    response = llm.generate("Hello")
except LLMServiceError as e:
    print(f"LLM failed: {e.message}")
    print(f"Operation: {e.operation}")
except MorganError as e:
    print(f"Morgan error: {e}")
```

## Design Principles

### 1. KISS (Keep It Simple)
- Single responsibility per module
- Clean, well-defined interfaces
- Minimal abstraction layers

### 2. Privacy First
- All processing on local hardware
- No external API dependencies (except self-hosted services)
- Data stays on your infrastructure

### 3. Fault Tolerance
- Multiple fallback strategies
- Automatic failover
- Health monitoring

### 4. Quality Over Speed
- 5-10s response time acceptable for thoughtful responses
- Comprehensive search and reranking
- Emotional intelligence processing

### 5. Modular Enhancement
- Keep excellent existing code
- Add new capabilities without breaking changes
- Clean separation of concerns

## Configuration

### Environment Variables

```bash
# LLM Configuration
MORGAN_LLM_ENDPOINT=http://localhost:11434/v1
MORGAN_LLM_MODEL=qwen2.5:32b-instruct-q4_K_M
MORGAN_LLM_FAST_MODEL=qwen2.5:7b-instruct-q5_K_M
MORGAN_LLM_MODE=single  # or "distributed"

# Embedding Configuration
MORGAN_EMBEDDING_ENDPOINT=http://localhost:11434/v1
MORGAN_EMBEDDING_MODEL=qwen3-embedding:4b
MORGAN_EMBEDDING_DIMENSIONS=2048

# Reranking Configuration
MORGAN_RERANKING_ENDPOINT=http://localhost:8080/rerank
MORGAN_RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Vector Database
MORGAN_QDRANT_URL=http://localhost:6333

# Cache
MORGAN_REDIS_URL=redis://localhost:6379
MORGAN_MODEL_CACHE=~/.morgan/models
```

### Defaults Module

Default values are centralized in `morgan/config/defaults.py`:

```python
from morgan.config.defaults import Defaults

# Access defaults
port = Defaults.MORGAN_PORT  # 8080
model = Defaults.LLM_MODEL   # "qwen2.5:32b-instruct-q4_K_M"
```

## Testing

### Unit Tests
```bash
cd morgan-rag
pytest tests/
```

### Integration Tests
```bash
pytest tests/integration/
```

### Coverage
```bash
pytest --cov=morgan tests/
```

## Performance Targets

| Operation | Target | Status |
|-----------|--------|--------|
| Embeddings (batch) | <200ms | ✅ Achieved |
| Search + rerank | <500ms | ✅ Achieved |
| Simple queries | 1-2s | ⏳ Target |
| Complex reasoning | 5-10s | ⏳ Acceptable |

---

**Morgan RAG** - The intelligent core of Morgan AI Assistant.
