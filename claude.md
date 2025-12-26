# Morgan - Personal AI Assistant Project

**Status**: Phase 1 - Infrastructure Complete (83%)
**Version**: 2.0.0-alpha (Active Development)
**Last Updated**: December 26, 2025

---

## Project Overview

Morgan is a fully self-hosted, distributed personal AI assistant designed as an intelligent companion with:

- **Deep emotional intelligence** - Understands and responds empathetically (95% complete)
- **Multi-step reasoning** - Chain-of-thought planning and problem decomposition (planned)
- **Proactive assistance** - Anticipates needs and offers help before being asked (planned)
- **Complete privacy** - All processing on local hardware, zero external APIs
- **Continuous learning** - Adapts and improves from every interaction

**Key Principle**: Quality over speed (5-10s response time acceptable for thoughtful, accurate responses)

---

## Hardware Architecture (6 Hosts)

### CPU Hosts
- **Host 1** (i9, 64GB RAM): Morgan Core Orchestrator + Qdrant + Redis
- **Host 2** (i9, 64GB RAM): Background Services + Monitoring

### GPU Hosts
- **Host 3** (RTX 3090, 12GB): Main LLM #1 (Qwen2.5-32B)
- **Host 4** (RTX 3090, 12GB): Main LLM #2 (Load balanced)
- **Host 5** (RTX 4070, 8GB): Embeddings + Fast LLM
- **Host 6** (RTX 2060, 6GB): Reranking + Utilities

**Network**: All hosts on 192.168.1.x subnet, 1Gbps minimum bandwidth

---

## Technology Stack

### Self-Hosted Models
- **Main LLM**: Qwen2.5-32B-Instruct (Q4_K_M, ~19GB) - Complex reasoning
- **Fast LLM**: Qwen2.5-7B-Instruct (Q5_K_M, ~4.4GB) - Simple queries
- **Embeddings**: Qwen3-Embedding:4b (2048 dims) via Ollama - RAG and semantic search
- **Reranking**: CrossEncoder ms-marco-MiniLM-L-6-v2 (~90MB) - Result relevance

### Infrastructure
- **LLM Serving**: Ollama (OpenAI-compatible API)
- **Vector Database**: Qdrant
- **Caching**: Redis
- **Services**: FastAPI
- **Language**: Python 3.11+

### Distributed Architecture
- Load balancing: Round-robin, random, least-loaded strategies
- Automatic failover: 3 consecutive errors triggers unhealthy state
- Health monitoring: Background checks every 60s
- Performance tracking: Response times, success rates per endpoint

---

## Current Progress

### ✅ Phase 1: Infrastructure Complete (83%)

**Unified Services Layer (NEW - December 2025):**

All services have been consolidated into a clean, unified architecture:

```
morgan-rag/morgan/services/
├── __init__.py           # Unified access: get_llm_service(), get_embedding_service(), etc.
├── llm/                  # LLM service (single + distributed modes)
│   ├── __init__.py
│   ├── models.py         # LLMResponse, LLMMode
│   └── service.py        # Unified LLMService class
├── embeddings/           # Embedding service (remote + local fallback)
│   ├── __init__.py
│   ├── models.py         # EmbeddingStats
│   └── service.py        # Unified EmbeddingService class
├── reranking/            # Reranking service (4-level fallback)
│   ├── __init__.py
│   ├── models.py         # RerankResult, RerankStats
│   └── service.py        # Unified RerankingService class
└── external_knowledge/   # MCP, Context7, Web Search
```

**Infrastructure Layer:**
```
morgan-rag/morgan/infrastructure/
├── distributed_llm.py        # ✅ Load balancing across LLM hosts
├── distributed_gpu_manager.py # ✅ Distributed GPU host management
├── multi_gpu_manager.py      # ✅ Single-host GPU management
└── factory.py                # ✅ Infrastructure factory
```

**Utilities:**
```
morgan-rag/morgan/utils/
├── singleton.py          # ✅ Thread-safe singleton factory
├── model_cache.py        # ✅ Unified model cache setup
├── deduplication.py      # ✅ Result deduplication utility
└── ...
```

**Configuration:**
```
morgan-rag/morgan/config/
├── defaults.py           # ✅ Centralized default values
├── settings.py           # ✅ Application settings
└── distributed_config.py # ✅ Distributed deployment config
```

**Exception Hierarchy:**
```
morgan-rag/morgan/exceptions.py  # ✅ MorganError base + service-specific exceptions
```

**Existing Strong Code (Preserved):**
- `morgan/intelligence/` - Emotional intelligence & empathy (excellent, 95% complete)
- `morgan/learning/` - Pattern learning & adaptation (strong)
- `morgan/memory/` - Conversation memory with emotional context (strong)
- `morgan/companion/` - Relationship management (good)
- `morgan/search/` - Multi-stage search pipeline (excellent)
- `morgan/reasoning/` - Multi-step reasoning (good)
- `morgan/proactive/` - Proactive assistance (good)

### ⏳ Remaining (Phase 1: 17%)

- [ ] Documentation updates (this task)
- [ ] Integration testing
- [ ] Performance validation

### ⏳ Planned (Phases 2-5)

**Phase 2 - Multi-Step Reasoning Enhancement:**
- Chain-of-thought reasoning engine improvements
- Task planning and decomposition
- Progress tracking system

**Phase 3 - Proactive Features:**
- Background monitoring service
- Task anticipation engine
- Contextual suggestion system

**Phase 4 - Enhanced Context:**
- Context aggregation across sources
- Temporal awareness (time/day patterns)
- Activity tracking and analysis

**Phase 5 - Polish & Production:**
- Personality consistency refinement
- End-to-end testing
- Production deployment

---

## Key Design Principles

1. **Privacy First** - All data stays on your hardware, no external APIs
2. **Quality Over Speed** - 5-10s for thoughtful responses is acceptable
3. **KISS (Keep It Simple)** - Simple, focused modules with clear responsibilities
4. **Modular Enhancement** - Keep excellent existing code, add missing capabilities
5. **Fault Tolerance** - Distributed architecture with failover and health monitoring

---

## Project Structure

```
Morgan/
├── morgan-rag/                    # Main RAG project
│   ├── morgan/
│   │   ├── services/              # ✅ Unified services layer (NEW)
│   │   │   ├── llm/               # LLM service
│   │   │   ├── embeddings/        # Embedding service
│   │   │   ├── reranking/         # Reranking service
│   │   │   └── external_knowledge/# External knowledge sources
│   │   │
│   │   ├── infrastructure/        # ✅ Distributed infrastructure
│   │   │   ├── distributed_llm.py
│   │   │   ├── distributed_gpu_manager.py
│   │   │   ├── multi_gpu_manager.py
│   │   │   └── factory.py
│   │   │
│   │   ├── intelligence/          # ✅ Emotional intelligence (excellent)
│   │   │   ├── emotions/          # Emotion detection
│   │   │   ├── empathy/           # Empathic responses
│   │   │   └── core/              # Intelligence engine
│   │   │
│   │   ├── learning/              # ✅ Pattern learning (strong)
│   │   ├── memory/                # ✅ Conversation memory (strong)
│   │   ├── companion/             # ✅ Relationship management (good)
│   │   ├── search/                # ✅ Multi-stage search (excellent)
│   │   ├── reasoning/             # ✅ Multi-step reasoning
│   │   ├── proactive/             # ✅ Proactive assistance
│   │   ├── communication/         # ✅ Communication preferences
│   │   │
│   │   ├── config/                # ✅ Configuration
│   │   │   ├── defaults.py
│   │   │   ├── settings.py
│   │   │   └── distributed_config.py
│   │   │
│   │   ├── utils/                 # ✅ Utilities
│   │   │   ├── singleton.py
│   │   │   ├── model_cache.py
│   │   │   ├── deduplication.py
│   │   │   └── logger.py
│   │   │
│   │   ├── core/                  # Core assistant logic
│   │   ├── exceptions.py          # ✅ Exception hierarchy
│   │   └── ...
│   │
│   ├── tests/                     # Test suite
│   ├── examples/                  # Usage examples
│   └── scripts/                   # Utility scripts
│
├── morgan-server/                 # Server component
├── morgan-cli/                    # CLI client
├── docker/                        # Docker configurations
│
├── archive/                       # Archived deprecated code
│   ├── deprecated-root-modules/   # Old CLI, services
│   ├── deprecated-modules/        # Old embeddings, infrastructure
│   └── abandoned-refactors/       # Incomplete refactors
│
├── shared/                        # Shared models and utilities
├── .kiro/                         # Kiro specs and planning docs
│
├── claude.md                      # ✅ This file (project context)
├── README.md                      # Project README
├── DOCUMENTATION.md               # Documentation index
└── MIGRATION.md                   # Migration guide
```

---

## Service Usage Examples

### Unified Services (Recommended)

```python
# Import from unified services layer
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

### Distributed LLM (Advanced)

```python
from morgan.services.llm import get_llm_service

# Distributed mode with multiple endpoints
llm = get_llm_service(
    mode="distributed",
    endpoints=[
        "http://192.168.1.20:11434/v1",  # Host 3 (3090 #1)
        "http://192.168.1.21:11434/v1"   # Host 4 (3090 #2)
    ]
)

response = await llm.agenerate(
    prompt="Explain quantum computing",
    temperature=0.7
)
```

### Infrastructure Factory (Full Stack)

```python
from morgan.infrastructure import get_infrastructure_services

# Initialize all services at once
services = get_infrastructure_services()

# Access individual services
llm = services.llm_service
embeddings = services.embedding_service
reranking = services.reranking_service

# Get combined stats
stats = services.get_stats()
```

---

## Exception Handling

Morgan provides a consistent exception hierarchy:

```python
from morgan.exceptions import (
    MorganError,           # Base exception
    LLMServiceError,       # LLM failures
    EmbeddingServiceError, # Embedding failures
    RerankingServiceError, # Reranking failures
    ConfigurationError,    # Configuration issues
    ValidationError,       # Input validation errors
)

try:
    response = llm.generate("Hello")
except LLMServiceError as e:
    print(f"LLM failed: {e.message}")
    print(f"Service: {e.service}")
    print(f"Operation: {e.operation}")
```

---

## Performance Targets

### Latency
- ✅ Embeddings: <200ms batch (achieved)
- ✅ Search + rerank: <500ms (achieved)
- ⏳ Simple queries: 1-2s (target)
- ⏳ Complex reasoning: 5-10s (acceptable)

### Resource Usage
- ⏳ GPU memory: <90% per host
- ⏳ CPU: <70% average
- ⏳ Uptime: >99.5%
- ✅ Network latency: +10-50ms (acceptable for distributed)

### User Experience
- ✅ Emotionally appropriate responses: >90%
- ⏳ Answer accuracy: >90% (target)
- ⏳ Reasoning coherence: >85% (target)
- ⏳ Proactive helpfulness: >70% (target)

---

## Development Guidelines

### Import Conventions

```python
# Services (preferred)
from morgan.services import get_llm_service, get_embedding_service
from morgan.services.llm import LLMService, LLMResponse

# Infrastructure (advanced)
from morgan.infrastructure import DistributedLLMClient, get_distributed_llm_client

# Exceptions
from morgan.exceptions import MorganError, LLMServiceError

# Configuration
from morgan.config import get_settings
from morgan.config.defaults import Defaults

# Utilities
from morgan.utils.logger import get_logger
from morgan.utils.singleton import SingletonFactory
```

### Testing Strategy

1. Unit tests for individual components
2. Integration tests for service interactions
3. End-to-end tests with real queries
4. Performance benchmarks against targets

### Git Workflow

- Branch: `main` (primary development)
- Feature branches for new work
- Clean commits with descriptive messages

---

## Archived Code

The following code has been archived (not deleted) in `/archive/`:

- `archive/deprecated-root-modules/` - Old CLI, standalone services
- `archive/deprecated-modules/embeddings/` - Old embeddings module
- `archive/deprecated-modules/infrastructure/` - Old local_embeddings.py, local_reranking.py
- `archive/abandoned-refactors/morgan_v2/` - Incomplete Clean Architecture attempt

---

## Success Criteria

**Phase 1 Complete When:**
- ✅ All infrastructure services implemented
- ✅ Service consolidation complete
- ✅ Exception hierarchy created
- ⏳ Integration tests passing
- ⏳ Documentation updated

**Overall Project Complete When:**
- ✅ Emotionally intelligent responses
- ⏳ Multi-step reasoning works well
- ⏳ Proactive suggestions are helpful
- ⏳ Feels like talking to a knowledgeable assistant
- ⏳ 5-10s response time for complex queries

---

## Quick Reference

| Component | Location | Status |
|-----------|----------|--------|
| LLM Service | `morgan/services/llm/` | ✅ Complete |
| Embedding Service | `morgan/services/embeddings/` | ✅ Complete |
| Reranking Service | `morgan/services/reranking/` | ✅ Complete |
| Infrastructure | `morgan/infrastructure/` | ✅ Complete |
| Emotional Intelligence | `morgan/intelligence/` | ✅ Excellent |
| Memory System | `morgan/memory/` | ✅ Strong |
| Search Pipeline | `morgan/search/` | ✅ Excellent |
| Configuration | `morgan/config/` | ✅ Complete |
| Exceptions | `morgan/exceptions.py` | ✅ Complete |
| Documentation | Various `.md` files | 🔄 In Progress |

---

**Remember**: Morgan is a personal AI companion focused on emotional intelligence, proactive assistance, and complete privacy through self-hosting. Quality over speed, privacy over convenience.
