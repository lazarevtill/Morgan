# Morgan - Personal AI Assistant Project

**Status**: Phase 1 - Infrastructure Complete (95%)
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

## System Architecture

### High-Level Architecture Diagram

```mermaid
graph TB
    subgraph Clients["Client Layer"]
        CLI[Morgan CLI<br/>Terminal UI]
        API[REST/WebSocket<br/>API Clients]
        WEB[Web Interface<br/>Future]
    end

    subgraph Server["Server Layer"]
        GW[API Gateway<br/>FastAPI]
        AUTH[Auth &<br/>Session]
    end

    subgraph Core["Core Intelligence Layer"]
        LLM[LLM Service]
        EMB[Embedding Service]
        RERANK[Reranking Service]
        INTEL[Emotional<br/>Intelligence]
        MEM[Memory<br/>System]
        SEARCH[Search<br/>Pipeline]
    end

    subgraph Infra["Infrastructure Layer"]
        DIST[Distributed<br/>LLM Client]
        GPU[GPU<br/>Manager]
        FACTORY[Service<br/>Factory]
    end

    subgraph External["External Services"]
        OLLAMA[Ollama<br/>LLM Server]
        QDRANT[Qdrant<br/>Vector DB]
        REDIS[Redis<br/>Cache]
    end

    CLI --> GW
    API --> GW
    WEB --> GW
    GW --> AUTH
    AUTH --> LLM
    AUTH --> EMB
    AUTH --> RERANK
    AUTH --> INTEL
    AUTH --> MEM
    AUTH --> SEARCH
    
    LLM --> DIST
    EMB --> FACTORY
    RERANK --> FACTORY
    
    DIST --> GPU
    GPU --> OLLAMA
    FACTORY --> OLLAMA
    
    MEM --> QDRANT
    SEARCH --> QDRANT
    EMB --> QDRANT
    
    MEM --> REDIS
    SEARCH --> REDIS
```

### Request Flow Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as Morgan CLI
    participant API as API Gateway
    participant LLM as LLM Service
    participant EMB as Embedding Service
    participant SEARCH as Search Pipeline
    participant MEM as Memory System
    participant INTEL as Intelligence Engine
    participant QDRANT as Qdrant
    participant OLLAMA as Ollama

    U->>CLI: Send message
    CLI->>API: POST /api/chat
    
    API->>EMB: Embed query
    EMB->>OLLAMA: Generate embedding
    OLLAMA-->>EMB: Vector [2048]
    EMB-->>API: Query embedding
    
    API->>SEARCH: Search relevant context
    SEARCH->>QDRANT: Vector search
    QDRANT-->>SEARCH: Similar documents
    SEARCH->>SEARCH: Rerank results
    SEARCH-->>API: Ranked context
    
    API->>MEM: Get conversation history
    MEM->>QDRANT: Fetch memories
    QDRANT-->>MEM: Past interactions
    MEM-->>API: Conversation context
    
    API->>INTEL: Analyze emotion
    INTEL-->>API: Emotional context
    
    API->>LLM: Generate response
    LLM->>OLLAMA: Chat completion
    OLLAMA-->>LLM: Generated text
    LLM-->>API: Response
    
    API->>MEM: Store interaction
    MEM->>QDRANT: Save memory
    
    API-->>CLI: Chat response
    CLI-->>U: Display response
```

---

## Hardware Architecture (6 Hosts)

### Network Topology Diagram

```mermaid
graph TB
    subgraph Network["192.168.1.x Network"]
        subgraph CPU["CPU Hosts"]
            H1[Host 1<br/>192.168.1.10<br/>i9, 64GB RAM<br/>Core + Qdrant + Redis]
            H2[Host 2<br/>192.168.1.11<br/>i9, 64GB RAM<br/>Background + Monitoring]
        end
        
        subgraph GPU["GPU Hosts"]
            H3[Host 3<br/>192.168.1.20<br/>RTX 3090, 12GB<br/>Main LLM #1]
            H4[Host 4<br/>192.168.1.21<br/>RTX 3090, 12GB<br/>Main LLM #2]
            H5[Host 5<br/>192.168.1.22<br/>RTX 4070, 8GB<br/>Embeddings + Fast LLM]
            H6[Host 6<br/>192.168.1.23<br/>RTX 2060, 6GB<br/>Reranking]
        end
    end
    
    H1 <--> H3
    H1 <--> H4
    H1 <--> H5
    H1 <--> H6
    H1 <--> H2
```

### Host Specifications

| Host | IP | Hardware | Role | Services |
|------|-----|----------|------|----------|
| **Host 1** | 192.168.1.10 | i9, 64GB RAM | Core | Morgan Orchestrator, Qdrant, Redis |
| **Host 2** | 192.168.1.11 | i9, 64GB RAM | Background | Prometheus, Grafana, Background Jobs |
| **Host 3** | 192.168.1.20 | RTX 3090, 12GB | LLM Primary | Ollama (Qwen2.5-32B) |
| **Host 4** | 192.168.1.21 | RTX 3090, 12GB | LLM Secondary | Ollama (Qwen2.5-32B) |
| **Host 5** | 192.168.1.22 | RTX 4070, 8GB | Embeddings | Ollama (Qwen3-Embedding, Qwen2.5-7B) |
| **Host 6** | 192.168.1.23 | RTX 2060, 6GB | Reranking | CrossEncoder Service |

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

---

## Services Layer Architecture

### Service Class Diagram

```mermaid
classDiagram
    class LLMService {
        +mode: LLMMode
        +endpoint: str
        +model: str
        +generate(prompt) LLMResponse
        +agenerate(prompt) LLMResponse
        +stream(prompt) AsyncIterator
        +astream(prompt) AsyncIterator
    }
    
    class EmbeddingService {
        +endpoint: str
        +model: str
        +dimensions: int
        +encode(text) List~float~
        +encode_batch(texts) List~List~float~~
        +aencode(text) List~float~
    }
    
    class RerankingService {
        +endpoint: str
        +model: str
        +rerank(query, docs) List~RerankResult~
        +arerank(query, docs) List~RerankResult~
    }
    
    class SingletonFactory {
        -_instances: Dict
        +get_or_create(cls, factory) T
        +reset(cls) void
    }
    
    class LLMResponse {
        +content: str
        +model: str
        +usage: Dict
        +finish_reason: str
    }
    
    class RerankResult {
        +index: int
        +score: float
        +text: str
    }
    
    LLMService --> LLMResponse
    RerankingService --> RerankResult
    SingletonFactory --> LLMService
    SingletonFactory --> EmbeddingService
    SingletonFactory --> RerankingService
```

### Service Fallback Strategy

```mermaid
flowchart TD
    subgraph LLM["LLM Service"]
        L1[Primary Endpoint] --> L2{Healthy?}
        L2 -->|Yes| L3[Generate]
        L2 -->|No| L4[Secondary Endpoint]
        L4 --> L5{Healthy?}
        L5 -->|Yes| L3
        L5 -->|No| L6[Fast Model Fallback]
    end
    
    subgraph EMB["Embedding Service"]
        E1[Remote Ollama] --> E2{Available?}
        E2 -->|Yes| E3[Generate Embedding]
        E2 -->|No| E4[Local sentence-transformers]
        E4 --> E3
    end
    
    subgraph RERANK["Reranking Service"]
        R1[Remote Endpoint] --> R2{Available?}
        R2 -->|Yes| R3[Rerank]
        R2 -->|No| R4[Local CrossEncoder]
        R4 --> R5{Available?}
        R5 -->|Yes| R3
        R5 -->|No| R6[Embedding Similarity]
        R6 --> R7{Available?}
        R7 -->|Yes| R3
        R7 -->|No| R8[BM25 Lexical]
        R8 --> R3
    end
```

---

## Project Structure

```
Morgan/
├── morgan-rag/                    # Main RAG project
│   ├── morgan/
│   │   ├── services/              # ✅ Unified services layer
│   │   │   ├── __init__.py        # Service exports
│   │   │   ├── llm/               # LLM service
│   │   │   │   ├── __init__.py
│   │   │   │   ├── models.py      # LLMResponse, LLMMode
│   │   │   │   └── service.py     # LLMService class
│   │   │   ├── embeddings/        # Embedding service
│   │   │   │   ├── __init__.py
│   │   │   │   ├── models.py      # EmbeddingStats
│   │   │   │   └── service.py     # EmbeddingService class
│   │   │   ├── reranking/         # Reranking service
│   │   │   │   ├── __init__.py
│   │   │   │   ├── models.py      # RerankResult, RerankStats
│   │   │   │   └── service.py     # RerankingService class
│   │   │   └── external_knowledge/# External knowledge sources
│   │   │
│   │   ├── infrastructure/        # ✅ Distributed infrastructure
│   │   │   ├── distributed_llm.py # Load balancing
│   │   │   ├── distributed_gpu_manager.py
│   │   │   ├── multi_gpu_manager.py
│   │   │   └── factory.py         # Infrastructure factory
│   │   │
│   │   ├── intelligence/          # ✅ Emotional intelligence
│   │   │   ├── emotions/          # Emotion detection
│   │   │   ├── empathy/           # Empathic responses
│   │   │   └── core/              # Intelligence engine
│   │   │
│   │   ├── learning/              # ✅ Pattern learning
│   │   ├── memory/                # ✅ Conversation memory
│   │   ├── companion/             # ✅ Relationship management
│   │   ├── search/                # ✅ Multi-stage search
│   │   ├── reasoning/             # ✅ Multi-step reasoning
│   │   ├── proactive/             # ✅ Proactive assistance
│   │   ├── communication/         # ✅ Communication preferences
│   │   │
│   │   ├── config/                # ✅ Configuration
│   │   │   ├── defaults.py        # Default values
│   │   │   ├── settings.py        # Application settings
│   │   │   └── distributed_config.py
│   │   │
│   │   ├── utils/                 # ✅ Utilities
│   │   │   ├── singleton.py       # Singleton factory
│   │   │   ├── model_cache.py     # Model caching
│   │   │   ├── deduplication.py   # Result deduplication
│   │   │   └── logger.py          # Logging
│   │   │
│   │   ├── core/                  # Core assistant logic
│   │   └── exceptions.py          # ✅ Exception hierarchy
│   │
│   ├── tests/                     # Test suite
│   ├── examples/                  # Usage examples
│   └── scripts/                   # Utility scripts
│
├── morgan-server/                 # Server component
├── morgan-cli/                    # CLI client
├── docker/                        # Docker configurations
├── archive/                       # Archived deprecated code
├── shared/                        # Shared models and utilities
└── .kiro/                         # Kiro specs and planning docs
```

---

## Service Usage Examples

### LLM Service Flow

```mermaid
sequenceDiagram
    participant App as Application
    participant LLM as LLMService
    participant Dist as DistributedLLMClient
    participant EP1 as Endpoint 1
    participant EP2 as Endpoint 2

    App->>LLM: get_llm_service()
    LLM->>LLM: Check mode (single/distributed)
    
    alt Single Mode
        App->>LLM: generate("Hello")
        LLM->>EP1: POST /v1/chat/completions
        EP1-->>LLM: Response
        LLM-->>App: LLMResponse
    else Distributed Mode
        App->>LLM: generate("Hello")
        LLM->>Dist: generate()
        Dist->>Dist: Select endpoint (round-robin)
        Dist->>EP1: POST /v1/chat/completions
        alt EP1 Fails
            EP1-->>Dist: Error
            Dist->>EP2: POST /v1/chat/completions
            EP2-->>Dist: Response
        else EP1 Succeeds
            EP1-->>Dist: Response
        end
        Dist-->>LLM: Response
        LLM-->>App: LLMResponse
    end
```

### Code Examples

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

---

## Exception Hierarchy

```mermaid
classDiagram
    class MorganError {
        +message: str
        +service: str
        +operation: str
    }
    
    class LLMServiceError {
        +endpoint: str
        +model: str
    }
    
    class EmbeddingServiceError {
        +model: str
        +batch_size: int
    }
    
    class RerankingServiceError {
        +fallback_used: str
    }
    
    class ConfigurationError {
        +config_key: str
    }
    
    class ValidationError {
        +field: str
        +value: any
    }
    
    class InfrastructureError {
        +host: str
    }
    
    class ConnectionError {
        +url: str
        +timeout: float
    }
    
    class TimeoutError {
        +operation: str
        +timeout: float
    }
    
    MorganError <|-- LLMServiceError
    MorganError <|-- EmbeddingServiceError
    MorganError <|-- RerankingServiceError
    MorganError <|-- ConfigurationError
    MorganError <|-- ValidationError
    MorganError <|-- InfrastructureError
    InfrastructureError <|-- ConnectionError
    InfrastructureError <|-- TimeoutError
```

### Usage

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

## Search Pipeline

### Multi-Stage Search Flow

```mermaid
flowchart TD
    Q[User Query] --> E[Embed Query]
    E --> S1[Stage 1: Vector Search]
    S1 --> S2[Stage 2: Keyword Search]
    S2 --> S3[Stage 3: Hybrid Fusion]
    S3 --> S4[Stage 4: Reranking]
    S4 --> S5[Stage 5: Deduplication]
    S5 --> R[Ranked Results]
    
    subgraph Vector["Vector Search"]
        S1 --> V1[Coarse Search<br/>Top 100]
        V1 --> V2[Fine Search<br/>Top 50]
    end
    
    subgraph Rerank["Reranking"]
        S4 --> R1{Remote Available?}
        R1 -->|Yes| R2[Remote Reranker]
        R1 -->|No| R3[Local CrossEncoder]
        R3 --> R4{Available?}
        R4 -->|No| R5[Embedding Similarity]
        R5 --> R6{Available?}
        R6 -->|No| R7[BM25 Fallback]
    end
```

---

## Memory System

### Memory Flow Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant A as Assistant
    participant MP as Memory Processor
    participant ES as Embedding Service
    participant Q as Qdrant
    participant EM as Emotional Memory

    U->>A: Send message
    A->>MP: Process interaction
    
    MP->>ES: Embed message
    ES-->>MP: Message embedding
    
    MP->>EM: Analyze emotion
    EM-->>MP: Emotional context
    
    MP->>Q: Store memory
    Note over Q: Stores: text, embedding,<br/>emotion, timestamp, user_id
    
    A->>MP: Retrieve context
    MP->>Q: Vector search
    Q-->>MP: Similar memories
    MP->>MP: Apply emotional weighting
    MP-->>A: Relevant context
    
    A-->>U: Response with context
```

---

## Emotional Intelligence

### Emotion Processing Flow

```mermaid
flowchart LR
    subgraph Input["Input Processing"]
        T[Text Input] --> ED[Emotion Detector]
        ED --> E1[Joy]
        ED --> E2[Sadness]
        ED --> E3[Anger]
        ED --> E4[Fear]
        ED --> E5[Surprise]
        ED --> E6[Neutral]
    end
    
    subgraph Analysis["Emotional Analysis"]
        E1 & E2 & E3 & E4 & E5 & E6 --> TA[Tone Analyzer]
        TA --> EV[Emotional Validator]
    end
    
    subgraph Response["Response Generation"]
        EV --> EG[Empathy Generator]
        EG --> EM[Emotional Mirror]
        EM --> ES[Emotional Support]
        ES --> R[Empathic Response]
    end
```

---

## Performance Targets

| Operation | Target | Status |
|-----------|--------|--------|
| Embeddings (batch) | <200ms | ✅ Achieved |
| Search + rerank | <500ms | ✅ Achieved |
| Simple queries | 1-2s | ⏳ Target |
| Complex reasoning | 5-10s | ⏳ Acceptable |

---

## Development Progress

```mermaid
gantt
    title Morgan Development Timeline
    dateFormat  YYYY-MM-DD
    section Phase 1
    Infrastructure Setup     :done, p1a, 2025-11-01, 30d
    Service Consolidation    :done, p1b, 2025-11-15, 20d
    Documentation           :done, p1c, 2025-12-20, 7d
    section Phase 2
    Multi-Step Reasoning    :p2, 2025-01-01, 14d
    section Phase 3
    Proactive Features      :p3, after p2, 14d
    section Phase 4
    Enhanced Context        :p4, after p3, 14d
    section Phase 5
    Production Polish       :p5, after p4, 14d
```

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
| Documentation | Various `.md` files | ✅ Complete |

---

## Key Design Principles

1. **Privacy First** - All data stays on your hardware, no external APIs
2. **Quality Over Speed** - 5-10s for thoughtful responses is acceptable
3. **KISS (Keep It Simple)** - Simple, focused modules with clear responsibilities
4. **Modular Enhancement** - Keep excellent existing code, add missing capabilities
5. **Fault Tolerance** - Distributed architecture with failover and health monitoring

---

**Remember**: Morgan is a personal AI companion focused on emotional intelligence, proactive assistance, and complete privacy through self-hosting. Quality over speed, privacy over convenience.
