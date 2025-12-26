# Morgan Documentation Index

Complete documentation for the Morgan AI Assistant.

**Last Updated**: December 26, 2025

---

## 📚 Quick Navigation

| Document | Description |
|----------|-------------|
| [claude.md](./claude.md) | **Project context** - Complete architecture, diagrams, and status |
| [README.md](./README.md) | Project overview and quick start |
| [MIGRATION.md](./MIGRATION.md) | Migration guide from old system |

---

## 🏗️ System Architecture

### High-Level Overview

```mermaid
graph TB
    subgraph Users["👤 Users"]
        U1[Terminal User]
        U2[API Client]
        U3[Web User]
    end
    
    subgraph Morgan["🤖 Morgan System"]
        subgraph Client["Client Layer"]
            CLI[Morgan CLI]
            SDK[Python SDK]
        end
        
        subgraph Server["Server Layer"]
            API[FastAPI Server]
            WS[WebSocket Handler]
        end
        
        subgraph Core["Core Layer"]
            SVC[Services]
            INT[Intelligence]
            MEM[Memory]
            SRC[Search]
        end
        
        subgraph Infra["Infrastructure"]
            DIST[Distributed LLM]
            GPU[GPU Manager]
        end
    end
    
    subgraph External["🔧 External"]
        OLL[Ollama]
        QDR[Qdrant]
        RED[Redis]
    end
    
    U1 --> CLI
    U2 --> SDK
    U3 --> API
    CLI --> API
    SDK --> API
    API --> SVC
    WS --> SVC
    SVC --> INT & MEM & SRC
    SVC --> DIST
    DIST --> GPU
    GPU --> OLL
    MEM --> QDR & RED
    SRC --> QDR
```

### Request Processing Flow

```mermaid
sequenceDiagram
    participant U as User
    participant C as Client
    participant A as API Gateway
    participant E as Embedding Service
    participant S as Search Pipeline
    participant M as Memory System
    participant I as Intelligence Engine
    participant L as LLM Service
    participant Q as Qdrant
    participant O as Ollama

    U->>C: Send message
    C->>A: POST /api/chat
    
    Note over A: Request Processing
    A->>E: Embed query
    E->>O: Generate embedding
    O-->>E: Vector [2048]
    E-->>A: Query vector
    
    Note over A: Context Retrieval
    A->>S: Search relevant docs
    S->>Q: Vector similarity search
    Q-->>S: Top-k results
    S->>S: Rerank results
    S-->>A: Ranked context
    
    A->>M: Get conversation history
    M->>Q: Fetch user memories
    Q-->>M: Past interactions
    M-->>A: Conversation context
    
    Note over A: Intelligence Processing
    A->>I: Analyze emotion & intent
    I-->>A: Emotional context
    
    Note over A: Response Generation
    A->>L: Generate response
    L->>O: Chat completion
    O-->>L: Generated text
    L-->>A: LLMResponse
    
    Note over A: Memory Storage
    A->>M: Store interaction
    M->>Q: Persist memory
    
    A-->>C: Chat response
    C-->>U: Display response
```

---

## 📁 Project Structure

```mermaid
graph LR
    subgraph Root["Morgan/"]
        direction TB
        RAG["morgan-rag/<br/>(Core Intelligence)"]
        SRV["morgan-server/<br/>(API Server)"]
        CLI["morgan-cli/<br/>(Terminal Client)"]
        DOC["docker/<br/>(Deployment)"]
        ARC["archive/<br/>(Deprecated)"]
    end
    
    subgraph RAG_Detail["morgan-rag/morgan/"]
        direction TB
        SVC["services/"]
        INT["intelligence/"]
        MEM["memory/"]
        SRC["search/"]
        INF["infrastructure/"]
        CFG["config/"]
        UTL["utils/"]
    end
    
    subgraph SVC_Detail["services/"]
        direction TB
        LLM["llm/"]
        EMB["embeddings/"]
        RRK["reranking/"]
        EXT["external_knowledge/"]
    end
    
    RAG --> RAG_Detail
    SVC --> SVC_Detail
```

### Directory Structure

```
Morgan/
├── morgan-rag/              # Core RAG intelligence (ACTIVE)
│   └── morgan/
│       ├── services/        # Unified service layer
│       │   ├── llm/         # LLM service
│       │   ├── embeddings/  # Embedding service
│       │   ├── reranking/   # Reranking service
│       │   └── external_knowledge/
│       ├── intelligence/    # Emotional intelligence
│       ├── memory/          # Conversation memory
│       ├── search/          # Multi-stage search
│       ├── infrastructure/  # Distributed infrastructure
│       ├── config/          # Configuration
│       └── utils/           # Utilities
├── morgan-server/           # FastAPI server (ACTIVE)
├── morgan-cli/              # Terminal client (ACTIVE)
├── docker/                  # Docker configs (ACTIVE)
├── shared/                  # Shared utilities
└── archive/                 # Archived deprecated code
```

---

## 🔧 Services Layer

### Service Architecture

```mermaid
classDiagram
    class ServiceFactory {
        +get_llm_service() LLMService
        +get_embedding_service() EmbeddingService
        +get_reranking_service() RerankingService
    }
    
    class LLMService {
        +mode: LLMMode
        +generate(prompt) LLMResponse
        +agenerate(prompt) LLMResponse
        +stream(prompt) Iterator
    }
    
    class EmbeddingService {
        +encode(text) List~float~
        +encode_batch(texts) List~List~float~~
        +aencode(text) List~float~
    }
    
    class RerankingService {
        +rerank(query, docs) List~RerankResult~
        +arerank(query, docs) List~RerankResult~
    }
    
    ServiceFactory --> LLMService
    ServiceFactory --> EmbeddingService
    ServiceFactory --> RerankingService
```

### Service Fallback Hierarchy

```mermaid
flowchart TD
    subgraph LLM["LLM Service Fallback"]
        L1[Primary Endpoint<br/>Host 3] -->|Fail| L2[Secondary Endpoint<br/>Host 4]
        L2 -->|Fail| L3[Fast Model<br/>Host 5]
    end
    
    subgraph EMB["Embedding Service Fallback"]
        E1[Remote Ollama<br/>qwen3-embedding] -->|Fail| E2[Local Model<br/>sentence-transformers]
    end
    
    subgraph RERANK["Reranking Service Fallback"]
        R1[Remote Endpoint] -->|Fail| R2[Local CrossEncoder]
        R2 -->|Fail| R3[Embedding Similarity]
        R3 -->|Fail| R4[BM25 Lexical]
    end
```

### Usage

```python
from morgan.services import (
    get_llm_service,
    get_embedding_service,
    get_reranking_service,
)

llm = get_llm_service()
embeddings = get_embedding_service()
reranking = get_reranking_service()
```

---

## 🧠 Intelligence Layer

### Emotional Intelligence Flow

```mermaid
flowchart LR
    subgraph Input["Input"]
        T[User Text]
    end
    
    subgraph Detection["Detection"]
        ED[Emotion Detector]
        T --> ED
        ED --> Joy & Sadness & Anger & Fear & Neutral
    end
    
    subgraph Analysis["Analysis"]
        Joy & Sadness & Anger & Fear & Neutral --> TA[Tone Analyzer]
        TA --> EV[Validator]
    end
    
    subgraph Response["Response Generation"]
        EV --> EG[Empathy Generator]
        EG --> EM[Emotional Mirror]
        EM --> ES[Support System]
        ES --> R[Empathic Response]
    end
```

### Memory System Flow

```mermaid
sequenceDiagram
    participant A as Application
    participant MP as Memory Processor
    participant ES as Embedding Service
    participant EM as Emotional Memory
    participant Q as Qdrant

    A->>MP: Store interaction
    MP->>ES: Embed content
    ES-->>MP: Vector
    MP->>EM: Analyze emotion
    EM-->>MP: Emotional weight
    MP->>Q: Store with metadata
    
    A->>MP: Retrieve context
    MP->>Q: Vector search
    Q-->>MP: Similar memories
    MP->>MP: Apply emotional weighting
    MP-->>A: Ranked memories
```

---

## 🔍 Search Pipeline

### Multi-Stage Search

```mermaid
flowchart TD
    Q[Query] --> E[Embed Query]
    
    subgraph Stage1["Stage 1: Vector Search"]
        E --> VS[Vector Similarity]
        VS --> C[Coarse: Top 100]
        C --> F[Fine: Top 50]
    end
    
    subgraph Stage2["Stage 2: Keyword"]
        F --> KW[Keyword Matching]
    end
    
    subgraph Stage3["Stage 3: Fusion"]
        KW --> HF[Hybrid Fusion]
    end
    
    subgraph Stage4["Stage 4: Rerank"]
        HF --> RR[Reranking]
    end
    
    subgraph Stage5["Stage 5: Dedupe"]
        RR --> DD[Deduplication]
    end
    
    DD --> R[Final Results]
```

---

## 🖥️ Hardware Architecture

### 6-Host Distributed Setup

```mermaid
graph TB
    subgraph Network["192.168.1.x Network"]
        subgraph CPU["CPU Hosts"]
            H1["Host 1 (192.168.1.10)<br/>i9, 64GB RAM<br/>━━━━━━━━━━━━━<br/>Morgan Core<br/>Qdrant<br/>Redis"]
            H2["Host 2 (192.168.1.11)<br/>i9, 64GB RAM<br/>━━━━━━━━━━━━━<br/>Prometheus<br/>Grafana<br/>Background Jobs"]
        end
        
        subgraph GPU["GPU Hosts"]
            H3["Host 3 (192.168.1.20)<br/>RTX 3090, 12GB<br/>━━━━━━━━━━━━━<br/>Main LLM #1<br/>Qwen2.5-32B"]
            H4["Host 4 (192.168.1.21)<br/>RTX 3090, 12GB<br/>━━━━━━━━━━━━━<br/>Main LLM #2<br/>Qwen2.5-32B"]
            H5["Host 5 (192.168.1.22)<br/>RTX 4070, 8GB<br/>━━━━━━━━━━━━━<br/>Embeddings<br/>Fast LLM"]
            H6["Host 6 (192.168.1.23)<br/>RTX 2060, 6GB<br/>━━━━━━━━━━━━━<br/>Reranking<br/>CrossEncoder"]
        end
    end
    
    H1 <--> H3 & H4 & H5 & H6
    H1 <--> H2
```

---

## 📋 Configuration

### Configuration Hierarchy

```mermaid
flowchart TD
    subgraph Sources["Configuration Sources"]
        ENV[Environment Variables<br/>Highest Priority]
        CFG[Config Files<br/>YAML/JSON/.env]
        DEF[Default Values<br/>Lowest Priority]
    end
    
    ENV --> M[Merged Config]
    CFG --> M
    DEF --> M
    
    M --> APP[Application]
```

### Key Environment Variables

```bash
# LLM
MORGAN_LLM_ENDPOINT=http://localhost:11434/v1
MORGAN_LLM_MODEL=qwen2.5:7b

# Embeddings
MORGAN_EMBEDDING_ENDPOINT=http://localhost:11434/v1
MORGAN_EMBEDDING_MODEL=qwen3-embedding:4b

# Vector Database
MORGAN_QDRANT_URL=http://localhost:6333

# Cache
MORGAN_REDIS_URL=redis://localhost:6379
```

---

## 📊 Status & Progress

### Development Progress

```mermaid
gantt
    title Morgan Development Timeline
    dateFormat YYYY-MM-DD
    
    section Phase 1
    Infrastructure    :done, p1a, 2025-11-01, 30d
    Services         :done, p1b, 2025-11-15, 20d
    Documentation    :done, p1c, 2025-12-20, 7d
    
    section Phase 2
    Reasoning        :p2, 2025-01-01, 14d
    
    section Phase 3
    Proactive        :p3, after p2, 14d
    
    section Phase 4
    Context          :p4, after p3, 14d
    
    section Phase 5
    Production       :p5, after p4, 14d
```

### Component Status

```mermaid
pie title Component Completion
    "LLM Service" : 100
    "Embedding Service" : 100
    "Reranking Service" : 100
    "Intelligence" : 95
    "Memory" : 90
    "Search" : 95
    "Documentation" : 100
```

---

## 📂 Documentation Links

### Core Documentation

| Document | Description |
|----------|-------------|
| [claude.md](./claude.md) | Complete project context with all diagrams |
| [README.md](./README.md) | Project overview and quick start |
| [morgan-rag/docs/ARCHITECTURE.md](./morgan-rag/docs/ARCHITECTURE.md) | Detailed architecture |

### Component Documentation

| Document | Description |
|----------|-------------|
| [morgan-server/README.md](./morgan-server/README.md) | Server documentation |
| [morgan-cli/README.md](./morgan-cli/README.md) | CLI documentation |
| [docker/README.md](./docker/README.md) | Docker deployment |

### Migration & Setup

| Document | Description |
|----------|-------------|
| [MIGRATION.md](./MIGRATION.md) | Migration guide |
| [SYSTEM_STATUS.md](./SYSTEM_STATUS.md) | Current system status |

### Planning & Specs

| Document | Description |
|----------|-------------|
| [.kiro/CODEBASE_REORGANIZATION_SUMMARY.md](./.kiro/CODEBASE_REORGANIZATION_SUMMARY.md) | Reorganization summary |
| [.kiro/specs/codebase-reorganization/tasks.md](./.kiro/specs/codebase-reorganization/tasks.md) | Implementation tasks |

---

## 💡 Getting Help

1. **Check Documentation** - Search this index
2. **Check claude.md** - Complete project context with diagrams
3. **Check Logs** - Server and service logs
4. **GitHub Issues** - Report bugs or request features

---

## 📄 License

```
Copyright 2025 Morgan AI Assistant Contributors
Licensed under the Apache License, Version 2.0
```

See [LICENSE](./LICENSE) for the full license text.

---

**Morgan** - Your private, emotionally intelligent AI companion.
