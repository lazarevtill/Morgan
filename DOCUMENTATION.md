# Morgan Documentation Index

Complete documentation for the Morgan AI Assistant.

**Last Updated**: December 26, 2025

## 📚 Quick Navigation

| Document | Description |
|----------|-------------|
| [claude.md](./claude.md) | **Project context** - Complete architecture and status |
| [README.md](./README.md) | Project overview and quick start |
| [MIGRATION.md](./MIGRATION.md) | Migration guide from old system |

---

## 🚀 Quick Start

- **[Server Quick Start](./morgan-server/README.md#quick-start)** - Get the server running
- **[Client Quick Start](./morgan-cli/README.md#quick-start)** - Start chatting
- **[Docker Quick Start](./docker/README.md#quick-start)** - Deploy with Docker Compose

---

## 📁 Project Structure

```
Morgan/
├── morgan-rag/              # Core RAG intelligence (ACTIVE)
│   └── morgan/
│       ├── services/        # Unified service layer
│       ├── intelligence/    # Emotional intelligence
│       ├── memory/          # Conversation memory
│       ├── search/          # Multi-stage search
│       └── ...
├── morgan-server/           # FastAPI server (ACTIVE)
├── morgan-cli/              # Terminal client (ACTIVE)
├── docker/                  # Docker configs (ACTIVE)
├── shared/                  # Shared utilities
└── archive/                 # Archived deprecated code
```

---

## 🔧 Core Documentation

### Services Layer (morgan-rag/morgan/services/)

The unified services layer provides clean access to all Morgan capabilities:

| Service | Location | Description |
|---------|----------|-------------|
| **LLM Service** | `services/llm/` | Text generation (single + distributed) |
| **Embedding Service** | `services/embeddings/` | Text embeddings with fallback |
| **Reranking Service** | `services/reranking/` | Document reranking |
| **External Knowledge** | `services/external_knowledge/` | MCP, Context7, Web Search |

**Usage:**
```python
from morgan.services import (
    get_llm_service,
    get_embedding_service,
    get_reranking_service,
)

llm = get_llm_service()
response = llm.generate("Hello!")
```

### Server Documentation

| Document | Description |
|----------|-------------|
| [Server README](./morgan-server/README.md) | Overview and installation |
| [Configuration Guide](./morgan-server/docs/CONFIGURATION.md) | Complete configuration reference |
| [Embedding Configuration](./morgan-server/docs/EMBEDDING_CONFIGURATION.md) | Embedding provider setup |
| [Deployment Guide](./morgan-server/docs/DEPLOYMENT.md) | Docker and bare metal deployment |
| [API Documentation](./morgan-server/docs/API.md) | REST and WebSocket API reference |

### Client Documentation

| Document | Description |
|----------|-------------|
| [Client README](./morgan-cli/README.md) | Overview and usage |

### Docker Documentation

| Document | Description |
|----------|-------------|
| [Docker README](./docker/README.md) | Docker deployment guide |

---

## 🏗️ Architecture Documentation

### Current Architecture (December 2025)

Morgan uses a clean, layered architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  (morgan-server, morgan-cli)                                │
├─────────────────────────────────────────────────────────────┤
│                     Services Layer                           │
│  ┌───────────┐  ┌────────────┐  ┌───────────┐              │
│  │    LLM    │  │ Embeddings │  │ Reranking │              │
│  │  Service  │  │  Service   │  │  Service  │              │
│  └───────────┘  └────────────┘  └───────────┘              │
├─────────────────────────────────────────────────────────────┤
│                   Intelligence Layer                         │
│  ┌───────────┐  ┌────────────┐  ┌───────────┐              │
│  │ Emotional │  │   Memory   │  │  Search   │              │
│  │Intelligence│  │   System   │  │ Pipeline  │              │
│  └───────────┘  └────────────┘  └───────────┘              │
├─────────────────────────────────────────────────────────────┤
│                  Infrastructure Layer                        │
│  ┌───────────┐  ┌────────────┐  ┌───────────┐              │
│  │Distributed│  │    GPU     │  │  Factory  │              │
│  │    LLM    │  │  Manager   │  │           │              │
│  └───────────┘  └────────────┘  └───────────┘              │
├─────────────────────────────────────────────────────────────┤
│                    External Services                         │
│  ┌───────────┐  ┌────────────┐  ┌───────────┐              │
│  │  Ollama   │  │   Qdrant   │  │   Redis   │              │
│  │   (LLM)   │  │ (Vector DB)│  │  (Cache)  │              │
│  └───────────┘  └────────────┘  └───────────┘              │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Unified Services Layer** - Single access point for all services
2. **Infrastructure Abstraction** - Distributed vs single-host transparent to consumers
3. **Fallback Strategies** - Each service has multiple fallback options
4. **Thread-Safe Singletons** - Services are safely shared across threads

---

## 📋 Configuration

### Configuration Files

| File | Location | Purpose |
|------|----------|---------|
| `defaults.py` | `morgan-rag/morgan/config/` | Default values |
| `settings.py` | `morgan-rag/morgan/config/` | Application settings |
| `distributed_config.py` | `morgan-rag/morgan/config/` | Distributed deployment |

### Environment Variables

Key environment variables:

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

## 🧪 Testing

### Running Tests

```bash
# RAG tests
cd morgan-rag && pytest

# Server tests
cd morgan-server && pytest

# Client tests
cd morgan-cli && pytest
```

### Test Coverage

```bash
cd morgan-rag && pytest --cov=morgan
```

---

## 📊 Status & Progress

### Phase 1: Infrastructure (83% Complete)

| Component | Status |
|-----------|--------|
| LLM Service | ✅ Complete |
| Embedding Service | ✅ Complete |
| Reranking Service | ✅ Complete |
| Infrastructure | ✅ Complete |
| Exception Hierarchy | ✅ Complete |
| Documentation | 🔄 In Progress |
| Testing | ⏳ Pending |

### Planned Phases

- **Phase 2**: Multi-Step Reasoning Enhancement
- **Phase 3**: Proactive Features
- **Phase 4**: Enhanced Context
- **Phase 5**: Production Polish

---

## 📂 Archived Documentation

The following documentation is for deprecated code (archived in `/archive/`):

| Document | Status |
|----------|--------|
| `morgan-rag/DEPRECATED.md` | Old deprecation notice |
| `docs/archive/` | Old streaming, TTS docs |
| `DEPRECATION_NOTICE.md` | Original deprecation notice |

---

## 🔗 Related Documents

### Planning & Specs

| Document | Description |
|----------|-------------|
| [.kiro/CODEBASE_REORGANIZATION_SUMMARY.md](./.kiro/CODEBASE_REORGANIZATION_SUMMARY.md) | Reorganization summary |
| [.kiro/specs/codebase-reorganization/tasks.md](./.kiro/specs/codebase-reorganization/tasks.md) | Implementation tasks |
| [.kiro/specs/codebase-reorganization/requirements.md](./.kiro/specs/codebase-reorganization/requirements.md) | Requirements |

### Error Handling

| Document | Description |
|----------|-------------|
| [docs/ERROR_HANDLING_GUIDE.md](./docs/ERROR_HANDLING_GUIDE.md) | Error handling guide |
| [docs/ERROR_HANDLING_QUICK_REFERENCE.md](./docs/ERROR_HANDLING_QUICK_REFERENCE.md) | Quick reference |

---

## 💡 Getting Help

1. **Check Documentation** - Search this index
2. **Check claude.md** - Complete project context
3. **Check Logs** - Server and service logs
4. **GitHub Issues** - Report bugs or request features

---

**Morgan** - Your private, emotionally intelligent AI companion.
