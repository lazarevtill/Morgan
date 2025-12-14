# Morgan - Personal AI Assistant Project

**Status**: Phase 1 Complete - Distributed Infrastructure
**Version**: 2.0.0-alpha

---

## Project Overview

Morgan is a fully self-hosted, distributed personal AI assistant with:

- **Emotional Intelligence** - Understands and responds empathetically
- **Distributed Architecture** - Multi-host GPU deployment with load balancing
- **Separate Providers** - Different hosts for LLM, Embeddings, and Reranking
- **Complete Privacy** - All processing on local hardware, zero external APIs

---

## Project Structure

```text
Morgan/
├── morgan-rag/               # Main application
│   ├── morgan/
│   │   ├── config/          # Settings (settings.py)
│   │   ├── infrastructure/  # Distributed services
│   │   │   ├── distributed_llm.py      # LLM with load balancing
│   │   │   ├── local_embeddings.py     # Embedding service
│   │   │   ├── local_reranking.py      # Reranking service
│   │   │   └── distributed_gpu_manager.py  # GPU monitoring
│   │   ├── services/        # Service layer
│   │   │   ├── llm_service.py          # LLM service
│   │   │   ├── embedding_service.py    # Embedding service
│   │   │   └── service_factory.py      # Unified factory
│   │   ├── emotional/       # Emotion detection
│   │   ├── memory/          # Conversation memory
│   │   ├── search/          # Multi-stage search
│   │   ├── cli/             # CLI commands
│   │   ├── core/            # Assistant logic
│   │   └── utils/           # Health checks, logging
│   ├── .env.example         # Configuration template
│   ├── docker-compose.yml   # Docker deployment
│   └── requirements.txt     # Dependencies
├── .github/                 # CI/CD workflows
├── README.md                # Project documentation
└── CLAUDE.md               # This file
```text

---

## Configuration

Morgan supports **SEPARATE hosts** for each service:

### Environment Variables

```bash
# LLM Provider (for chat/generation)
LLM_BASE_URL=http://192.168.1.20:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen2.5:32b-instruct-q4_K_M

# Distributed LLM (optional - load balancing)
LLM_DISTRIBUTED_ENABLED=true
LLM_ENDPOINTS=http://192.168.1.20:11434/v1,http://192.168.1.21:11434/v1
LLM_LOAD_BALANCING_STRATEGY=round_robin

# Embedding Provider (SEPARATE from LLM)
EMBEDDING_BASE_URL=http://192.168.1.22:11434
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSIONS=768

# Reranking Provider (SEPARATE)
RERANKING_ENABLED=true
RERANKING_ENDPOINT=http://192.168.1.23:8081/rerank

# Vector Database
QDRANT_URL=http://localhost:6333
```

### 6-Host Architecture

```
Host 1 (192.168.1.10): Morgan Core + Qdrant + Redis
Host 2 (192.168.1.11): Background Services + Monitoring
Host 3 (192.168.1.20): RTX 3090 - Main LLM #1
Host 4 (192.168.1.21): RTX 3090 - Main LLM #2 (load balanced)
Host 5 (192.168.1.22): RTX 4070 - Embeddings
Host 6 (192.168.1.23): RTX 2060 - Reranking
```

---

## Key Files

### Configuration
- `morgan-rag/morgan/config/settings.py` - All settings with validation
- `morgan-rag/.env.example` - Configuration template

### Services
- `morgan-rag/morgan/services/llm_service.py` - LLM service (single/distributed)
- `morgan-rag/morgan/services/embedding_service.py` - Embedding service
- `morgan-rag/morgan/services/service_factory.py` - Unified service factory

### Infrastructure
- `morgan-rag/morgan/infrastructure/distributed_llm.py` - Load-balanced LLM client
- `morgan-rag/morgan/infrastructure/local_embeddings.py` - Embedding generation
- `morgan-rag/morgan/infrastructure/local_reranking.py` - Reranking service

### Health & Monitoring
- `morgan-rag/morgan/utils/health.py` - Health checker for all services

---

## CLI Commands

```bash
cd morgan-rag

# Interactive chat
python -m morgan chat

# Ask a question
python -m morgan ask "What is Python?"

# Learn from documents
python -m morgan learn ./documents

# Check health
python -m morgan health

# Start web server
python -m morgan serve
```

---

## Testing Configuration

```bash
# Test settings
python -m morgan.config.settings

# Test health
python -m morgan.utils.health

# Test services
python -m morgan.services.service_factory
```

---

## Design Principles

1. **Privacy First** - All data stays on your hardware
2. **Separate Providers** - LLM, Embedding, Reranking on different hosts
3. **Distributed** - Load balancing across multiple LLM hosts
4. **Quality Over Speed** - 5-10s for thoughtful responses is acceptable
5. **Local Fallback** - Services fall back to local models when remote unavailable

---

## Performance Targets

- Embeddings: <200ms batch
- Search + rerank: <500ms
- Simple queries: 1-2s
- Complex reasoning: 5-10s

---

## DO NOT MODIFY

The following modules are production-ready:
- `morgan/emotional/` - Emotion detection (excellent)
- `morgan/memory/` - Conversation memory (strong)
- `morgan/search/` - Multi-stage search (excellent)
