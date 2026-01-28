# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Morgan is a self-hosted, distributed personal AI assistant with emotional intelligence and RAG capabilities. It's a Python monorepo with three main components that communicate via REST/WebSocket APIs.

**Key Principle**: Quality over speed (5-10s response time acceptable for thoughtful responses). Privacy first - all processing on local hardware.

## Build & Development Commands

### Installation (editable mode for development)
```bash
# Core RAG library
cd morgan-rag && pip install -e ".[server,dev]"

# FastAPI server
cd morgan-server && pip install -e ".[dev]"

# Terminal client
cd morgan-cli && pip install -e ".[dev]"
```

### Running Tests
```bash
# Run all tests in a component
cd morgan-rag && pytest tests/
cd morgan-server && pytest tests/
cd morgan-cli && pytest tests/

# Run a single test file
pytest tests/test_emotional_intelligence.py

# Run a specific test
pytest tests/test_emotional_intelligence.py::test_emotion_detection -v

# With coverage
pytest --cov=morgan tests/
```

### Code Formatting & Linting
```bash
# Format (Black, line-length: 100 for server/cli, 88 for morgan-rag)
black morgan_server/ morgan_cli/
cd morgan-rag && make format

# Lint
ruff check .
ruff check --fix .

# Type checking
mypy morgan_server

# Pre-commit (runs all checks)
pre-commit run --all-files
```

### Running the Server
```bash
# Start server
cd morgan-server && python -m morgan_server --host 0.0.0.0 --port 8080

# Start with dependencies (Docker)
cd docker && docker-compose up -d
```

## Architecture

### Component Relationships
```
morgan-cli (Terminal UI) ──HTTP/WS──► morgan-server (FastAPI) ──► morgan-rag (Core Intelligence)
                                              │                           │
                                              └───► Qdrant (Vector DB)    ├── services/llm/
                                              └───► Redis (Cache)         ├── services/embeddings/
                                              └───► Ollama (LLM)          ├── services/reranking/
                                                                          ├── intelligence/ (emotions)
                                                                          ├── memory/
                                                                          └── search/
```

### Service Layer Pattern (morgan-rag)
Services are singletons accessed via factory functions:
```python
from morgan.services import get_llm_service, get_embedding_service, get_reranking_service

llm = get_llm_service()  # Singleton instance
response = await llm.agenerate("prompt")
```

Each service has multi-level fallback (e.g., reranking: remote → CrossEncoder → embedding similarity → BM25).

### Key Entry Points
- **morgan-server**: `morgan_server/__main__.py` → FastAPI app in `morgan_server/app.py`
- **morgan-rag**: `morgan/services/` for service layer, `morgan/core/` for assistant logic
- **API routes**: `morgan-server/morgan_server/api/routes/` (chat.py, memory.py, knowledge.py, profile.py)

### Data Flow
1. Client sends message to `/api/chat`
2. Server embeds query via `EmbeddingService` → Ollama
3. `SearchPipeline` finds context from Qdrant (vector search + reranking)
4. `MemoryProcessor` retrieves conversation history
5. `IntelligenceEngine` analyzes emotion
6. `LLMService` generates response with context
7. Response stored in Qdrant, returned to client

### Exception Hierarchy
All exceptions inherit from `MorganError` in `morgan/exceptions.py`:
- `LLMServiceError`, `EmbeddingServiceError`, `RerankingServiceError`
- `ConfigurationError`, `ValidationError`
- `InfrastructureError` → `ConnectionError`, `TimeoutError`

## Configuration

Environment variables prefixed with `MORGAN_`:
```bash
MORGAN_LLM_ENDPOINT=http://localhost:11434/v1
MORGAN_LLM_MODEL=qwen2.5:7b
MORGAN_QDRANT_URL=http://localhost:6333
MORGAN_REDIS_URL=redis://localhost:6379
MORGAN_EMBEDDING_MODEL=qwen3-embedding:4b
```

Server config can also use YAML (`morgan-server/config.example.yaml`).

## External Services

- **Ollama** (localhost:11434): LLM serving with OpenAI-compatible API
- **Qdrant** (localhost:6333): Vector database for memories and search
- **Redis** (localhost:6379): Caching and session storage

## Code Style Notes

- **Line length**: 100 for morgan-server/morgan-cli, 88 for morgan-rag
- **Async**: Prefer async methods (`agenerate`, `aencode`, `arerank`)
- **Pre-commit excludes morgan-rag** from some hooks (has its own formatting rules via Makefile)
- **Python 3.11+** required
