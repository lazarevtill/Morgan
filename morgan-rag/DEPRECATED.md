# Morgan RAG - Status Update

**Last Updated:** December 26, 2025

## Current Status: ✅ ACTIVE

Morgan RAG is the **active core intelligence engine** of the Morgan AI Assistant. It is **not deprecated**.

## What Was Deprecated?

Some **older implementations** within morgan-rag have been consolidated and archived:

| Old Module | Status | Replacement |
|------------|--------|-------------|
| `embeddings/service.py` | 📦 Archived | `services/embeddings/` |
| `services/llm_service.py` | 🗑️ Deleted | `services/llm/` |
| `services/distributed_llm_service.py` | 🗑️ Deleted | `services/llm/` |
| `services/distributed_embedding_service.py` | 🗑️ Deleted | `services/embeddings/` |
| `infrastructure/local_embeddings.py` | 📦 Archived | `services/embeddings/` |
| `infrastructure/local_reranking.py` | 📦 Archived | `services/reranking/` |

Archived code is in `/archive/deprecated-modules/`.

## Current Architecture

The current unified services are:

```
morgan-rag/morgan/services/
├── llm/                 # Unified LLM service (single + distributed)
├── embeddings/          # Unified embedding service (remote + local)
├── reranking/           # Unified reranking service (4-level fallback)
└── external_knowledge/  # External knowledge sources
```

## Usage

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

## Documentation

- [README.md](./README.md) - Morgan RAG overview
- [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) - Architecture details
- [../claude.md](../claude.md) - Complete project context

---

**Morgan RAG** - The intelligent core of Morgan AI Assistant.
