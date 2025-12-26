# Deprecation Notice - Archived Code

**Last Updated:** December 26, 2025

## Summary

Several components of the Morgan codebase have been archived as part of the codebase reorganization. This document explains what was archived and why.

## Archived Components

All deprecated code has been moved to the `/archive/` directory:

```
archive/
├── README.md                      # Archive documentation
├── deprecated-root-modules/       # Old root-level modules
│   ├── cli.py.old                 # Old CLI (replaced by morgan-cli)
│   ├── core/                      # Orphaned emotional_handler
│   └── services/                  # Standalone Docker services
├── deprecated-modules/            # Old morgan-rag modules
│   ├── embeddings/                # Old embeddings module
│   └── infrastructure/            # Old local_embeddings.py, local_reranking.py
└── abandoned-refactors/           # Incomplete refactoring attempts
    └── morgan_v2/                 # Incomplete Clean Architecture attempt
```

## What Replaced What

| Archived Component | Replacement | Location |
|-------------------|-------------|----------|
| `cli.py.old` | `morgan-cli/` | Terminal client package |
| `embeddings/` module | `morgan/services/embeddings/` | Unified embedding service |
| `local_embeddings.py` | `morgan/services/embeddings/` | Unified embedding service |
| `local_reranking.py` | `morgan/services/reranking/` | Unified reranking service |
| `llm_service.py` | `morgan/services/llm/` | Unified LLM service |
| `distributed_llm_service.py` | `morgan/services/llm/` | Unified LLM service |
| `distributed_embedding_service.py` | `morgan/services/embeddings/` | Unified embedding service |
| `morgan_v2/` | N/A | Abandoned (incomplete) |

## Why Archived?

### Service Consolidation

The codebase had multiple duplicate implementations of core services:

- **LLM**: `llm_service.py` + `distributed_llm_service.py` (~400 duplicate lines)
- **Embeddings**: `embeddings/service.py` + `distributed_embedding_service.py` + `local_embeddings.py` (~700 duplicate lines)
- **Reranking**: `jina/reranking/service.py` + `local_reranking.py` (~600 duplicate lines)

These have been consolidated into unified services in `morgan/services/`:

- `morgan/services/llm/` - Single LLM service with single + distributed modes
- `morgan/services/embeddings/` - Single embedding service with remote + local fallback
- `morgan/services/reranking/` - Single reranking service with 4-level fallback

### Abandoned Refactors

The `morgan_v2/` directory contained an incomplete Clean Architecture refactoring attempt that was never completed. This has been archived rather than deleted to preserve the work for reference.

### Old CLI

The old `cli.py` has been replaced by the `morgan-cli` package which provides a better user experience with:
- Rich terminal UI
- HTTP/WebSocket client
- Multiple commands (chat, learn, memory, knowledge, health)

## Using Archived Code

If you need to reference archived code:

```bash
# View archived files
ls archive/

# Read archived code
cat archive/deprecated-modules/embeddings/service.py
```

**Note:** Archived code is not maintained and should not be used in production.

## Current Architecture

The current active architecture is:

```
morgan-rag/morgan/
├── services/                    # Unified Services Layer
│   ├── llm/                     # LLM service
│   ├── embeddings/              # Embedding service
│   ├── reranking/               # Reranking service
│   └── external_knowledge/      # External knowledge sources
├── intelligence/                # Emotional intelligence
├── memory/                      # Memory system
├── search/                      # Search pipeline
├── infrastructure/              # Distributed infrastructure
├── config/                      # Configuration
├── utils/                       # Utilities
└── exceptions.py                # Exception hierarchy
```

## Documentation

For current documentation, see:

- [claude.md](./claude.md) - Complete project context
- [README.md](./README.md) - Project overview
- [DOCUMENTATION.md](./DOCUMENTATION.md) - Documentation index
- [morgan-rag/docs/ARCHITECTURE.md](./morgan-rag/docs/ARCHITECTURE.md) - Architecture details

---

**Morgan** - Your private, emotionally intelligent AI companion.
