# Codebase Reorganization Specification

> **Date**: 2025-12-26
> **Status**: Planning Complete - Ready for Implementation
> **Priority**: High - Technical Debt Reduction

---

## Purpose

This specification defines a comprehensive reorganization plan for the Morgan codebase to:

1. **Eliminate security vulnerabilities** (committed secrets)
2. **Remove code duplications** (50+ singleton patterns, 3 embedding services, etc.)
3. **Consolidate fragmented services** (LLM, embeddings, reranking)
4. **Standardize patterns** (configuration, error handling, logging)
5. **Clean up dead code** and orphaned modules

---

## Quick Reference

### Documents

| Document | Purpose | Size |
|----------|---------|------|
| [requirements.md](./requirements.md) | What needs to be fixed (25 requirements) | 320 lines |
| [tasks.md](./tasks.md) | How to fix it (54 tasks in 12 phases) | 450 lines |

### Issue Summary

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| Security | 1 | 1 | 0 | 0 | 2 |
| Architecture | 0 | 1 | 2 | 0 | 3 |
| Services | 0 | 2 | 1 | 0 | 3 |
| Deduplication | 0 | 1 | 3 | 0 | 4 |
| Configuration | 0 | 0 | 3 | 1 | 4 |
| Integration | 0 | 1 | 2 | 0 | 3 |
| Patterns | 0 | 0 | 2 | 2 | 4 |
| Documentation | 0 | 0 | 0 | 2 | 2 |
| **Total** | **1** | **6** | **13** | **5** | **25** |

---

## Key Issues Found

### 🔴 CRITICAL: Security

1. **API keys committed to repository**
   - `morgan-rag/.env` contains real API keys
   - `.env` contains real API keys
   - HuggingFace tokens exposed

### 🟠 HIGH: Code Duplication

2. **3 Parallel Architectures**
   - Old: `morgan-rag/morgan/`
   - New: `morgan-server/`, `morgan-cli/`
   - Abandoned: `morgan_v2/`

3. **Duplicate Service Implementations**
   - `llm_service.py` + `distributed_llm_service.py` (same functionality)
   - `embeddings/service.py` + `distributed_embedding_service.py` + `local_embeddings.py`
   - `jina/reranking/service.py` + `local_reranking.py`

4. **Duplicate Milestone Logic**
   - `emotional_processor.py:72-104` + `milestone_tracker.py:48-72`
   - Same celebration messages in two places

### 🟡 MEDIUM: Configuration Chaos

5. **6 places to change LLM config**
6. **Hardcoded IPs** (`192.168.1.x`) throughout distributed config
7. **Inconsistent env var naming** (`LLM_*`, `MORGAN_LLM_*`, `OLLAMA_*`)
8. **50+ repeated singleton patterns**

---

## Implementation Timeline

| Phase | Description | Duration | Priority |
|-------|-------------|----------|----------|
| **1** | Security & Cleanup | 0.5 days | P0 |
| **2** | LLM Service Consolidation | 1-2 days | P1 |
| **3** | Embedding Service Consolidation | 1-2 days | P1 |
| **4** | Reranking Service Consolidation | 1 day | P1 |
| **5** | Core Component Consolidation | 1 day | P1 |
| **6** | Shared Utilities Extraction | 1 day | P2 |
| **7** | Configuration Standardization | 1-2 days | P2 |
| **8** | Integration Layer Updates | 1-2 days | P2 |
| **9** | Dead Code Cleanup | 0.5 days | P3 |
| **10** | Pattern Standardization | 1-2 days | P3 |
| **11** | Documentation Updates | 0.5 days | P3 |
| **12** | Testing & Validation | 1 day | P3 |

**Total: 8-12 days**

---

## Before vs After

### Current State (Problematic)

```
Morgan/
├── .env                          # ⚠️ Contains real API keys!
├── cli.py.old                    # Dead file
├── core/
│   └── emotional_handler.py      # ⚠️ Broken imports
├── morgan-rag/
│   ├── .env                      # ⚠️ Contains real API keys!
│   ├── morgan/
│   │   ├── services/
│   │   │   ├── llm_service.py            # Duplicate #1
│   │   │   └── distributed_llm_service.py # Duplicate #2
│   │   ├── embeddings/
│   │   │   └── service.py                 # Duplicate #1
│   │   └── infrastructure/
│   │       └── local_embeddings.py        # Duplicate #2
│   └── morgan_v2/                # ⚠️ Abandoned code
├── morgan-server/                # New active code
└── morgan-cli/                   # New active code
```

### Target State (Clean)

```
Morgan/
├── .env.example                  # ✅ Template only
├── archive/
│   ├── morgan-rag-legacy/        # ✅ Archived
│   └── morgan_v2-abandoned/      # ✅ Archived
├── morgan-server/
│   └── morgan_server/
│       └── services/
│           ├── llm/              # ✅ Single implementation
│           │   ├── client.py
│           │   └── models.py
│           ├── embeddings/       # ✅ Single implementation
│           │   ├── service.py
│           │   └── models.py
│           └── reranking/        # ✅ Single implementation
│               ├── service.py
│               └── models.py
├── morgan-cli/                   # ✅ Client code
└── docker/
    └── config/
        └── distributed.yaml      # ✅ Single location
```

---

## Getting Started

### Review Documents

1. Read [requirements.md](./requirements.md) to understand what needs fixing
2. Read [tasks.md](./tasks.md) for step-by-step implementation plan

### Start Implementation

```bash
# Start with Phase 1: Security & Cleanup
git checkout -b refactor/codebase-reorganization

# Begin with Task 1.1: Remove secrets
# See tasks.md for details
```

### Priority Order

1. **P0 (Immediate)**: Phase 1 - Security fixes
2. **P1 (This Week)**: Phases 2-5 - Service consolidation
3. **P2 (Next Week)**: Phases 6-8 - Utilities and config
4. **P3 (Following)**: Phases 9-12 - Cleanup and polish

---

## Success Criteria

The reorganization is **complete** when:

- [ ] No secrets in version control
- [ ] Single LLM service implementation
- [ ] Single embedding service implementation
- [ ] Single reranking service implementation
- [ ] Single milestone management system
- [ ] All environment variables use `MORGAN_*` prefix
- [ ] No hardcoded IP addresses
- [ ] No orphaned or dead code
- [ ] All tests passing
- [ ] Documentation updated

---

## Analysis Methodology

This specification was created by running **8 parallel exploration agents** that analyzed:

1. **Project Structure** - Directory organization, package layout
2. **Infrastructure Layer** - `distributed_llm.py`, `local_embeddings.py`, etc.
3. **Services Layer** - All `*_service.py` files
4. **Emotional/Learning Modules** - `intelligence/`, `learning/`
5. **Memory/Search Modules** - `memory/`, `search/`
6. **Core/Companion Modules** - `core/`, `companion/`
7. **Duplicate Code Patterns** - Singleton patterns, HTTP clients, etc.
8. **Configuration Management** - `.env` files, `settings.py`, docker-compose

**Total issues identified**: 150+

---

*Generated by Claude Code on 2025-12-26*
