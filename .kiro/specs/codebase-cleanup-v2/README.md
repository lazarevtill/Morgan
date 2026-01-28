# Codebase Cleanup v2

## Summary

Comprehensive cleanup and reorganization of the Morgan codebase to eliminate technical debt, duplications, and inconsistencies.

**Status**: Ready for Implementation
**Date**: 2025-12-27
**Note**: App is in development - NO backward compatibility required

## Problem Statement

Analysis by 7 parallel exploration agents identified **87 issues** across the codebase:

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| Exceptions | 1 | 1 | 0 | 0 | 2 |
| Singletons | 1 | 1 | 0 | 0 | 2 |
| Deduplication | 0 | 2 | 2 | 0 | 4 |
| Configuration | 1 | 1 | 1 | 0 | 3 |
| API Consistency | 0 | 1 | 1 | 0 | 2 |
| Missing Impl | 1 | 0 | 2 | 0 | 3 |
| Dead Code | 0 | 0 | 0 | 3 | 3 |
| Data Integrity | 0 | 1 | 0 | 0 | 1 |

## Key Issues

### Critical (Must Fix)
1. **Empty cultural.py module** - Imported but has no code (will crash)
2. **Duplicate exceptions** - MorganError defined in 2 places with different APIs
3. **Duplicate singletons** - 4+ different singleton patterns
4. **Config mismatch** - settings.py uses `gemma3:latest`, defaults.py uses `qwen2.5:32b`

### High Priority
1. **5 deduplication implementations** - `ResultDeduplicator` exists but unused
2. **140 lines emotion detection duplicated** - detector.py vs intelligence_engine.py
3. **Race conditions** - No locks on availability flags
4. **Reranking async-first** - Inconsistent with other sync-first services
5. **Silent error suppression** - `except: pass` in multiple files

## Solution

Delete duplicates directly (no backward compatibility). Consolidate to single implementations:

- Single `exceptions.py` for all exceptions
- Single `SingletonFactory` for all singletons
- Single `ResultDeduplicator` for all deduplication
- Single `intelligence/constants.py` for emotion patterns
- Single `defaults.py` for all configuration values

## Documents

| Document | Purpose |
|----------|---------|
| [requirements.md](requirements.md) | 20 requirements with acceptance criteria |
| [design.md](design.md) | Architecture diagrams and code samples |
| [tasks.md](tasks.md) | 34 tasks in 11 phases |

## Implementation Plan

| Phase | Focus | Duration | Priority |
|-------|-------|----------|----------|
| 1 | Critical Fixes (cultural.py, locks, silent errors) | 2-3 hrs | P0 |
| 2 | Exception Consolidation | 2-3 hrs | P0 |
| 3 | Singleton Consolidation | 2-3 hrs | P1 |
| 4 | Configuration Unification | 1-2 hrs | P1 |
| 5 | Intelligence Consolidation | 2-3 hrs | P1 |
| 6 | Deduplication Consolidation | 2-3 hrs | P1 |
| 7 | API Standardization | 1-2 hrs | P2 |
| 8 | Dead Code Removal | 1-2 hrs | P3 |
| 9 | Memory Search Consolidation | 1-2 hrs | P2 |
| 10 | Companion Storage Integration | 1-2 hrs | P2 |
| 11 | Validation & Testing | 2-3 hrs | P3 |

**Total: 17-27 hours**

## Files to DELETE

| File | Reason |
|------|--------|
| `cli.py` (root) | DEPRECATED |
| `shared/utils/singleton.py` | Duplicate |
| `shared/utils/deduplication.py` | Duplicate |
| `communication/cultural.py` | Empty |

## Files to CREATE

| File | Purpose |
|------|---------|
| `intelligence/constants.py` | Centralized emotion patterns, valence, modifiers |

## Key Line Numbers Reference

| File | Lines | Issue |
|------|-------|-------|
| `error_handling.py` | 94-262 | Duplicate exception classes |
| `error_handling.py` | 1109-1136 | Manual singletons |
| `validators.py` | 10 | Duplicate ValidationError |
| `settings.py` | 42, 51 | Wrong defaults (gemma3) |
| `detector.py` | 40-82, 85-98 | Duplicate patterns |
| `intelligence_engine.py` | 51-102, 364, 393, 438 | Duplicate emotion logic |
| `multi_stage_search.py` | 1660-2334 | 3 dedup methods |
| `reranking/service.py` | 109, 245-249, 497 | Config, event loop, silent except |

## Expected Results

- ~1,500 lines of duplicate code removed
- Single source of truth for exceptions, singletons, deduplication
- Consistent async/sync API patterns
- Thread-safe service singletons
- Unified configuration defaults

## How to Start

1. Read `tasks.md` Phase 1
2. Start with Task 1.1 (fix cultural.py)
3. Complete Phase 1 before moving to Phase 2
4. Run tests after each phase

---

*Generated from comprehensive codebase review on 2025-12-27*
