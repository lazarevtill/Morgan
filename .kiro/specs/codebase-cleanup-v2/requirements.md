# Requirements Document: Codebase Cleanup v2

## Introduction

This requirements specification defines the comprehensive cleanup and reorganization of the Morgan codebase based on findings from 7 parallel exploration agents. The analysis identified **87 issues** including code duplications, inconsistent patterns, missing implementations, dead code, and configuration chaos.

**Date**: 2025-12-27
**Status**: Requirements Complete - Ready for Implementation
**Priority**: Critical - Technical Debt & Data Integrity
**Note**: App is in development - NO backward compatibility required

## Glossary

- **morgan-rag** - Core RAG library with services, infrastructure, and intelligence modules
- **morgan-server** - FastAPI server exposing morgan-rag functionality
- **morgan-cli** - Command-line client for interacting with morgan-server
- **shared** - Cross-project shared utilities, models, and configuration
- **Collection** - Qdrant vector database collection for storing embeddings
- **Singleton Factory** - Thread-safe pattern for creating single instances of services

---

## 1. Critical Exception & Error Handling Requirements

### REQ-EXC-1: Single Exception Hierarchy (CRITICAL)
**Priority**: CRITICAL
**WHEN** custom exceptions are needed
**SHALL** use single `MorganError` base class in `morgan/exceptions.py`
**AND** DELETE all duplicate exception definitions

**Current State**:
- `MorganError` defined in `exceptions.py:47-58` AND `error_handling.py:97-108` with DIFFERENT APIs
- `ValidationError` defined in 3 places (exceptions.py:249, error_handling.py:241, validators.py:10)
- `ConfigurationError` defined in 2 places

**Acceptance Criteria**:
- [ ] `morgan/exceptions.py` is ONLY exception source (delete all duplicates)
- [ ] Delete `MorganError` from `utils/error_handling.py:94-262`
- [ ] Delete `ValidationError` from `utils/validators.py:10`
- [ ] Delete exception definitions from `utils/companion_error_handling.py`
- [ ] All imports use `from morgan.exceptions import X`

### REQ-EXC-2: Fix Bare Exception Handling
**Priority**: HIGH
**WHEN** exceptions are caught
**SHALL** use specific exception types and log errors
**AND** eliminate bare `except:` or `except Exception: pass`

**Current State**:
- `reranking/service.py:497-498`: `except Exception: pass` (silent failure)
- `profile.py:131,139`: `except: pass` (silent suppression)

**Acceptance Criteria**:
- [ ] Fix `reranking/service.py:497` - add logging
- [ ] Fix `profile.py:131,139` - add logging
- [ ] No bare `except:` or silent `pass` in codebase

---

## 2. Singleton Pattern Consolidation Requirements

### REQ-SING-1: Unified Singleton Factory (CRITICAL)
**Priority**: CRITICAL
**WHEN** singleton services are created
**SHALL** use single `SingletonFactory` from `utils/singleton.py`
**AND** DELETE all duplicate implementations

**Current State (4+ implementations)**:
- `morgan-rag/morgan/utils/singleton.py:49-141` - Instance-based with generics
- `shared/utils/singleton.py:15-126` - Static class methods (DIFFERENT!)
- `error_handling.py:1109-1136` - Manual `_instance + _lock` pattern
- `distributed_config.py:511-534` - Manual `_cached_config` pattern
- Each service has own `_service_instance + _service_lock` pattern

**Acceptance Criteria**:
- [ ] Keep `morgan/utils/singleton.py` as authoritative (more complete)
- [ ] Update or delete `shared/utils/singleton.py`
- [ ] Delete manual singletons from `error_handling.py:1109-1136`
- [ ] Delete `_cached_config` pattern from `distributed_config.py`
- [ ] Migrate all services to use `SingletonFactory`

### REQ-SING-2: Thread-Safe Service Singletons
**Priority**: HIGH
**WHEN** service singletons are accessed
**SHALL** be thread-safe with proper locking
**AND** use consistent cleanup patterns

**Current State**:
- `llm/service.py:683-735` - Has lock but different cleanup
- `embeddings/service.py:771-819` - Has lock, calls `clear_cache()`
- `reranking/service.py:536-582` - Has lock, NO cleanup
- Race conditions in availability flags (embeddings:240, reranking:285)

**Acceptance Criteria**:
- [ ] Add locks to availability flag updates in `embeddings/service.py:240`
- [ ] Add locks to availability flags in `reranking/service.py:285`
- [ ] Consistent `shutdown()` or `cleanup()` in all reset functions

---

## 3. Code Deduplication Requirements

### REQ-DUP-1: Unified Deduplication Logic (HIGH)
**Priority**: HIGH
**WHEN** search results or memories need deduplication
**SHALL** use `utils/deduplication.py` `ResultDeduplicator`
**AND** DELETE all 5 duplicate implementations

**Current State (5 implementations, utility unused)**:
- `search/multi_stage_search.py:1729-1775` - `_deduplicate_results()`
- `search/multi_stage_search.py:1660-1727` - `_apply_rrf_deduplication()`
- `search/multi_stage_search.py:2300-2334` - `_deduplicate_memory_results()`
- `search/companion_memory_search.py:1057-1075` - `_deduplicate_search_results()`
- `memory/memory_processor.py:608-623` - `_deduplicate_memories()`
- `utils/deduplication.py` - `ResultDeduplicator` EXISTS BUT UNUSED!

**Acceptance Criteria**:
- [ ] Use `ResultDeduplicator` in all search modules
- [ ] Delete 5 duplicate dedup methods
- [ ] Consolidate `shared/utils/deduplication.py` and `morgan/utils/deduplication.py`

### REQ-DUP-2: Unified Emotion Detection (HIGH)
**Priority**: HIGH
**WHEN** emotion detection is performed
**SHALL** use single implementation in `emotions/detector.py`
**AND** DELETE duplicate logic from `intelligence_engine.py`

**Current State (~140 lines duplicated)**:
- `detector.py:197` AND `intelligence_engine.py:364` - `_detect_emotions_rule_based()`
- `detector.py:250` AND `intelligence_engine.py:393` - `_detect_emotions_llm()`
- `detector.py:342` AND `intelligence_engine.py:438` - `_combine_emotion_results()`
- `detector.py:40-82` AND `intelligence_engine.py:51-88` - `EMOTION_PATTERNS`
- `detector.py:85-98` AND `intelligence_engine.py:91-102` - `INTENSITY_MODIFIERS`

**Acceptance Criteria**:
- [ ] Make `detector.py` the single source for emotion detection
- [ ] Have `intelligence_engine.py` delegate to detector
- [ ] Delete duplicate methods from `intelligence_engine.py`
- [ ] Create `intelligence/constants.py` for shared patterns

### REQ-DUP-3: Unified Memory Search (MEDIUM)
**Priority**: MEDIUM
**WHEN** memory search is performed
**SHALL** use single implementation
**AND** DELETE duplicate memory search logic

**Current State (3 implementations)**:
- `multi_stage_search.py:1022` - `_memory_search()`
- `multi_stage_search.py:1185` - `_basic_memory_search()`
- `companion_memory_search.py:727` - `_execute_enhanced_memory_search()`

**Acceptance Criteria**:
- [ ] Consolidate to single `_memory_search()` implementation
- [ ] Companion search delegates to main implementation
- [ ] Delete redundant `_basic_memory_search()`

### REQ-DUP-4: Cross-Package Duplicates (MEDIUM)
**Priority**: MEDIUM
**WHEN** shared utilities are needed
**SHALL** exist in single location
**AND** be imported from there

**Current State**:
- `shared/utils/deduplication.py` (195 lines) vs `morgan/utils/deduplication.py` (154 lines)
- `shared/utils/singleton.py` vs `morgan/utils/singleton.py`
- `shared/utils/exceptions.py` (784 lines) vs `morgan/exceptions.py` (359 lines) vs `morgan/utils/error_handling.py` (1136 lines)

**Acceptance Criteria**:
- [ ] Choose ONE location for each utility
- [ ] Delete duplicates
- [ ] Update all imports

---

## 4. Configuration Requirements

### REQ-CFG-1: Unified Default Values (CRITICAL)
**Priority**: CRITICAL
**WHEN** default configuration values are defined
**SHALL** have single source of truth
**AND** DELETE conflicting defaults

**Current State (3 sources with DIFFERENT values!)**:
- `defaults.py`: `llm_model = "qwen2.5:32b-instruct-q4_K_M"`
- `settings.py`: `llm_model = "gemma3:latest"` (DIFFERENT MODEL FAMILY!)
- `distributed_config.py`: `main_model = "qwen2.5:32b-instruct-q4_K_M"`

**Acceptance Criteria**:
- [ ] Keep `defaults.py` as authoritative
- [ ] Update `settings.py` to use `defaults.py` values
- [ ] Update `distributed_config.py` to use `defaults.py` values
- [ ] Single model family (Qwen) across all configs

### REQ-CFG-2: Consistent Configuration Access (HIGH)
**Priority**: HIGH
**WHEN** configuration values are accessed
**SHALL** use settings object consistently
**AND** NOT use direct `os.environ.get()`

**Current State**:
- `llm/service.py` - Uses `getattr(self.settings, ...)` CORRECT
- `embeddings/service.py` - Uses settings with fallback
- `reranking/service.py:109` - Uses `os.environ.get()` INCONSISTENT!

**Acceptance Criteria**:
- [ ] Fix `reranking/service.py:109` to use settings
- [ ] All services use `getattr(self.settings, "key", default)` pattern
- [ ] Remove direct `os.environ.get()` from service code

### REQ-CFG-3: Single Configuration Paradigm (MEDIUM)
**Priority**: MEDIUM
**WHEN** configuration classes are defined
**SHALL** use Pydantic BaseSettings consistently
**AND** DELETE dataclass-based configs

**Current State**:
- `settings.py` - Pydantic BaseSettings
- `distributed_config.py` - Python dataclasses (DIFFERENT!)

**Acceptance Criteria**:
- [ ] Convert `distributed_config.py` to Pydantic if needed, or
- [ ] Document why dataclasses are used for YAML configs

---

## 5. API Consistency Requirements

### REQ-API-1: Consistent Async/Sync Naming (HIGH)
**Priority**: HIGH
**WHEN** async/sync method pairs are defined
**SHALL** use consistent naming convention
**AND** all services follow same pattern

**Current State**:
- LLM: `generate()` → `agenerate()` BUT `stream_generate()` → `astream()` (MISMATCH)
- Embedding: `encode()` → `aencode()` CONSISTENT
- Reranking: `rerank()` (async) → `rerank_sync()` REVERSED (async-first)

**Acceptance Criteria**:
- [ ] Fix LLM: `stream_generate()` → `astream_generate()` or `stream()` → `astream()`
- [ ] Fix Reranking: Make sync-first like other services
- [ ] All services: `method()` (sync) + `amethod()` (async)

### REQ-API-2: Fix Deprecated Event Loop Patterns (MEDIUM)
**Priority**: MEDIUM
**WHEN** async code runs in sync context
**SHALL** use `asyncio.run()` or proper patterns
**AND** NOT use deprecated `asyncio.new_event_loop()`

**Current State**:
- `llm/service.py:288-298` - Creates new loop each call
- `reranking/service.py:245-249` - Creates new loop each call

**Acceptance Criteria**:
- [ ] Replace `asyncio.new_event_loop()` with `asyncio.run()`
- [ ] Or use `asyncio.get_running_loop()` where appropriate

---

## 6. Missing Implementation Requirements

### REQ-IMPL-1: Implement CulturalEmotionalAwareness (CRITICAL)
**Priority**: CRITICAL
**WHEN** cultural awareness module is used
**SHALL** have actual implementation
**AND** NOT just docstring

**Current State**:
- `communication/cultural.py:1-7` - EMPTY (only docstring)
- `communication/__init__.py` imports `CulturalEmotionalAwareness` - WILL FAIL

**Acceptance Criteria**:
- [ ] EITHER: Implement `CulturalEmotionalAwareness` class
- [ ] OR: Remove from `__init__.py` exports

### REQ-IMPL-2: Fix Stub Methods in Learning (MEDIUM)
**Priority**: MEDIUM
**WHEN** learning pattern methods are called
**SHALL** return meaningful results
**AND** NOT always return empty/hardcoded values

**Current State (7 stub methods)**:
- `learning/patterns.py:552-557` - `_identify_learning_areas()` returns `[]`
- `learning/patterns.py:559-564` - `_identify_avoided_topics()` returns `[]`
- `learning/patterns.py:566-571` - `_analyze_topic_transitions()` returns `{}`
- `learning/patterns.py:622-625` - `_determine_interaction_style()` returns `"conversational"`
- `learning/patterns.py:639-644` - `_analyze_help_seeking_behavior()` returns `"direct"`
- `learning/patterns.py:651-654` - `_determine_learning_style()` returns `"textual"`
- `learning/patterns.py:661-664` - `_estimate_attention_span()` returns `"medium"`

**Acceptance Criteria**:
- [ ] EITHER: Implement these methods properly
- [ ] OR: Add `# TODO: Implement` comments and create tracking issue
- [ ] OR: Remove if not needed

### REQ-IMPL-3: Connect Companion to Storage (MEDIUM)
**Priority**: MEDIUM
**WHEN** companion profiles are created
**SHALL** persist to Qdrant via `CompanionStorage`
**AND** NOT only store in memory

**Current State**:
- `relationship_manager.py` stores in `self.profiles: Dict` (memory only)
- `companion/storage.py` EXISTS with Qdrant methods but NOT USED

**Acceptance Criteria**:
- [ ] Add `self.storage = CompanionStorage()` to `__init__`
- [ ] Call `storage.store_*` methods after profile updates
- [ ] Load profiles from storage on startup

---

## 7. Dead Code Removal Requirements

### REQ-DEAD-1: Remove Unused Imports (LOW)
**Priority**: LOW
**WHEN** code is imported
**SHALL** be used
**AND** unused imports deleted

**Current State**:
- `embeddings/service.py:13` - `from pathlib import Path` UNUSED
- `reranking/service.py:15` - `from pathlib import Path` UNUSED

**Acceptance Criteria**:
- [ ] Delete unused `Path` imports
- [ ] Run linter to find other unused imports

### REQ-DEAD-2: Remove Deprecated Files (LOW)
**Priority**: LOW
**WHEN** codebase is cleaned
**SHALL** remove deprecated files
**AND** document or delete archive

**Current State**:
- `cli.py` (root) - DEPRECATED notice but still exists
- `archive/` - 18 Python files
- `morgan-server/tests_archive/` - 15 test files

**Acceptance Criteria**:
- [ ] Delete `cli.py` from root
- [ ] Review `archive/` - delete or document
- [ ] Review `tests_archive/` - delete or document

### REQ-DEAD-3: Remove Unused Methods (LOW)
**Priority**: LOW
**WHEN** methods are defined
**SHALL** be called somewhere
**AND** unused methods deleted

**Current State**:
- `companion/relationship_manager.py:361-427` - `suggest_conversation_topics()` UNUSED
- `companion/storage.py:329-367` - `search_similar_emotional_states()` UNUSED
- `utils/validators.py:109-130` - `validate_file_path()` UNUSED
- `empathy/validator.py` - Exported but never imported

**Acceptance Criteria**:
- [ ] Delete or integrate unused methods
- [ ] Remove orphaned modules from exports

---

## 8. Data Integrity Requirements

### REQ-DATA-1: Collection Name Consistency
**Priority**: HIGH
**WHEN** memory is stored and searched
**SHALL** use same collection name
**AND** fix mismatched collection names

**Current State**:
- `memory_processor.py:124` uses `morgan_memories`
- `multi_stage_search.py:176` may use `morgan_turns` (MISMATCH)

**Acceptance Criteria**:
- [ ] Verify and fix collection name consistency
- [ ] Add integration test for store-then-search

---

## Summary

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| Exceptions | 1 | 1 | 0 | 0 | 2 |
| Singletons | 1 | 1 | 0 | 0 | 2 |
| Deduplication | 0 | 2 | 2 | 0 | 4 |
| Configuration | 1 | 1 | 1 | 0 | 3 |
| API Consistency | 0 | 1 | 1 | 0 | 2 |
| Missing Implementation | 1 | 0 | 2 | 0 | 3 |
| Dead Code | 0 | 0 | 0 | 3 | 3 |
| Data Integrity | 0 | 1 | 0 | 0 | 1 |
| **Total** | **4** | **7** | **6** | **3** | **20** |

**Estimated Impact**: ~1,500+ lines of duplicate code removed

---

## Traceability Matrix

| Requirement | Files Affected | Priority |
|-------------|----------------|----------|
| REQ-EXC-1 | exceptions.py, error_handling.py, validators.py | CRITICAL |
| REQ-EXC-2 | reranking/service.py, profile.py | HIGH |
| REQ-SING-1 | utils/singleton.py, error_handling.py, distributed_config.py | CRITICAL |
| REQ-SING-2 | embeddings/service.py, reranking/service.py | HIGH |
| REQ-DUP-1 | multi_stage_search.py, companion_memory_search.py, memory_processor.py | HIGH |
| REQ-DUP-2 | detector.py, intelligence_engine.py | HIGH |
| REQ-CFG-1 | defaults.py, settings.py, distributed_config.py | CRITICAL |
| REQ-CFG-2 | reranking/service.py | HIGH |
| REQ-API-1 | llm/service.py, reranking/service.py | HIGH |
| REQ-IMPL-1 | communication/cultural.py | CRITICAL |

---

*Last Updated: 2025-12-27*
