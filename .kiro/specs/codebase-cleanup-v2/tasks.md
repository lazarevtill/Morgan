# Implementation Tasks: Codebase Cleanup v2

## Overview

This document contains detailed implementation tasks for the Morgan codebase cleanup. Since app is in development, NO backward compatibility is required - delete duplicates directly.

**Date**: 2025-12-27
**Status**: Ready for Implementation
**Priority Order**: P0 (Critical) → P1 (High) → P2 (Medium) → P3 (Low)

---

## Phase 1: Critical Fixes (P0)

**Duration**: 2-3 hours
**Dependencies**: None
**Status**: ✅ COMPLETE (All issues already fixed in codebase)

### Task 1.1: Fix Empty Cultural Module

- [x] 1.1.1 Read `morgan-rag/morgan/communication/__init__.py` ✅ No broken import exists
- [x] 1.1.2 ✅ No `CulturalEmotionalAwareness` import in `__init__.py`
- [x] 1.1.3 ✅ Not in `__all__`
- [x] 1.1.4 ✅ `cultural.py` file doesn't exist - already deleted
- [x] 1.1.5 ✅ No references to this class in codebase

**Files**: `communication/__init__.py`, `communication/cultural.py`
**Requirements**: REQ-IMPL-1
**Note**: Issue was already resolved - file and import don't exist

### Task 1.2: Add Thread Safety Locks

- [x] 1.2.1 Read `morgan-rag/morgan/services/embeddings/service.py`
- [x] 1.2.2 ✅ Already has `self._availability_lock = threading.Lock()` (line 110)
- [x] 1.2.3 ✅ Already wraps `_remote_available` updates with lock (lines 226-234, 256-264, 275-277, 285-291)
- [x] 1.2.4 Read `morgan-rag/morgan/services/reranking/service.py`
- [x] 1.2.5 ✅ Already has `self._availability_lock = threading.Lock()` (line 104)
- [x] 1.2.6 ✅ Already wraps availability flag updates with lock (lines 291-305, 350-366)

**Files**: `services/embeddings/service.py`, `services/reranking/service.py`
**Requirements**: REQ-SING-2
**Note**: Thread safety locks already implemented correctly

### Task 1.3: Fix Silent Error Suppression

- [x] 1.3.1 Read `morgan-rag/morgan/services/reranking/service.py`
- [x] 1.3.2 ✅ Line 504 already has proper logging: `logger.debug("Remote reranking endpoint not available: %s", e)`
- [x] 1.3.3 ✅ No silent `except Exception: pass` found
- [x] 1.3.4 Read `morgan-server/morgan_server/api/routes/profile.py`
- [x] 1.3.5 ✅ Lines 129-133 and 141-146 already use `logger.warning()` with proper exception details

**Files**: `reranking/service.py`, `profile.py`
**Requirements**: REQ-EXC-2
**Note**: All exception handlers already include proper logging

---

## Phase 2: Exception Consolidation (P0)

**Duration**: 2-3 hours
**Dependencies**: None
**Status**: ✅ COMPLETE

### Task 2.1: Enhance exceptions.py

- [x] 2.1.1 Read `morgan-rag/morgan/exceptions.py`
- [x] 2.1.2 Verify it has: MorganError, ConfigurationError, ValidationError ✅ Already present
- [x] 2.1.3 Add if missing: CompanionError, EmotionalProcessingError, MemoryProcessingError ✅ Already present
- [x] 2.1.4 Add if missing: ServiceError, LLMError, EmbeddingError, RerankingError ✅ Already present
- [x] 2.1.5 Add if missing: InfrastructureError, ConnectionError, TimeoutError ✅ Already present
- [x] 2.1.6 Added: RelationshipTrackingError, EmpathyGenerationError (moved from companion_error_handling.py)

**Files**: `morgan/exceptions.py`
**Requirements**: REQ-EXC-1

### Task 2.2: Delete Duplicate Exceptions from error_handling.py

- [x] 2.2.1 Read `morgan-rag/morgan/utils/error_handling.py`
- [x] 2.2.2 Identify exception class definitions (lines 94-262)
- [x] 2.2.3 Deleted duplicate CompanionError, EmotionalProcessingError, MemoryProcessingError
- [x] 2.2.4 Added imports: `from morgan.exceptions import MorganError, ValidationError, ConfigurationError, CompanionError, EmotionalProcessingError, MemoryProcessingError`
- [x] 2.2.5 Kept VectorizationError, EmbeddingError, StorageError, SearchError, CacheError, NetworkError (with ErrorHandlingMixin for rich context)
- [x] 2.2.6 Kept decorator implementations and other utility code

**Files**: `utils/error_handling.py`
**Requirements**: REQ-EXC-1

### Task 2.3: Delete Duplicate ValidationError from validators.py

- [x] 2.3.1 Read `morgan-rag/morgan/utils/validators.py`
- [x] 2.3.2 ✅ Already fixed - imports ValidationError from morgan.exceptions (line 9)
- [x] 2.3.3 ✅ Already has import: `from morgan.exceptions import ValidationError`
- [x] 2.3.4 Verified all usages still work (syntax check passed)

**Files**: `utils/validators.py`
**Requirements**: REQ-EXC-1

### Task 2.4: Delete Duplicate Exceptions from companion_error_handling.py

- [x] 2.4.1 Read `morgan-rag/morgan/utils/companion_error_handling.py`
- [x] 2.4.2 Deleted RelationshipTrackingError and EmpathyGenerationError class definitions
- [x] 2.4.3 Added import from `morgan.exceptions` for all exception classes
- [x] 2.4.4 Verified all usages still work (syntax check passed)

**Files**: `utils/companion_error_handling.py`
**Requirements**: REQ-EXC-1

---

## Phase 3: Singleton Consolidation (P1)

**Duration**: 2-3 hours
**Dependencies**: Phase 2
**Status**: Pending

### Task 3.1: Delete shared/utils/singleton.py

- [ ] 3.1.1 Check if `shared/utils/singleton.py` exists
- [ ] 3.1.2 Search for any imports from `shared.utils.singleton`
- [ ] 3.1.3 Update any imports to use `morgan.utils.singleton`
- [ ] 3.1.4 Delete `shared/utils/singleton.py`

**Files**: `shared/utils/singleton.py`
**Requirements**: REQ-SING-1

### Task 3.2: Delete Manual Singletons from error_handling.py

- [ ] 3.2.1 Read `morgan-rag/morgan/utils/error_handling.py` lines 1109-1136
- [ ] 3.2.2 Delete `_degradation_manager_instance`, `_recovery_manager_instance`
- [ ] 3.2.3 Delete `_degradation_lock`, `_recovery_lock`
- [ ] 3.2.4 Delete `get_degradation_manager()`, `get_recovery_manager()`
- [ ] 3.2.5 Replace with SingletonFactory usage

**Files**: `utils/error_handling.py`
**Requirements**: REQ-SING-1

### Task 3.3: Delete Manual Singleton from distributed_config.py

- [ ] 3.3.1 Read `morgan-rag/morgan/config/distributed_config.py` lines 511-534
- [ ] 3.3.2 Delete `_cached_config` global variable
- [ ] 3.3.3 Update `get_distributed_config()` to use SingletonFactory
- [ ] 3.3.4 Verify config loading still works

**Files**: `config/distributed_config.py`
**Requirements**: REQ-SING-1

### Task 3.4: Migrate Services to SingletonFactory

- [ ] 3.4.1 Update `services/llm/service.py` get_llm_service()
- [ ] 3.4.2 Update `services/embeddings/service.py` get_embedding_service()
- [ ] 3.4.3 Update `services/reranking/service.py` get_reranking_service()
- [ ] 3.4.4 Delete `_*_service_instance` and `_*_service_lock` variables
- [ ] 3.4.5 Use `SingletonFactory.get_or_create(ServiceClass)` pattern

**Files**: All service files
**Requirements**: REQ-SING-1

---

## Phase 4: Configuration Unification (P1)

**Duration**: 1-2 hours
**Dependencies**: Phase 3
**Status**: Pending

### Task 4.1: Fix Configuration Defaults Mismatch

- [ ] 4.1.1 Read `morgan-rag/morgan/config/defaults.py`
- [ ] 4.1.2 Verify `LLM_MODEL = "qwen2.5:32b-instruct-q4_K_M"`
- [ ] 4.1.3 Read `morgan-rag/morgan/config/settings.py`
- [ ] 4.1.4 Change `llm_model` default from `gemma3:latest` to `Defaults.LLM_MODEL`
- [ ] 4.1.5 Import `from .defaults import Defaults`
- [ ] 4.1.6 Update all Field defaults to use `Defaults.X`

**Files**: `config/settings.py`
**Requirements**: REQ-CFG-1

### Task 4.2: Fix distributed_config.py Defaults

- [ ] 4.2.1 Read `morgan-rag/morgan/config/distributed_config.py`
- [ ] 4.2.2 Import `from .defaults import Defaults`
- [ ] 4.2.3 Update dataclass defaults to use `Defaults.X`
- [ ] 4.2.4 Verify model names are consistent

**Files**: `config/distributed_config.py`
**Requirements**: REQ-CFG-1

### Task 4.3: Fix Reranking Service Config Access

- [ ] 4.3.1 Read `services/reranking/service.py`
- [ ] 4.3.2 Find line 109: `os.environ.get("MORGAN_RERANKING_ENDPOINT")`
- [ ] 4.3.3 Replace with `getattr(self.settings, "reranking_endpoint", None)`
- [ ] 4.3.4 Remove any other direct `os.environ.get()` calls

**Files**: `services/reranking/service.py`
**Requirements**: REQ-CFG-2

---

## Phase 5: Intelligence Consolidation (P1)

**Duration**: 2-3 hours
**Dependencies**: Phase 4
**Status**: Pending

### Task 5.1: Create intelligence/constants.py

- [ ] 5.1.1 Create new file `morgan-rag/morgan/intelligence/constants.py`
- [ ] 5.1.2 Add `EmotionType` enum
- [ ] 5.1.3 Add `EMOTION_VALENCE` mapping
- [ ] 5.1.4 Add `EMOTION_PATTERNS` dictionary
- [ ] 5.1.5 Add `INTENSITY_MODIFIERS` dictionary
- [ ] 5.1.6 Add `FORMALITY_INDICATORS` dictionary
- [ ] 5.1.7 Update `intelligence/__init__.py` to export constants

**Files**: `intelligence/constants.py`
**Requirements**: REQ-DUP-2

### Task 5.2: Update detector.py to Use Constants

- [ ] 5.2.1 Read `morgan-rag/morgan/intelligence/emotions/detector.py`
- [ ] 5.2.2 Add import: `from morgan.intelligence.constants import ...`
- [ ] 5.2.3 Delete `EMOTION_PATTERNS` (lines 40-82)
- [ ] 5.2.4 Delete `INTENSITY_MODIFIERS` (lines 85-98)
- [ ] 5.2.5 Update code to use imported constants

**Files**: `intelligence/emotions/detector.py`
**Requirements**: REQ-DUP-2

### Task 5.3: Update intelligence_engine.py to Delegate

- [ ] 5.3.1 Read `morgan-rag/morgan/intelligence/core/intelligence_engine.py`
- [ ] 5.3.2 Delete `EMOTION_PATTERNS` (lines 51-88)
- [ ] 5.3.3 Delete `INTENSITY_MODIFIERS` (lines 91-102)
- [ ] 5.3.4 Delete `_detect_emotions_rule_based()` (line 364)
- [ ] 5.3.5 Delete `_detect_emotions_llm()` (line 393)
- [ ] 5.3.6 Delete `_combine_emotion_results()` (line 438)
- [ ] 5.3.7 Add import: `from morgan.intelligence.emotions.detector import EmotionDetector`
- [ ] 5.3.8 Update `analyze_emotion()` to use `EmotionDetector`

**Files**: `intelligence/core/intelligence_engine.py`
**Requirements**: REQ-DUP-2

---

## Phase 6: Deduplication Consolidation (P1)

**Duration**: 2-3 hours
**Dependencies**: Phase 5
**Status**: Pending

### Task 6.1: Delete shared/utils/deduplication.py

- [ ] 6.1.1 Check if `shared/utils/deduplication.py` exists
- [ ] 6.1.2 Search for any imports from `shared.utils.deduplication`
- [ ] 6.1.3 Update imports to use `morgan.utils.deduplication`
- [ ] 6.1.4 Delete `shared/utils/deduplication.py`

**Files**: `shared/utils/deduplication.py`
**Requirements**: REQ-DUP-1

### Task 6.2: Update multi_stage_search.py

- [ ] 6.2.1 Read `morgan-rag/morgan/search/multi_stage_search.py`
- [ ] 6.2.2 Add import: `from morgan.utils.deduplication import ResultDeduplicator`
- [ ] 6.2.3 Add `self.deduplicator = ResultDeduplicator()` in `__init__`
- [ ] 6.2.4 Delete `_deduplicate_results()` (lines 1729-1775)
- [ ] 6.2.5 Delete `_apply_rrf_deduplication()` (lines 1660-1727)
- [ ] 6.2.6 Delete `_deduplicate_memory_results()` (lines 2300-2334)
- [ ] 6.2.7 Update callers to use `self.deduplicator.deduplicate_by_similarity()`

**Files**: `search/multi_stage_search.py`
**Requirements**: REQ-DUP-1

### Task 6.3: Update companion_memory_search.py

- [ ] 6.3.1 Read `morgan-rag/morgan/search/companion_memory_search.py`
- [ ] 6.3.2 Add import: `from morgan.utils.deduplication import ResultDeduplicator`
- [ ] 6.3.3 Delete `_deduplicate_search_results()` (lines 1057-1075)
- [ ] 6.3.4 Update callers to use `ResultDeduplicator`

**Files**: `search/companion_memory_search.py`
**Requirements**: REQ-DUP-1

### Task 6.4: Update memory_processor.py

- [ ] 6.4.1 Read `morgan-rag/morgan/memory/memory_processor.py`
- [ ] 6.4.2 Add import: `from morgan.utils.deduplication import ResultDeduplicator`
- [ ] 6.4.3 Delete `_deduplicate_memories()` (lines 608-623)
- [ ] 6.4.4 Update callers to use `ResultDeduplicator`

**Files**: `memory/memory_processor.py`
**Requirements**: REQ-DUP-1

---

## Phase 7: API Standardization (P2)

**Duration**: 1-2 hours
**Dependencies**: Phase 6
**Status**: Pending

### Task 7.1: Fix Reranking Async/Sync Pattern

- [ ] 7.1.1 Read `morgan-rag/morgan/services/reranking/service.py`
- [ ] 7.1.2 Rename `rerank()` to `arerank()` (async version)
- [ ] 7.1.3 Create new sync `rerank()` method as primary
- [ ] 7.1.4 Delete `rerank_sync()` wrapper method
- [ ] 7.1.5 Update callers to use new pattern

**Files**: `services/reranking/service.py`
**Requirements**: REQ-API-1

### Task 7.2: Fix LLM Streaming Method Names

- [ ] 7.2.1 Read `morgan-rag/morgan/services/llm/service.py`
- [ ] 7.2.2 Find `stream_generate()` method
- [ ] 7.2.3 Find `astream()` method
- [ ] 7.2.4 Rename to consistent pattern: `stream()` / `astream()` OR `stream_generate()` / `astream_generate()`
- [ ] 7.2.5 Update all callers

**Files**: `services/llm/service.py`
**Requirements**: REQ-API-1

### Task 7.3: Fix Deprecated Event Loop Patterns

- [ ] 7.3.1 Read `services/llm/service.py` lines 288-298
- [ ] 7.3.2 Replace `asyncio.new_event_loop()` with `asyncio.run()`
- [ ] 7.3.3 Read `services/reranking/service.py` lines 245-249
- [ ] 7.3.4 Replace `asyncio.new_event_loop()` with `asyncio.run()`

**Files**: `services/llm/service.py`, `services/reranking/service.py`
**Requirements**: REQ-API-2

---

## Phase 8: Dead Code Removal (P3)

**Duration**: 1-2 hours
**Dependencies**: All previous phases
**Status**: Pending

### Task 8.1: Delete Deprecated Files

- [ ] 8.1.1 Delete `cli.py` from project root
- [ ] 8.1.2 Verify no scripts reference it

**Files**: `cli.py`
**Requirements**: REQ-DEAD-2

### Task 8.2: Delete Unused Imports

- [ ] 8.2.1 Read `services/embeddings/service.py`
- [ ] 8.2.2 Delete `from pathlib import Path` (line 13) if unused
- [ ] 8.2.3 Read `services/reranking/service.py`
- [ ] 8.2.4 Delete `from pathlib import Path` (line 15) if unused
- [ ] 8.2.5 Run linter to find other unused imports

**Files**: `services/embeddings/service.py`, `services/reranking/service.py`
**Requirements**: REQ-DEAD-1

### Task 8.3: Handle Unused Methods (Document or Delete)

- [ ] 8.3.1 Review `companion/relationship_manager.py:361-427` `suggest_conversation_topics()`
- [ ] 8.3.2 Either delete or add `# TODO: Implement and integrate` comment
- [ ] 8.3.3 Review `companion/storage.py:329-367` `search_similar_emotional_states()`
- [ ] 8.3.4 Either delete or add `# TODO: Implement and integrate` comment
- [ ] 8.3.5 Review `utils/validators.py:109-130` `validate_file_path()`
- [ ] 8.3.6 Either delete or add `# TODO: Implement and integrate` comment

**Files**: `companion/relationship_manager.py`, `companion/storage.py`, `utils/validators.py`
**Requirements**: REQ-DEAD-3

### Task 8.4: Handle Stub Methods in Learning

- [ ] 8.4.1 Read `learning/patterns.py`
- [ ] 8.4.2 Add `# TODO: Implement - currently returns stub value` to:
  - `_identify_learning_areas()` (line 552)
  - `_identify_avoided_topics()` (line 559)
  - `_analyze_topic_transitions()` (line 566)
  - `_determine_interaction_style()` (line 622)
  - `_analyze_help_seeking_behavior()` (line 639)
  - `_determine_learning_style()` (line 651)
  - `_estimate_attention_span()` (line 661)

**Files**: `learning/patterns.py`
**Requirements**: REQ-IMPL-2

---

## Phase 9: Memory Search Consolidation (P2)

**Duration**: 1-2 hours
**Dependencies**: Phase 6
**Status**: Pending

### Task 9.1: Consolidate Memory Search Methods

- [ ] 9.1.1 Read `search/multi_stage_search.py`
- [ ] 9.1.2 Identify `_memory_search()` (line 1022)
- [ ] 9.1.3 Identify `_basic_memory_search()` (line 1185)
- [ ] 9.1.4 Merge functionality into single `_memory_search()` method
- [ ] 9.1.5 Delete `_basic_memory_search()`
- [ ] 9.1.6 Update all callers

**Files**: `search/multi_stage_search.py`
**Requirements**: REQ-DUP-3

### Task 9.2: Update companion_memory_search.py

- [ ] 9.2.1 Read `search/companion_memory_search.py`
- [ ] 9.2.2 Review `_execute_enhanced_memory_search()` (line 727)
- [ ] 9.2.3 Have it delegate to `MultiStageSearchEngine._memory_search()`
- [ ] 9.2.4 Or import shared functionality

**Files**: `search/companion_memory_search.py`
**Requirements**: REQ-DUP-3

---

## Phase 10: Companion Storage Integration (P2)

**Duration**: 1-2 hours
**Dependencies**: Phase 8
**Status**: Pending

### Task 10.1: Connect Companion to Storage

- [ ] 10.1.1 Read `companion/relationship_manager.py`
- [ ] 10.1.2 Add import: `from morgan.companion.storage import CompanionStorage`
- [ ] 10.1.3 Add `self.storage = CompanionStorage()` in `__init__`
- [ ] 10.1.4 In `build_user_profile()`, add `self.storage.store_companion_profile(profile)`
- [ ] 10.1.5 In `update_interaction()`, add storage update call
- [ ] 10.1.6 Add `_load_profiles_from_storage()` method
- [ ] 10.1.7 Call `_load_profiles_from_storage()` in `__init__`

**Files**: `companion/relationship_manager.py`
**Requirements**: REQ-IMPL-3

---

## Phase 11: Validation & Testing (P3)

**Duration**: 2-3 hours
**Dependencies**: All phases
**Status**: Pending

### Task 11.1: Verify No Duplicate Definitions

- [ ] 11.1.1 Run: `grep -r "class.*Error.*Exception" morgan-rag/`
- [ ] 11.1.2 Verify only `exceptions.py` defines exception classes
- [ ] 11.1.3 Run: `grep -r "_instance = None" morgan-rag/`
- [ ] 11.1.4 Verify only `SingletonFactory` uses this pattern
- [ ] 11.1.5 Run: `grep -r "def.*dedup" morgan-rag/`
- [ ] 11.1.6 Verify only `deduplication.py` defines dedup methods

**Requirements**: All

### Task 11.2: Run Tests

- [ ] 11.2.1 Run `pytest morgan-rag/tests/`
- [ ] 11.2.2 Run `pytest morgan-cli/tests/`
- [ ] 11.2.3 Run `pytest morgan-server/tests/`
- [ ] 11.2.4 Fix any broken tests

**Requirements**: All

### Task 11.3: Verify Imports

- [ ] 11.3.1 Run `python -c "import morgan"`
- [ ] 11.3.2 Run `python -c "from morgan.exceptions import MorganError"`
- [ ] 11.3.3 Run `python -c "from morgan.utils.singleton import SingletonFactory"`
- [ ] 11.3.4 Run `python -c "from morgan.intelligence.constants import EMOTION_PATTERNS"`
- [ ] 11.3.5 Fix any import errors

**Requirements**: All

---

## Summary

| Phase | Tasks | Duration | Priority | Dependencies |
|-------|-------|----------|----------|--------------|
| Phase 1: Critical Fixes | 3 | 2-3 hrs | P0 | None |
| Phase 2: Exception Consolidation | 4 | 2-3 hrs | P0 | None |
| Phase 3: Singleton Consolidation | 4 | 2-3 hrs | P1 | Phase 2 |
| Phase 4: Configuration Unification | 3 | 1-2 hrs | P1 | Phase 3 |
| Phase 5: Intelligence Consolidation | 3 | 2-3 hrs | P1 | Phase 4 |
| Phase 6: Deduplication Consolidation | 4 | 2-3 hrs | P1 | Phase 5 |
| Phase 7: API Standardization | 3 | 1-2 hrs | P2 | Phase 6 |
| Phase 8: Dead Code Removal | 4 | 1-2 hrs | P3 | All previous |
| Phase 9: Memory Search Consolidation | 2 | 1-2 hrs | P2 | Phase 6 |
| Phase 10: Companion Storage Integration | 1 | 1-2 hrs | P2 | Phase 8 |
| Phase 11: Validation & Testing | 3 | 2-3 hrs | P3 | All |
| **Total** | **34** | **17-27 hrs** | | |

---

## Quick Reference: Files to DELETE

| File | Reason |
|------|--------|
| `cli.py` (root) | DEPRECATED stub |
| `shared/utils/singleton.py` | Duplicate |
| `shared/utils/deduplication.py` | Duplicate |
| `communication/cultural.py` | Empty |

## Quick Reference: Key Line Numbers

| File | Lines | What to Delete/Fix |
|------|-------|-------------------|
| `error_handling.py` | 94-262 | Exception class definitions |
| `error_handling.py` | 1109-1136 | Manual singletons |
| `validators.py` | 10 | ValidationError class |
| `distributed_config.py` | 511-534 | _cached_config pattern |
| `detector.py` | 40-82 | EMOTION_PATTERNS |
| `detector.py` | 85-98 | INTENSITY_MODIFIERS |
| `intelligence_engine.py` | 51-88 | EMOTION_PATTERNS |
| `intelligence_engine.py` | 91-102 | INTENSITY_MODIFIERS |
| `intelligence_engine.py` | 364, 393, 438 | Duplicate emotion methods |
| `multi_stage_search.py` | 1660-1775, 2300-2334 | Dedup methods |
| `companion_memory_search.py` | 1057-1075 | Dedup method |
| `memory_processor.py` | 608-623 | Dedup method |
| `reranking/service.py` | 109 | os.environ.get() |
| `reranking/service.py` | 245-249, 497 | Event loop, silent except |
| `llm/service.py` | 288-298 | Event loop |
| `embeddings/service.py` | 13 | Unused Path import |
| `reranking/service.py` | 15 | Unused Path import |
| `settings.py` | 42, 51, 68 | Wrong defaults (gemma3) |

---

## Progress Tracking

### Completed Tasks
- [ ] None yet

### In Progress
- [ ] None yet

### Blocked
- [ ] None yet

---

*Last Updated: 2025-12-27*
