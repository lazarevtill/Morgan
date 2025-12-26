# Implementation Tasks: Codebase Cleanup v2

## Overview

This document contains the detailed implementation tasks for the Morgan codebase cleanup. Tasks are organized in phases with dependencies clearly marked.

**Estimated Total Time**: 3-5 days
**Priority Order**: P0 (Critical) → P1 (High) → P2 (Medium) → P3 (Low)

---

## Phase 1: Critical Data Fixes (P0)

**Duration**: 2-3 hours
**Dependencies**: None
**Status**: Pending

### Task 1.1: Fix Collection Name Mismatch

- [ ] 1.1.1 Read `morgan-rag/morgan/search/multi_stage_search.py`
- [ ] 1.1.2 Change line 176 from `self.memory_collection = "morgan_turns"` to `self.memory_collection = "morgan_memories"`
- [ ] 1.1.3 Search for any other references to `morgan_turns` collection
- [ ] 1.1.4 Verify `memory_processor.py` uses `morgan_memories`
- [ ] 1.1.5 Add integration test for store-then-search

**Files**: `morgan-rag/morgan/search/multi_stage_search.py`
**Requirements**: REQ-DATA-1

### Task 1.2: Fix Test Import Errors

- [ ] 1.2.1 Read `morgan-cli/tests/test_client_properties.py`
- [ ] 1.2.2 Add `WebSocketClient` to imports at line 15-22
- [ ] 1.2.3 Change `ClientConnectionError` to `ConnectionError` at line 372
- [ ] 1.2.4 Verify test file syntax is valid
- [ ] 1.2.5 Run tests to confirm fixes work

**Files**: `morgan-cli/tests/test_client_properties.py`
**Requirements**: REQ-DATA-2

### Task 1.3: Fix Client cleanup_memory Bug

- [ ] 1.3.1 Read `morgan-cli/morgan_cli/client.py`
- [ ] 1.3.2 Find `cleanup_memory()` method (around line 437-450)
- [ ] 1.3.3 Update `delete()` method to accept `params` parameter
- [ ] 1.3.4 Pass `params` to `delete()` call in `cleanup_memory()`
- [ ] 1.3.5 Add test for cleanup_memory

**Files**: `morgan-cli/morgan_cli/client.py`
**Requirements**: REQ-CLI-1

---

## Phase 2: Shared Module Structure (P1)

**Duration**: 2-3 hours
**Dependencies**: None
**Status**: Pending

### Task 2.1: Create Shared Config Module

- [ ] 2.1.1 Create `shared/config/__init__.py`
- [ ] 2.1.2 Create `shared/config/base.py` with `MorganBaseSettings`
- [ ] 2.1.3 Create `shared/config/defaults.py` with all default values
- [ ] 2.1.4 Create `shared/config/validators.py` with shared validation logic
- [ ] 2.1.5 Export classes from `__init__.py`

**Files**: `shared/config/`
**Requirements**: REQ-CFG-1

### Task 2.2: Create Shared Exceptions Module

- [ ] 2.2.1 Create `shared/exceptions/__init__.py`
- [ ] 2.2.2 Create `shared/exceptions/base.py` with full exception hierarchy
- [ ] 2.2.3 Define `MorganError` base class with context
- [ ] 2.2.4 Define all service-specific exceptions
- [ ] 2.2.5 Export exceptions from `__init__.py`

**Files**: `shared/exceptions/`
**Requirements**: REQ-EXC-1

### Task 2.3: Create Shared Utils Module

- [ ] 2.3.1 Create `shared/utils/__init__.py`
- [ ] 2.3.2 Create `shared/utils/singleton.py` with `SingletonFactory`
- [ ] 2.3.3 Create `shared/utils/deduplication.py`
- [ ] 2.3.4 Create `shared/utils/text_extraction.py`
- [ ] 2.3.5 Create `shared/utils/health_monitor.py`
- [ ] 2.3.6 Export utilities from `__init__.py`

**Files**: `shared/utils/`
**Requirements**: REQ-DUP-3, REQ-DUP-4, REQ-SVC-1, REQ-SVC-3

### Task 2.4: Create Shared Models Module

- [ ] 2.4.1 Create `shared/models/__init__.py`
- [ ] 2.4.2 Create `shared/models/enums.py` with `HostRole`, `GPURole`, etc.
- [ ] 2.4.3 Export enums from `__init__.py`

**Files**: `shared/models/`
**Requirements**: REQ-DUP-2

---

## Phase 3: Exception Consolidation (P1)

**Duration**: 2-3 hours
**Dependencies**: Phase 2.2
**Status**: Pending

### Task 3.1: Update morgan/exceptions.py

- [ ] 3.1.1 Read current `morgan-rag/morgan/exceptions.py`
- [ ] 3.1.2 Remove unused exception classes (lines 84-208)
- [ ] 3.1.3 Import and re-export from `shared.exceptions`
- [ ] 3.1.4 Keep backward compatibility with existing imports

**Files**: `morgan-rag/morgan/exceptions.py`
**Requirements**: REQ-EXC-2

### Task 3.2: Update error_handling.py

- [ ] 3.2.1 Read `morgan-rag/morgan/utils/error_handling.py`
- [ ] 3.2.2 Remove duplicate exception definitions
- [ ] 3.2.3 Import exceptions from `morgan.exceptions`
- [ ] 3.2.4 Keep decorator implementations

**Files**: `morgan-rag/morgan/utils/error_handling.py`
**Requirements**: REQ-EXC-1

### Task 3.3: Update companion_error_handling.py

- [ ] 3.3.1 Read `morgan-rag/morgan/utils/companion_error_handling.py`
- [ ] 3.3.2 Remove duplicate `EmotionalProcessingError`
- [ ] 3.3.3 Remove duplicate `MemoryProcessingError`
- [ ] 3.3.4 Import from `morgan.exceptions`

**Files**: `morgan-rag/morgan/utils/companion_error_handling.py`
**Requirements**: REQ-EXC-1

### Task 3.4: Update validators.py

- [ ] 3.4.1 Read `morgan-rag/morgan/utils/validators.py`
- [ ] 3.4.2 Remove duplicate `ValidationError` class
- [ ] 3.4.3 Import from `morgan.exceptions`

**Files**: `morgan-rag/morgan/utils/validators.py`
**Requirements**: REQ-EXC-1

### Task 3.5: Fix Service Exceptions

- [ ] 3.5.1 Update `services/ocr_service.py` exceptions to inherit from `MorganError`
- [ ] 3.5.2 Update `services/external_knowledge/mcp_client.py` exceptions

**Files**: `services/ocr_service.py`, `services/external_knowledge/mcp_client.py`
**Requirements**: REQ-EXC-1

---

## Phase 4: Code Deduplication (P1)

**Duration**: 3-4 hours
**Dependencies**: Phase 2
**Status**: Pending

### Task 4.1: Remove Duplicate setup_model_cache

- [ ] 4.1.1 Verify `morgan/utils/model_cache.py` has complete implementation
- [ ] 4.1.2 Read `services/embeddings/service.py`
- [ ] 4.1.3 Remove `setup_model_cache()` function (lines 39-87)
- [ ] 4.1.4 Add import from `morgan.utils.model_cache`
- [ ] 4.1.5 Read `services/reranking/service.py`
- [ ] 4.1.6 Remove `setup_model_cache()` function (lines 47-84)
- [ ] 4.1.7 Add import from `morgan.utils.model_cache`
- [ ] 4.1.8 Check `config/distributed_config.py` for duplicates

**Files**: `services/embeddings/service.py`, `services/reranking/service.py`
**Requirements**: REQ-DUP-1

### Task 4.2: Consolidate HostRole Enum

- [ ] 4.2.1 Verify `shared/models/enums.py` has `HostRole`
- [ ] 4.2.2 Update `infrastructure/distributed_gpu_manager.py` to import from shared
- [ ] 4.2.3 Remove local `HostRole` definition (lines 29-36)
- [ ] 4.2.4 Update `infrastructure/distributed_manager.py` to import from shared
- [ ] 4.2.5 Remove local `HostRole` definition (lines 41-49)
- [ ] 4.2.6 Update all references to use shared enum

**Files**: `infrastructure/distributed_gpu_manager.py`, `infrastructure/distributed_manager.py`
**Requirements**: REQ-DUP-2

### Task 4.3: Create Intelligence Constants

- [ ] 4.3.1 Create `intelligence/constants.py`
- [ ] 4.3.2 Add `EMOTION_VALENCE` mapping
- [ ] 4.3.3 Add `FORMALITY_INDICATORS`
- [ ] 4.3.4 Add `INTENSITY_MODIFIERS`
- [ ] 4.3.5 Update `emotions/analyzer.py` to use constants
- [ ] 4.3.6 Update `emotions/context.py` to use constants
- [ ] 4.3.7 Update `emotions/recovery.py` to use constants

**Files**: `intelligence/constants.py`, `emotions/*.py`
**Requirements**: REQ-DUP-5

---

## Phase 5: Configuration Consolidation (P2)

**Duration**: 2-3 hours
**Dependencies**: Phase 2.1
**Status**: Pending

### Task 5.1: Update morgan-rag Settings

- [ ] 5.1.1 Read `morgan-rag/morgan/config/settings.py`
- [ ] 5.1.2 Make `Settings` inherit from `MorganBaseSettings`
- [ ] 5.1.3 Remove duplicate field definitions
- [ ] 5.1.4 Keep RAG-specific settings only
- [ ] 5.1.5 Test settings loading

**Files**: `morgan-rag/morgan/config/settings.py`
**Requirements**: REQ-CFG-1

### Task 5.2: Update morgan-server Config

- [ ] 5.2.1 Read `morgan-server/morgan_server/config.py`
- [ ] 5.2.2 Make `ServerConfig` inherit from `MorganBaseSettings`
- [ ] 5.2.3 Remove duplicate field definitions
- [ ] 5.2.4 Keep server-specific settings only

**Files**: `morgan-server/morgan_server/config.py`
**Requirements**: REQ-CFG-1

### Task 5.3: Update morgan-cli Config

- [ ] 5.3.1 Read `morgan-cli/morgan_cli/config.py`
- [ ] 5.3.2 Make `Config` inherit from `MorganBaseSettings`
- [ ] 5.3.3 Remove duplicate field definitions
- [ ] 5.3.4 Keep CLI-specific settings only

**Files**: `morgan-cli/morgan_cli/config.py`
**Requirements**: REQ-CFG-1

### Task 5.4: Merge Docker Example Files

- [ ] 5.4.1 Read `docker/.env.example`
- [ ] 5.4.2 Read `docker/env.example`
- [ ] 5.4.3 Merge into single `docker/.env.example`
- [ ] 5.4.4 Use consistent default values
- [ ] 5.4.5 Delete `docker/env.example`

**Files**: `docker/.env.example`, `docker/env.example`
**Requirements**: REQ-CFG-3

---

## Phase 6: Service Pattern Updates (P2)

**Duration**: 2-3 hours
**Dependencies**: Phase 2.3
**Status**: Pending

### Task 6.1: Add Missing Service Methods

- [ ] 6.1.1 Add `health_check()` to `EmbeddingService`
- [ ] 6.1.2 Add `health_check()` to `RerankingService`
- [ ] 6.1.3 Add `shutdown()` to `EmbeddingService`
- [ ] 6.1.4 Add `shutdown()` to `RerankingService`
- [ ] 6.1.5 Add `shutdown()` to `OCRService`

**Files**: `services/embeddings/service.py`, `services/reranking/service.py`, `services/ocr_service.py`
**Requirements**: REQ-SVC-2

### Task 6.2: Refactor to Use Singleton Factory

- [ ] 6.2.1 Update LLM service to use `SingletonFactory`
- [ ] 6.2.2 Update Embedding service to use `SingletonFactory`
- [ ] 6.2.3 Update Reranking service to use `SingletonFactory`
- [ ] 6.2.4 Update OCR service to use `SingletonFactory`
- [ ] 6.2.5 Update external knowledge services

**Files**: All service files
**Requirements**: REQ-SVC-1

---

## Phase 7: Client/Server Fixes (P2)

**Duration**: 2-3 hours
**Dependencies**: Phase 2
**Status**: Pending

### Task 7.1: Extract Common Client Code

- [ ] 7.1.1 Create `BaseClient` class with shared methods
- [ ] 7.1.2 Move `status` property to base
- [ ] 7.1.3 Move `add_status_callback` to base
- [ ] 7.1.4 Move `_set_status` to base
- [ ] 7.1.5 Move `__aenter__` and `__aexit__` to base
- [ ] 7.1.6 Have `HTTPClient` and `WebSocketClient` inherit from base

**Files**: `morgan-cli/morgan_cli/client.py`
**Requirements**: REQ-CLI-2

### Task 7.2: Fix Server Global State

- [ ] 7.2.1 Replace global `_assistant` in `chat.py` with `Depends()`
- [ ] 7.2.2 Replace global `_health_system` in `health.py` with `Depends()`
- [ ] 7.2.3 Replace global `_session_manager` in `session.py` with `Depends()`
- [ ] 7.2.4 Create dependency injection functions

**Files**: `morgan-server/morgan_server/api/routes/*.py`
**Requirements**: REQ-SRV-1

### Task 7.3: Fix Silent Error Suppression

- [ ] 7.3.1 Read `morgan-server/morgan_server/api/routes/profile.py`
- [ ] 7.3.2 Replace `except: pass` at line 131 with logging
- [ ] 7.3.3 Replace `except: pass` at line 139 with logging
- [ ] 7.3.4 Search for other silent error suppressions

**Files**: `morgan-server/morgan_server/api/routes/profile.py`
**Requirements**: REQ-SRV-2

---

## Phase 8: Dead Code Removal (P3)

**Duration**: 1-2 hours
**Dependencies**: Phases 1-7
**Status**: Pending

### Task 8.1: Remove Deprecated Files

- [ ] 8.1.1 Delete `cli.py` (deprecated stub)
- [ ] 8.1.2 Delete `infrastructure/consul_client.py` (unused)
- [ ] 8.1.3 Search for other unused files

**Files**: `cli.py`, `infrastructure/consul_client.py`
**Requirements**: REQ-DEAD-1

### Task 8.2: Clean Up Unused Imports

- [ ] 8.2.1 Run linter to identify unused imports
- [ ] 8.2.2 Remove unused imports from each file
- [ ] 8.2.3 Verify code still works

**Files**: All Python files
**Requirements**: REQ-DEAD-2

---

## Phase 9: Intelligence Module Cleanup (P3)

**Duration**: 1-2 hours
**Dependencies**: Phase 4.3
**Status**: Pending

### Task 9.1: Consolidate Emotion Detection

- [ ] 9.1.1 Identify duplicate detection logic in `intelligence_engine.py`
- [ ] 9.1.2 Make `detector.py` the single source
- [ ] 9.1.3 Have `intelligence_engine.py` delegate to detector
- [ ] 9.1.4 Remove duplicate patterns

**Files**: `intelligence/core/intelligence_engine.py`, `intelligence/emotions/detector.py`
**Requirements**: REQ-INT-1

### Task 9.2: Document Stub Methods

- [ ] 9.2.1 Add TODO comments to stub methods in `learning/adaptation.py`
- [ ] 9.2.2 Add TODO comments to stub methods in `learning/patterns.py`
- [ ] 9.2.3 Add TODO comments to stub methods in `communication/style.py`
- [ ] 9.2.4 Create tracking issue for implementation

**Files**: `learning/adaptation.py`, `learning/patterns.py`, `communication/style.py`
**Requirements**: REQ-INT-2

---

## Phase 10: Validation & Testing (P3)

**Duration**: 2-3 hours
**Dependencies**: All phases
**Status**: Pending

### Task 10.1: Run Existing Tests

- [ ] 10.1.1 Run morgan-rag tests
- [ ] 10.1.2 Run morgan-cli tests
- [ ] 10.1.3 Run morgan-server tests
- [ ] 10.1.4 Fix any broken tests

**Files**: All test files
**Requirements**: All

### Task 10.2: Add Integration Tests

- [ ] 10.2.1 Add test for memory store-then-search
- [ ] 10.2.2 Add test for exception hierarchy
- [ ] 10.2.3 Add test for configuration loading
- [ ] 10.2.4 Add test for singleton factory

**Files**: New test files
**Requirements**: All

### Task 10.3: Final Validation

- [ ] 10.3.1 Verify no duplicate exception definitions
- [ ] 10.3.2 Verify no duplicate setup_model_cache implementations
- [ ] 10.3.3 Verify single HostRole enum definition
- [ ] 10.3.4 Verify collection names are consistent
- [ ] 10.3.5 Verify all imports resolve correctly
- [ ] 10.3.6 Update documentation

**Requirements**: All

---

## Summary

| Phase | Tasks | Duration | Priority | Dependencies |
|-------|-------|----------|----------|--------------|
| Phase 1: Critical Fixes | 3 | 2-3 hrs | P0 | None |
| Phase 2: Shared Module | 4 | 2-3 hrs | P1 | None |
| Phase 3: Exceptions | 5 | 2-3 hrs | P1 | Phase 2.2 |
| Phase 4: Deduplication | 3 | 3-4 hrs | P1 | Phase 2 |
| Phase 5: Configuration | 4 | 2-3 hrs | P2 | Phase 2.1 |
| Phase 6: Service Patterns | 2 | 2-3 hrs | P2 | Phase 2.3 |
| Phase 7: Client/Server | 3 | 2-3 hrs | P2 | Phase 2 |
| Phase 8: Dead Code | 2 | 1-2 hrs | P3 | Phases 1-7 |
| Phase 9: Intelligence | 2 | 1-2 hrs | P3 | Phase 4.3 |
| Phase 10: Validation | 3 | 2-3 hrs | P3 | All |
| **Total** | **31** | **20-29 hrs** | | |

---

## Progress Tracking

### Completed Tasks
- [ ] None yet

### In Progress
- [ ] None yet

### Blocked
- [ ] None yet

---

*Last Updated: 2025-12-26*
