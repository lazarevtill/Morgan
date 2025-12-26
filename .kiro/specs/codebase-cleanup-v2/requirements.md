# Requirements Document: Codebase Cleanup v2

## Introduction

This requirements specification defines the comprehensive cleanup and reorganization of the Morgan codebase based on findings from 11 parallel exploration agents. The analysis identified 150+ issues including code duplications, inconsistent patterns, missing implementations, dead code, and configuration chaos.

**Date**: 2025-12-26
**Status**: Planning Complete - Ready for Implementation
**Priority**: Critical - Technical Debt & Data Integrity

## Glossary

- **morgan-rag** - Core RAG (Retrieval Augmented Generation) library with services, infrastructure, and intelligence modules
- **morgan-server** - FastAPI server exposing morgan-rag functionality via REST/WebSocket APIs
- **morgan-cli** - Command-line client for interacting with morgan-server
- **shared** - Cross-project shared utilities, models, and configuration
- **Collection** - Qdrant vector database collection for storing embeddings
- **Singleton Factory** - Thread-safe pattern for creating single instances of services

---

## 1. Critical Data Integrity Requirements

### REQ-DATA-1: Collection Name Consistency
**Priority**: CRITICAL
**WHEN** memory is stored via `MemoryProcessor`
**SHALL** use collection name `morgan_memories`
**AND** search operations must query the same collection name

**Acceptance Criteria**:
- [ ] `memory_processor.py:124` uses `morgan_memories`
- [ ] `multi_stage_search.py:176` uses `morgan_memories` (not `morgan_turns`)
- [ ] All memory-related searches find stored memories
- [ ] Integration test validates store-then-search works

### REQ-DATA-2: Test Import Fixes
**Priority**: CRITICAL
**WHEN** tests in morgan-cli are executed
**SHALL** have all required imports defined
**AND** no NameError exceptions during test execution

**Acceptance Criteria**:
- [ ] `test_client_properties.py:372` uses `ConnectionError` (not undefined `ClientConnectionError`)
- [ ] `test_client_properties.py:15-22` imports `WebSocketClient`
- [ ] All morgan-cli tests pass without import errors

---

## 2. Exception Handling Requirements

### REQ-EXC-1: Single Exception Hierarchy
**Priority**: HIGH
**WHEN** custom exceptions are needed
**SHALL** inherit from single `MorganError` base class in `morgan/exceptions.py`
**AND** no duplicate exception definitions shall exist

**Acceptance Criteria**:
- [ ] `morgan/exceptions.py` is single source of truth
- [ ] `utils/error_handling.py` imports from `exceptions.py` (no redefinitions)
- [ ] `utils/companion_error_handling.py` imports from `exceptions.py`
- [ ] `utils/validators.py` imports `ValidationError` from `exceptions.py`
- [ ] `services/ocr_service.py` exceptions inherit from `MorganError`
- [ ] `services/external_knowledge/mcp_client.py` exceptions inherit from `MorganError`

### REQ-EXC-2: Remove Unused Exceptions
**Priority**: HIGH
**WHEN** exception classes are defined
**SHALL** be used somewhere in the codebase
**AND** unused exceptions must be removed

**Acceptance Criteria**:
- [ ] Remove unused `LLMServiceError` from `exceptions.py:84-103`
- [ ] Remove unused `EmbeddingServiceError` from `exceptions.py:105-124`
- [ ] Remove unused `RerankingServiceError` from `exceptions.py:126-145`
- [ ] Remove unused `VectorDBError` from `exceptions.py:147-166`
- [ ] Remove unused `MemoryServiceError` from `exceptions.py:168-187`
- [ ] Remove unused `SearchServiceError` from `exceptions.py:189-208`

### REQ-EXC-3: Specific Exception Handling
**Priority**: MEDIUM
**WHEN** exceptions are caught
**SHALL** use specific exception types instead of bare `Exception`
**AND** bare `except:` clauses must be eliminated

**Acceptance Criteria**:
- [ ] Reduce 708 `except Exception` to <200 with specific types
- [ ] Eliminate 15+ bare `except:` clauses
- [ ] Critical paths (vector_db, search, memory) use typed exceptions

---

## 3. Code Deduplication Requirements

### REQ-DUP-1: Single setup_model_cache Implementation
**Priority**: HIGH
**WHEN** model cache directories are initialized
**SHALL** use single `setup_model_cache()` from `morgan/utils/model_cache.py`
**AND** no duplicate implementations shall exist

**Acceptance Criteria**:
- [ ] Remove `setup_model_cache()` from `embeddings/service.py:39-87`
- [ ] Remove `setup_model_cache()` from `reranking/service.py:47-84`
- [ ] Remove `setup_model_cache()` from `config/distributed_config.py`
- [ ] All services import from `morgan.utils.model_cache`

### REQ-DUP-2: Single HostRole Enum
**Priority**: HIGH
**WHEN** host roles are defined for distributed deployment
**SHALL** use single `HostRole` enum in shared location
**AND** no conflicting definitions shall exist

**Acceptance Criteria**:
- [ ] Create `shared/models/enums.py` with `HostRole`
- [ ] Remove `HostRole` from `distributed_gpu_manager.py:29-36`
- [ ] Remove `HostRole` from `distributed_manager.py:41-49`
- [ ] All infrastructure uses shared `HostRole`

### REQ-DUP-3: Unified Deduplication Logic
**Priority**: MEDIUM
**WHEN** search results or memories need deduplication
**SHALL** use single deduplication utility
**AND** consistent strategy across all modules

**Acceptance Criteria**:
- [ ] Create `shared/utils/deduplication.py`
- [ ] Consolidate 5 deduplication implementations
- [ ] Memory, search, companion modules use shared utility

### REQ-DUP-4: Unified Text Extraction
**Priority**: MEDIUM
**WHEN** entities, keywords, or topics are extracted from text
**SHALL** use single extraction utility
**AND** consistent patterns across all modules

**Acceptance Criteria**:
- [ ] Create `shared/utils/text_extraction.py`
- [ ] Consolidate entity detection from `memory_processor.py:286-314`
- [ ] Consolidate concept extraction from `memory_processor.py:523-548`
- [ ] Consolidate keyword extraction from `multi_stage_search.py:1852-1902`

### REQ-DUP-5: Unified Valence Mapping
**Priority**: MEDIUM
**WHEN** emotion valence values are needed
**SHALL** use single constant definition
**AND** no duplicate mappings in multiple files

**Acceptance Criteria**:
- [ ] Create `intelligence/constants.py` with `EMOTION_VALENCE`
- [ ] Remove duplicate from `emotions/analyzer.py:44-52`
- [ ] Remove duplicate from `emotions/context.py:135-143`
- [ ] Remove duplicate from `emotions/recovery.py:110-119`

---

## 4. Configuration Requirements

### REQ-CFG-1: Unified Configuration Base
**Priority**: HIGH
**WHEN** settings classes are defined
**SHALL** inherit from shared base configuration
**AND** consistent field naming across all services

**Acceptance Criteria**:
- [ ] Create `shared/config/base.py` with `BaseSettings`
- [ ] `morgan-rag/config/settings.py` inherits from base
- [ ] `morgan-server/config.py` inherits from base
- [ ] `morgan-cli/config.py` inherits from base
- [ ] Unified field names: `llm_endpoint`, `vector_db_url`, etc.

### REQ-CFG-2: Single Environment Variable Convention
**Priority**: MEDIUM
**WHEN** environment variables are used
**SHALL** follow `MORGAN_*` prefix convention
**AND** no conflicting variable names

**Acceptance Criteria**:
- [ ] All env vars use `MORGAN_` prefix consistently
- [ ] Remove `LLM_BASE_URL` in favor of `MORGAN_LLM_ENDPOINT`
- [ ] Remove `QDRANT_URL` in favor of `MORGAN_VECTOR_DB_URL`
- [ ] Migration guide for old variable names

### REQ-CFG-3: Merge Duplicate Example Files
**Priority**: LOW
**WHEN** example configuration files exist
**SHALL** have single canonical example per type
**AND** no duplicate example files

**Acceptance Criteria**:
- [ ] Merge `docker/.env.example` and `docker/env.example`
- [ ] Single set of default values for all examples
- [ ] Consistent model names across all examples

---

## 5. Service Pattern Requirements

### REQ-SVC-1: Unified Singleton Factory
**Priority**: HIGH
**WHEN** singleton services are created
**SHALL** use shared factory pattern
**AND** thread-safe with consistent cleanup

**Acceptance Criteria**:
- [ ] All 8+ services use `SingletonFactory` or `@singleton`
- [ ] Consistent cleanup in `shutdown()` methods
- [ ] Thread-safe double-checked locking

### REQ-SVC-2: Complete Service Interface
**Priority**: MEDIUM
**WHEN** services are defined
**SHALL** implement complete interface including health_check and shutdown
**AND** consistent async/sync method pairs

**Acceptance Criteria**:
- [ ] Add `health_check()` to EmbeddingService
- [ ] Add `health_check()` to RerankingService
- [ ] Add `shutdown()` to all services
- [ ] Both sync and async variants for all methods

### REQ-SVC-3: Health Monitoring Consolidation
**Priority**: MEDIUM
**WHEN** health monitoring is needed
**SHALL** use shared `HealthMonitorMixin`
**AND** consistent monitoring patterns

**Acceptance Criteria**:
- [ ] Create `shared/utils/health_monitor.py`
- [ ] Consolidate from `distributed_llm.py:173-195`
- [ ] Consolidate from `distributed_gpu_manager.py:400-422`
- [ ] Consolidate from `distributed_manager.py:571-661`

---

## 6. Client Requirements

### REQ-CLI-1: Fix Client Bugs
**Priority**: HIGH
**WHEN** morgan-cli client methods are called
**SHALL** work correctly without bugs
**AND** all parameters properly used

**Acceptance Criteria**:
- [ ] Fix `cleanup_memory()` to pass `params` to `delete()` method
- [ ] Add `params` support to `delete()` method signature
- [ ] All client methods properly use their parameters

### REQ-CLI-2: Extract Common Client Code
**Priority**: MEDIUM
**WHEN** HTTP and WebSocket clients are defined
**SHALL** share common functionality via base class
**AND** no duplicate method implementations

**Acceptance Criteria**:
- [ ] Create `BaseClient` with shared methods
- [ ] `HTTPClient` and `WebSocketClient` inherit from base
- [ ] Eliminate 5 duplicate methods (~40 lines)

### REQ-SRV-1: Remove Global State Anti-Pattern
**Priority**: MEDIUM
**WHEN** morgan-server routes need dependencies
**SHALL** use FastAPI `Depends()` injection
**AND** no module-level global state

**Acceptance Criteria**:
- [ ] Replace global `_assistant` in `chat.py` with `Depends()`
- [ ] Replace global `_health_system` in `health.py` with `Depends()`
- [ ] Replace global `_session_manager` in `session.py` with `Depends()`

### REQ-SRV-2: Fix Silent Error Suppression
**Priority**: HIGH
**WHEN** errors occur in profile handling
**SHALL** log errors instead of silently suppressing
**AND** no bare `pass` in except blocks

**Acceptance Criteria**:
- [ ] Fix `profile.py:131` - log error instead of `pass`
- [ ] Fix `profile.py:139` - log error instead of `pass`
- [ ] Replace all silent `except: pass` patterns

---

## 7. Dead Code Requirements

### REQ-DEAD-1: Remove Deprecated Files
**Priority**: LOW
**WHEN** codebase is cleaned
**SHALL** remove deprecated files and stubs
**AND** no orphaned code remains

**Acceptance Criteria**:
- [ ] Delete `cli.py` (deprecated stub)
- [ ] Delete `infrastructure/consul_client.py` (unused)
- [ ] Archive or delete incomplete stub methods in learning module

### REQ-DEAD-2: Remove Unused Imports
**Priority**: LOW
**WHEN** modules are imported
**SHALL** only import what is used
**AND** no unused imports

**Acceptance Criteria**:
- [ ] Remove unused imports across codebase
- [ ] Run linter to identify unused imports
- [ ] Clean up conditional imports that are never used

---

## 8. Intelligence Module Requirements

### REQ-INT-1: Emotion Detection Consolidation
**Priority**: MEDIUM
**WHEN** emotion detection is needed
**SHALL** use single implementation in `detector.py`
**AND** `intelligence_engine.py` delegates to detector

**Acceptance Criteria**:
- [ ] `detector.py` is single source of emotion detection
- [ ] `intelligence_engine.py` imports from detector
- [ ] Remove duplicate patterns from `intelligence_engine.py:119-176`

### REQ-INT-2: Complete Stub Implementations
**Priority**: LOW
**WHEN** learning/communication methods are called
**SHALL** provide meaningful implementations
**AND** no methods returning empty collections as stubs

**Acceptance Criteria**:
- [ ] Implement or document `_adapt_vocabulary_level()` in `adaptation.py`
- [ ] Implement or document stub methods in `patterns.py:552-571`
- [ ] Implement or document `_create_adapted_style()` in `style.py`

---

## Summary

| Category | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| Data Integrity | 2 | 0 | 0 | 0 |
| Exceptions | 0 | 2 | 1 | 0 |
| Deduplication | 0 | 2 | 3 | 0 |
| Configuration | 0 | 1 | 1 | 1 |
| Services | 0 | 1 | 2 | 0 |
| Clients | 0 | 2 | 2 | 0 |
| Dead Code | 0 | 0 | 0 | 2 |
| Intelligence | 0 | 0 | 1 | 1 |
| **Total** | **2** | **8** | **10** | **4** |

**Total Requirements**: 24
**Estimated Impact**: ~2,500 lines of duplicate code removed

---

## Traceability Matrix

| Requirement | Design Component | Files Affected | Priority |
|-------------|------------------|----------------|----------|
| REQ-DATA-1 | Collection Names | memory_processor.py, multi_stage_search.py | CRITICAL |
| REQ-DATA-2 | Test Imports | test_client_properties.py | CRITICAL |
| REQ-EXC-1 | Exception Hierarchy | exceptions.py, error_handling.py, validators.py | HIGH |
| REQ-EXC-2 | Dead Exceptions | exceptions.py | HIGH |
| REQ-DUP-1 | Model Cache | embeddings/service.py, reranking/service.py | HIGH |
| REQ-DUP-2 | HostRole Enum | distributed_gpu_manager.py, distributed_manager.py | HIGH |
| REQ-CFG-1 | Config Base | shared/config/, settings.py, config.py | HIGH |
| REQ-SVC-1 | Singleton Factory | All services | HIGH |
| REQ-CLI-1 | Client Bugs | client.py | HIGH |
| REQ-SRV-2 | Error Suppression | profile.py | HIGH |
