# Implementation Plan: Codebase Reorganization

## Overview

This implementation plan outlines all tasks for reorganizing the Morgan codebase to eliminate duplications, fix inconsistencies, consolidate services, and establish clean architecture patterns.

**Estimated Total Time**: 8-12 days
**Priority Order**: P0 → P1 → P2 → P3

---

## Phase 1: Critical Security & Cleanup (P0)

**Duration**: 0.5 days
**Dependencies**: None

- [ ] 1.1 Remove secrets from repository
  - Remove API keys from `morgan-rag/.env` (Lines 11, 68)
  - Remove API keys from `.env` (Lines 13, 14)
  - Remove HuggingFace tokens from committed files
  - Update `.gitignore` to exclude all `.env` files
  - _Requirements: REQ-SEC-1_

- [ ] 1.2 Create secure example files
  - Create `morgan-rag/.env.example` with placeholder values
  - Create `morgan-server/.env.example` with placeholder values
  - Create `docker/.env.example` with placeholder values
  - Add "CHANGE_ME" comments for required secrets
  - _Requirements: REQ-SEC-1, REQ-SEC-2_

- [ ] 1.3 Fix default credentials in docker-compose
  - Replace Grafana default password "admin"/"morgan" in `docker/docker-compose.distributed.yml` (Line 242)
  - Replace Grafana default password in `morgan-rag/docker-compose.yml` (Line 264)
  - Update session secret default in `settings.py` (Line 260)
  - Add startup warnings for default credentials
  - _Requirements: REQ-SEC-2_

- [ ] 1.4 Archive deprecated code
  - Create `/archive/` directory
  - Move `morgan-rag/` to `/archive/morgan-rag-legacy/`
  - Move `cli.py.old` to `/archive/`
  - Move `morgan-rag/morgan_v2/` to `/archive/morgan_v2-abandoned/`
  - Update root `.gitignore` for archive
  - _Requirements: REQ-ARCH-1_

- [ ] 1.5 Fix broken imports
  - Fix or remove `core/emotional_handler.py` (broken `shared.utils.emotional` import)
  - Remove or implement missing `shared/utils/` modules
  - Verify all imports resolve correctly
  - _Requirements: REQ-ARCH-3_

---

## Phase 2: Service Consolidation - LLM (P1)

**Duration**: 1-2 days
**Dependencies**: Phase 1

- [ ] 2.1 Create unified LLM service structure
  - Create `morgan/services/llm/` directory
  - Create `morgan/services/llm/__init__.py` with exports
  - Create `morgan/services/llm/models.py` for dataclasses
  - _Requirements: REQ-SVC-1_

- [ ] 2.2 Consolidate LLMResponse dataclass
  - Merge definitions from `llm_service.py:33-42` and `distributed_llm_service.py:25-34`
  - Standardize field names: `latency_ms`, `endpoint_used`, `model`
  - Add to `morgan/services/llm/models.py`
  - _Requirements: REQ-SVC-1, REQ-PAT-4_

- [ ] 2.3 Merge LLM service implementations
  - Create `morgan/services/llm/client.py`
  - Merge `llm_service.py` (450+ lines) and `distributed_llm_service.py` (445 lines)
  - Keep distributed architecture from `distributed_llm_service.py`
  - Keep single-mode fallback from `llm_service.py`
  - Use infrastructure layer `DistributedLLMClient`
  - _Requirements: REQ-SVC-1, REQ-INT-1_

- [ ] 2.4 Update LLM service singleton
  - Create single `get_llm_service()` function
  - Use shared singleton factory pattern
  - Support both distributed and single modes
  - Add proper cleanup methods
  - _Requirements: REQ-SVC-1, REQ-DUP-2_

- [ ] 2.5 Update all LLM service imports
  - Find all files importing from old locations
  - Update imports to new `morgan/services/llm/`
  - Remove old `llm_service.py` and `distributed_llm_service.py`
  - Run tests to verify
  - _Requirements: REQ-SVC-1_

---

## Phase 3: Service Consolidation - Embeddings (P1)

**Duration**: 1-2 days
**Dependencies**: Phase 2

- [ ] 3.1 Create unified embedding service structure
  - Create `morgan/services/embeddings/` directory
  - Create `morgan/services/embeddings/__init__.py` with exports
  - Create `morgan/services/embeddings/models.py` for dataclasses
  - _Requirements: REQ-SVC-2_

- [ ] 3.2 Consolidate embedding implementations
  - Create `morgan/services/embeddings/service.py`
  - Merge `embeddings/service.py` (218 lines)
  - Merge `services/distributed_embedding_service.py` (274 lines)
  - Use `LocalEmbeddingService` from infrastructure as backend
  - _Requirements: REQ-SVC-2, REQ-INT-1_

- [ ] 3.3 Fix thread-safety in embedding service
  - Add `threading.Lock()` to singleton pattern
  - Match pattern from `distributed_embedding_service.py:245-246`
  - Remove unsafe singleton from `embeddings/service.py:209-217`
  - _Requirements: REQ-SVC-2, REQ-DUP-2_

- [ ] 3.4 Add async interface consistency
  - Ensure both `embed()` and `embed_async()` methods exist
  - Add proper sync wrappers using `asyncio.run()`
  - Document threading model
  - _Requirements: REQ-SVC-2, REQ-PAT-1_

- [ ] 3.5 Update all embedding service imports
  - Find all files importing from old locations
  - Update imports to new `morgan/services/embeddings/`
  - Remove old files
  - Run tests to verify
  - _Requirements: REQ-SVC-2_

---

## Phase 4: Service Consolidation - Reranking (P1)

**Duration**: 1 day
**Dependencies**: Phase 3

- [ ] 4.1 Create unified reranking service structure
  - Create `morgan/services/reranking/` directory
  - Create `morgan/services/reranking/__init__.py` with exports
  - Create `morgan/services/reranking/models.py` for dataclasses
  - _Requirements: REQ-SVC-3_

- [ ] 4.2 Consolidate reranking implementations
  - Create `morgan/services/reranking/service.py`
  - Merge `jina/reranking/service.py` with `local_reranking.py`
  - Use `LocalRerankingService` from infrastructure as backend
  - Keep fallback strategy hierarchy (remote → local → embedding → BM25)
  - _Requirements: REQ-SVC-3, REQ-INT-1_

- [ ] 4.3 Integrate reranking into search pipeline
  - Update `morgan/search/multi_stage_search.py`
  - Make reranking mandatory (not optional)
  - Remove separate `SearchReranker` class from `reranker.py`
  - _Requirements: REQ-SVC-3_

- [ ] 4.4 Update all reranking imports
  - Find all files importing from old locations
  - Update imports to new `morgan/services/reranking/`
  - Remove old files
  - Run tests to verify
  - _Requirements: REQ-SVC-3_

---

## Phase 5: Core Component Consolidation (P1)

**Duration**: 1 day
**Dependencies**: Phase 1

- [ ] 5.1 Consolidate milestone management
  - Keep `milestone_tracker.py` as single source
  - Remove duplicate `check_for_milestones()` from `emotional_processor.py:72-104`
  - Remove duplicate `generate_milestone_celebration()` from `emotional_processor.py:168-182`
  - Update `EmotionalProcessor` to delegate to `MilestoneTracker`
  - _Requirements: REQ-DUP-1_

- [ ] 5.2 Unify milestone code paths
  - Audit `assistant.py` for multiple milestone paths (Lines 239, 296, 327)
  - Create single milestone handling path through orchestrator
  - Remove direct `MilestoneTracker` usage from `MorganAssistant`
  - All milestone operations through `ConversationOrchestrator`
  - _Requirements: REQ-DUP-1_

- [ ] 5.3 Extract shared emotional context utilities
  - Create `morgan/emotional/context_utils.py`
  - Move `_calculate_emotional_weight()` from `memory_processor.py:485-515`
  - Move `_extract_emotional_context_from_turn()` from `multi_stage_search.py:1937-1983`
  - Move `_extract_emotional_context_from_content()` from `companion_memory_search.py:446-496`
  - Update all modules to use shared utility
  - _Requirements: REQ-DUP-4_

---

## Phase 6: Shared Utilities Extraction (P2)

**Duration**: 1 day
**Dependencies**: Phases 2-5

- [ ] 6.1 Create singleton factory utility
  - Create `morgan/utils/singleton.py`
  - Implement `SingletonFactory` class with thread-safety
  - Support optional `force_new` parameter for testing
  - Add proper cleanup/reset methods
  - _Requirements: REQ-DUP-2_

- [ ] 6.2 Refactor services to use singleton factory
  - Update LLM service singleton
  - Update embedding service singleton
  - Update reranking service singleton
  - Update all 50+ singleton patterns gradually
  - _Requirements: REQ-DUP-2_

- [ ] 6.3 Create unified model cache setup
  - Create `morgan/utils/model_cache.py`
  - Extract from `local_embeddings.py:43` (`setup_model_cache()`)
  - Extract from `local_reranking.py:42` (`setup_reranker_cache()`)
  - Extract from `distributed_config.py:209`
  - Single function with configurable paths
  - _Requirements: REQ-DUP-3_

- [ ] 6.4 Update infrastructure to use shared cache setup
  - Update `local_embeddings.py` to import shared function
  - Update `local_reranking.py` to import shared function
  - Update `distributed_config.py` to import shared function
  - Remove duplicate implementations
  - _Requirements: REQ-DUP-3_

- [ ] 6.5 Create unified deduplication utility
  - Create `morgan/utils/deduplication.py`
  - Implement `ResultDeduplicator` class
  - Support string-based and embedding-based strategies
  - Configurable similarity thresholds
  - _Requirements: REQ-DUP-4_

- [ ] 6.6 Update modules to use shared deduplicator
  - Update `memory_processor.py:608-623`
  - Update `multi_stage_search.py:1729-1775` and `1660-1727`
  - Update `companion_memory_search.py:1057-1075`
  - Remove duplicate implementations
  - _Requirements: REQ-DUP-4_

---

## Phase 7: Configuration Standardization (P2)

**Duration**: 1-2 days
**Dependencies**: Phase 1

- [ ] 7.1 Create defaults module
  - Create `morgan/config/defaults.py`
  - Define all port defaults (8080, 8000, 6333, 6379, 11434)
  - Define all timeout defaults
  - Define all model name defaults
  - _Requirements: REQ-CFG-2_

- [ ] 7.2 Update settings to use defaults module
  - Update `morgan/config/settings.py` to import from defaults
  - Remove inline default values
  - Reference `Defaults.MORGAN_PORT` instead of hardcoded `8080`
  - _Requirements: REQ-CFG-2_

- [ ] 7.3 Standardize environment variable naming
  - Create migration map: `LLM_BASE_URL` → `MORGAN_LLM_ENDPOINT`
  - Update all env var references to use `MORGAN_*` prefix
  - Add backward compatibility layer for old names
  - Log deprecation warnings for old names
  - _Requirements: REQ-CFG-1_

- [ ] 7.4 Remove hardcoded IP addresses
  - Replace `192.168.1.10-23` in `distributed.yaml` with `${HOST_*}` vars
  - Replace IPs in `docker-compose.distributed.yml`
  - Replace IPs in `prometheus-distributed.yml`
  - Document required host environment variables
  - _Requirements: REQ-CFG-3_

- [ ] 7.5 Consolidate docker-compose files
  - Keep `docker/docker-compose.yml` (development)
  - Keep `docker/docker-compose.prod.yml` (production)
  - Keep `docker/docker-compose.distributed.yml` (6-host)
  - Archive `morgan-rag/docker-compose.yml` (duplicate)
  - _Requirements: REQ-CFG-4_

- [ ] 7.6 Consolidate distributed config files
  - Keep `docker/config/distributed.yaml` as single source
  - Remove `morgan-rag/config/distributed.yaml` (duplicate)
  - Update references to use canonical path
  - _Requirements: REQ-CFG-4_

---

## Phase 8: Integration Layer Updates (P2)

**Duration**: 1-2 days
**Dependencies**: Phases 2-4

- [ ] 8.1 Wire infrastructure to services
  - Update `morgan/services/llm/client.py` to use `DistributedLLMClient`
  - Update `morgan/services/embeddings/service.py` to use `LocalEmbeddingService`
  - Update `morgan/services/reranking/service.py` to use `LocalRerankingService`
  - Remove duplicate HTTP client code
  - _Requirements: REQ-INT-1_

- [ ] 8.2 Integrate learning with emotional engine
  - Update `morgan/learning/engine.py` to import emotion detector
  - Add `emotion_detector = get_emotion_detector()` in `__init__`
  - Use emotional state in `process_feedback()`
  - Weight feedback based on emotional context
  - _Requirements: REQ-INT-2_

- [ ] 8.3 Update memory to use embedding infrastructure
  - Update `morgan/memory/memory_processor.py`
  - Replace individual `encode()` calls with batch encoding
  - Use `LocalEmbeddingService` from infrastructure
  - Leverage embedding caching
  - _Requirements: REQ-INT-3_

- [ ] 8.4 Fix orchestrator dependency flow
  - Audit `ConversationOrchestrator` dependencies
  - Remove lazy loading with silent failures
  - Add proper initialization validation
  - Log clear errors on missing dependencies
  - _Requirements: REQ-INT-1_

---

## Phase 9: Dead Code Cleanup (P3)

**Duration**: 0.5 days
**Dependencies**: Phases 5-8

- [ ] 9.1 Remove stub methods from learning module
  - Remove `_identify_learning_areas()` from `learning/patterns.py:553-558`
  - Remove `_identify_avoided_topics()` from `learning/patterns.py:560-565`
  - Remove `_analyze_topic_transitions()` from `learning/patterns.py:567-572`
  - Remove `_determine_interaction_style()` from `learning/patterns.py:623-626`
  - Remove `_analyze_help_seeking_behavior()` from `learning/patterns.py:640-645`
  - Remove `_estimate_error_tolerance()` from `learning/patterns.py:647-650`
  - Remove `_determine_learning_style()` from `learning/patterns.py:652-655`
  - Or implement if needed
  - _Requirements: REQ-ARCH-3_

- [ ] 9.2 Remove dead code expressions
  - Remove `len(messages)` on `learning/patterns.py:428` (unused expression)
  - Remove `_enhance_memory_result()` from `companion_memory_search.py:367-444` (dead code)
  - Remove `_apply_companion_aware_ranking()` from `multi_stage_search.py:2265-2298` (duplicate)
  - _Requirements: REQ-ARCH-3_

- [ ] 9.3 Clean up orphaned directories
  - Evaluate `/core/` directory (only has broken `emotional_handler.py`)
  - Evaluate `/services/embedding/` (standalone, not integrated)
  - Evaluate `/services/reranking/` (standalone, not integrated)
  - Archive or remove orphaned code
  - _Requirements: REQ-ARCH-3_

- [ ] 9.4 Remove unused service methods
  - Audit `llm_service.py stream_generate():356-397` (never called)
  - Audit `distributed_embedding_service.py get_dimension():223-225` (unused)
  - Remove or document as internal
  - _Requirements: REQ-ARCH-3_

---

## Phase 10: Pattern Standardization (P3)

**Duration**: 1-2 days
**Dependencies**: Phases 6-9

- [ ] 10.1 Create base exception hierarchy
  - Create `morgan/exceptions.py`
  - Define `MorganError(Exception)` as base
  - Define `LLMServiceError(MorganError)`
  - Define `EmbeddingServiceError(MorganError)`
  - Define `RerankingServiceError(MorganError)`
  - Define `ConfigurationError(MorganError)`
  - _Requirements: REQ-PAT-2_

- [ ] 10.2 Update services to use exception hierarchy
  - Update LLM service to raise `LLMServiceError`
  - Update embedding service to raise `EmbeddingServiceError`
  - Update reranking service to raise `RerankingServiceError`
  - Include context in all exceptions (service, operation, details)
  - _Requirements: REQ-PAT-2_

- [ ] 10.3 Standardize async/sync patterns
  - Audit all services for consistency
  - Add missing sync wrappers
  - Add missing async variants
  - Document threading model per service
  - _Requirements: REQ-PAT-1_

- [ ] 10.4 Standardize logging usage
  - Replace `logging.getLogger(__name__)` with `get_logger(__name__)`
  - Audit log levels for consistency
  - Add structured context to log messages
  - _Requirements: REQ-PAT-3_

- [ ] 10.5 Complete type hints
  - Add missing type hints to all public methods
  - Standardize field names in dataclasses
  - Use consistent naming: `confidence_score`, `timestamp`, `latency_ms`
  - _Requirements: REQ-PAT-4_

---

## Phase 11: Documentation Updates (P3)

**Duration**: 0.5 days
**Dependencies**: All phases

- [ ] 11.1 Update architecture documentation
  - Update `CLAUDE.md` with new structure
  - Create architecture diagram reflecting changes
  - Document service dependencies
  - _Requirements: REQ-DOC-1_

- [ ] 11.2 Create migration guide
  - Document environment variable changes
  - Document import path changes
  - Document configuration file changes
  - Provide migration script if possible
  - _Requirements: REQ-DOC-2_

- [ ] 11.3 Update README
  - Reflect new project structure
  - Update installation instructions
  - Update configuration section
  - _Requirements: REQ-DOC-1_

---

## Phase 12: Testing & Validation (P3)

**Duration**: 1 day
**Dependencies**: All phases

- [ ] 12.1 Run existing tests
  - Ensure all unit tests pass
  - Fix any broken tests from refactoring
  - Update test imports as needed
  - _Requirements: All_

- [ ] 12.2 Add integration tests
  - Test LLM service with infrastructure
  - Test embedding service with infrastructure
  - Test reranking service with infrastructure
  - Test end-to-end conversation flow
  - _Requirements: REQ-INT-1, REQ-INT-2, REQ-INT-3_

- [ ] 12.3 Validate configuration
  - Test all environment variable patterns
  - Test default value loading
  - Test configuration validation
  - _Requirements: REQ-CFG-1, REQ-CFG-2_

- [ ] 12.4 Security validation
  - Scan for remaining secrets
  - Verify no credentials in git history (consider git-filter-repo)
  - Test default credential warnings
  - _Requirements: REQ-SEC-1, REQ-SEC-2_

---

## Summary

| Phase | Tasks | Duration | Priority |
|-------|-------|----------|----------|
| Phase 1: Security & Cleanup | 5 | 0.5 days | P0 |
| Phase 2: LLM Consolidation | 5 | 1-2 days | P1 |
| Phase 3: Embedding Consolidation | 5 | 1-2 days | P1 |
| Phase 4: Reranking Consolidation | 4 | 1 day | P1 |
| Phase 5: Core Consolidation | 3 | 1 day | P1 |
| Phase 6: Shared Utilities | 6 | 1 day | P2 |
| Phase 7: Configuration | 6 | 1-2 days | P2 |
| Phase 8: Integration | 4 | 1-2 days | P2 |
| Phase 9: Dead Code | 4 | 0.5 days | P3 |
| Phase 10: Patterns | 5 | 1-2 days | P3 |
| Phase 11: Documentation | 3 | 0.5 days | P3 |
| Phase 12: Testing | 4 | 1 day | P3 |
| **Total** | **54** | **8-12 days** | |

---

## Task Dependencies

```
Phase 1 (P0)
    ↓
┌───┴───┬───────┬───────┐
↓       ↓       ↓       ↓
Phase 2 Phase 3 Phase 5 Phase 7
(LLM)   (Embed) (Core)  (Config)
    ↓       ↓       ↓
    └───┬───┘       ↓
        ↓           ↓
    Phase 4     Phase 6
    (Rerank)    (Utils)
        ↓           ↓
        └─────┬─────┘
              ↓
          Phase 8
          (Integration)
              ↓
          Phase 9
          (Dead Code)
              ↓
          Phase 10
          (Patterns)
              ↓
          Phase 11
          (Docs)
              ↓
          Phase 12
          (Testing)
```

---

## Critical Path

The minimum path to a working reorganization:

1. **Phase 1** (0.5 days) - Security fixes are mandatory
2. **Phase 2** (1 day) - LLM is core functionality
3. **Phase 3** (1 day) - Embeddings needed for search
4. **Phase 5** (1 day) - Remove confusing duplications
5. **Phase 8** (1 day) - Wire everything together
6. **Phase 12** (1 day) - Validate it works

**Minimum viable reorganization: 5.5 days**

---

*Generated by Claude Code on 2025-12-26*
