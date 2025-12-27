# Codebase Reorganization Requirements

## Overview

This document defines the requirements for reorganizing and cleaning up the Morgan codebase to eliminate duplications, fix inconsistencies, and establish a clean architecture foundation.

**Date**: 2025-12-26
**Status**: Planning Complete - Ready for Implementation
**Priority**: High - Technical Debt Reduction

---

## 1. Security Requirements

### REQ-SEC-1: Remove Committed Secrets
**Priority**: CRITICAL
**WHEN** the codebase is scanned for sensitive data
**SHALL** contain zero API keys, tokens, or credentials in version control
**AND** all secrets must be moved to environment variables or secret managers

**Acceptance Criteria**:
- [ ] No API keys in `.env` files committed to git
- [ ] `.env.example` templates contain placeholder values only
- [ ] `.gitignore` properly excludes all secret files
- [ ] HuggingFace tokens removed from committed files

### REQ-SEC-2: Secure Default Credentials
**Priority**: HIGH
**WHEN** default credentials are defined in configuration files
**SHALL** use obviously placeholder values (e.g., "CHANGE_ME_IN_PRODUCTION")
**AND** log warnings when default credentials are detected in production mode

**Acceptance Criteria**:
- [ ] Grafana password is not "admin" or "morgan" by default
- [ ] Session secrets are not hardcoded
- [ ] CORS origins are not "*" by default

---

## 2. Architectural Requirements

### REQ-ARCH-1: Single Active Architecture
**Priority**: HIGH
**WHEN** the project structure is examined
**SHALL** have only one active codebase architecture
**AND** deprecated code must be clearly archived

**Acceptance Criteria**:
- [ ] `morgan-rag/` moved to `/archive/` directory
- [ ] `morgan_v2/` (abandoned) removed or archived
- [ ] `cli.py.old` archived
- [ ] Only `morgan-server/` and `morgan-cli/` remain as active code

### REQ-ARCH-2: Unified Data Directory
**Priority**: MEDIUM
**WHEN** runtime data is stored
**SHALL** use a single canonical data directory path
**AND** all services must reference the same data location

**Acceptance Criteria**:
- [ ] Single `/data/` directory for all runtime data
- [ ] Duplicate `/morgan-rag/data/` removed
- [ ] Configuration points to unified data path

### REQ-ARCH-3: No Orphaned Code
**Priority**: MEDIUM
**WHEN** the codebase is analyzed for dead code
**SHALL** contain no orphaned modules with broken imports
**AND** all code paths must be reachable

**Acceptance Criteria**:
- [ ] `core/emotional_handler.py` fixed or removed (broken imports)
- [ ] No stub methods returning hardcoded values
- [ ] All imports resolve correctly

---

## 3. Service Consolidation Requirements

### REQ-SVC-1: Single LLM Service Implementation
**Priority**: HIGH
**WHEN** LLM functionality is needed
**SHALL** use a single unified LLM service class
**AND** no duplicate implementations shall exist

**Acceptance Criteria**:
- [ ] `llm_service.py` and `distributed_llm_service.py` merged into one
- [ ] Single `LLMResponse` dataclass definition
- [ ] Unified singleton pattern for LLM service

### REQ-SVC-2: Single Embedding Service Implementation
**Priority**: HIGH
**WHEN** embedding functionality is needed
**SHALL** use a single unified embedding service class
**AND** no duplicate implementations shall exist

**Acceptance Criteria**:
- [ ] `embeddings/service.py`, `distributed_embedding_service.py`, `local_embeddings.py` consolidated
- [ ] Single embedding interface with consistent methods
- [ ] Unified caching strategy

### REQ-SVC-3: Single Reranking Service Implementation
**Priority**: MEDIUM
**WHEN** reranking functionality is needed
**SHALL** use a single unified reranking service class
**AND** reranking must be integrated into the search pipeline

**Acceptance Criteria**:
- [ ] `jina/reranking/service.py` and `local_reranking.py` consolidated
- [ ] Reranking is mandatory in search pipeline (not optional)
- [ ] Single fallback strategy hierarchy

---

## 4. Code Deduplication Requirements

### REQ-DUP-1: Single Milestone Management System
**Priority**: HIGH
**WHEN** milestone detection or celebration is needed
**SHALL** use a single implementation
**AND** no duplicate detection logic or celebration messages shall exist

**Acceptance Criteria**:
- [ ] `EmotionalProcessor.check_for_milestones()` removed (use MilestoneTracker)
- [ ] Single celebration messages dictionary
- [ ] Single code path for milestone handling

### REQ-DUP-2: Shared Singleton Factory
**Priority**: MEDIUM
**WHEN** singleton services are created
**SHALL** use a shared factory/decorator pattern
**AND** no repeated singleton boilerplate code

**Acceptance Criteria**:
- [ ] `SingletonFactory` utility class created
- [ ] All 50+ singletons refactored to use factory
- [ ] Thread-safety guaranteed by factory

### REQ-DUP-3: Unified Model Cache Setup
**Priority**: MEDIUM
**WHEN** ML model caches are initialized
**SHALL** use a single shared utility function
**AND** no duplicate cache setup code

**Acceptance Criteria**:
- [ ] Single `setup_model_cache()` function in utils
- [ ] `local_embeddings.py`, `local_reranking.py`, `distributed_config.py` use shared function

### REQ-DUP-4: Single Deduplication Implementation
**Priority**: MEDIUM
**WHEN** search results or memories need deduplication
**SHALL** use a single deduplication utility
**AND** consistent deduplication strategy across all modules

**Acceptance Criteria**:
- [ ] Single `ResultDeduplicator` class
- [ ] Memory, search, and companion modules use shared deduplicator
- [ ] Configurable strategy (string hash vs embedding similarity)

---

## 5. Configuration Requirements

### REQ-CFG-1: Unified Environment Variable Naming
**Priority**: MEDIUM
**WHEN** environment variables are used
**SHALL** follow consistent `MORGAN_*` prefix naming convention
**AND** no duplicate variable names for same configuration

**Acceptance Criteria**:
- [ ] All env vars use `MORGAN_` prefix
- [ ] No duplicate `LLM_*` and `MORGAN_LLM_*` patterns
- [ ] Migration guide for old variable names

### REQ-CFG-2: Single Source of Truth for Defaults
**Priority**: MEDIUM
**WHEN** default configuration values are needed
**SHALL** be defined in a single `defaults.py` file
**AND** no scattered hardcoded defaults

**Acceptance Criteria**:
- [ ] `morgan/config/defaults.py` contains all defaults
- [ ] Docker compose files reference defaults module
- [ ] No hardcoded values in service code

### REQ-CFG-3: No Hardcoded IP Addresses
**Priority**: MEDIUM
**WHEN** distributed deployment is configured
**SHALL** use environment variables for all host addresses
**AND** no hardcoded `192.168.x.x` addresses in code or config

**Acceptance Criteria**:
- [ ] All IPs replaced with environment variables
- [ ] Template config files use `${HOST_*}` placeholders
- [ ] Documentation updated with required variables

### REQ-CFG-4: Single Configuration Location per Type
**Priority**: LOW
**WHEN** configuration files are organized
**SHALL** have one canonical location per configuration type
**AND** no duplicate config files in different directories

**Acceptance Criteria**:
- [ ] Single `distributed.yaml` location
- [ ] Single `docker-compose.yml` per deployment type
- [ ] Clear configuration hierarchy documented

---

## 6. Integration Requirements

### REQ-INT-1: Infrastructure Layer Integration
**Priority**: HIGH
**WHEN** services require LLM, embedding, or reranking
**SHALL** use the infrastructure layer implementations
**AND** no direct API calls bypassing infrastructure

**Acceptance Criteria**:
- [ ] All services use `DistributedLLMClient` from infrastructure
- [ ] All services use `LocalEmbeddingService` from infrastructure
- [ ] All services use `LocalRerankingService` from infrastructure

### REQ-INT-2: Learning-Emotional Integration
**Priority**: MEDIUM
**WHEN** the learning module processes feedback
**SHALL** incorporate emotional state from the intelligence module
**AND** emotional context influences learning outcomes

**Acceptance Criteria**:
- [ ] `LearningEngine` imports and uses emotion detector
- [ ] Emotional state affects feedback weighting
- [ ] Emotional patterns inform preference learning

### REQ-INT-3: Memory-Infrastructure Integration
**Priority**: MEDIUM
**WHEN** the memory module stores or retrieves memories
**SHALL** use the new embedding infrastructure
**AND** batch encoding instead of individual calls

**Acceptance Criteria**:
- [ ] Memory processor uses `LocalEmbeddingService`
- [ ] Batch encoding for memory storage
- [ ] Embedding caching leveraged

---

## 7. Pattern Consistency Requirements

### REQ-PAT-1: Consistent Async/Sync Interfaces
**Priority**: MEDIUM
**WHEN** services provide async methods
**SHALL** also provide sync wrappers
**AND** consistent naming (`method_async()` or `amethod()`)

**Acceptance Criteria**:
- [ ] All services have both async and sync variants
- [ ] Consistent naming convention across codebase
- [ ] No mixing of patterns in same module

### REQ-PAT-2: Consistent Error Handling
**Priority**: MEDIUM
**WHEN** exceptions are raised
**SHALL** use custom exception hierarchy rooted in `MorganError`
**AND** consistent error context in all services

**Acceptance Criteria**:
- [ ] `MorganError` base exception class
- [ ] Service-specific exceptions inherit from `MorganError`
- [ ] Error context includes service name, operation, and details

### REQ-PAT-3: Consistent Logging
**Priority**: LOW
**WHEN** logging is performed
**SHALL** use `get_logger(__name__)` utility consistently
**AND** appropriate log levels for different message types

**Acceptance Criteria**:
- [ ] All modules use `get_logger()` not direct `logging.getLogger()`
- [ ] DEBUG for internal operations
- [ ] INFO for state changes
- [ ] WARNING for recoverable issues
- [ ] ERROR for failures with context

### REQ-PAT-4: Consistent Type Hints
**Priority**: LOW
**WHEN** function signatures are defined
**SHALL** include complete type hints
**AND** standardized field names across dataclasses

**Acceptance Criteria**:
- [ ] All public methods have type hints
- [ ] Standard names: `confidence_score`, `timestamp`, `latency_ms`
- [ ] Optional return types documented

---

## 8. Documentation Requirements

### REQ-DOC-1: Architecture Documentation
**Priority**: LOW
**WHEN** developers need to understand the system
**SHALL** have up-to-date architecture documentation
**AND** diagrams reflecting current structure

**Acceptance Criteria**:
- [ ] Updated architecture diagram
- [ ] Service dependency graph
- [ ] Data flow documentation

### REQ-DOC-2: Migration Guide
**Priority**: LOW
**WHEN** old code references are found
**SHALL** have migration documentation
**AND** clear mapping from old to new implementations

**Acceptance Criteria**:
- [ ] Environment variable migration guide
- [ ] Service import migration guide
- [ ] Configuration file migration guide

---

## Summary

| Category | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| Security | 1 | 1 | 0 | 0 |
| Architecture | 0 | 1 | 2 | 0 |
| Services | 0 | 2 | 1 | 0 |
| Deduplication | 0 | 1 | 3 | 0 |
| Configuration | 0 | 0 | 3 | 1 |
| Integration | 0 | 1 | 2 | 0 |
| Patterns | 0 | 0 | 2 | 2 |
| Documentation | 0 | 0 | 0 | 2 |
| **Total** | **1** | **6** | **13** | **5** |

**Total Requirements**: 25
