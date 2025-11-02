# Requirements Document: Morgan Refactoring Program

## Introduction

This requirements specification aligns Morgan's refactoring roadmap with concrete user stories and measurable acceptance criteria. The goal is to bridge the current implementation and the advanced architecture by modernising ingestion, search, learning, reasoning, and background optimisation while preserving Morgan's human-first experience.

## Glossary

- **Morgan_System** – The complete Morgan assistant system including all components and services.
- **Ingestion_Pipeline** – The document processing pipeline that handles embedding generation and storage.
- **Search_Engine** – The search component that handles query processing and result retrieval.
- **Search_Pipeline** – The complete search workflow including multi-stage search and reranking.
- **Background_Processing_Service** – Service that handles continuous background tasks like reindexing and cache priming.
- **Learning_Engine** – Personalisation modules that adapt behaviour based on conversation analytics.
- **Conversation_Manager** – Component that manages conversation flow and personalisation integration.
- **Reasoning_Engine** – Structured problem-solving modules that expose chain-of-thought style insights.
- **Metrics_System** – System component responsible for collecting and reporting performance metrics.
- **Jina_Model_Selector** – Component that selects appropriate Jina models for embedding tasks.
- **Hierarchical Ingestion** – Ingesting documents with coarse/medium/fine embeddings stored as named vectors in Qdrant.
- **Jina Stack** – The collection of Jina services (embeddings, reranking, scraping) accessed through the new integration layer.
- **Legacy Pipeline** – Existing single-vector ingestion and search paths that remain available as fallbacks.
- **Observability Dashboard** – Monitoring components tracking ingestion/search/learning metrics.
- **CLI Diagnostics** – Command-line tools for operating Morgan (e.g., `morgan doctor`, `morgan background status`).

## Requirements

### Requirement R1 – Hierarchical Document Ingestion

**User Story:** As a Morgan maintainer, I want knowledge ingestion to produce hierarchical embeddings so that multi-stage search reaches the promised candidate reduction.

#### Acceptance Criteria

1. WHEN `KnowledgeBase.ingest_documents` runs, THE **Morgan_System** SHALL compute coarse, medium, and fine embeddings for every chunk via `HierarchicalEmbeddingService`.
2. WHEN storing chunks, THE **Morgan_System** SHALL upsert named vectors (`coarse`, `medium`, `fine`) into Qdrant collections created with `VectorDBClient.create_hierarchical_collection`.
3. WHEN chunks come from unchanged Git hashes, THE **Ingestion_Pipeline** SHALL reuse cached embeddings and skip recomputation.
4. WHEN ingestion completes, THE **Morgan_System** SHALL report counts per embedding scale and cache hit rate.
5. IF hierarchical ingestion fails, THEN THE **Ingestion_Pipeline** SHALL fall back to legacy single-vector ingestion and log the downgrade.

### Requirement R2 – Jina Embedding Integration

**User Story:** As a Morgan operator, I want all new embeddings to use Jina models so that semantic accuracy meets 2025 expectations.

#### Acceptance Criteria

1. WHEN embedding text, THE **Morgan_System** SHALL route through `morgan/jina/models/selector.py` to determine the correct Jina model.
2. WHEN the chosen Jina model is unavailable, THE **Morgan_System** SHALL attempt the configured fallback model before raising an error.
3. WHEN embedding batches exceed configured size, THE **Ingestion_Pipeline** SHALL chunk batches without exceeding memory or latency limits.
4. WHEN using non-English content, THE **Jina_Model_Selector** SHALL choose appropriate multilingual models automatically.
5. WHEN embeddings are generated, THE **Morgan_System** SHALL log model name, batch size, and latency in monitoring metrics.
6. WHEN the configured OpenAI-compatible embedding endpoint (from Morgan settings) is reachable, THE **Embedding_Service** SHALL send requests to it; only when it is unavailable SHALL it load the approved local HuggingFace/`sentence-transformers` model, logging the fallback decision.

### Requirement R3 – Multi-Stage Search Adoption

**User Story:** As a Morgan user, I want search to leverage hierarchical embeddings and reranking, so that answers feel precise and empathetic.

#### Acceptance Criteria

1. WHEN `SmartSearch.find_relevant_info` runs, THE **Search_Engine** SHALL call `MultiStageSearchEngine.search` with hierarchical embeddings enabled by default.
2. WHEN Jina reranking succeeds, THE **Search_Pipeline** SHALL apply reranked scores to the final result list.
3. WHEN reranking is disabled or fails, THE **Morgan_System** SHALL provide results from hierarchical search with a warning flag.
4. WHEN searches complete, THE **Metrics_System** SHALL record total candidates, filtered candidates, and reranking improvement.
5. WHEN contextual or emotional signals exist, THE **Search_Pipeline** SHALL pass them through to multi-stage search and reranking.

### Requirement R4 – Background Optimisation Enablement

**User Story:** As a Morgan operator, I want background tasks running continuously so that ingestion, caching, and reranking remain fresh.

#### Acceptance Criteria

1. WHEN Morgan starts, THE **Morgan_System** SHALL invoke `BackgroundProcessingService.start()` and confirm running status.
2. WHEN the service runs, THE **Background_Processing_Service** SHALL schedule default reindexing, reranking, and cache priming tasks per configuration.
3. WHEN the CLI command `morgan background status` is executed, THE **Morgan_System** SHALL display task queues, last run times, and errors.
4. WHEN resource usage exceeds configured thresholds, THE **Background_Processing_Service** SHALL throttle or pause while logging the decision.
5. WHEN the service stops (`stop()`), THE **Background_Processing_Service** SHALL gracefully finish executing tasks or cancel them with clear reporting.

### Requirement R5 – Learning & Personalisation Engine

**User Story:** As a frequent Morgan user, I want the assistant to adapt to my preferences over time so that conversations feel personalised.

#### Acceptance Criteria

1. WHEN a conversation turn completes, THE **Learning_Engine** SHALL process the turn, update preference weights, and store them in profile storage.
2. WHEN a user returns, THE **Conversation_Manager** SHALL consult personalisation data to adjust tone, suggestions, and contextual retrieval.
3. WHEN preferences age out, THE **Learning_Engine** SHALL decay weights to prevent over-fitting to old habits.
4. WHEN users request a reset, THE **Morgan_System** SHALL clear learned preferences and confirm the reset.
5. WHEN personalisation decisions affect responses, THE **Morgan_System** SHALL record the factors used (topics, tone, habits) for observability.

### Requirement R6 – Reasoning Engine Deployment

**User Story:** As a power user, I want Morgan to surface structured reasoning so that I can trust complex answers.

#### Acceptance Criteria

1. WHEN the assistant detects multi-step problems, THE **Morgan_System** SHALL invoke the Reasoning Engine to build a reasoning chain.
2. WHEN reasoning completes, THE **Morgan_System** SHALL provide an optional "show reasoning" view summarising decomposition and conclusions.
3. WHEN the Reasoning Engine flags low confidence or hallucination risk, THE **Morgan_System** SHALL communicate the uncertainty.
4. WHEN reasoning calls external tools or searches, THE **Reasoning_Engine** SHALL log tool usage and results for auditing.
5. WHEN reasoning times out, THE **Morgan_System** SHALL fall back to a concise answer with an explanation of the timeout.

### Requirement R7 – Memory & Preference Alignment

**User Story:** As a returning user, I want Morgan to recall relevant memories without overwhelming me, so that interactions stay fluid.

#### Acceptance Criteria

1. WHEN searching memories, THE **Morgan_System** SHALL include preference-weighted ranking from the Learning Engine.
2. WHEN memories conflict with updated preferences, THE **Morgan_System** SHALL reconcile conflicts before surfacing context.
3. WHEN new memories are stored, THE **Morgan_System** SHALL capture emotional context and relationship significance for future ranking.
4. WHEN a user opts out of memory retention, THE **Morgan_System** SHALL delete associated memories and refrain from storing new ones.
5. WHEN memory retrieval influences an answer, THE **Morgan_System** SHALL highlight the memory source on demand.

### Requirement R8 – CLI & API Compatibility

**User Story:** As a maintainer, I need current CLI and API contracts to keep working so that users experience a smooth upgrade.

#### Acceptance Criteria

1. WHEN running `morgan chat` or `morgan ask`, THE **Morgan_System** SHALL remain backward compatible with current JSON/schema responses.
2. WHEN new CLI diagnostics are added, THE **Morgan_System** SHALL provide optional commands that do not break existing scripts.
3. WHEN environment variables or config options change, THE **Morgan_System** SHALL provide defaults that maintain prior behaviour.
4. WHEN refactoring introduces new logs, THE **Morgan_System** SHALL respect the existing logging format and verbosity flags.
5. WHEN REST/websocket endpoints are used, THE **Morgan_System** SHALL return compatible payloads or versioned responses.
6. WHEN generating LLM completions, THE **LLM_Service** SHALL use the configured OpenAI-compatible `LLM_BASE_URL`/`LLM_API_KEY`; only when that endpoint is unreachable SHALL it fall back to a documented local provider and log the fallback.
7. WHEN exposing new functionality, THE **Morgan_System** SHALL provide CLI/bash-first entry points (`morgan` commands or scripts) so that operators can access features without a GUI.

### Requirement R9 – Observability & Metrics

**User Story:** As an operator, I want full visibility into ingestion, search, learning, and reasoning so that regressions are easy to detect.

#### Acceptance Criteria

1. WHEN ingestion runs, THE **Metrics_System** SHALL capture documents processed, hierarchical cache hit rate, and per-scale latency.
2. WHEN search executes, THE **Metrics_System** SHALL include candidate reduction, reranking improvement, and total latency.
3. WHEN background tasks run, THE **Metrics_System** SHALL record task duration, success/failure counts, and queue depth.
4. WHEN learning updates preferences, THE **Metrics_System** SHALL track preference drift and adaptation counts.
5. WHEN reasoning chains execute, THE **Metrics_System** SHALL record chain length, tools invoked, and confidence levels.

### Requirement R10 – Testing & Migration Safety

**User Story:** As a developer, I need confidence that each refactor preserves functionality so that releases remain stable.

#### Acceptance Criteria

1. WHEN hierarchical ingestion is enabled, THE **Morgan_System** SHALL validate dual-write compatibility with legacy vectors through automated tests.
2. WHEN Jina integration is active, THE **Morgan_System** SHALL mock remote services and verify fallback paths through tests.
3. WHEN learning and reasoning engines run, THE **Morgan_System** SHALL cover tip-of-spear behaviours (preference decay, reasoning timeout, etc.) through tests.
4. WHEN migrations execute, THE **Morgan_System** SHALL provide dry-run and rollback modes with logging through scripts.
5. WHEN CI runs, THE **Morgan_System** SHALL execute both unit suites and integration suites that include background tasks and Jina stack simulations.

### Requirement R11 – Chat MVP Validation

**User Story:** As a product owner, I need an end-to-end chat experience that proves Morgan can combine emotional intelligence, memory, learning, and reasoning so that stakeholders can validate the refactor.

#### Acceptance Criteria

1. WHEN `morgan chat` is launched, THE **Morgan_System** SHALL expose a configuration to enable hierarchical search, reranking, learning, and reasoning features simultaneously.
2. WHEN a conversation expresses a clear emotion, THE **Morgan_System** SHALL surface the detected emotion and the empathetic response elements (tone, acknowledgement).
3. WHEN a returning user references prior topics, THE **Morgan_System** SHALL retrieve relevant memories and cite them in debug mode (or upon request) to prove recall.
4. WHEN complex questions are asked, THE **Morgan_System** SHALL optionally display reasoning traces or enable a command to reveal them.
5. WHEN running scripted chat scenarios (happy path, negative emotion, preference update), THE **Morgan_System** SHALL log checkpoints indicating each advanced subsystem executed successfully.
6. WHEN the chat MVP is executed entirely from CLI/bash (e.g., `morgan chat`, scripted REPLs), THE **Morgan_System** SHALL expose all advanced toggles and debug views without requiring a web UI.

## Traceability Matrix

| Requirement | Design Component | Implementation Focus | Metric |
|-------------|------------------|----------------------|--------|
| R1 | Vector & Ingestion Modernisation | Hierarchical ingestion pipeline | 90% candidate reduction |
| R2 | Vector & Ingestion Modernisation | Jina embedding selector integration | Embedding accuracy + latency |
| R3 | Search & Reranking Refinement | Multi-stage search, reranking | Reranking ≥ 25% improvement |
| R4 | Background Optimisation | BackgroundProcessingService activation | Cache hit ≥ 90% |
| R5 | Learning & Personalisation | Learning engine, profile storage | Personalisation feedback |
| R6 | Reasoning & Transparency | Reasoning engine, explainable answers | Reasoning coverage |
| R7 | Memory Alignment | Memory + learning integration | Preference-aligned recall |
| R8 | Compatibility | CLI/API preservation | Zero breaking changes |
| R9 | Observability | Monitoring dashboard, metrics pipeline | Metrics completeness |
| R10 | Testing | Regression + integration suites | CI pass rate |
| R11 | Chat Experience & Validation | End-to-end chat MVP instrumentation | Scenario pass rate |

## Validation Checklist

- ✅ Hierarchical ingestion, Jina integration, reranking, learning, reasoning, and background optimisation have explicit requirements.
- ✅ Acceptance criteria emphasise measurable outcomes (latency, candidate reduction, cache hit rate, explainability).
- ✅ Traceability connects requirements to design workstreams and implementation tasks.
- ✅ Compatibility and testing safeguards ensure a safe migration path.
