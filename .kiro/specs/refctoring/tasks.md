# Implementation Plan: Morgan Refactoring Program

Primary Goal: Align Morgan’s codebase with the advanced architecture by delivering hierarchical ingestion, Jina-powered search, continuous optimisation, learning, and reasoning—while keeping the assistant human-first and maintainable.

## Phase 1: Hierarchical Ingestion & Storage (Priority: CRITICAL)

- [ ] 1. Replace legacy ingestion pathway
  - [x] 1.1 Wire `KnowledgeBase.ingest_documents` to call `HierarchicalEmbeddingService.create_hierarchical_embeddings` (R1.1, R1.2)
  - [x] 1.2 Extend `VectorDBClient` helpers to dual-write named vectors and manage hierarchical collections (R1.2, R1.3)
  - [x] 1.3 Implement Git hash cache reuse with cache-hit metrics (R1.3, R9.1)
  - [-] 1.4 Provide migration scripts to re-embed existing knowledge bases (R10.4, R10.5)
  - [ ]* 1.5 Add regression tests covering hierarchical + legacy ingestion modes (R10.1, R10.2)

## Phase 2: Jina Search & Reranking Integration (Priority: CRITICAL)

- [ ] 2. Activate Jina embeddings end-to-end
  - [ ] 2.1 Route all local embedding calls through `morgan/jina/models/selector.py` and `JinaEmbeddingService` (R2.1, R2.2)
  - [ ] 2.2 Implement batching, multilingual support, and logging hooks (R2.3, R2.4, R2.5)
  - [ ] 2.3 Update CLI/config defaults to opt-in while preserving legacy fallbacks (R8.1, R8.3)
  - [ ] 2.4 Enforce OpenAI-compatible embedding endpoint usage with logged fallback to approved local transformers when offline but main time we use embedding from gpt.lazarev.cloud and local jina based rerancers and etc (R2.6)

- [ ] 3. Finalise multi-stage search with reranking
  - [ ] 3.1 Ensure `SmartSearch` delegates to `MultiStageSearchEngine` with hierarchical embeddings (R3.1)
  - [ ] 3.2 Integrate `JinaRerankingService` and record fusion metrics (R3.2, R3.4, R9.2)
  - [ ] 3.3 Provide fallback behaviour and tests when reranking is unavailable (R3.3, R10.2)

## Phase 3: Background Optimisation & Observability (Priority: HIGH)

- [ ] 4. Enable continuous background processing
  - [ ] 4.1 Start `BackgroundProcessingService` during assistant boot and expose health status (R4.1, R4.3)
  - [ ] 4.2 Configure default schedules for reindexing, reranking, cache priming (R4.2, R4.5)
  - [ ] 4.3 Implement resource-aware throttling and logging (R4.4, R9.3)

- [ ] 5. Expand observability
  - [ ] 5.1 Add ingestion/search/learning/reasoning metrics to monitoring dashboards (R9.1–R9.5)
  - [ ] 5.2 Create CLI diagnostics (`morgan doctor`, `morgan background status`) (R4.3, R8.2)
  - [ ] 5.3 Document runbooks for operators covering new metrics and toggles (R8.4, R10.4)

## Phase 4: Learning & Personalisation System (Priority: HIGH)

- [ ] 6. Implement learning engine modules
  - [ ] 6.1 Build `morgan/learning/engine.py` with preference weighting, decay, and resets (R5.1–R5.4)
  - [ ] 6.2 Integrate learning updates into ConversationManager and profile storage (R5.2, R7.1, R7.3)
  - [ ] 6.3 Extend memory search to respect personalised rankings (R7.1, R7.2)
  - [ ] 6.4 Add opt-out and data deletion flows (R5.4, R7.4, R8.5)
  - [ ]* 6.5 Create test suites covering learning decay, preference reset, and personalised retrieval (R5.5, R10.3)

## Phase 5: Reasoning & Transparency (Priority: MEDIUM)

- [ ] 7. Ship reasoning engine
  - [ ] 7.1 Implement reasoning orchestrator, decomposition, assumption tracking, and explainer (R6.1, R6.2)
  - [ ] 7.2 Add tool invocation hooks with observability logs (R6.4, R9.5)
  - [ ] 7.3 Surface “show reasoning” responses and fallback messaging (R6.2, R6.5)
  - [ ] 7.4 Integrate hallucination risk detection and confidence markers (R6.3, R9.2)
  - [ ]* 7.5 Test reasoning timeout paths, uncertainty communication, and tool logging (R6.5, R10.3)

## Phase 6: Chat MVP Enablement (Priority: MEDIUM)

- [ ] 8. Wire chat orchestration
  - [ ] 8.1 Update `MorganAssistant.ask/ask_stream` to toggle advanced subsystems (hierarchical search, reranking, learning, reasoning) via config or runtime flags (R11.1)
  - [ ] 8.2 Ensure emotional analysis outputs are surfaced in chat responses with empathetic phrasing and optional debug annotations (R11.2)
  - [ ] 8.3 Integrate memory recall summaries (with opt-in display) to demonstrate personalised context usage (R11.3, R7.5)
  - [ ] 8.4 Confirm the full chat experience (advanced toggles, debug views) operates from CLI/bash only, without GUI dependencies (R8.7, R11.6)

- [ ] 9. Deliver chat validation toolset
  - [ ] 9.1 Provide scripted scenario runners (happy path, emotional support, returning user) and capture subsystem checkpoints (R11.5)
  - [ ] 9.2 Add chat debug overlay/CLI flags to reveal emotion detection, memory hits, learning adjustments, and reasoning traces (R11.2–R11.4)
  - [ ]* 9.3 Write integration tests or golden transcripts validating chat scenarios end-to-end (R11.5, R10.5)

## Phase 7: Hardening & Release (Priority: MEDIUM)

- [ ] 10. Hardening sweep
  - [ ] 10.1 Conduct performance benchmarking across ingestion, search, reasoning, and chat latency (R3.4, R9.2, R11.1)
  - [ ] 10.2 Finalise migration guides, configuration tables, chat playbooks, and developer docs (R8.3, R10.4, R11.5)
  - [ ] 10.3 Run full CI plus manual acceptance scenarios (legacy fallbacks, opt-outs, background stop/start, chat playbooks) (R8.1, R10.5, R11.5)
  - [ ] 10.4 Validate that both embeddings and LLM completions honour the OpenAI-compatible endpoints with documented, logged fallbacks (R2.6, R8.6)
  - [ ] 10.5 Prepare release checklist summarising metrics vs. targets (R1.4, R3.4, R9.x, R11.5)

_* Tasks marked with an asterisk denote test or validation deliverables that must be completed alongside implementation._
