# Design Document: Morgan Refactoring Program

## Overview

This design defines the refactoring program that evolves Morgan into a production‑ready companion assistant while preserving the existing human‑first ergonomics. The scope complements the **Advanced Vectorization System** by aligning ingestion, search, reasoning, learning, and background optimisation code with the 2025 architecture envisioned for Morgan.

## Goals

- Close the gap between current implementation and specification by delivering hierarchical embeddings, Jina integration, continuous background optimisation, and adaptive behaviours.
- Simplify orchestration so Morgan can grow into a smart, self‑learning companion with memory, reranking, and transparent reasoning.
- Maintain KISS principles—each module keeps a single responsibility with clear, composable interfaces.
- Ensure every embedding/LLM call honours the configured OpenAI-compatible API base URL/IP, with explicit fallbacks to approved local HuggingFace/`sentence-transformers` models (e.g., self-hosted Jina models); no opaque third-party endpoints.
- Keep the experience CLI-first—chat, debugging, and validation must run comfortably from bash without requiring a GUI front end.

## Guiding Principles

1. **Incremental Replacement** – Refactor feature slices (ingestion, search, memory, etc.) in isolation, verify, then merge.
2. **Compatibility First** – Preserve existing CLI/API contracts; new capabilities must slot in without breaking Morgan’s current workflows.
3. **Instrumentation Everywhere** – Add metrics and logging hooks to prove performance targets (latency, cache hit rate, candidate reduction).
4. **Fallback Ready** – Provide safe fallbacks (legacy embeddings, single‑stage search) when advanced components are unavailable.
5. **Documentation Driven** – Each refactor produces code comments, usage docs, and migration notes that mirror this specification.

## Target Architecture

```text
User ➜ Morgan CLI ➜ Assistant Orchestrator
  │                              │
  ├─▶ Conversation Memory ◀──────┤
  │                              │
  ├─▶ Emotional + Relationship Engines
  │                              │
  └─▶ Knowledge Layer (Hierarchical Ingestion, Vector DB, Background Optimiser)
             │
             ├─▶ Hierarchical Embedding Pipeline (coarse/medium/fine)
             ├─▶ Multi-Stage Search + Jina Reranker
             ├─▶ Background Services (Reindex, Cache, Rerank, Metrics)
             └─▶ Learning & Reasoning Engines (self-learning, critique, reranking)
```

## Architecture

### System Architecture Overview

The refactored Morgan system maintains a layered architecture that preserves existing CLI/API contracts while introducing advanced capabilities:

```text
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                     │
│  ┌─────────────┐  ┌────────────────────┐  ┌─────────────────────┐ │
│  │ Morgan CLI  │  │ Morgan Web(future)  │  │ REST/WebSocket APIs │ │
│  └─────────────┘  └────────────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                Assistant Orchestration Layer                │
│  ┌─────────────────┐  ┌─────────────────────────────────┐   │
│  │ MorganAssistant │  │ ConversationManager             │   │
│  │ - ask()         │  │ - Emotional Context Integration │   │
│  │ - ask_stream()  │  │ - Memory Recall Coordination    │   │
│  └─────────────────┘  └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Intelligence Layer                        │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│ │ Learning    │ │ Reasoning   │ │ Emotional Analysis      │ │
│ │ Engine      │ │ Engine      │ │ - Emotion Detection     │ │
│ │ - Preferences│ │ - Chain     │ │ - Empathetic Response   │ │
│ │ - Adaptation │ │ - Tools     │ │ - Relationship Context  │ │
│ └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Knowledge Layer                           │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│ │ Search Engine   │ │ Memory System   │ │ Ingestion       │ │
│ │ - Multi-stage   │ │ - Conversation  │ │ - Hierarchical  │ │
│ │ - Jina Reranker │ │ - Preference    │ │ - Git Caching   │ │
│ │ - Fallbacks     │ │ - Emotional     │ │ - Jina Models   │ │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                Infrastructure Layer                         │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│ │ Vector Database │ │ Background      │ │ Metrics &       │ │
│ │ - Qdrant        │ │ Processing      │ │ Observability   │ │
│ │ - Named Vectors │ │ - Reindexing    │ │ - Performance   │ │
│ │ - Collections   │ │ - Cache Priming │ │ - Health Checks │ │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Workstream Breakdown

1. **Vector & Ingestion Modernisation** (Requirements R1, R2)
   - Replace single vector ingestion with Matryoshka hierarchical embeddings supporting 90% candidate reduction target.
   - Integrate Jina model selector with automatic fallback to legacy models when services are unavailable.
   - Ensure Qdrant collections use named vectors (`coarse`, `medium`, `fine`) with Git hash caching for unchanged content.
   - Guarantee all embedding calls respect the configured OpenAI-compatible API base URL/IP; if unreachable, load the approved local HuggingFace/`sentence-transformers` model (e.g., self-hosted Jina checkpoint) and log the fallback.
   - **Design Decision**: Dual-write approach during migration enables safe rollback while validating hierarchical performance.

2. **Search & Reranking Refinement** (Requirement R3)
   - Drive all query embeddings through Jina models via the model selector with multilingual support.
   - Run multi-stage search and pipe results through Jina reranker targeting ≥25% improvement in result quality.
   - Provide graceful fallback to legacy search when Jina services are down, maintaining user experience.
   - Keep the search orchestration CLI-first—`morgan ask`/`chat` and bash scripts should exercise the full retrieval stack without relying on web middleware.
   - **Design Decision**: Search gateway pattern abstracts complexity while preserving existing API contracts.

3. **Background Optimisation Activation** (Requirement R4)
   - Bootstrap `BackgroundProcessingService` during assistant startup with health monitoring.
   - Schedule reindex, reranking, and cache priming tasks targeting ≥90% cache hit rate.
   - Implement resource-aware throttling to prevent system overload during peak usage.
   - **Design Decision**: Separate background service enables independent scaling and resource management.

4. **Learning & Personalisation Engine** (Requirements R5, R7)
   - Materialise the `morgan/learning` modules with preference weighting and automatic decay.
   - Feed ConversationMemory + MemoryProcessor insights into profile storage with emotional context.
   - Adapt ConversationManager suggestions, style, and retrievals based on learned preferences.
   - **Design Decision**: Stateless learning engine with centralized storage enables horizontal scaling.

5. **Reasoning & Transparency** (Requirement R6)
   - Implement the `morgan/reasoning` package with structured chain construction and tool integration.
   - Allow users to request visible reasoning traces with confidence indicators and uncertainty communication.
   - Add safety rails including hallucination detection and timeout handling for complex reasoning.
   - **Design Decision**: Optional reasoning display maintains backward compatibility while enabling transparency.

6. **Observability & Tooling** (Requirement R9)
   - Extend monitoring dashboards with comprehensive metrics across all system components.
   - Add CLI diagnostics (`morgan doctor`, `morgan background status`) for operational visibility.
   - Implement structured logging with performance metrics and error categorization.
   - **Design Decision**: Centralized metrics system enables unified monitoring across distributed components.

7. **Chat Experience & Validation** (Requirement R11)
   - Deliver an MVP chat loop that exercises emotional analysis, memory recall, learning adaptation, and reasoning.
   - Provide conversation scenario playbooks with scripted validation of advanced subsystems.
   - Instrument chat flows with toggleable debug traces for rapid validation and troubleshooting.
   - Validate every scenario via CLI/bash (e.g., `morgan chat`, scripted REPL sessions) to confirm headless, automation-friendly operation.
   - **Design Decision**: Configuration-driven feature toggles enable gradual rollout and A/B testing.

## Components and Interfaces

### Core System Components

| Component | Owner Module | Interface Highlights |
|-----------|--------------|----------------------|
| Hierarchical Ingestion | `morgan/ingestion/enhanced_processor.py` + `morgan/vectorization/hierarchical_embeddings.py` | `KnowledgeBase.ingest_documents(...)` returns hierarchical stats; stores `coarse/medium/fine` vectors through `VectorDBClient.upsert_points` |
| Search Gateway | `morgan/core/search.py` | Accepts query, optional emotion + user id; internally orchestrates multi-stage search, Jina reranker, fallback search |
| Background Engine | `morgan/background/service.py` | `start()`, `stop()`, `schedule_reindexing(...)`, `precompute_queries(...)` |
| Learning Engine | `morgan/learning/engine.py` | `process_turn(...)`, `update_preferences(...)`, `get_recommendations(...)` |
| Reasoning Engine | `morgan/reasoning/engine.py` | `solve(problem, context)` returning structured reasoning chain |

### Interface Design Rationale

**Hierarchical Ingestion Interface**: The design separates embedding generation (`HierarchicalEmbeddingService`) from storage operations (`VectorDBClient`) to enable independent scaling and testing. This allows fallback to legacy ingestion when hierarchical processing fails, satisfying requirement R1.5.

**Search Gateway Interface**: The unified search interface abstracts multi-stage complexity from callers while providing optional emotional context integration. This design enables gradual migration from legacy search without breaking existing API contracts (R8.1).

**Background Processing Interface**: Simple start/stop semantics with configurable scheduling allow operators to control resource usage dynamically. The interface exposes health status for monitoring integration (R4.3).

**Learning Engine Interface**: Stateless processing methods enable horizontal scaling while preference storage remains centralized. The design includes explicit reset capabilities to prevent over-personalization (R5.4).

**Reasoning Engine Interface**: The solve method returns structured chains rather than just answers, enabling transparency features while maintaining backward compatibility for callers who only need final results (R6.2).

## Data Models

### Hierarchical Embedding Storage

```python
@dataclass
class HierarchicalEmbedding:
    coarse: List[float]      # 256-dim for broad semantic matching
    medium: List[float]      # 512-dim for balanced precision/recall  
    fine: List[float]        # 1024-dim for precise semantic matching
    metadata: Dict[str, Any] # Git hash, chunk info, processing flags
```

**Design Rationale**: Named vector storage in Qdrant enables independent querying at different granularities. The metadata includes Git hash for cache reuse (R1.3) and processing flags for fallback detection.

### Learning Profile Schema

```python
@dataclass
class UserProfile:
    user_id: str
    preferences: Dict[str, float]    # Topic weights with decay timestamps
    conversation_style: Dict[str, Any] # Tone, formality, detail preferences
    memory_priorities: List[str]     # Ranked topics for memory retrieval
    opt_out_flags: Dict[str, bool]   # Privacy controls per feature
```

**Design Rationale**: Preference weights include timestamps for automatic decay (R5.3). Opt-out flags provide granular privacy controls (R7.4). The schema supports both learning adaptation and memory personalization (R7.1).

### Reasoning Chain Structure

```python
@dataclass
class ReasoningChain:
    problem: str
    decomposition: List[str]         # Sub-problems identified
    assumptions: List[str]           # Explicit assumptions made
    tool_calls: List[ToolInvocation] # External tools used
    confidence: float                # Overall confidence score
    explanation: str                 # Human-readable reasoning trace
```

**Design Rationale**: Structured chains enable transparency features (R6.2) while confidence scores support uncertainty communication (R6.3). Tool call logging satisfies audit requirements (R6.4).

## Error Handling

### Fallback Strategy

The system implements graceful degradation across all major components:

1. **Embedding Failures**: When Jina models are unavailable, the system falls back to legacy embedding models with clear logging (R2.2, R8.3).

2. **Search Degradation**: Multi-stage search failures trigger fallback to single-stage search with performance warnings (R3.3).

3. **Background Service Resilience**: Background tasks implement exponential backoff and circuit breaker patterns to prevent cascade failures (R4.4).

4. **Learning Engine Safeguards**: Preference updates include validation to prevent corruption, with automatic rollback on detection of anomalous changes (R5.1).

5. **Reasoning Timeouts**: Complex reasoning chains implement configurable timeouts with partial result return (R6.5).

### Error Monitoring

All error conditions feed into the metrics system with categorization by component and severity. Critical errors (data corruption, service unavailability) trigger immediate alerts, while degraded performance conditions generate warnings for operational review.

## Testing Strategy

### Unit Testing Focus

- **Hierarchical Embedding Generation**: Test dimensional consistency, cache hit/miss scenarios, and fallback behavior (R10.1).
- **Multi-stage Search Logic**: Verify candidate reduction, reranking integration, and fallback paths (R10.2).
- **Learning Engine Behavior**: Cover preference decay, reset functionality, and over-personalization prevention (R10.3).
- **Reasoning Chain Construction**: Test decomposition accuracy, tool integration, and timeout handling (R10.3).

### Integration Testing Approach

- **End-to-End Chat Scenarios**: Scripted conversations covering emotional support, memory recall, and reasoning transparency (R11.5).
- **Background Service Integration**: Test service lifecycle, task scheduling, and resource management under load (R10.5).
- **Migration Safety**: Dual-write validation during hierarchical ingestion rollout with data consistency checks (R10.4).

### Performance Benchmarking

- **Ingestion Throughput**: Measure hierarchical embedding generation vs. legacy pipeline with cache hit rate analysis (R9.1).
- **Search Latency**: Compare multi-stage search performance against single-stage baseline with reranking impact assessment (R9.2).
- **Chat Response Time**: End-to-end latency measurement including emotion detection, memory retrieval, and reasoning chain generation (R11.1).

**Testing Rationale**: The strategy prioritizes regression prevention during incremental rollout while ensuring new capabilities meet performance targets. Mock services enable testing of Jina integration without external dependencies (R10.2).

## Requirements Traceability

This design directly addresses all requirements from the specification:

### Performance Targets
- **R1**: 90% candidate reduction through hierarchical embeddings
- **R3**: ≥25% improvement in search quality through Jina reranking  
- **R4**: ≥90% cache hit rate via background optimization
- **R8**: Zero breaking changes to existing CLI/API contracts

### System Capabilities
- **R2**: Jina model integration with automatic fallback handling
- **R5**: Learning engine with preference decay and personalization
- **R6**: Reasoning engine with transparency and confidence indicators
- **R7**: Memory system with preference-weighted ranking
- **R9**: Comprehensive observability across all components
- **R10**: Migration safety through dual-write and rollback capabilities
- **R11**: End-to-end chat validation with scripted scenarios

### Design Alignment
Each architectural component maps directly to specific requirements, ensuring complete coverage while maintaining system coherence. The layered approach enables independent development and testing of each capability while preserving integration points.

## Non-Goals

- Replacing the LLM provider or rewriting UI clients.
- Shipping multimodal interfaces (image/audio) beyond text-based interactions.
- Building new deployment infrastructure (Docker/Kubernetes scripts stay untouched).

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Embedding dimensional mismatch between existing data and hierarchical collections | Maintain migration scripts that dual-write during transition, include validation to detect misaligned vectors |
| Increased latency from reranking | Benchmark and cache reranker results; allow config flag to disable reranking per query |
| Feature regressions from parallel refactors | Add regression suites per module, run CI nightly with background tasks enabled |
| Learning engine over-personalises | Store preference weights with decay, expose user settings to reset or adjust |

## Rollout Strategy

1. **Phase A: Parallel-path ingestion/search** – Build hierarchical ingestion and Jina pipelines while keeping legacy defaults; run shadow evaluations.
2. **Phase B: Activate background services & reranking** – Start background tasks, expose CLI toggles, gather metrics.
3. **Phase C: Launch learning + reasoning** – Introduce personalisation, habit detection, reasoning chain, and wire them into chat responses; gather user feedback.
4. **Phase D: Chat MVP & Hardening** – Validate chat playbooks, optimise latency, document CLI flows, polish monitoring dashboards, prepare release notes.

This refactoring blueprint ensures Morgan progresses toward a smart, self-learning companion that can reason, remember, and adapt, while keeping the existing human-first experience intact.
