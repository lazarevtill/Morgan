# Morgan v2 — Modular Brain Architecture

> Ground-up redesign based on SkillOpt (Microsoft, 2026), MAPLE (AAMAS 2026),
> XSkill dual-stream learning, and the Agent / Skills / MCP three-layer pattern.

---

## Design Principles

| # | Principle | Rationale |
|---|-----------|-----------|
| 1 | **Skills are the trainable state** | SkillOpt: markdown skill documents evolve via trajectory feedback, not model fine-tuning |
| 2 | **Memory ≠ Learning ≠ Personalization** | MAPLE: three sub-agents on different timescales, not a conflated monolith |
| 3 | **Perception is multimodal** | Audio prosody (mood, sarcasm), text semantics, visual context — fused before reasoning |
| 4 | **Integrations are MCP servers** | Calendar, email, tools are external MCP services — the brain never hardcodes them |
| 5 | **Docker-first, microservice-native** | Each brain module = independent container, composed via docker-compose |
| 6 | **Event-driven reactive core** | All inter-module communication via typed events on a message bus |
| 7 | **Privacy-first, local hardware** | All processing on-prem; external APIs are opt-in |

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         CLIENTS                                      │
│   CLI  ·  Web UI  ·  Telegram  ·  Discord  ·  API consumers         │
└──────────────┬───────────────────────────────────────────────────────┘
               │ HTTP / WebSocket / gRPC
┌──────────────▼───────────────────────────────────────────────────────┐
│                      GATEWAY SERVICE                                 │
│  Auth · Rate limit · Session · Route to brain modules                │
│  Channel adapters (Telegram, Discord, etc.)                          │
└──────┬──────────┬──────────┬──────────┬──────────┬───────────────────┘
       │          │          │          │          │
┌──────▼───┐ ┌───▼────┐ ┌───▼────┐ ┌───▼────┐ ┌───▼──────────────────┐
│PERCEPTION│ │ MEMORY │ │LEARNING│ │PERSONAL│ │   SKILL ENGINE       │
│ MODULE   │ │ MODULE │ │ MODULE │ │-IZATION│ │ (SkillOpt runtime)   │
│          │ │ (M)    │ │ (L)    │ │ (P)    │ │                      │
│ Audio    │ │ Store  │ │ Async  │ │ Real-  │ │ Skill registry       │
│ Text     │ │ Recall │ │ Trait  │ │ time   │ │ Trajectory collector │
│ Vision   │ │ Index  │ │ Extract│ │ Context│ │ Optimizer loop       │
│ Emotion  │ │ Vector │ │ User   │ │ Inject │ │ Validation gate      │
│ Sarcasm  │ │ Graph  │ │ Model  │ │        │ │ best_skill.md deploy │
└──────┬───┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬──────────────────┘
       │         │          │          │           │
┌──────▼─────────▼──────────▼──────────▼───────────▼───────────────────┐
│                      REASONING ENGINE                                │
│  Orchestrator · Context assembly · LLM routing · Tool planning       │
│  Multi-step planning · Reflection · Response generation              │
└──────┬──────────┬──────────┬─────────────────────────────────────────┘
       │          │          │
┌──────▼───┐ ┌───▼────┐ ┌───▼──────────────────────────────────────────┐
│  TOOL    │ │  MCP   │ │           EVENT BUS                          │
│ EXECUTOR │ │ CLIENT │ │  Redis Streams / NATS / in-process           │
│          │ │ HUB    │ │  Typed events · Fan-out · Replay             │
│ Built-in │ │        │ │                                              │
│ tools    │ │ Mail   │ └──────────────────────────────────────────────┘
│          │ │ Cal    │
│          │ │ Search │
│          │ │ Custom │
└──────────┘ └────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE                                    │
│  Qdrant (vectors) · Redis (cache/bus) · Ollama (LLM) · PostgreSQL   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Module Specifications

### 1. Gateway Service (`morgan-gateway`)

The single entry point for all clients. Replaces the current `morgan-server` + `morgan-cli` split.

```
morgan-gateway/
├── gateway/
│   ├── app.py                 # FastAPI app factory
│   ├── config.py              # Unified config (single source of truth)
│   ├── auth/
│   │   ├── middleware.py       # JWT / API key auth
│   │   └── sessions.py        # Session lifecycle (Redis-backed)
│   ├── channels/
│   │   ├── base.py            # BaseChannel protocol
│   │   ├── http.py            # REST + WebSocket
│   │   ├── telegram.py        # Telegram adapter
│   │   └── discord.py         # Discord adapter
│   ├── routing/
│   │   ├── router.py          # Route to brain modules
│   │   └── middleware.py      # Logging, CORS, rate limiting
│   └── models/
│       ├── requests.py        # Inbound message models
│       └── responses.py       # Outbound response models
├── Dockerfile
└── pyproject.toml
```

**Key changes from current:**
- Single config system (no more `ServerConfig` vs `get_settings()` split)
- Auth is real (JWT + API keys), not "no auth on any endpoint"
- Channel adapters are thin protocol implementations, not Morgan-coupled
- FastAPI `Depends()` for all injection (no module-level `_assistant` globals)
- WebSocket paths are consistent and documented

---

### 2. Perception Module (`morgan-perception`)

Multimodal input analysis. Runs as a separate service with GPU access.

```
morgan-perception/
├── perception/
│   ├── __init__.py
│   ├── service.py             # PerceptionService (gRPC + REST)
│   ├── pipeline.py            # MultimodalPipeline orchestrator
│   ├── audio/
│   │   ├── processor.py       # Audio preprocessing (WAV/MP3 → features)
│   │   ├── emotion.py         # Wav2Vec2 / EmotionThinker emotion detection
│   │   ├── sarcasm.py         # Prosody-text incongruence detection
│   │   ├── transcription.py   # Whisper ASR
│   │   └── models.py          # AudioAnalysis, EmotionResult, SarcasmResult
│   ├── text/
│   │   ├── analyzer.py        # Semantic emotion, intent, entity extraction
│   │   ├── sentiment.py       # Fine-grained sentiment (beyond pos/neg)
│   │   └── models.py          # TextAnalysis, Intent, Entity
│   ├── vision/
│   │   ├── analyzer.py        # Image/screenshot understanding
│   │   ├── ocr.py             # Text extraction from images
│   │   └── models.py          # VisionAnalysis
│   ├── fusion/
│   │   ├── multimodal.py      # Cross-modal fusion (audio+text agreement)
│   │   └── models.py          # FusedPerception (unified output)
│   └── config.py
├── Dockerfile                  # GPU-capable image
└── pyproject.toml
```

**Output contract** — every perception call returns a `FusedPerception`:

```python
@dataclass
class FusedPerception:
    text: str                          # Original or transcribed text
    intent: Intent                     # What the user wants
    entities: list[Entity]             # Named entities extracted
    emotion: EmotionState              # Detected emotion (multimodal)
    emotion_confidence: float          # How sure we are
    sarcasm: SarcasmResult | None      # Sarcasm detection if audio present
    sentiment: SentimentScore          # Valence/arousal/dominance
    modalities_used: list[Modality]    # Which inputs were available
    raw_audio_features: dict | None    # Prosody features for downstream
    raw_text_features: dict | None     # Embeddings for downstream
```

**Audio emotion pipeline:**
1. Whisper ASR → text transcription
2. Wav2Vec2 / ONNX quantized → emotion embeddings (7 classes + neutral)
3. Prosody extraction (pitch CV, energy variance, speech rate)
4. Text sentiment analysis
5. Sarcasm = prosody-sentiment incongruence score
6. Fusion: weighted combination based on modality reliability

---

### 3. Memory Module (`morgan-memory`) — MAPLE "M"

Storage and retrieval infrastructure. Answers "what did the user say/do?"

```
morgan-memory/
├── memory/
│   ├── __init__.py
│   ├── service.py             # MemoryService (the single memory authority)
│   ├── stores/
│   │   ├── base.py            # MemoryStore protocol
│   │   ├── vector.py          # Qdrant vector store (semantic search)
│   │   ├── graph.py           # Knowledge graph (entity relationships)
│   │   ├── temporal.py        # Time-indexed conversation log
│   │   └── workspace.py       # File-based workspace (SOUL.md, MEMORY.md)
│   ├── indexing/
│   │   ├── embedder.py        # Embedding service (Ollama / local)
│   │   ├── chunker.py         # Document chunking strategies
│   │   └── dedup.py           # Deduplication
│   ├── retrieval/
│   │   ├── search.py          # Unified search (vector + keyword + temporal)
│   │   ├── reranker.py        # Single reranking layer (no more triple)
│   │   └── context.py         # Context window assembly + compaction
│   ├── models.py              # Memory, MemoryQuery, SearchResult
│   └── config.py
├── Dockerfile
└── pyproject.toml
```

**Key changes from current:**
- **Single `MemoryService`** — no more `MemoryProcessor` vs `MemoryService` vs `core/memory` split
- **Single reranker** — `retrieval/reranker.py` wraps the reranking model once
- **Compaction lives here** (context window management is a memory concern)
- **Workspace files** (SOUL.md, MEMORY.md) are just another store type
- **Memory gating** is enforced at the `MemoryService` level (single `MemoryGate`)
- Daily logs → `stores/temporal.py` (no more workspace vs memory_consolidation duplication)

---

### 4. Learning Module (`morgan-learning`) — MAPLE "L"

Async background intelligence extraction. Runs on a schedule, not in the request path.

```
morgan-learning/
├── learning/
│   ├── __init__.py
│   ├── service.py             # LearningService
│   ├── extractors/
│   │   ├── trait.py           # Extract stable user traits from history
│   │   ├── preference.py     # Learn communication preferences
│   │   ├── pattern.py        # Behavioral patterns (time-of-day, topics)
│   │   └── relationship.py   # Relationship dynamics evolution
│   ├── consolidation/
│   │   ├── consolidator.py   # LLM-driven memory consolidation
│   │   ├── dedup.py          # Cross-session deduplication
│   │   └── decay.py          # Memory importance decay over time
│   ├── skillopt/
│   │   ├── trainer.py        # SkillOpt training loop integration
│   │   ├── trajectory.py     # Conversation trajectory collector
│   │   ├── evaluator.py      # Validation gate (held-out scoring)
│   │   └── registry.py       # Skill document version management
│   ├── user_model/
│   │   ├── model.py          # UserModel (the stable trait store)
│   │   ├── schema.py         # Trait categories and structure
│   │   └── persistence.py    # Save/load user models
│   ├── scheduler.py           # Cron/interval scheduling for learning jobs
│   └── config.py
├── Dockerfile
└── pyproject.toml
```

**SkillOpt integration:**
- Morgan's existing skill documents (SOUL.md, conversation skills) become **trainable artifacts**
- `trajectory.py` captures conversation rollouts (user message → morgan response → user satisfaction signals)
- `trainer.py` wraps SkillOpt's optimizer loop: rollout → reflect → aggregate → edit → validate
- `evaluator.py` implements the validation gate against held-out conversations
- Trained `best_skill.md` artifacts are deployed to the Skill Engine at runtime
- Training runs asynchronously (nightly or on-demand), never in the request path

**User model output** — the Learning module produces a stable `UserModel`:

```python
@dataclass
class UserModel:
    user_id: str
    traits: list[Trait]                  # Stable personality traits
    preferences: CommunicationPrefs      # Tone, length, formality
    topics_of_interest: list[Topic]      # Weighted topic affinities
    behavioral_patterns: list[Pattern]   # Time patterns, routines
    emotional_baseline: EmotionProfile   # Typical emotional state
    relationship_stage: RelationshipStage
    last_updated: datetime
    confidence: float                    # How much data we have
```

---

### 5. Personalization Module (`morgan-personalization`) — MAPLE "P"

Real-time adaptation layer. Lives in the request path.

```
morgan-personalization/
├── personalization/
│   ├── __init__.py
│   ├── service.py             # PersonalizationService
│   ├── context/
│   │   ├── assembler.py       # Build personalized context for LLM
│   │   ├── selector.py       # Select relevant traits for this turn
│   │   └── injector.py       # Inject traits into system prompt
│   ├── adaptation/
│   │   ├── tone.py            # Tone/style adaptation
│   │   ├── complexity.py     # Response complexity matching
│   │   └── proactive.py     # Proactive suggestion triggers
│   ├── models.py              # PersonalizedContext
│   └── config.py
├── Dockerfile
└── pyproject.toml
```

**Request-path flow:**
1. Receives `FusedPerception` + `UserModel` + conversation history
2. `selector.py` picks traits relevant to this turn (budget-aware, not full dump)
3. `injector.py` builds the system prompt section with trait signals
4. `assembler.py` produces `PersonalizedContext` for the Reasoning Engine

---

### 6. Skill Engine (`morgan-skills`)

Self-evolving skill management based on SkillOpt principles.

```
morgan-skills/
├── skills/
│   ├── __init__.py
│   ├── engine.py              # SkillEngine (main coordinator)
│   ├── registry.py            # Skill registry (discover, load, version)
│   ├── executor.py            # Execute a skill (inject into LLM context)
│   ├── optimizer.py           # SkillOpt wrapper (offline training)
│   ├── validator.py           # Validation gate for skill updates
│   ├── models.py              # Skill, SkillVersion, SkillMetrics
│   ├── bundled/               # Built-in skill documents
│   │   ├── conversation.md    # Core conversation skill
│   │   ├── empathy.md         # Emotional support skill
│   │   ├── research.md        # Web research skill
│   │   ├── coding.md          # Code assistance skill
│   │   ├── planning.md        # Task planning skill
│   │   └── calendar.md        # Calendar management skill
│   ├── agents/
│   │   ├── spawner.py         # Agent spawning from skill definitions
│   │   ├── definition.py      # AgentDefinition (YAML frontmatter)
│   │   └── builtin/           # Built-in agent definitions
│   └── config.py
├── Dockerfile
└── pyproject.toml
```

**Skill lifecycle:**
1. Author writes `skill.md` with YAML frontmatter (name, triggers, tools, model)
2. Skill is registered in the registry with version tracking
3. At inference: `executor.py` injects the skill into the LLM's system prompt
4. Trajectory data is collected during execution
5. Periodically: `optimizer.py` runs SkillOpt training loop offline
6. `validator.py` gates the update — new version only accepted if it beats current
7. Updated `best_skill.md` is deployed to registry

---

### 7. Reasoning Engine (`morgan-reasoning`)

The central LLM orchestrator. Replaces the current god-class `ConversationOrchestrator`.

```
morgan-reasoning/
├── reasoning/
│   ├── __init__.py
│   ├── engine.py              # ReasoningEngine (thin orchestrator)
│   ├── llm/
│   │   ├── router.py         # Model routing (fast/strong/vision)
│   │   ├── client.py         # Unified LLM client (OpenAI-compat)
│   │   ├── fallback.py       # Multi-level fallback chain
│   │   └── models.py         # LLMRequest, LLMResponse
│   ├── planning/
│   │   ├── planner.py        # Multi-step task decomposition
│   │   ├── reflection.py     # Self-critique and correction
│   │   └── models.py         # Plan, Step, PlanResult
│   ├── context/
│   │   ├── builder.py        # Assemble full context window
│   │   ├── compactor.py      # Token-aware compaction
│   │   └── models.py         # ContextWindow
│   ├── response/
│   │   ├── generator.py      # Response generation
│   │   ├── formatter.py      # Output formatting (markdown, code, etc.)
│   │   └── quality.py        # Response quality scoring
│   └── config.py
├── Dockerfile
└── pyproject.toml
```

**Key changes from current:**
- **No god class** — `engine.py` is a thin pipeline coordinator (~200 lines max)
- LLM client is **one unified wrapper** (not scattered across services)
- Planning/reflection are explicit modules, not buried in the orchestrator
- Quality scoring feeds back into SkillOpt trajectory data

---

### 8. Tool Executor (`morgan-tools`)

Clean, permission-gated tool execution.

```
morgan-tools/
├── tools/
│   ├── __init__.py
│   ├── executor.py            # ToolExecutor (registry + execution)
│   ├── base.py                # BaseTool protocol
│   ├── permissions.py         # Unified permission model (one enum, one gate)
│   ├── builtin/
│   │   ├── calculator.py
│   │   ├── file_read.py
│   │   ├── bash.py
│   │   ├── web_search.py
│   │   ├── memory_search.py
│   │   └── fetch_url.py
│   ├── models.py              # ToolResult, ToolContext, ToolSchema
│   └── config.py
├── Dockerfile
└── pyproject.toml
```

**Key change:** One `PermissionMode` enum, one `PermissionGate` — no more tools/permissions vs security/permission_modes split.

---

### 9. MCP Client Hub (`morgan-mcp-hub`)

MCP protocol client that connects to external MCP servers for integrations.

```
morgan-mcp-hub/
├── mcp_hub/
│   ├── __init__.py
│   ├── hub.py                 # MCPHub (discover, connect, route)
│   ├── client.py              # MCP client (stdio / SSE / WebSocket)
│   ├── registry.py            # Registered MCP servers + capabilities
│   ├── auth/
│   │   ├── oauth.py           # OAuth2 flows for MCP servers
│   │   └── tokens.py         # Token storage and refresh
│   ├── adapters/
│   │   ├── calendar.py       # Calendar abstraction over MCP
│   │   ├── email.py          # Email abstraction over MCP
│   │   └── search.py         # Web search abstraction
│   ├── models.py              # MCPServer, MCPTool, MCPResource
│   └── config.py              # MCP server configuration
├── Dockerfile
└── pyproject.toml
```

**How calendar/email works:**
- User configures MCP servers in `config.yaml` (e.g., `renfield-mcp-calendar`, `outlook-mcp`, `google-workspace-mcp`)
- `hub.py` connects to configured servers, discovers their tools
- `adapters/calendar.py` provides a unified `CalendarAdapter` interface
- The Reasoning Engine calls `mcp_hub.calendar.list_events()` — hub routes to the right MCP server
- OAuth tokens are managed by `auth/oauth.py`, stored encrypted in Redis
- New integrations = just add an MCP server to config, no code changes

**Supported MCP servers (out of the box config):**
- `renfield-mcp-calendar` — Google Calendar, Exchange, CalDAV
- `outlook-mcp` — Microsoft 365 (mail, calendar, contacts, tasks)
- `google-workspace-mcp` — Gmail, Calendar, Drive, Docs
- `speechpulse` — Audio emotion analysis (MCP-compatible)
- Custom MCP servers via config

---

### 10. Event Bus (`morgan-events`)

Shared library (not a service) for typed inter-module events.

```
morgan-events/
├── events/
│   ├── __init__.py
│   ├── bus.py                 # EventBus (Redis Streams backend)
│   ├── types.py               # All event type definitions
│   ├── models.py              # Event base class, envelope
│   ├── handlers.py            # Handler registration decorators
│   └── config.py
└── pyproject.toml
```

**Event types:**

```python
class EventType(str, Enum):
    # Perception
    MESSAGE_RECEIVED = "message.received"
    AUDIO_ANALYZED = "audio.analyzed"
    PERCEPTION_COMPLETE = "perception.complete"

    # Memory
    MEMORY_STORED = "memory.stored"
    MEMORY_RECALLED = "memory.recalled"
    CONTEXT_COMPACTED = "context.compacted"

    # Learning
    TRAIT_EXTRACTED = "trait.extracted"
    USER_MODEL_UPDATED = "user_model.updated"
    SKILL_OPTIMIZED = "skill.optimized"

    # Reasoning
    RESPONSE_GENERATED = "response.generated"
    TOOL_INVOKED = "tool.invoked"
    PLAN_CREATED = "plan.created"

    # Lifecycle
    SESSION_START = "session.start"
    SESSION_END = "session.end"
    HEARTBEAT = "heartbeat"
```

---

## Request Flow (end-to-end)

```
1. User sends message (text/audio/image)
       │
2. Gateway receives, authenticates, creates session context
       │
3. Perception Module analyzes input
   ├── Audio → Whisper ASR + Wav2Vec2 emotion + prosody sarcasm
   ├── Text → intent + entities + sentiment
   └── Fusion → FusedPerception
       │
4. Personalization Module (real-time path)
   ├── Retrieves UserModel from Learning store
   ├── Selects relevant traits for this turn
   └── Produces PersonalizedContext
       │
5. Memory Module retrieves context
   ├── Semantic search for relevant memories
   ├── Conversation history (compacted if needed)
   └── Workspace context (SOUL.md, MEMORY.md)
       │
6. Skill Engine selects active skill
   ├── Match skill triggers to current intent
   └── Load best_skill.md for matched skill
       │
7. Reasoning Engine assembles and generates
   ├── Build context window: perception + personality + memories + skill
   ├── Route to appropriate LLM (fast/strong/vision)
   ├── If tool needed → Tool Executor or MCP Hub
   ├── If multi-step → Planner decomposes, executes steps
   └── Generate response with quality scoring
       │
8. Post-response (async, off request path)
   ├── Memory Module stores conversation turn
   ├── Event Bus emits RESPONSE_GENERATED
   ├── Trajectory collector logs rollout for SkillOpt
   └── Learning Module queues interaction for batch processing
       │
9. Gateway formats and returns response to client
```

---

## Docker Compose (Development)

```yaml
# docker/docker-compose.yml
name: morgan-brain

services:
  # --- Core Brain Modules ---
  gateway:
    build:
      context: ..
      dockerfile: morgan-gateway/Dockerfile
      target: development
    ports:
      - "8080:8080"
    environment:
      MORGAN_ENV: development
      MORGAN_REDIS_URL: redis://redis:6379/0
      MORGAN_QDRANT_URL: http://qdrant:6333
    depends_on:
      redis: { condition: service_healthy }
      qdrant: { condition: service_healthy }
    volumes:
      - ../morgan-gateway:/app/morgan-gateway
      - workspace:/home/morgan/.morgan

  perception:
    build:
      context: ..
      dockerfile: morgan-perception/Dockerfile
      target: development
    environment:
      MORGAN_WHISPER_MODEL: base
      MORGAN_EMOTION_MODEL: wav2vec2-emotion
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]    # GPU for audio/vision models
    volumes:
      - models:/app/models

  memory:
    build:
      context: ..
      dockerfile: morgan-memory/Dockerfile
      target: development
    environment:
      MORGAN_QDRANT_URL: http://qdrant:6333
      MORGAN_REDIS_URL: redis://redis:6379/0
    depends_on:
      qdrant: { condition: service_healthy }
      redis: { condition: service_healthy }
    volumes:
      - workspace:/home/morgan/.morgan
      - memory-data:/app/data

  learning:
    build:
      context: ..
      dockerfile: morgan-learning/Dockerfile
      target: development
    environment:
      MORGAN_QDRANT_URL: http://qdrant:6333
      MORGAN_REDIS_URL: redis://redis:6379/0
      MORGAN_SKILLOPT_ENABLED: "true"
    depends_on:
      memory: { condition: service_started }
    volumes:
      - skills:/app/skills
      - user-models:/app/user-models

  personalization:
    build:
      context: ..
      dockerfile: morgan-personalization/Dockerfile
      target: development
    environment:
      MORGAN_REDIS_URL: redis://redis:6379/0
    depends_on:
      learning: { condition: service_started }

  skills:
    build:
      context: ..
      dockerfile: morgan-skills/Dockerfile
      target: development
    environment:
      MORGAN_REDIS_URL: redis://redis:6379/0
    volumes:
      - skills:/app/skills

  reasoning:
    build:
      context: ..
      dockerfile: morgan-reasoning/Dockerfile
      target: development
    environment:
      MORGAN_LLM_ENDPOINT: http://host.docker.internal:11434/v1
      MORGAN_LLM_MODEL: ${MORGAN_LLM_MODEL:-qwen3.5:35b}
      MORGAN_LLM_FAST_MODEL: ${MORGAN_LLM_FAST_MODEL:-gemma3:12b}

  tools:
    build:
      context: ..
      dockerfile: morgan-tools/Dockerfile
      target: development
    environment:
      MORGAN_REDIS_URL: redis://redis:6379/0

  mcp-hub:
    build:
      context: ..
      dockerfile: morgan-mcp-hub/Dockerfile
      target: development
    environment:
      MORGAN_MCP_CONFIG: /app/config/mcp-servers.yaml
    volumes:
      - ./config/mcp-servers.yaml:/app/config/mcp-servers.yaml:ro
      - mcp-tokens:/app/tokens

  # --- Infrastructure ---
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
    volumes:
      - redis-data:/data

  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:6333/readyz"]
    volumes:
      - qdrant-storage:/qdrant/storage

  # --- Monitoring (optional) ---
  prometheus:
    image: prom/prometheus:latest
    ports: ["9090:9090"]
    profiles: [monitoring]
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml:ro

  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    profiles: [monitoring]

volumes:
  redis-data:
  qdrant-storage:
  workspace:
  memory-data:
  models:
  skills:
  user-models:
  mcp-tokens:
```

---

## Shared Libraries (internal packages, not services)

```
morgan-shared/
├── shared/
│   ├── __init__.py
│   ├── config.py              # Unified config loader (single MORGAN_ prefix)
│   ├── models/
│   │   ├── base.py            # Base models (timestamps, IDs)
│   │   ├── emotion.py         # EmotionType, EmotionState (shared vocabulary)
│   │   ├── message.py         # Message, Conversation
│   │   └── user.py            # UserProfile, UserModel
│   ├── events/
│   │   ├── bus.py             # EventBus (Redis Streams)
│   │   ├── types.py           # EventType enum
│   │   └── models.py          # Event envelope
│   ├── exceptions.py          # MorganError hierarchy (no builtin shadowing)
│   ├── logging.py             # Unified structlog config
│   ├── singleton.py           # SingletonFactory (used everywhere)
│   └── utils.py               # Common utilities
└── pyproject.toml
```

**Key fixes from current codebase:**
- `ConnectionError` / `TimeoutError` renamed to `MorganConnectionError` / `MorganTimeoutError`
- **One singleton pattern** (`SingletonFactory`) used by all modules
- **One logging setup** (structlog everywhere, no `print()` for errors)
- **One config system** (`MORGAN_` prefix, pydantic-settings, no dual config)

---

## Migration Strategy

### Phase 1: Foundation (Week 1-2)
1. Create `morgan-shared` with unified config, events, models, exceptions
2. Create `morgan-gateway` (extract from morgan-server, proper auth + DI)
3. Create `morgan-memory` (consolidate MemoryProcessor + MemoryService + workspace + compaction + reranking into one)
4. Docker compose with gateway + memory + infrastructure

### Phase 2: Brain Modules (Week 3-4)
5. Create `morgan-reasoning` (extract from ConversationOrchestrator, slim it to ~200 lines)
6. Create `morgan-tools` (clean extract, unified permissions)
7. Create `morgan-personalization` (extract from intelligence/emotions → real-time path only)
8. Event bus wiring between modules

### Phase 3: Learning & Skills (Week 5-6)
9. Create `morgan-learning` (async extractors + SkillOpt integration)
10. Create `morgan-skills` (SkillOpt runtime + skill registry + agent spawning)
11. Wire trajectory collection into reasoning engine
12. First SkillOpt training run on conversation data

### Phase 4: Perception & Integrations (Week 7-8)
13. Create `morgan-perception` (Whisper ASR + Wav2Vec2 emotion + sarcasm)
14. Create `morgan-mcp-hub` (calendar, email, search via MCP)
15. End-to-end multimodal flow (audio → perception → reasoning → response)
16. MCP server configuration for Google/Outlook

### Phase 5: Polish & Optimize (Week 9-10)
17. CLI client rewrite (thin, uses gateway API)
18. Web UI (optional)
19. Comprehensive test suite
20. Production Docker compose + Swarm/K8s manifests
21. Documentation

---

## What Gets Reused vs. Rewritten

### Reuse (with refactoring)
- `services/llm/` → becomes `morgan-reasoning/reasoning/llm/client.py`
- `services/embeddings/` → becomes `morgan-memory/memory/indexing/embedder.py`
- `tools/` → becomes `morgan-tools/tools/` (mostly as-is, unified permissions)
- `channels/` adapters → become `morgan-gateway/gateway/channels/`
- `workspace/` file I/O → becomes `morgan-memory/memory/stores/workspace.py`
- `config/defaults.py` → becomes `morgan-shared/shared/config.py`
- `vector_db/client.py` → becomes `morgan-memory/memory/stores/vector.py`
- `search/reranker.py` → becomes `morgan-memory/memory/retrieval/reranker.py`

### Rewrite from scratch
- `core/application/orchestrators.py` (god class → thin `ReasoningEngine`)
- `core/assistant.py` (god class → distributed across modules)
- `intelligence/` (split into Perception + Learning + Personalization)
- `memory/memory_processor.py` + `core/memory.py` (merge into single MemoryService)
- `morgan-server/app.py` (lifespan + DI → proper FastAPI patterns)
- All configuration (single unified system)
- All singletons (consistent SingletonFactory)
- All logging (consistent structlog)
- Docker setup (clean per-module Dockerfiles)

### Remove
- `morgan-rag/morgan/companion/` (absorbed into Personalization)
- `morgan-rag/morgan/communication/` (absorbed into Personalization)
- `morgan-rag/morgan/personality/` (absorbed into Learning)
- `morgan-rag/morgan/relationships/` (absorbed into Learning)
- `morgan-rag/morgan/habits/` (absorbed into Learning)
- `morgan-rag/morgan/proactive/` (absorbed into Personalization)
- `morgan-rag/morgan/expertise/` (absorbed into Skills)
- Duplicate reranking layers
- Duplicate memory systems
- Duplicate daily log implementations
- Duplicate permission enums

---

## Key Differences from Current Morgan

| Aspect | Current | V2 |
|--------|---------|-----|
| Architecture | Monolithic (morgan-rag + thin server) | Microservices (10 modules) |
| Skills | Static markdown templates | Self-evolving via SkillOpt training loop |
| Memory | 3 overlapping systems | Single MemoryService with pluggable stores |
| Emotions | God class in intelligence_engine.py | Split: Perception (detect) + Learning (track) + Personalization (adapt) |
| Audio | Not supported | Wav2Vec2 emotion, Whisper ASR, sarcasm detection |
| Calendar/Email | Not supported | MCP protocol → any MCP server |
| Config | Dual system (server vs core) | Single `MORGAN_` prefix everywhere |
| Singletons | 3 use factory, 40+ use bare globals | All use `SingletonFactory` |
| Logging | Mix of structlog + logging + print | Structlog everywhere |
| Auth | None | JWT + API keys |
| DI | Module globals | FastAPI `Depends()` |
| Orchestrator | 1600-line god class | ~200-line thin pipeline |
| Inter-module | Direct imports | Event bus (Redis Streams) |
| Docker | 2 divergent compose stacks | Single clean compose |

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.12+ |
| Web framework | FastAPI + Uvicorn |
| Event bus | Redis Streams (upgrade path: NATS) |
| Vector DB | Qdrant |
| Cache | Redis |
| LLM | Ollama (local) / OpenAI-compat API |
| ASR | Whisper (local or API) |
| Audio emotion | Wav2Vec2 + ONNX Runtime |
| Embeddings | Ollama / sentence-transformers |
| Reranking | CrossEncoder (single layer) |
| Skill optimization | SkillOpt (pip install skillopt) |
| MCP client | mcp Python SDK |
| Containerization | Docker + Docker Compose |
| Monitoring | Prometheus + Grafana |
| Testing | pytest + hypothesis |
| CI/CD | GitHub Actions |
