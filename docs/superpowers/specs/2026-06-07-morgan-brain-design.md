# Morgan Brain — Design Spec

**Date:** 2026-06-07
**Status:** Approved (conversation), pending written review
**Supersedes:** `docs/ARCHITECTURE_V2.md` (deleted 2026-08-02; see
[the local-first reshape design](2026-08-02-morgan-reshape-local-first-design.md))
**Predecessor codebase:** archived in git tag `legacy-v0.0.3-monolith`

---

## 1. Goal

A self-hosted, privacy-first **personal assistant that knows and learns from its user** —
where every interaction measurably improves how well it knows you, and that knowledge
visibly changes the next response. "Knows me" is defined concretely as:

> **stable traits + evolving facts + learned procedures + emotional baseline**, all keyed to one user.

Quality over speed (5–10 s thoughtful responses acceptable). All processing local.

## 2. Grounding (research)

- **MAPLE** (AAMAS 2026, arXiv:2602.13258) — Memory ≠ Learning ≠ Personalization; three
  mechanisms on three timescales. Reference impl uses files + an LLM extraction prompt and
  still lifts trait incorporation 45% → 75%.
- **SkillOpt** (Microsoft 2026, arXiv:2605.23904) — skills as *trainable markdown*, improved
  by trajectory feedback behind a validation gate, zero inference-time cost.
- **2026 memory landscape** — memory splits into episodic / semantic / procedural; the open
  problems are **temporal evolution** (not overwrite), **staleness**, and **identity**.
  Retrieval is multi-signal (vector + BM25 + entity). Benchmarks: LoCoMo, LongMemEval, BEAM.

## 3. Locked decisions

| Decision | Choice |
|----------|--------|
| **Topology** | 3 services: `brain-api` (request path), `learning-worker` (async), `perception-gpu` (deferred, interface only) |
| **Tenancy** | Single-owner now, **everything keyed by `user_id`**, multi-tenant by config flip later |
| **Modalities** | Text-first; audio/vision plug into the `Perception` seam later |
| **Memory engine** | Own Qdrant + Redis substrate, adopting proven patterns (temporal facts, vector+BM25+entity, actor attribution, validation-gated skills) |
| **Migration** | Greenfield `morgan-brain`; port `llm`/`embeddings`/`tools` selectively from the archive branch; retire monolith after Phase 3 |
| **Proactivity** | Proactive core (heartbeat + cron + pattern-triggered), consent-gated |

## 4. Principles

1. **Memory ≠ Learning ≠ Personalization** — three subsystems, three timescales, three owners.
2. **Knowledge evolves, never overwrites** — facts carry validity intervals; history is queryable.
3. **Skills are trainable state** — versioned markdown improved by SkillOpt; no fine-tuning.
4. **The seam is the contract** — modules reachable only via typed Protocols + events.
5. **Proactive but consent-gated** — autonomy only via approved rules.
6. **Privacy-first, single owner, multi-tenant-ready** — one `MemoryGate`, all `user_id`-keyed.
7. **Always shippable** — every phase yields a working assistant.

## 5. Topology

- **`brain-api`** — the whole request path in one process; modules behind interfaces; in-process
  event bus (interface-identical to Redis Streams). No network hops per turn.
- **`learning-worker`** — all async/off-path work: trait extraction, user-model maintenance,
  SkillOpt training, consolidation, pattern mining. Communicates only via store + event stream.
  Can be down without breaking chat.
- **`perception-gpu`** — interface defined now, **not built**. Text analysis runs inline in
  brain-api; voice/vision later implement the same `Perception` protocol with zero downstream change.

The in-process and Redis-Streams event buses share one interface, so promoting a module from
in-proc to its own service is a deployment change, not a code change.

## 6. Cognitive loop (one turn)

```
1. Gateway      resolve session + user_id, load SessionContext
2. Perception   text → FusedPerception{intent, entities, emotion, sentiment, sarcasm?, modalities}
3. Personalize  UserModel → select traits relevant to THIS turn → PersonalizedContext (≤~15% ctx)
4. Memory       multi-signal recall (vector+BM25+entity), temporal-aware (current facts only)
                + compacted history + workspace (SOUL.md, MEMORY.md)
5. Skills       match triggers → load best_skill.md
6. Reasoning    assemble context → route LLM (fast/strong) → plan/reflect → tools/MCP → generate
7. Respond      stream to client
8. Post-turn    (async) Memory.store(turn) w/ actor attribution; emit events + trajectory;
                learning-worker queues turn for extraction
```

**Discipline:** steps 2–7 only *read* learned knowledge; step 8 only *writes* and never blocks
the response. Reads in the hot path, writes in the cold path.

## 7. Memory architecture

Four memory types:

| Type | Answers | Store |
|------|---------|-------|
| Working | "what's in play now?" | context window + Redis |
| Episodic | "what happened, when?" | temporal log (SQLite→PG) + Qdrant |
| Semantic | "what's true about user/world?" | Qdrant + temporal facts |
| Procedural | "how does Morgan do X well?" | Skill registry (markdown), owned by SkillOpt |

**Three hard problems, addressed:**

1. **Temporal / evolution-not-overwrite** — every semantic fact carries `valid_from`/`valid_to`/
   `superseded_by`. Updates close the old interval and open a new one. Recall defaults to
   currently-valid; history is queryable. Implemented as a **bi-temporal fact table** with
   entity/relation columns (grows into a Graphiti-style temporal KG only if needed).
2. **Staleness** — facts carry `confidence` + `last_confirmed`; the Learning worker decays
   confidence and flags old high-impact facts for re-confirmation.
3. **Identity** — one `user_id` anchor; an `identity` table maps channel ids (telegram/discord/cli)
   to it. Same person across devices = same memory.

**Retrieval** = vector + BM25 + entity, merged and re-ranked by a **single** rerank layer.
**Actor attribution** on every memory: `source ∈ {user_stated, agent_inferred, tool_observed}`.
**Forgetting** = importance-decay + consolidation collapse redundant episodics into semantic memory.

## 8. Learning subsystem (async worker)

Runs off the request path (session-end + periodic batch + event-triggered). Three levels:

1. **Extraction** — LLM emits `{facts, preferences, behaviors}` w/ confidence from completed sessions.
2. **User-model maintenance** — maintains the stable `UserModel`:
   ```
   UserModel(user_id): traits[], comm_prefs, topics_of_interest[], behavioral_patterns[],
                       emotional_baseline, relationship_stage, confidence
   ```
3. **Procedural learning (SkillOpt)** — collects trajectories (msg→response→satisfaction),
   runs the optimizer loop offline (rollout→reflect→propose bounded edit→**validate held-out**→
   accept only if it beats current), deploys `best_skill.md`.

Plus **consolidation** (MEMORY.md curation, dedup, decay) and **pattern mining** (feeds proactivity).
`relationship_stage` is the trust lever gating proactivity.

## 9. Personalization (request path)

Thin and stateless — reads UserModel + FusedPerception, writes nothing.
- **Selector** picks only traits relevant to this query (budget-aware, ~15% of context).
- **Injector** composes them into the system prompt as signals.
- **Adapters** tune tone, complexity, proactive-suggestion thresholds.

## 10. Skills, Tools & MCP (the platform dimension)

- **Skills** — markdown + YAML frontmatter (`name, triggers, tools, model`); bundled + user-authored;
  versioned, trainable (§8).
- **Tools** — `BaseTool` registry, **one** `PermissionMode` enum + `PermissionGate`.
- **MCP Hub** — calendar/email/search are external MCP servers; add a server to config → new
  capability, no code. OAuth tokens encrypted in Redis.
- **Agents** — spawnable sub-agents from skill definitions for multi-step work.

## 11. Proactivity (consent-gated)

- **HeartbeatManager** (jittered tick), **CronService** (scheduled jobs),
  **pattern-triggered** (from mined patterns).
- **Consent gate** — every proactive action checks an approved-rules list + `relationship_stage`;
  delivery via channels (Telegram/Discord).

## 12. Interfaces, events, data model

- Every module implements a **Protocol** (`Perception`, `MemoryStore`, `Learner`, `Personalizer`,
  `Reasoner`, `SkillEngine`, `ToolExecutor`, `EventBus`). Orchestrator depends on protocols only.
- **Typed events** (`MESSAGE_RECEIVED`, `PERCEPTION_COMPLETE`, `MEMORY_STORED`, `TRAIT_EXTRACTED`,
  `USER_MODEL_UPDATED`, `SKILL_OPTIMIZED`, `RESPONSE_GENERATED`, `HEARTBEAT`, …); in-proc now,
  Redis Streams for cross-service.
- **Data model** — all `user_id`-keyed; `MemoryGate` is the single read/write choke point.
- **One** config system (`MORGAN_` prefix, pydantic-settings), **one** singleton pattern,
  **one** structured logger (structlog).

## 13. Reliability, privacy, testing, stack

- **Degradation ladder** — rerank remote→cross-encoder→embedding→BM25; LLM strong→fast→cached;
  learning-worker down → chat unaffected; perception-gpu down → text path unaffected.
- **Privacy** — all-local processing; MCP external calls opt-in; `MemoryGate` enforces store/recall;
  user can inspect/delete any memory.
- **Testing** — unit per module (protocols make mocking trivial); integration on the loop;
  **memory-quality regression suite** modeled on LoCoMo/LongMemEval (multi-hop, temporal,
  knowledge-update); SkillOpt has its held-out validation gate by construction.
- **Stack** — Python 3.12+, FastAPI+Uvicorn, Qdrant, Redis, SQLite→Postgres, Ollama, structlog,
  pytest+hypothesis, Docker Compose. Whisper/Wav2Vec2/ONNX deferred to `perception-gpu`.

## 14. Phasing (greenfield, always-shippable)

| Phase | Ships | Port from `legacy-v0.0.3-monolith` |
|-------|-------|-----|
| 0 Foundation | skeleton: config, events, protocols, MemoryGate, data model, gateway, CLI | config patterns |
| 1 Memory + reasoning | working text assistant w/ recall (vector+BM25+entity, temporal facts) | `services/llm`, `services/embeddings`, `vector_db`, `search/reranker` |
| 2 Learning + Personalization | assistant that measurably adapts | `intelligence` emotion logic |
| 3 Skills + Tools + MCP | platform: skills, unified tools, MCP calendar/email | `tools/`, `channels/`, `workspace/` |
| 4 Proactivity | heartbeat, cron, pattern push | `scheduling/`, `proactive/` |
| 5 Perception-gpu | voice: Whisper + Wav2Vec2 + sarcasm | new |

Monolith retires after Phase 3.

## 15. Cut from V2 (YAGNI)

- 10 services → 3 (other concerns are modules behind interfaces).
- Full temporal KG → bi-temporal fact table (grows into KG only if needed).
- gRPC → HTTP/events. NATS → just Redis Streams. JWT machinery → single API key now.

## 16. Open items for written review

- Confirm bi-temporal fact table over a graph DB for v1.
- Confirm `relationship_stage` as the proactivity-trust lever.
- Confirm phase order (memory-before-skills).
