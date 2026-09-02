# Architecture

One installable package, `morgan_brain`, that runs as up to two services. Built on the
**MAPLE** decomposition (Memory ≠ Learning ≠ Personalization — three mechanisms on three
timescales) and **SkillOpt** (skills + champion preprompt as trainable, validation-gated state).

Design authority: [`design/morgan-brain.md`](design/morgan-brain.md). Current direction:
[`design/local-first-reshape.md`](design/local-first-reshape.md). The memory and governance
graft: [`design/dual-brain-memory-and-pattern-register.md`](design/dual-brain-memory-and-pattern-register.md).
Status: [`ROADMAP.md`](ROADMAP.md). Run guide: [`WIRING.md`](WIRING.md).

## Topology (two services, one package)

| Service | Role |
|---------|------|
| `brain-api` | The request path — perceive → personalize → recall → skills → tools → reason → store → signal. Hosts the REST/SSE gateway. |
| `learning-worker` | Off the request path — consolidate episodic memory into valid-time facts, mine signals, run the eval-gated GEPA optimizer, schedule nightly jobs. |

Modules talk over typed Protocols (`interfaces/`) and an event bus whose in-process and
Redis-Streams backends share one interface, so any module can be promoted to its own service
with no code change. For local use everything runs in one process (`MORGAN_EVENT_BUS=inproc`),
and the `morgan` CLI and the `morgan-mcp` server are thin adapters over the same wiring.

## Package layout (`morgan_brain/`)

| Package | Responsibility |
|---------|----------------|
| `config.py` | The single `MORGAN_`-prefixed settings source (`get_settings()`). Reads `~/.config/morgan/.env`, then `./.env`, then the environment; the one database defaults to `~/.local/share/morgan/`. |
| `logging_setup.py` | The one logging configuration: every entrypoint logs to stderr, because the CLI's stdout is its `--json` contract and the MCP server's stdout is the JSON-RPC channel. |
| `interfaces/` | Protocols — the contracts every module implements (llm, memory, learning, personalization, reasoning, skills, tools, perception, events). `llm.ProviderUnreachable` is the one error every surface reports by name. |
| `models/` | Shared domain models; everything that persists is `user_id`-keyed (`UserScoped`) and, for memories and facts, `project`-keyed. |
| `bus/` | Event bus — in-proc + Redis-Streams backends behind one interface. |
| `security/` | The single `MemoryGate` (all memory access), the unified `PermissionMode`/`PermissionGate`, and the bind guard that refuses an unauthenticated listener beyond loopback. |
| `providers/` | Provider adapters, wire types, capability registry, role router (strong/fast/judge/reflection), structured-output ladder. **The only place a provider SDK is imported.** |
| `modules/perception/` | Raw input → `FusedPerception` (text built; audio/vision not built). `text/entities.py` is the one entity extractor both paths use. |
| `modules/memory/` | Episodic + valid-time fact store on one SQLite database, and multi-signal retrieval fused by reciprocal rank — sqlite-vec vectors, an FTS5 keyword index, and an entity index, all durable. `retrieval/semantic_index.py` sits above them: schema → entity routing that narrows every signal to a candidate pool before it searches. |
| `modules/personalization/` | Request-path `AdaptivePersonalizer` — budget-aware trait selection, injected every turn — plus `persona_graph.py`, which separates intrinsic dispositions from attitudes anchored to a specific entity. Read-only on the request path. |
| `modules/reasoning/` | Context assembly + role-routed LLM call + tool loop + generation (streaming when the bound model cannot call tools). |
| `modules/skills/` | Markdown+frontmatter skill registry, trigger-matched, champion-versioned. |
| `modules/tools/` | `BaseTool` registry + executor behind the PermissionGate; SSRF/DoS-hardened built-ins. |
| `learning/` | Signal capture (recorder/signals), consolidation, profile, `ChampionTrainer`, optimizer — plus the cold-path writers for the index and persona graph (`semantic_index_builder.py`, `persona_attribution.py`, `cluster_emergence.py`) and the governance layer (`patterns.py`, `receipts.py`). |
| `learning_lifecycle/` | `PromptRegistry` + `Optimizer` seam (local SQLite or MLflow backend). |
| `eval/` | 3-layer golden eval harness + calibrated cross-family LLM judge — the self-learning gate. `gate_integrity.py` protects it from what it judges. |
| `scheduling/` | CronService + nightly learning jobs (APScheduler optional). |
| `core/` | The thin cognitive-loop orchestrator (coordinates only; owns no domain logic). |
| `composition.py` | Wires concrete implementations into the orchestrator + app context. |
| `apps/` | Entrypoints: `brain_api`, `learning_worker`. |
| `cli/` | The `morgan` terminal client — `remember`/`recall`/`facts`/`forget`/`ask`/`doctor`/`receipts`; project auto-detected from the current git repository. |
| `ports/` | `morgan-mcp` — five MCP tools over stdio or streamable-HTTP, calling the same command handlers the CLI does. |

## The cognitive loop (per turn, `brain-api`)

**perceive → personalize → recall → skills → tools → reason → store → signal.** Steps 2–6 only
**read** learned knowledge (hot path); the store + signal steps only **write** and never block the
response (cold path). Memory access goes through `MemoryGate`; facts are valid-time (update = close
the old interval, open a new one) with actor attribution on every record.

## The self-learning loop (`learning-worker`)

Off the request path: **consolidate** recent episodics into durable valid-time facts
(ADD/UPDATE/DELETE/NOOP, contradiction → supersede, confidence decay), then **eval-gated optimize** —
mine high-value signals (edits > retries > thumbs), ask the **reflection** model to propose an
improved champion preprompt, score it on the 3-layer golden eval, and promote **only on a strict
beats-current win** (versioned for instant rollback), and only when
`MORGAN_ENABLE_CHAMPION_PROMOTION` is on — it ships off. No learned update ever degrades the assistant.

## Tests (`tests/`)

`unit/` per package; `integration/` for the CLI as a subprocess, the API over ASGI, the MCP server
over raw stdio pipes, cross-process durability and erasure; `e2e/` a deterministic conversation
harness with an optional live mode; `memory_quality/` a plumbing check over a hash embedder (not a
relevance measurement); `live/` smoke tests behind `--live`. `pip install -e ".[dev]"` installs
exactly what the suite needs.
