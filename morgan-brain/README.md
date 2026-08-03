# morgan-brain

The implementation of Morgan — a self-hosted, self-learning, provider-agnostic personal agent
kernel. One installable package (`morgan_brain`) that runs as up to two services.

Built on the **MAPLE** decomposition (Memory ≠ Learning ≠ Personalization — three mechanisms on
three timescales) and **SkillOpt** (skills + champion preprompt as trainable, validation-gated
state). Design authority: [`docs/superpowers/specs/2026-06-07-morgan-brain-design.md`](../docs/superpowers/specs/2026-06-07-morgan-brain-design.md);
current direction: [`docs/superpowers/specs/2026-08-02-morgan-reshape-local-first-design.md`](../docs/superpowers/specs/2026-08-02-morgan-reshape-local-first-design.md);
status: [`docs/ROADMAP.md`](../docs/ROADMAP.md); run guide: [`docs/WIRING.md`](../docs/WIRING.md).

> The previous monolithic implementation is archived in the git tag
> `legacy-v0.0.3-monolith` (branch `origin/legacy/v0.0.3-monolith`) and is the source for any
> selectively ported code.

## Topology (2 services, one package)

| Service | Role | Status |
|---------|------|--------|
| `brain-api` | The request path — perceive → personalize → recall → skills → tools → reason → store → signal. Hosts the REST/SSE gateway. | active |
| `learning-worker` | Async, off the request path — consolidate episodic memory into valid-time facts, mine signals, run the eval-gated GEPA optimizer, schedule nightly jobs. | active |

Both services run from the single `morgan_brain` package. Modules talk over typed Protocols
(`interfaces/`) and an event bus whose in-process and Redis-Streams backends share one interface —
so any module can be promoted to its own service with no code change. For local dev everything can
run in one process (`MORGAN_EVENT_BUS=inproc`).

## Package layout (`morgan_brain/`)

| Package | Responsibility |
|---------|----------------|
| `config.py` | The single `MORGAN_`-prefixed settings source (`get_settings()`). |
| `interfaces/` | Protocols — the contracts every module implements (llm, memory, learning, personalization, reasoning, skills, tools, perception, events). |
| `models/` | Shared domain models; everything that persists is `user_id`-keyed (`UserScoped`). |
| `bus/` | Event bus — in-proc + Redis-Streams backends behind one interface. |
| `security/` | The single `MemoryGate` (all memory access), the unified `PermissionMode`/`PermissionGate`, and the bind guard that refuses an unauthenticated listener beyond loopback. |
| `providers/` | Provider adapters, wire types, capability registry, role router, structured-output ladder. **The only place a provider SDK is imported.** |
| `modules/perception/` | Raw input → `FusedPerception` (text built; audio/vision not built). |
| `modules/memory/` | Episodic + valid-time fact store on one SQLite database, and multi-signal retrieval fused by reciprocal rank — sqlite-vec vectors, an FTS5 keyword index, and an entity index, all durable. |
| `modules/personalization/` | Request-path `AdaptivePersonalizer` — budget-aware trait selection, injected every turn. |
| `modules/reasoning/` | Context assembly + role-routed LLM call + tool loop + generation. |
| `modules/skills/` | Markdown+frontmatter skill registry, trigger-matched, champion-versioned. |
| `modules/tools/` | `BaseTool` registry + executor behind the PermissionGate; SSRF/DoS-hardened built-ins. |
| `learning/` | Signal capture (recorder/signals), consolidation, profile, `ChampionTrainer`, optimizer. |
| `learning_lifecycle/` | `PromptRegistry` + `Optimizer` seam (local SQLite or MLflow backend). |
| `eval/` | 3-layer golden eval harness + calibrated cross-family LLM judge — the self-learning gate. |
| `scheduling/` | CronService + nightly learning jobs (APScheduler optional). |
| `core/` | The thin cognitive-loop orchestrator (coordinates only; owns no domain logic). |
| `composition.py` | Wires concrete implementations into the orchestrator + app context. |
| `apps/` | Entrypoints: `brain_api`, `learning_worker`. |

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
beats-current win** (versioned for instant rollback). No learned update ever degrades the assistant.

## Build / test / run

```bash
cd morgan-brain
cp .env.example .env                 # set MORGAN_API_KEY before any remote exposure
pip install -e ".[dev]"

# Quality (CI runs these on Python 3.12)
ruff check .
ruff format --check .
mypy morgan_brain                    # strict
pytest -q                            # 633 passed, 7 skipped

# Run (two-process)
docker compose up -d redis qdrant
python -m morgan_brain.apps.brain_api          # http://localhost:8080/health
MORGAN_ENABLE_SCHEDULING=true python -m morgan_brain.apps.learning_worker
```

## Optional extras

```bash
pip install -e ".[learning]"     # MLflow-backed PromptRegistry + GEPA optimizer
pip install -e ".[scheduling]"   # APScheduler (cron; in-proc scheduler otherwise)
pip install -e ".[tracing]"      # slim mlflow-tracing for the hot path
pip install -e ".[tokens]"       # tiktoken
```

## Status

**755 tests pass** (8 skipped), mypy-strict clean, bandit clean. Reshape milestones 0 and 1 are
delivered: one durable SQLite database behind `MemoryGate`, three retrieval signals that survive a
restart, project scoping enforced on every read and write, cascading `forget()`, llama.cpp as the
default provider, and the `morgan` CLI + `morgan-mcp` server as usage surfaces. Milestone 2
(concept annotation, retrieval-quality measurement) is specified, not started. LoRA fine-tuning is
deferred by design.

See [`docs/ROADMAP.md`](../docs/ROADMAP.md) for the milestone table and
[the reshape design spec](../docs/superpowers/specs/2026-08-02-morgan-reshape-local-first-design.md)
for the diagnosis and target architecture.
