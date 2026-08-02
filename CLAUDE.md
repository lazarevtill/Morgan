# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Morgan is a self-hosted, **self-learning, provider-agnostic personal agent kernel** — it owns its
owner's identity, memory, learning, and policy, and the chat assistant is one app on top of it, not
the product. The implementation lives under `morgan-brain/` (one package, two services).
**All phases (0–5) + Wave 0.5 + the self-learning engine are built and green** (633 tests, 7
skipped, mypy-strict clean). LoRA fine-tuning is deferred by design (only if the 4-condition
escalation test in the self-learning decision record fires). The previous monolith is archived in
the git tag **`legacy-v0.0.3-monolith`** (branch `origin/legacy/v0.0.3-monolith`).

**Key principle:** Quality over speed (5–10 s thoughtful responses acceptable). Provider-agnostic,
runs fully local (Ollama/llama.cpp/vLLM) or against any OpenAI-compatible remote.

### Read these first
- **Status (authoritative):** `docs/ROADMAP.md`
- **Run guide / endpoints / config:** `docs/WIRING.md`
- **Current direction (authoritative):** `docs/superpowers/specs/2026-08-02-morgan-reshape-local-first-design.md`
  — diagnosis, target architecture, and milestone plan for the local-first reshape.
- **Design authority (kernel semantics):** `docs/superpowers/specs/2026-06-07-morgan-brain-design.md`
- **Decision records:** `docs/superpowers/specs/2026-06-08-self-learning-decision.md`,
  `…-platform-architecture-decision.md`

### Current direction
Morgan is mid-reshape per the design spec linked above: one SQLite database (episodics, facts,
entities, history, signals, FTS5, sqlite-vec) behind `MemoryGate` on both the hot and cold path,
llama.cpp as the default provider, project-scoped memory, and a `morgan` CLI + MCP server + Python
library as the first-class surfaces. None of the reshape milestones are implemented yet — do not
describe any of it as shipped until its milestone lands.

## Architecture (one package, two services)

Built on **MAPLE** (Memory ≠ Learning ≠ Personalization — three mechanisms, three timescales)
and **SkillOpt** (skills + champion preprompt as trainable, validation-gated state). One
installable package `morgan_brain` runs as up to two processes:

| Service | Role | Status |
|---------|------|--------|
| `brain-api` | request path: perceive → personalize → recall → skills → tools → reason → store → signal; hosts the REST/SSE gateway | active |
| `learning-worker` | async: consolidate episodics → valid-time facts, mine signals, eval-gated GEPA optimizer, nightly scheduler | active |

**The seam is the contract.** Every module implements a `Protocol` in
`morgan_brain/interfaces/` and is reachable only through it. `core/orchestrator.py` depends on
those protocols, never on concretions — which is what makes a module swappable and promotable
(in-proc → its own service) with no code change.

### Package map (`morgan-brain/morgan_brain/`)
Hot path = `brain-api` request path; cold path = `learning-worker`.
- `config.py` — the single `MORGAN_`-prefixed settings source (`get_settings()`).
- `interfaces/` — Protocols: llm, perception, memory, learning, personalization, reasoning,
  skills, tools, events. **Change a contract here before changing an implementation.**
- `models/` — shared domain models; **everything that persists is `user_id`-keyed** (`UserScoped`).
- `bus/` — event bus; in-proc and Redis-Streams backends share one interface.
- `security/` — the single `MemoryGate` (all memory access) and single `PermissionMode`/`PermissionGate`.
- `providers/` — adapters, wire types, capability registry, role router (fast/strong/vision/
  reflection/judge), structured-output ladder. **The only place a provider SDK is imported**
  (hot path).
- `modules/` — domain logic. **Hot path:** perception (text), memory, personalization, reasoning,
  skills (select), tools. Each `__init__.py` states its responsibility, owner service, and wiring.
- `learning/` — signal capture, consolidation, profile, `ChampionTrainer`, optimizer (cold path;
  the recorder runs on the cold path of each turn).
- `learning_lifecycle/` — `PromptRegistry` + `Optimizer` seam (local SQLite or MLflow backend).
- `eval/` — 3-layer golden eval harness + calibrated cross-family judge — the self-learning gate.
- `scheduling/` — CronService + nightly learning jobs (worker).
- `core/orchestrator.py` — the thin cognitive loop (design spec §6). Coordinates only; owns no domain logic.
- `composition.py` — wires concrete implementations into the orchestrator + app context.
- `apps/` — entrypoints: `brain_api`, `learning_worker`.

### API surface (`brain-api`)
`/health` (open). All `/api/*` require `Authorization: Bearer <MORGAN_API_KEY>` (or `X-API-Key`)
**when a key is set**: `POST /api/chat`, `POST /api/chat/stream` (SSE, terminal `[DONE]`),
`POST /api/feedback` (`kind`: edit|retry|thumb), `GET/POST /api/tools[/{name}]`,
`GET/POST /api/skills[/{name}]`, `GET /api/profile`.

### Non-negotiable invariants
- **Hot path reads, cold path writes.** The request path (orchestrator steps 2–6) only *reads*
  learned knowledge; all learning happens off-path in `learning-worker` (step 7 publishes an
  event, never blocks the response). Do not add synchronous "learn now" calls to the request path.
- **All memory access goes through `MemoryGate`.** No module touches a `MemoryStore` directly.
- **Facts evolve, they don't overwrite.** Semantic facts are `TemporalFact`s with
  `valid_from`/`valid_to`/`superseded_by`. Update = close the old interval, open a new one.
- **Actor attribution.** Every memory records `MemorySource` (user_stated / agent_inferred /
  tool_observed). Never treat an inference as a user-stated fact.
- **One of each.** One config system, one event-bus interface, one permission model, one
  `structlog` logger.
- **Provider SDKs are isolated to `providers/adapters/`.** Nothing above the provider layer imports
  `openai`/`anthropic`/etc. directly; the brain talks to `ChatClient` seams and a role router.
  `composition.py` currently constructs `OllamaEmbedder` directly rather than routing through the
  seam, so composition is not yet provider-neutral end to end — see the reshape design spec §4.2.
- **Self-learning is eval-gated.** No learned update (champion preprompt or, later, weights) ships
  unless it beats the current version on the held-out 3-layer eval. Optimize a *candidate*, never
  mutate the live champion; promote only on a strict beats-current win; keep versions for rollback.

### How the self-learning loop works (`learning-worker`, off the request path)
1. Every turn records training signals on the cold path (edits > retries > thumbs; a base signal
   for every turn). Eval items are firewalled from what the assistant may consolidate.
2. **Consolidate:** recent episodics → durable valid-time facts (ADD/UPDATE/DELETE/NOOP,
   contradiction → supersede via `invalid_at`, confidence decay). Scheduled nightly.
3. **Optimize (GEPA, gated):** mine high-value signals → the **reflection** model proposes an
   improved champion preprompt → score on the golden eval → promote only on a full-valset win.
   The champion is a versioned string prepend (zero inference-time cost), read once at startup.
4. **Personalize (hot path):** `AdaptivePersonalizer` injects the compact profile + turn-relevant
   traits every turn — this is where learning becomes visible.

**Deferred by design:** LoRA (only if the 4-condition escalation test fires).

## Build & Development Commands

```bash
cd morgan-brain
cp .env.example .env
pip install -e ".[dev]"

# Run services (separate terminals)
docker compose up -d redis qdrant
python -m morgan_brain.apps.brain_api                                  # http://localhost:8080/health
MORGAN_ENABLE_SCHEDULING=true python -m morgan_brain.apps.learning_worker

# Tests
pytest -q                                       # 633 passed, 7 skipped
pytest tests/unit/test_foundation.py            # one file
pytest tests/unit/test_foundation.py::test_everything_is_user_scoped -v

# Quality (CI runs all of these on Python 3.12)
ruff check .
ruff format --check .          # line-length 100
mypy morgan_brain              # strict
```

Optional extras: `pip install -e ".[learning]"` (MLflow GEPA) · `[scheduling]` (APScheduler) ·
`[tracing]` (slim mlflow-tracing) · `[tokens]` (tiktoken).

## Configuration

All env vars are `MORGAN_`-prefixed (see `morgan-brain/.env.example` for the full list). Notable:
`MORGAN_API_KEY` (set a real key before remote exposure), `MORGAN_EVENT_BUS` (`inproc`|`redis`),
`MORGAN_VECTOR_BACKEND` (`memory`|`qdrant`), `MORGAN_LEARNING_BACKEND` (`local`|`mlflow`),
`MORGAN_ROLE_BINDINGS`/`MORGAN_PROVIDERS` (mix backends, add a reflection/judge model),
`MORGAN_ENABLE_SCHEDULING`. There is exactly one `Settings` object — access it via
`get_settings()`, never re-read env directly.

## External Services
- **Ollama / any OpenAI-compatible endpoint** — LLM + embeddings. **Qdrant** — vectors
  (`MORGAN_VECTOR_BACKEND=qdrant`; default `memory` is ephemeral). **Redis** — cache + Redis-Streams bus.
- **SQLite→Postgres** — valid-time fact store (`MORGAN_TEMPORAL_DB_URL`); MLflow tracking store
  when `MORGAN_LEARNING_BACKEND=mlflow` (telemetry forced off).

## Working in this repo
- Single-owner now, but **never hardcode the owner** — key everything by `user_id` so the
  multi-tenant flip stays a config change.
- Memory changes must be measured by the `tests/memory_quality/` harness
  (LoCoMo/LongMemEval-style: single-hop, multi-hop, temporal, knowledge-update).
- Any self-learning change must pass the `tests/eval/` golden gate — no promotion without a
  beats-current win.
- Python 3.12+, line-length 100, prefer async (`agenerate`/`aencode`). Keep `main` green
  (pytest + ruff check + ruff format --check + mypy strict).
