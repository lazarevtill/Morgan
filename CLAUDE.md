# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Morgan is a self-hosted, privacy-first **personal assistant that knows and learns from you**.
The codebase was reset to a greenfield design (`morgan-brain/`) on 2026-06-07. The previous
monolithic implementation is archived in the git branch/tag **`legacy/v0.0.3-monolith`** and is
the source for any selectively ported code.

**Key principle:** Quality over speed (5–10 s thoughtful responses acceptable). All processing local.

### Read these first
- **Design authority:** `docs/superpowers/specs/2026-06-07-morgan-brain-design.md`
- **Background/rationale:** `docs/ARCHITECTURE_V2.md`
- Implementation lives under `morgan-brain/`.

## Architecture (one package, three services)

Built on **MAPLE** (Memory ≠ Learning ≠ Personalization — three mechanisms, three timescales)
and **SkillOpt** (skills as trainable markdown). One installable package `morgan_brain` runs as
three processes:

| Service | Role | Status |
|---------|------|--------|
| `brain-api` | request path: Perception → Personalization → Memory → Skills → Reasoning → Tools | active |
| `learning-worker` | async: trait extraction, UserModel, SkillOpt, consolidation, pattern mining | active |
| `perception-gpu` | voice/vision (Whisper, Wav2Vec2) — interface defined, **not built** | deferred |

**The seam is the contract.** Every module implements a `Protocol` in
`morgan_brain/interfaces/` and is reachable only through it. `core/orchestrator.py` depends on
those protocols, never on concretions — which is what makes a module swappable (text→audio
perception) and promotable (in-proc → its own service) with no code change.

### Package map (`morgan-brain/morgan_brain/`)
- `config.py` — the single `MORGAN_`-prefixed settings source (`get_settings()`).
- `interfaces/` — Protocols: Perception, MemoryStore, Learner, Personalizer, Reasoner,
  SkillEngine, ToolExecutor, EventBus. **Change a contract here before changing an implementation.**
- `models/` — shared domain models; **everything that persists is `user_id`-keyed** (`UserScoped`).
- `bus/` — event bus; in-proc and Redis-Streams backends share one interface (`get_event_bus()`).
- `security/` — the single `MemoryGate` (all memory access) and single `PermissionMode`/`PermissionGate`.
- `modules/` — domain logic: perception, memory, learning, personalization, reasoning, skills,
  tools, mcp, proactivity. Each `__init__.py` states its responsibility, owner service, and phase.
- `core/orchestrator.py` — the thin cognitive loop (design spec §6). Coordinates only; owns no domain logic.
- `apps/` — entrypoints: `brain_api`, `learning_worker`, `perception_gpu`.
- `clients/cli/` — thin terminal client over HTTP.

### Non-negotiable invariants
- **Hot path reads, cold path writes.** The request path (orchestrator steps 2–6) only *reads*
  learned knowledge; all learning happens off-path in `learning-worker` (step 7 publishes an
  event, never blocks the response). Do not add synchronous "learn now" calls to the request path.
- **All memory access goes through `MemoryGate`.** No module touches a `MemoryStore` directly.
- **Facts evolve, they don't overwrite.** Semantic facts are `TemporalFact`s with
  `valid_from`/`valid_to`/`superseded_by`. Update = close the old interval, open a new one.
- **Actor attribution.** Every memory records `MemorySource` (user_stated / agent_inferred /
  tool_observed). Never treat an inference as a user-stated fact.
- **One of each.** One config system, one event-bus interface, one permission model, one rerank
  layer, one `structlog` logger. The V2 doc exists largely to kill the old duplications.

## Build & Development Commands

```bash
cd morgan-brain
cp .env.example .env
pip install -e ".[dev]"

# Run services (separate terminals)
docker compose up -d redis qdrant
python -m morgan_brain.apps.brain_api          # http://localhost:8080/health
python -m morgan_brain.apps.learning_worker

# Tests
pytest                                          # all
pytest tests/unit/test_foundation.py            # one file
pytest tests/unit/test_foundation.py::test_everything_is_user_scoped -v

# Quality
black .            # line-length 100
ruff check .
mypy morgan_brain  # strict
```

Optional extras: `pip install -e ".[mcp]"` · `[channels]` · `[scheduling]` · `[tokens]` ·
`[perception]` (deferred voice/vision).

## Configuration

All env vars are `MORGAN_`-prefixed (see `morgan-brain/.env.example`). Notable:
`MORGAN_EVENT_BUS` (`inproc`|`redis`), `MORGAN_OWNER_USER_ID`, `MORGAN_LLM_ENDPOINT`,
`MORGAN_QDRANT_URL`, `MORGAN_REDIS_URL`, `MORGAN_TEMPORAL_DB_URL`. There is exactly one
`Settings` object — access it via `get_settings()`, never re-read env directly.

## External Services
- **Ollama** — LLM (OpenAI-compatible). **Qdrant** — vectors. **Redis** — cache + event bus.
- **SQLite→Postgres** — bi-temporal fact store (`MORGAN_TEMPORAL_DB_URL`).

## Working in this repo
- Single-owner now, but **never hardcode the owner** — key everything by `user_id` so the
  multi-tenant flip stays a config change.
- New external integration (calendar/email/etc.) = an MCP server in config, **not** native code.
- Memory changes must be measured by the `tests/memory_quality/` harness
  (LoCoMo/LongMemEval-style: single-hop, multi-hop, temporal, knowledge-update).
- Follow the phase plan in the design spec §14; each phase must leave a working assistant.
- Python 3.12+, line-length 100, prefer async (`agenerate`/`aencode`).
