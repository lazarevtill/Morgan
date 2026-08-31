# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Morgan is a self-hosted, **self-learning, provider-agnostic personal agent kernel** — it owns its
owner's identity, memory, learning, and policy, and the chat assistant is one app on top of it, not
the product. The implementation lives under `morgan-brain/` (one package, two services). Memory is
**local-first**: one SQLite database under `MORGAN_DATA_DIR` holds episodics, facts, entities,
vectors (sqlite-vec), the FTS5 keyword index, training signals, session history, and the champion
prompt registry. Redis and Qdrant are optional extras, not requirements. The default provider is
**llama.cpp** (`llama-server`), remote-first: a GPU box on the homelab reached over an overlay
network, with local loopback as the offline/dev fallback. The previous monolith is archived in the
git tag **`legacy-v0.0.3-monolith`**; the pre-reshape
platform build is archived in tag **`legacy-v0.0.4-full`**.

**Key principle:** Quality over speed (5–10 s thoughtful responses acceptable). Provider-agnostic —
any OpenAI-compatible endpoint works; Ollama remains a supported non-default provider key.

### Read these first
- **Status (authoritative):** `docs/ROADMAP.md`
- **Run guide / endpoints / config:** `docs/WIRING.md`
- **Design authority (kernel semantics):** `docs/superpowers/specs/2026-06-07-morgan-brain-design.md`
- **Local-first reshape design:** `docs/superpowers/specs/2026-08-02-morgan-reshape-local-first-design.md`
- **Decision records:** `docs/superpowers/specs/2026-06-08-self-learning-decision.md`,
  `…-platform-architecture-decision.md`

### Four surfaces
- **`brain-api`** — the REST/SSE gateway (`/api/chat`, `/api/chat/stream`, `/api/feedback`,
  `/api/tools`, `/api/skills`, `/api/profile`).
- **`learning-worker`** — the async consolidation + eval-gated optimizer process.
- **`morgan` CLI** — a terminal client (`remember`/`recall`/`facts`/`forget`/`ask`/`doctor`/
  `receipts`); project is auto-detected from the current git repository's directory name.
- **`morgan-mcp`** — an MCP server exposing five tools (`remember`, `recall`, `facts`, `forget`,
  `ask_morgan`) over stdio or streamable-HTTP with a bearer token, for any MCP client (Claude
  Code, Claude Desktop, etc.). Both the CLI and the MCP server are thin adapters over the same
  `composition.build_memory_context`/`build_app_context` wiring and the one `MemoryGate` —
  no memory logic is duplicated between them.

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
- `config.py` — the single `MORGAN_`-prefixed settings source (`get_settings()`). Derives the
  four role bindings (strong/fast/judge/reflection) and the shared SQLite path from `data_dir`.
- `interfaces/` — Protocols: llm, perception, memory, learning, personalization, reasoning,
  skills, tools, events. **Change a contract here before changing an implementation.**
- `models/` — shared domain models; **everything that persists is `user_id`-keyed** (`UserScoped`)
  and, for `Memory`/`TemporalFact`, **`project`-keyed** too.
- `bus/` — event bus; in-proc (bounded queue, drops counted) and Redis-Streams backends share
  one interface.
- `security/` — the single `MemoryGate` (every memory read/write, hot **and** cold path) and
  single `PermissionMode`/`PermissionGate`.
- `providers/` — adapters, wire types, capability registry, role router (strong/fast/judge/
  reflection), structured-output ladder. **The only place a provider SDK is imported.**
  `providers/factory.py::build_embedder` is the single seam decision point for the embedder too
  (provider vs. deterministic hash stub) — composition.py never constructs one directly.
- `modules/` — domain logic. **Hot path:** perception (text), memory, personalization, reasoning,
  skills (select), tools. Each `__init__.py` states its responsibility, owner service, and wiring.
  `modules/memory/stores/` + `retrieval/` are the durable SQLite-backed indexes (vector via
  sqlite-vec, keyword via FTS5, entity) that recall fuses with reciprocal rank fusion.
  `modules/memory/retrieval/semantic_index.py` is the **upper index** above them: schemas route
  coarsely, entities locate concretely, and `recall` narrows every signal to its candidate pool
  before searching. `modules/personalization/persona_graph.py` is the **persona graph**: intrinsic
  dispositions vs. attitudes anchored to an entity.
  `modules/perception/text/entities.py` is the one entity extractor both paths use.
- `learning/` — signal capture, consolidation, profile, `ChampionTrainer`, optimizer (cold path;
  the recorder runs on the cold path of each turn), plus the cold-path writers for the two
  brains and the governance layer: `semantic_index_builder.py`, `persona_attribution.py`,
  `cluster_emergence.py`, `patterns.py` (the register of correction *classes*), and
  `receipts.py` (why the champion is what it is).
- `learning_lifecycle/` — `PromptRegistry` + `Optimizer` seam (local SQLite or MLflow backend).
- `eval/` — 3-layer golden eval harness + calibrated cross-family judge — the self-learning gate.
  `eval/gate_integrity.py` protects it from what it judges: the gate is fingerprinted, and a
  candidate scored on a different or weaker one is refused.
- `scheduling/` — CronService + nightly learning jobs (worker).
- `core/orchestrator.py` — the thin cognitive loop (design spec §6). Coordinates only; owns no domain logic.
- `composition.py` — wires concrete implementations into the orchestrator + app context.
- `apps/` — entrypoints: `brain_api` (owns the bus lifespan — starts it on entry, stops it on
  exit), `learning_worker`.
- `cli/` — the `morgan` terminal client (`__main__.py`): `remember`/`recall`/`facts`/`forget`/
  `ask`/`doctor`/`receipts`; project is auto-detected from the current git repository's
  directory name (`cli/project.py::detect_project`).
- `ports/mcp_server.py` — the `morgan-mcp` MCP server: five tools over stdio or streamable-HTTP
  with a bearer token, calling the exact same `cli.__main__` command handlers the CLI uses.

### API surface (`brain-api`)
`/health` (open). All `/api/*` require `Authorization: Bearer <MORGAN_API_KEY>` (or `X-API-Key`)
**when a key is set**: `POST /api/chat` (project-scoped, `project` required), `POST /api/chat/stream`
(SSE, terminal `[DONE]`, threads the learned champion preprompt through `system_override`),
`POST /api/feedback` (`kind`: edit|retry|thumb, `project` required), `GET/POST /api/tools[/{name}]`,
`GET/POST /api/skills[/{name}]`, `GET /api/profile`.

### Non-negotiable invariants
- **Hot path reads, cold path writes.** The request path (orchestrator steps 2–6) only *reads*
  learned knowledge; all learning happens off-path in `learning-worker` (step 7 publishes an
  event, never blocks the response). Do not add synchronous "learn now" calls to the request path.
- **All memory access goes through `MemoryGate`.** No module touches a `MemoryStore` directly —
  this covers `forget()`, `distinct_projects()`, and fact mutation, not just `store`/`recall`.
- **Every read and write is project-scoped.** `Memory` and `TemporalFact` carry a required
  `project`; `MemoryGate` rejects an empty one. `--all-projects` / `all_projects=True` is the
  explicit cross-project escape hatch, never the default.
- **Facts evolve, they don't overwrite.** Semantic facts are `TemporalFact`s with
  `valid_from`/`valid_to`/`superseded_by`. Update = close the old interval, open a new one.
- **Actor attribution.** Every memory records `MemorySource` (user_stated / agent_inferred /
  tool_observed). Never treat an inference as a user-stated fact.
- **Routing never costs recall.** `SemanticIndex.route()` returns `None` — "search everything" —
  and never an empty list, whenever it has nothing useful to say. The candidate pool is pushed
  *into* each signal's query, never applied to its output: a post-filter over the top-k cannot
  reach a memory that ranks below the cut, which is the entire point of routing. Cross-project
  recall is never routed.
- **A situational reading is not a trait.** The persona graph keeps attitudes anchored to the
  entity they concern. Promotion to an intrinsic disposition requires recurrence across several
  *different anchors* and several *distinct sessions* — impatience with one recurring meeting is
  a fact about that meeting. An untargeted inference is dropped; only the user's own statement
  about themselves enters unanchored.
- **The gate may not be weakened by what it judges.** The eval gate is fingerprinted (item count
  and ids, judge model, scorer set, epsilon); a candidate scored on a different or weaker gate is
  refused, and a candidate whose body addresses the evaluator is refused *before* it is scored.
  Every promotion decision — including rejections and why — is recorded as a receipt.
- **One of each.** One config system, one event-bus interface, one permission model, one
  `structlog` logger, one SQLite database, one entity extractor
  (`modules/perception/text/entities.py`, used by both the hot and the cold path), one
  `SemanticIndex` instance per assembly.
- **Provider SDKs are isolated to `providers/adapters/`.** Nothing above the provider layer imports
  `openai`/`anthropic`/etc. directly; the brain talks to `ChatClient`/`Embedder` seams and a role
  router. No provider is hardcoded.
- **Self-learning is eval-gated, and disarmed by default.** No learned update (champion preprompt
  or, later, weights) ships unless it beats the current version on the held-out 3-layer eval.
  `MORGAN_ENABLE_CHAMPION_PROMOTION` defaults to `false` — the current promotion logic is a bare
  `>` on a single scored run over a 12-item golden set, too statistically weak to trust
  unattended. Optimize a *candidate*, never mutate the live champion; keep versions for rollback.

### How the self-learning loop works (`learning-worker`, off the request path)
1. Every turn records training signals on the cold path (edits > retries > thumbs; a base signal
   for every turn). Eval items are firewalled from what the assistant may consolidate.
2. **Consolidate:** recent episodics → durable valid-time facts (ADD/UPDATE/DELETE/NOOP,
   contradiction → supersede via `valid_to`/`superseded_by`, confidence decay). Scheduled nightly.
3. **Optimize (GEPA, gated, disarmed by default):** mine high-value signals → the **reflection**
   model proposes an improved champion preprompt → score on the golden eval → promote only when
   `MORGAN_ENABLE_CHAMPION_PROMOTION=true` and it wins. The champion is a versioned string prepend
   (zero inference-time cost), cached with a short TTL so a promotion reaches live traffic without
   a `brain-api` restart.
4. **Personalize (hot path):** `AdaptivePersonalizer` injects the compact profile + turn-relevant
   traits every turn, plus the persona nodes this turn activated — this is where learning becomes
   visible.

### The two brains (VoiceMem, arXiv:2608.26005 — design spec `2026-08-31-…-design.md`)
- **Left brain — the semantic upper index.** Schemas route coarsely, entities locate concretely,
  one-hop co-occurrence expands. `recall` narrows every signal to the resulting candidate pool
  *before* searching, which is what makes a small top-k dense rather than merely small. Built on
  the cold path (`semantic_index_builder.py`); re-partitioned nightly by `cluster_emergence.py`
  when queries keep activating a subgroup together and a judge agrees it is one topic.
- **Right brain — the persona graph.** Intrinsic dispositions and entity-anchored attitudes, with
  the short horizon (per turn, cold path) and the long horizon (nightly generalisation) kept
  apart. Joint retrieval: the anchor set is the turn's entities expanded one hop through the left
  brain, so an attitude toward something the turn implies but does not name still surfaces.

### Known limitations (current state, not regressions)
- **`recall` has no relevance floor.** Vector, FTS5, and entity search each return their top-k
  regardless of score, and all currently-valid facts are always included — once a project holds
  anything, a query always returns something. There is no "no matches" state except on an empty
  project.
- **`forget()` does not erase vectors under `vector_backend=qdrant`.** The sqlite-vec default is
  fully covered (same transaction as everything else); Qdrant vectors must be removed separately.
- **Both listeners serve unauthenticated when `MORGAN_API_KEY` is unset or left at `change-me`** —
  `/api/*` and the MCP HTTP transport share that policy. It is confined to loopback:
  `security/network.py::assert_safe_bind` refuses to start either surface on a non-loopback bind
  without a real key, so the open case cannot reach the overlay network.
- **Retrieval quality is unmeasured.** `tests/memory_quality/` is a stub harness over a hash
  embedder — it exercises the plumbing, not real relevance. The upper index's accuracy claim is
  the paper's, measured on LoCoMo/LongMemEval; it has not been reproduced here, and cannot be
  until that harness runs against a real embedder.
- **Entity extraction has no model behind it yet.** The deterministic extractor handles cased
  scripts (Latin, Cyrillic, CamelCase brands); scripts without letter case — Chinese, Japanese,
  Arabic, Hebrew — yield nothing rather than a guess. The model-backed path exists for schema
  classification but not yet for extraction itself.
- **An entity is classified into a schema once.** Deliberate — reclassifying nightly would let a
  slot flap, and each flap rewrites what a query can route to — but it means an early
  misclassification persists until cluster emergence re-partitions around it.
- **Persona attribution and cluster emergence need a reachable model.** Both record/promote
  nothing without one, by design: guessing at the owner's personality, or re-partitioning the
  index on a heuristic, is worse than an empty graph and an unchanged partition.

**Deferred by design:** LoRA (only if the 4-condition escalation test fires).

## Build & Development Commands

```bash
cd morgan-brain
cp .env.example .env                            # point MORGAN_LLM_ENDPOINT at your llama-server
pip install -e ".[dev]"

# The CLI — no Redis/Qdrant/Docker required, just a reachable llama-server (or
# MORGAN_EMBEDDING_BACKEND=hash for memory commands with no model server at all)
morgan doctor                                   # diagnose the local install
morgan remember "prefers terse, code-first answers"
morgan recall "how do I like answers"
morgan ask "what do you know about me"          # a full chat turn — needs the LLM endpoint

# Run the REST/SSE gateway + async worker (separate terminals; optional, for remote/multi-process use)
python -m morgan_brain.apps.brain_api                                  # http://localhost:8080/health
MORGAN_ENABLE_SCHEDULING=true python -m morgan_brain.apps.learning_worker

# Tests
pytest -q                                       # 945 passed, 8 skipped
pytest tests/unit/test_foundation.py            # one file
pytest tests/unit/test_foundation.py::test_everything_is_user_scoped -v

# Quality (CI runs all of these on Python 3.12)
ruff check .
ruff format --check .          # line-length 100
mypy morgan_brain              # strict
```

Optional extras: `pip install -e ".[mcp]"` (the `morgan-mcp` server) · `[scale]` (Redis +
Qdrant, for `MORGAN_EVENT_BUS=redis` / `MORGAN_VECTOR_BACKEND=qdrant`) · `[learning]` (MLflow
GEPA) · `[scheduling]` (APScheduler) · `[tracing]` (slim mlflow-tracing) · `[tokens]` (tiktoken).

## Configuration

All env vars are `MORGAN_`-prefixed (see `morgan-brain/.env.example` for the full list). Notable:
`MORGAN_API_KEY` (INBOUND, required for any non-loopback bind), `MORGAN_API_HOST`/`MORGAN_API_PORT`
and `MORGAN_MCP_HOST`/`MORGAN_MCP_PORT` (listener binds, loopback by default), `MORGAN_DATA_DIR` (where the
shared `morgan.db` and workspace files live), `MORGAN_LLM_ENDPOINT`/`MORGAN_LLM_API_KEY`
(OUTBOUND, to the model server — opposite direction from `MORGAN_API_KEY`), `MORGAN_EVENT_BUS`
(`inproc`|`redis`), `MORGAN_VECTOR_BACKEND` (`sqlite`|`memory`|`qdrant`), `MORGAN_LEARNING_BACKEND`
(`local`|`mlflow`), `MORGAN_ROLE_BINDINGS`/`MORGAN_PROVIDERS` (mix backends, add a
reflection/judge model), `MORGAN_ENABLE_CHAMPION_PROMOTION` (off by default), `MORGAN_ENABLE_SCHEDULING`.
There is exactly one `Settings` object — access it via `get_settings()`, never re-read env directly.

## External Services
- **Required:** an OpenAI-compatible LLM endpoint — `llama-server` (llama.cpp) by default,
  remote-first over an overlay network (NetBird), local loopback as the offline/dev fallback.
  Ollama and any other OpenAI-compatible endpoint remain supported non-default provider keys.
- **Optional:** Qdrant (`MORGAN_VECTOR_BACKEND=qdrant`; the `sqlite` default needs nothing
  external) and Redis (`MORGAN_EVENT_BUS=redis`, for multi-process deployments; `inproc` needs
  nothing external). Both ship behind the `[scale]` extra.
- **The database:** one SQLite file at `{MORGAN_DATA_DIR}/morgan.db` (override via
  `MORGAN_TEMPORAL_DB_URL`) — facts, episodics, vectors, FTS5, entities, signals, session
  history, and the champion prompt registry all share this one file. MLflow's own SQLite
  tracking store is separate, used only when `MORGAN_LEARNING_BACKEND=mlflow` (telemetry forced off).

## Working in this repo
- Single-owner now, but **never hardcode the owner** — key everything by `user_id` so the
  multi-tenant flip stays a config change.
- Memory changes must be measured by the `tests/memory_quality/` harness
  (LoCoMo/LongMemEval-style: single-hop, multi-hop, temporal, knowledge-update) — it currently
  runs over a hash embedder, so treat it as a plumbing check, not a relevance measurement.
- Any self-learning change must pass the `tests/eval/` golden gate — no promotion without a
  beats-current win.
- Python 3.12+, line-length 100, prefer async (`agenerate`/`aencode`). Keep `main` green
  (pytest + ruff check + ruff format --check + mypy strict).
