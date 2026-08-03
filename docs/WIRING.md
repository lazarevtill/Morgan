# Wiring Morgan to your models + running it

How to point Morgan at `llama-server`, run it, and use the two usage surfaces (`morgan` CLI,
`morgan-mcp`). Reflects the **current** state on `main`: local-first memory (one SQLite database),
project-scoped reads/writes, and an eval-gated (disarmed by default) learning loop.

> Morgan is **provider-agnostic**: any OpenAI-compatible endpoint works. The default and
> documented topology is **llama.cpp** (`llama-server`), remote-first — a GPU box on the homelab
> reached over an overlay network (NetBird), with local loopback as the offline/dev fallback.
> Ollama and other OpenAI-compatible endpoints remain supported non-default provider keys.

## 1. Prerequisites
- Python 3.12, the repo, deps: `cd morgan-brain && pip install -e ".[dev]"`.
- A running **`llama-server`** (from [llama.cpp](https://github.com/ggml-org/llama.cpp)) serving
  a chat model and an embedding model. `llama-server` only serves one model per process, so run
  two instances (or point separate roles at separate ports/hosts):
  ```bash
  llama-server -m qwen2.5-7b-instruct.gguf --port 8081                 # chat
  llama-server -m mxbai-embed-large.gguf --embedding --port 8082       # embeddings
  ```
  Either bind these on the homelab GPU box and reach them over NetBird, or run them on
  `localhost` for offline/dev work — both are the same OpenAI-compatible `/v1` protocol, only
  the endpoint URL differs.
- Optional, only if you need multi-process scale: Qdrant (`MORGAN_VECTOR_BACKEND=qdrant`) and
  Redis (`MORGAN_EVENT_BUS=redis`), behind the `[scale]` extra (`pip install -e ".[scale]"`).
  Neither is required — the defaults (`sqlite`, `inproc`) need nothing external.
- Optional extras: `pip install -e ".[mcp]"` (the `morgan-mcp` server), `".[learning]"` (MLflow
  GEPA), `".[scheduling]"` (cron for nightly consolidation).

## 2. Configure (`.env`, all `MORGAN_`-prefixed)
Copy `morgan-brain/.env.example` → `.env` and set:
```bash
# Identity
MORGAN_OWNER_USER_ID=you
MORGAN_API_KEY=change-me            # INBOUND: set a real key before any remote exposure

# Chat model — the default provider key is 'llamacpp' (llama-server's OpenAI-compatible /v1)
MORGAN_LLM_ENDPOINT=http://localhost:8081/v1
MORGAN_LLM_MODEL=qwen2.5-7b-instruct
MORGAN_LLM_FAST_MODEL=qwen2.5-7b-instruct
# MORGAN_LLM_API_KEY=               # OUTBOUND, to the model server — opposite direction from
                                     # MORGAN_API_KEY above. Empty by default (most homelab
                                     # llama-server setups run without --api-key).

# Embedding model — point at the embedding llama-server instance
MORGAN_EMBEDDING_MODEL=mxbai-embed-large
MORGAN_EMBEDDING_DIM=1024           # must match the model's output dim

# Where the shared database lives (episodics, facts, vectors, FTS5, entities, signals, history,
# champion prompt registry — one SQLite file at {MORGAN_DATA_DIR}/morgan.db)
MORGAN_DATA_DIR=./data

# Learning lifecycle (champion prompt registry + optimizer)
MORGAN_LEARNING_BACKEND=local        # 'local' (SQLite, dependency-light) or 'mlflow'
MORGAN_ENABLE_CHAMPION_PROMOTION=false   # see §7 — off by default, deliberately
```

**All four roles (advanced):** `MORGAN_ROLE_BINDINGS` maps each logical role to an ordered list of
`"provider:model"` fallbacks; `MORGAN_PROVIDERS` carries per-provider `base_url`/`api_key`/`timeout`.
Defaults derive from the `MORGAN_LLM_*` vars above and bind **all four roles** (`strong`, `fast`,
`judge`, `reflection`) to the same `llamacpp` endpoint — an unbound role makes the router raise.
Override to mix backends or run a bigger **reflection** model for the optimizer:
```bash
MORGAN_ROLE_BINDINGS={"strong":["llamacpp:qwen2.5-7b-instruct"],"fast":["llamacpp:qwen2.5-7b-instruct"],"judge":["llamacpp:qwen2.5-7b-instruct"],"reflection":["llamacpp:qwen2.5-32b-instruct"]}
MORGAN_PROVIDERS={"llamacpp":{"base_url":"http://localhost:8081/v1","api_key":"llamacpp","timeout":120.0}}
```
The **judge** role should be a *different model family* than the assistant (eval bias).

## 3. Verify — `morgan doctor`
```bash
cd morgan-brain
morgan doctor
```
Reports, independently per subsystem so one failure doesn't hide the rest: the database path,
whether `sqlite-vec` and FTS5 loaded, whether the configured LLM endpoint is reachable, and row
counts for the current project (or `--all-projects`). This is the standard first check after any
config change — run it before filing a wiring problem as a bug.

## 4. Use it — the `morgan` CLI
`remember`/`recall`/`facts`/`forget`/`doctor` are direct memory operations: they go through the
`MemoryGate` over the real database with no LLM router required (works even with
`MORGAN_EMBEDDING_BACKEND=hash`, a deterministic stub, if you have no model server running yet).
`ask` is a full chat turn and needs a reachable LLM endpoint. Every command accepts `--project`
(default: the current git repository's directory name), `--all-projects` (the explicit
cross-project escape hatch, where it applies), and `--json`.
```bash
morgan remember "prefers terse, code-first answers"
morgan recall "how do I like answers"          # vector + FTS5 keyword + entity, fused
morgan facts                                    # currently-valid facts for this project
morgan ask "what do you know about me"          # a full chat turn — recalls, reasons, stores
morgan forget                                   # cascading erasure for this project, one report
```

## 5. Use it — `morgan-mcp` (any MCP client)
The same memory, exposed as five MCP tools (`remember`, `recall`, `facts`, `forget`, `ask_morgan`)
to any client that speaks the Model Context Protocol — Claude Code, Claude Desktop, or your own.
It calls the exact same command handlers the CLI does, through the same `MemoryGate`; no memory
logic is duplicated. `project` is an explicit tool argument here (the server is a long-lived
daemon with no meaningful cwd of its own), falling back to a system-wide default when omitted.
```bash
pip install -e ".[mcp]"
morgan-mcp --transport stdio                                       # a client on this machine
morgan-mcp --transport http --host 0.0.0.0 --port 8090              # laptops over NetBird
```
The HTTP transport enforces `MORGAN_API_KEY` as a bearer token — the same policy `/api/*` uses,
and the same caveat: **it serves openly whenever the key is unset or left at `change-me`.** Set a
real key before exposing the port beyond loopback. See [`docs/OPERATIONS.md`](OPERATIONS.md) for
client config examples (`.mcp.json`, `claude mcp add`).

## 6. Use it — `brain-api` (REST/SSE gateway)
```bash
cd morgan-brain
python -m morgan_brain.apps.brain_api          # http://localhost:8080
curl -s localhost:8080/health
curl -s -X POST localhost:8080/api/chat -H 'content-type: application/json' \
     -d '{"message":"My name is Sam and I prefer terse, code-first answers.","project":"demo"}'
curl -s -X POST localhost:8080/api/chat -H 'content-type: application/json' \
     -d '{"message":"What is my name?","project":"demo"}'   # → recalls "Sam"
```
`project` is required on `/api/chat`, `/api/chat/stream`, and `/api/feedback` — there is no
implicit default project over the API. The second reply uses cross-turn memory recalled from the
same project. `/api/chat/stream` returns Server-Sent Events (`data: {"delta": "..."}`, terminal
`data: [DONE]`) and threads the learned champion preprompt through exactly like `/api/chat` does.

`/health` is open; every other `/api/*` route requires `Authorization: Bearer <MORGAN_API_KEY>`
(or `X-API-Key`) **when a key is set** — the default `change-me` leaves it open for local dev.
> ⚠️ **Before exposing remotely you MUST set a real `MORGAN_API_KEY`.** Primary control per the
> architecture is **network posture** — run behind the **NetBird overlay network with no public
> ports**; do not bind `0.0.0.0` to the internet. See [`docs/OPERATIONS.md`](OPERATIONS.md).

Run the worker alongside the API to automate learning:
`MORGAN_ENABLE_SCHEDULING=true python -m morgan_brain.apps.learning_worker`.

## 7. How learning works
- **Per turn (automatic):** the turn is stored as episodic memory off the response path — the
  in-process event bus enqueues it and a background drain task (started by `brain-api`'s lifespan
  hook) dispatches it, so the request never blocks on it. `AdaptivePersonalizer` injects your
  compact profile + turn-relevant traits every turn; current facts are merged into recall.
- **Feedback (capture it):** every turn returns a `turn_id`. Record feedback via
  **`POST /api/feedback`** `{turn_id, kind: "edit"|"retry"|"thumb", project, edited_reply?, thumb?}`
  — an edit is the highest-value signal. `project` is required. These feed consolidation + the
  optimizer. A base signal is recorded for every turn automatically.
- **Consolidation (automated):** with `MORGAN_ENABLE_SCHEDULING=true`, `learning-worker` runs a
  `LearningScheduler` that fires nightly consolidation — `ConsolidationLearner.consolidate(user_id)`
  turns recent episodics into durable valid-time facts per project (ADD/UPDATE/DELETE/NOOP,
  contradiction → close the old interval via `valid_to`/`superseded_by`, confidence decay).
- **Self-optimization (offline, gated, disarmed by default):** `ChampionTrainer.train(...)` mines
  positive examples from your high-value signals, asks the **reflection** model to propose an
  improved champion preprompt, and scores it on the golden-set eval. It promotes **only if**
  `MORGAN_ENABLE_CHAMPION_PROMOTION=true` *and* it beats the current champion — the flag defaults
  to `false` because the gate itself (a bare `>` on one scored run over a 12-item golden set) is
  too statistically weak to trust unattended. Enable MLflow GEPA with the `[learning]` extra
  (`MORGAN_LEARNING_BACKEND=mlflow`; telemetry is forced off).

## 8. Optional capabilities
- **Tools:** built-in calculator / clock / memory-search / fetch-url (SSRF-hardened), permission-gated
  (default-deny for side-effecting). Register your own `BaseTool`s.
- **Skills:** drop markdown+frontmatter skills in a skills dir; trigger-matched, champion-versioned
  (they participate in the optimizer loop).

## 9. Verifying quality (the eval gate)
The 3-layer eval harness (`morgan_brain/eval/`) + golden set (`tests/eval/golden_set.json`) is the
"did it learn me / don't regress" gate for any champion-preprompt change. Run the suite:
`pytest -q`.

## 10. Erasure — `forget()`
`morgan forget` (or the MCP `forget` tool) cascades a project's rows out of `memories`,
`fts_memories`, `memory_entities`, `facts`, `interaction_signals`, and `session_history` in one
transaction, and out of `vec_items`/`vec_meta` too — but **only under the default `sqlite` vector
backend**. Under `vector_backend=qdrant`, vectors are **not** erased by `forget()` and must be
removed from Qdrant separately; the CLI's JSON/text output says so explicitly rather than implying
a clean sweep. The report also distinguishes "erased zero rows" from "this table doesn't exist in
your database" (`tables_skipped`) so a fresh install doesn't read as a bug.

## 11. Known limitations
- **`recall` has no relevance floor.** Vector, FTS5, and entity search each return their top-k
  regardless of score, and all currently-valid facts are always included — once a project holds
  anything, a query always returns something. There is no "no matches" state except on an empty
  project.
- **The MCP HTTP transport serves openly** whenever `MORGAN_API_KEY` is unset or left at
  `change-me` — set a real key before exposing the port beyond loopback (see §5).
- **Retrieval quality is unmeasured.** `tests/memory_quality/` is a stub harness over a hash
  embedder — it exercises the plumbing, not real relevance.

## 12. What remains
- **LoRA fine-tuning** — deliberately deferred; only build it if the 4-condition escalation test in
  the [self-learning decision](superpowers/specs/2026-06-08-self-learning-decision.md) ever fires.
  RAG + the GEPA-optimized champion preprompt are the default and cover the vast majority of gains.
