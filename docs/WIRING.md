# Wiring Morgan to your models + running it

How to point Morgan at your own LLM backend(s), run it, and start the learning loop. Reflects the
**current** state on `main` (Phases 0–5 + Wave 0.5; self-learning loop live). Honest about what's automated vs. deferred.

> Morgan is **provider-agnostic**: any OpenAI-compatible endpoint works (local Ollama / llama.cpp /
> vLLM / LM Studio, or a remote provider). Ollama is just the easy local default.

> **Coming (design stage — not in this build):** Morgan is mid-reshape toward a local-first,
> durable, project-scoped memory kernel — one SQLite database, llama.cpp as the default provider,
> and a `morgan` CLI + MCP server + Python library as first-class surfaces. See
> [the reshape design spec](superpowers/specs/2026-08-02-morgan-reshape-local-first-design.md).
> **None of that is runnable yet**; this guide describes only what exists on `main` today.

## 1. Prerequisites
- Python 3.12, the repo, deps: `cd morgan-brain && pip install -e ".[dev]"`.
- An **OpenAI-compatible LLM endpoint** + an **embeddings** model. Local example (Ollama):
  `ollama pull qwen2.5:7b` (chat) and `ollama pull qwen3-embedding:4b` (embeddings); Ollama exposes
  `http://localhost:11434/v1`.
- **Qdrant** + **Redis** (for vectors / event bus when running multi-process):
  `docker compose up -d redis qdrant`.
- Optional extras: `pip install -e ".[learning]"` (MLflow GEPA), `".[scheduling]"` (cron for
  nightly consolidation).

## 2. Configure (`.env`, all `MORGAN_`-prefixed)
Copy `morgan-brain/.env.example` → `.env` and set:
```bash
# Identity
MORGAN_OWNER_USER_ID=you
MORGAN_API_KEY=change-me

# Model backend (any OpenAI-compatible base_url). Ollama example:
MORGAN_LLM_ENDPOINT=http://localhost:11434/v1
MORGAN_LLM_MODEL=qwen2.5:7b          # default 'strong' role
MORGAN_LLM_FAST_MODEL=qwen2.5:7b     # 'fast' role
MORGAN_EMBEDDING_MODEL=qwen3-embedding:4b

# Stores
MORGAN_QDRANT_URL=http://localhost:6333
MORGAN_REDIS_URL=redis://localhost:6379/0
MORGAN_TEMPORAL_DB_URL=sqlite:///./data/morgan.db

# Learning lifecycle (champion prompt registry + optimizer)
MORGAN_LEARNING_BACKEND=local        # 'local' (SQLite, dependency-light) or 'mlflow'
```
**Provider-agnostic routing (advanced):** `MORGAN_ROLE_BINDINGS` maps logical roles to backends, e.g.
`{"strong": ["ollama:qwen2.5:7b"], "fast": ["ollama:qwen2.5:7b"], "reflection": ["ollama:qwen2.5:32b"]}`
and `MORGAN_PROVIDERS` carries per-provider `base_url`/`api_key`. Defaults are derived from the
`MORGAN_LLM_*` vars, so you only need this to mix backends or add a bigger **reflection** model for the
optimizer. The **judge** role should be a *different model family* than the assistant (eval bias).

## 3. Run
```bash
cd morgan-brain
docker compose up -d redis qdrant
python -m morgan_brain.apps.brain_api          # http://localhost:8080
# health:
curl -s localhost:8080/health
# chat (it perceives → recalls → personalizes → reasons → stores the turn):
curl -s -X POST localhost:8080/api/chat -H 'content-type: application/json' \
     -d '{"message":"My name is Sam and I prefer terse, code-first answers."}'
curl -s -X POST localhost:8080/api/chat -H 'content-type: application/json' \
     -d '{"message":"What is my name?"}'   # → recalls "Sam"
```
The second reply uses cross-turn memory; over time your `USER.md` profile (terse, code-first) is
injected every turn so responses adapt.

## 4. How learning works (today)
- **Per turn (automatic):** the turn is stored as episodic memory off the response path; the
  `AdaptivePersonalizer` injects your compact profile + turn-relevant traits every turn; current
  facts are merged into recall.
- **Feedback (capture it):** every turn returns a `turn_id`. Record feedback via
  **`POST /api/feedback`** `{turn_id, kind: "edit"|"retry"|"thumb", edited_reply?, thumb?}` — an edit
  is the highest-value signal. These feed consolidation + the optimizer. A base signal is recorded
  for every turn automatically.
- **Consolidation (automated):** with `MORGAN_ENABLE_SCHEDULING=true`, the `learning-worker` runs a
  `LearningScheduler` that fires nightly consolidation — `ConsolidationLearner.consolidate(user_id)`
  turns recent episodics into durable valid-time facts (ADD/UPDATE/DELETE/NOOP, contradiction →
  supersede, confidence decay). (APScheduler optional; an in-process scheduler is the default.)
- **Self-optimization (offline, gated):** `ChampionTrainer.train(...)` mines positive examples from
  your high-value signals, asks the **reflection** model to propose an improved champion preprompt,
  scores it on the golden-set eval, and promotes it **only if it beats the current champion**. Enable
  MLflow GEPA with the `[learning]` extra (`MORGAN_LEARNING_BACKEND=mlflow`; telemetry is forced off).

## 5. Optional capabilities
- **Tools:** built-in calculator / clock / memory-search / fetch-url (SSRF-hardened), permission-gated
  (default-deny for side-effecting). Register your own `BaseTool`s.
- **Skills:** drop markdown+frontmatter skills in a skills dir; trigger-matched, champion-versioned
  (they participate in the optimizer loop).

## 6. Verifying quality (the eval gate)
The 3-layer eval harness (`morgan_brain/eval/`) + golden set (`tests/eval/golden_set.json`) is the
"did it learn me / don't regress" gate. Run the suite: `pytest -q`. Extend the golden set with your
own preference probes; any self-learned promotion must beat the current champion on it.

## 7. Remote access
- **Run the worker** (automates learning) alongside the API:
  `MORGAN_ENABLE_SCHEDULING=true python -m morgan_brain.apps.learning_worker`.
- **Streaming:** `POST /api/chat/stream` returns Server-Sent Events (`data: {...}` deltas, terminal
  `data: [DONE]`).
- **Auth (defense-in-depth):** `/api/*` requires `Authorization: Bearer <MORGAN_API_KEY>` (or
  `X-API-Key`) **when a key is set**. `/health` is open.
  > ⚠️ **Before exposing remotely you MUST set a real `MORGAN_API_KEY`** (the default `change-me`
  > leaves `/api/*` open for local dev). Primary control per the architecture is **network posture**:
  > run behind the **NetBird overlay network with no public ports** — do not bind `0.0.0.0` to the
  > internet. See [`docs/OPERATIONS.md`](OPERATIONS.md).

## 8. What remains
- **LoRA fine-tuning** — deliberately deferred; only build it if the 4-condition escalation test in
  the [self-learning decision](superpowers/specs/2026-06-08-self-learning-decision.md) ever fires.
  RAG + the GEPA-optimized champion preprompt are the default and cover the vast majority of gains.
