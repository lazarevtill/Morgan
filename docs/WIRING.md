# Wiring Morgan to your models + running it

How to point Morgan at your own LLM backend(s), run it, and start the learning loop. Reflects the
**current** state on `main` (Phases 0–3 + Wave 0.5). Honest about what's automated vs. manual today.

> Morgan is **provider-agnostic**: any OpenAI-compatible endpoint works (local Ollama / llama.cpp /
> vLLM / LM Studio, or a remote provider). Ollama is just the easy local default.

## 1. Prerequisites
- Python 3.12, the repo, deps: `cd morgan-brain && pip install -e ".[dev]"`.
- An **OpenAI-compatible LLM endpoint** + an **embeddings** model. Local example (Ollama):
  `ollama pull qwen2.5:7b` (chat) and `ollama pull qwen3-embedding:4b` (embeddings); Ollama exposes
  `http://localhost:11434/v1`.
- **Qdrant** + **Redis** (for vectors / event bus when running multi-process):
  `docker compose up -d redis qdrant`.
- Optional extras: `pip install -e ".[learning]"` (MLflow GEPA), `".[mcp]"` (MCP servers),
  `".[privacy]"` (Presidio + encryption), `".[scheduling]"` (cron — Phase 4).

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
MORGAN_MLFLOW_TRACKING_URI=sqlite:///./data/mlflow.db

# Privacy (opt-in)
MORGAN_REDACT_EGRESS=false           # true → redact PII before REMOTE providers (local passes through)
MORGAN_ENCRYPTION=false              # true → SQLCipher + envelope encryption (needs [privacy] + passphrase)
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
- **Feedback (capture it):** record edits/retries/thumbs against a turn via the `SignalRecorder`
  (edit = highest-value signal). These feed consolidation + the optimizer. *(A thin HTTP/CLI surface
  for feedback is part of Phase 4/Wave 6; the `SignalRecorder` API exists now.)*
- **Consolidation (currently triggered, automated in Phase 4):** `ConsolidationLearner.consolidate(user_id)`
  turns recent episodics into durable bi-temporal facts (ADD/UPDATE/DELETE/NOOP, contradiction →
  supersede, confidence decay). **Phase 4 adds the nightly/idle cron that runs this automatically.**
- **Self-optimization (offline, gated):** `ChampionTrainer.train(...)` mines positive examples from
  your high-value signals, asks the **reflection** model to propose an improved champion preprompt,
  scores it on the golden-set eval, and promotes it **only if it beats the current champion**. Enable
  MLflow GEPA with the `[learning]` extra (`MORGAN_LEARNING_BACKEND=mlflow`; telemetry is forced off).

## 5. Optional capabilities
- **Tools:** built-in calculator / clock / memory-search / fetch-url (SSRF-hardened), permission-gated
  (default-deny for side-effecting). Register your own `BaseTool`s.
- **Skills:** drop markdown+frontmatter skills in a skills dir; trigger-matched, champion-versioned
  (they participate in the optimizer loop).
- **MCP servers:** add to `MORGAN_MCP_SERVERS` (needs `[mcp]`). Tool descriptions are sanitized +
  **fingerprint-pinned** (rug-pull defense), allowlisted, and **default-deny** until you grant them.
- **Privacy:** set `MORGAN_REDACT_EGRESS=true` to redact PII before any *remote* provider (local
  models get full context); `MORGAN_ENCRYPTION=true` (+ `[privacy]` + passphrase) for at-rest encryption.

## 6. Verifying quality (the eval gate)
The 3-layer eval harness (`morgan_brain/eval/`) + golden set (`tests/eval/golden_set.json`) is the
"did it learn me / don't regress" gate. Run the suite: `pytest -q` (542 green). Extend the golden set
with your own preference probes; any self-learned promotion must beat the current champion on it.

## 7. What's not automated yet (remaining waves)
- **Phase 4:** heartbeat + cron to run consolidation/optimization on a schedule, and proactive
  suggestions (consent-gated).
- **Phase 5:** voice (Whisper + emotion) behind the perception seam; a hardened **remote gateway**
  (Tailscale-first, JWT/API-key, SSE streaming, Telegram/Discord channels).
- **Wave 6:** full end-to-end integration + this guide's automation.

Until Phase 4 lands the scheduler, run consolidation/optimization on demand (call
`ConsolidationLearner.consolidate` / `ChampionTrainer.train` from a script or REPL against your stores).
