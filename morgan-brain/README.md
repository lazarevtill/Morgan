# morgan-brain

The implementation of Morgan — a self-hosted, self-learning, provider-agnostic, privacy-first
personal agent platform, and (per the 2026-06-09 reframe) the **kernel of a Personal Agent OS**.
One installable package (`morgan_brain`) that runs as up to three services.

Built on the **MAPLE** decomposition (Memory ≠ Learning ≠ Personalization — three mechanisms on
three timescales) and **SkillOpt** (skills + champion preprompt as trainable, validation-gated
state). Design authority: [`docs/superpowers/specs/2026-06-07-morgan-brain-design.md`](../docs/superpowers/specs/2026-06-07-morgan-brain-design.md);
strategic authority: [`…/2026-06-09-personal-agent-os-vision.md`](../docs/superpowers/specs/2026-06-09-personal-agent-os-vision.md);
status: [`docs/ROADMAP.md`](../docs/ROADMAP.md); run guide: [`docs/WIRING.md`](../docs/WIRING.md).

> The previous monolithic implementation is archived in the git branch/tag
> `legacy/v0.0.3-monolith` and is the source for any selectively ported code.

## Topology (3 services, one package)

| Service | Role | Status |
|---------|------|--------|
| `brain-api` | The request path — perceive → personalize → recall → skills → tools → reason → store → signal. Hosts the REST/SSE gateway. | active |
| `learning-worker` | Async, off the request path — consolidate episodic memory into bi-temporal facts, mine signals, run the eval-gated GEPA optimizer, schedule nightly jobs. | active |
| `perception-gpu` | Voice/vision (PersonaPlex serving, Whisper/Wav2Vec2). Interface defined; **GPU serving deferred**. | deferred |

All services run from the single `morgan_brain` package. Modules talk over typed Protocols
(`interfaces/`) and an event bus whose in-process and Redis-Streams backends share one interface —
so any module can be promoted to its own service with no code change. For local dev everything can
run in one process (`MORGAN_EVENT_BUS=inproc`).

## Package layout (`morgan_brain/`)

| Package | Responsibility |
|---------|----------------|
| `config.py` | The single `MORGAN_`-prefixed settings source (`get_settings()`). |
| `interfaces/` | Protocols — the contracts every module implements (llm, embedding, rerank, memory, learning, personalization, reasoning, skills, tools, perception, voice, events). |
| `models/` | Shared domain models; everything that persists is `user_id`-keyed (`UserScoped`). |
| `bus/` | Event bus — in-proc + Redis-Streams backends behind one interface. |
| `security/` | The single `MemoryGate` (all memory access) + unified `PermissionMode`/`PermissionGate`. |
| `privacy/` | Data classification, egress PII redaction, field-level encryption (all opt-in via flags). |
| `providers/` | Provider adapters, wire types, capability registry, role router, structured-output ladder, resilience. **The only place a provider SDK is imported.** |
| `modules/perception/` | Raw input → `FusedPerception` (text built; audio/vision deferred). |
| `modules/memory/` | Episodic/semantic/procedural store + multi-signal retrieval (vector + BM25 + entity), bi-temporal facts. |
| `modules/personalization/` | Request-path `AdaptivePersonalizer` — budget-aware trait selection, injected every turn. |
| `modules/reasoning/` | Context assembly + role-routed LLM call + tool loop + generation. |
| `modules/skills/` | Markdown+frontmatter skill registry, trigger-matched, champion-versioned. |
| `modules/tools/` | `BaseTool` registry + executor behind the PermissionGate; SSRF/DoS-hardened built-ins. |
| `modules/mcp/` | Hardened MCP host — sanitize/fingerprint-pin/allowlist/default-deny. |
| `modules/proactivity/` | Consent-gated proactive suggestions (default-deny). |
| `learning/` | Signal capture (recorder/signals), consolidation, profile, `ChampionTrainer`, optimizer. |
| `learning_lifecycle/` | `PromptRegistry` + `Optimizer` seam (local SQLite or MLflow backend). |
| `eval/` | 3-layer golden eval harness + calibrated cross-family LLM judge — the self-learning gate. |
| `scheduling/` | CronService + InProcessScheduler + HeartbeatManager + nightly learning jobs (APScheduler optional). |
| `channels/` | Multi-channel gateway with per-chat allowlist (Telegram seam). |
| `voice/` | Brain-side voice utilities — `persona_bridge` (learned persona → role prompt + voice). |
| `core/` | The thin cognitive-loop orchestrator (coordinates only; owns no domain logic). |
| `composition.py` | Wires concrete implementations into the orchestrator + app context. |
| `apps/` | Entrypoints: `brain_api`, `learning_worker`, `perception_gpu`. |

## The cognitive loop (per turn, `brain-api`)

**perceive → personalize → recall → skills → tools → reason → store → signal.** Steps 2–6 only
**read** learned knowledge (hot path); the store + signal steps only **write** and never block the
response (cold path). Memory access goes through `MemoryGate`; facts are bi-temporal (update = close
the old interval, open a new one) with actor attribution on every record.

## The self-learning loop (`learning-worker`)

Off the request path: **consolidate** recent episodics into durable bi-temporal facts
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
pytest -q                            # 820 passed, 11 skipped, 1 xfailed

# Run (two-process)
docker compose up -d redis qdrant
python -m morgan_brain.apps.brain_api          # http://localhost:8080/health
MORGAN_ENABLE_SCHEDULING=true python -m morgan_brain.apps.learning_worker
```

## Optional extras

```bash
pip install -e ".[mcp]"          # MCP host (mcp SDK)
pip install -e ".[learning]"     # MLflow-backed PromptRegistry + GEPA optimizer
pip install -e ".[privacy]"      # Presidio NER + cryptography + argon2 (egress redaction, encryption)
pip install -e ".[scheduling]"   # APScheduler (cron/heartbeat; in-proc scheduler otherwise)
pip install -e ".[channels]"     # python-telegram-bot / discord.py
pip install -e ".[tracing]"      # slim mlflow-tracing for the hot path
pip install -e ".[perception]"   # deferred voice/vision (Whisper, transformers, onnxruntime)
```

## Status

All phases (0–5) + Wave 0.5 + the self-learning engine are built and green: **820 tests pass**
(11 skipped, 1 xfailed), mypy-strict clean. Deferred by design: the voice GPU serving service and
LoRA fine-tuning. See [`docs/ROADMAP.md`](../docs/ROADMAP.md).

## OS vocabulary (vision spec stage — ports/profiles are not built yet)

The Personal Agent OS reframe ([vision doc](../docs/superpowers/specs/2026-06-09-personal-agent-os-vision.md) §3)
maps OS role names onto this package. Package names do not churn; only docs adopt the vocabulary.
The **Ports row is design-stage only** — `ports/` does not exist; it is the first H1 deliverable
per the [horizons roadmap](../docs/superpowers/specs/2026-06-09-horizons-roadmap.md) (H1/H2/H3 +
kill criteria), alongside the [ports design](../docs/superpowers/specs/2026-06-09-ports-design.md)
and [deployment profiles & sync](../docs/superpowers/specs/2026-06-09-deployment-profiles-and-sync-design.md) specs.

| OS term | What it is in `morgan_brain/` today |
|---------|--------------------------------------|
| Kernel  | `core/`, `security/`, `privacy/`, `models/`, `modules/memory`, `learning/`, `modules/personalization`, `providers/` |
| Drivers | `providers/adapters/`, vector/temporal stores, `bus/` backends, `modules/mcp` (client), `voice/` |
| Ports   | new `ports/` surface: MCP server, `/v1` facade, passport, SKILL.md I/O (A2A card static) |
| Shells  | `clients/cli`, `channels/`, any OpenAI-compatible UI via Port 2 |
| Apps    | the orchestrated assistant; external MCP clients; `proactivity/` routines |
