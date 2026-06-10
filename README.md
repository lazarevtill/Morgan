# Morgan

A self-hosted, **self-learning, provider-agnostic, privacy-first personal agent platform** — and,
by direction, the **operating system for a person's agent life**: the kernel that owns your
identity, memory, learning, and policy; that any agent can plug into; and that provably gets
smarter the more it is used. The chat assistant is the **first app on that OS**, not the product.
It runs great fully local (Ollama / llama.cpp / vLLM / LM Studio) or against any OpenAI-compatible
remote provider.

> **Status: complete platform; OS reframe at spec stage.** Phases 0–5 + Wave 0.5 + the
> self-learning engine are built, green, and shippable (820 tests). The Personal Agent OS layers
> below the kernel exist today; the **ports, deployment profiles, and device replica are committed
> design specs, not code** (Horizon 1 has not started). Deferred by design: the voice GPU serving
> service and LoRA fine-tuning.

## What exists today

- **Learns you, safely.** A signal→consolidation→personalization loop: every turn logs training
  signals (edits > retries > thumbs); a nightly worker consolidates episodic memory into durable
  **bi-temporal facts** (knowledge evolves, never overwrites); an `AdaptivePersonalizer` injects
  your compact profile + turn-relevant traits on every turn.
- **Optimizes itself, gated.** A **GEPA champion-preprompt optimizer** mines high-value signals,
  proposes an improved system prompt, and promotes it **only if it beats the current champion** on a
  3-layer held-out eval (retrieval recall → golden preference probes → A/B with a cross-family LLM
  judge). No learned update ever ships that regresses the assistant.
- **Agentic.** Permission-gated, SSRF/DoS-hardened built-in **tools** (calculator, clock,
  memory-search, fetch-url) plus your own `BaseTool`s; default-deny for side effects.
- **Skills.** Markdown + frontmatter skills, trigger-matched and champion-versioned (they
  participate in the optimizer loop).
- **Hardened MCP host.** External integrations as config, not code — tool descriptions sanitized,
  fingerprint-pinned (rug-pull defense), allowlisted, default-deny.
- **Remote gateway.** API-key auth on `/api/*`, SSE streaming (`/api/chat/stream`), and a channel
  gateway with per-chat allowlist (Telegram seam).
- **Voice seam.** A `VoiceConversation` interface + `persona_bridge` (learned persona → role prompt
  + voice) targeting **NVIDIA PersonaPlex** full-duplex speech; GPU serving deferred.
- **Provider-agnostic & private.** Role router (fast/strong/vision/reflection/judge) behind typed
  seams; opt-in PII egress redaction before remote providers; opt-in SQLCipher at-rest encryption.
- **Deploys two-process or single-process.** `brain-api` (request path) + `learning-worker` (async
  learning), or everything in one process for local dev.

## Status (capabilities)

| Phase / Wave | Outcome | State |
|---|---|---|
| 0 Foundation | config, events, protocols, MemoryGate, data model | ✅ |
| 0.5 Provider seam + privacy | provider Protocols + role router + capability descriptors + structured-output ladder; encryption + classification + egress redaction; MLflow scaffold | ✅ |
| 1 Memory + Reasoning | text assistant w/ cross-turn recall (vector+BM25+entity, bi-temporal facts) | ✅ |
| 2 Learning + Personalization | signal capture + consolidation worker + adaptive personalization + 3-layer eval gate | ✅ |
| 3 Skills + Tools + MCP | permission-gated tools + champion-versioned skills + hardened MCP host + **GEPA optimizer** | ✅ |
| 4 Proactivity | Cron + Heartbeat + LearningScheduler + consent-gated ProactivityEngine | ✅ |
| 5 Perception/voice + remote | remote gateway (auth + SSE + channels) + voice seam | ✅ (voice GPU service deferred) |
| Self-learning engine | GEPA preprompt optimizer loop, eval-gated | ✅ (LoRA deferred by design) |

**Deferred by design:** the **voice GPU service** (PersonaPlex serving needs an A100/H100-class GPU;
`[perception]` extra) and **LoRA fine-tuning** (only built if the 4-condition escalation test fires — RAG +
the GEPA champion preprompt cover the vast majority of gains).

## Where this is going (vision, spec stage)

The 2026-06-09 spec set reframes Morgan as a Personal Agent OS. **None of the new layers are
implemented yet** — they are designs the next implementation waves (H1–H3) build:

```
┌─ APPS ────────────────────────────────────────────────────────┐
│  Morgan Assistant (chat/voice)  ·  external agents (Claude,   │
│  IDE agents, future agents)  ·  routines & proactive agents   │
├─ SHELLS ──────────────────────────────────────────────────────┤
│  CLI · Telegram/channels · any OpenAI-compat UI (Open WebUI,  │
│  LibreChat, HA Voice) · phone app (thin client)               │
├─ PORTS (the standardized syscall surface) ────────────────────┤
│  1. MCP server   — memory/profile/skills as tools+resources   │
│  2. /v1 facade   — OpenAI-compatible chat completions         │
│  3. SKILL.md     — skills read/written in the open standard   │
│  4. Memory Passport — versioned export/import + audit log     │
│  (5. A2A Agent Card — published, thin, endpoint deferred)     │
├─ KERNEL (existing morgan_brain, renamed in role) ─────────────┤
│  MemoryGate · bi-temporal facts · learning engine (signals →  │
│  consolidation → GEPA, eval-gated) · personalization ·        │
│  policy (permissions, privacy, egress) · role-routed providers│
├─ DRIVERS ─────────────────────────────────────────────────────┤
│  providers/adapters (Ollama/llama.cpp/remote) · Qdrant ·      │
│  Redis · SQLite/Postgres · MCP client · voice (PersonaPlex)   │
└───────────────────────────────────────────────────────────────┘
```

Kernel and drivers exist (the green platform above). Ports, deployment profiles
(homelab / desktop / phone / hybrid-burst), and the read-only memory replica are **specs**. The
key inversion: exposing the kernel as an MCP server makes external agents (Claude Code, IDE
agents) contribute to and draw from the same brain — a single-user flywheel on your own hardware.

The 2026-06-09 spec set:

- [Personal Agent OS vision](docs/superpowers/specs/2026-06-09-personal-agent-os-vision.md) —
  north star, layers, flywheel, defensibility, success criteria, non-goals (master doc).
- [Ports design](docs/superpowers/specs/2026-06-09-ports-design.md) — MCP server, `/v1` facade,
  SKILL.md, Memory Passport, A2A card.
- [Deployment profiles & device sync](docs/superpowers/specs/2026-06-09-deployment-profiles-and-sync-design.md) —
  homelab / desktop / phone / hybrid-burst + read-only memory replica.
- [Horizons roadmap](docs/superpowers/specs/2026-06-09-horizons-roadmap.md) — H1/H2/H3 sequencing,
  standards bets, kill criteria.
- [Ecosystem research 2026 H1](docs/superpowers/specs/2026-06-09-ecosystem-research-2026H1.md) —
  the ground truth the vision is built on.

## Quick start

```bash
cd morgan-brain
cp .env.example .env                 # set MORGAN_API_KEY before any remote exposure
pip install -e ".[dev]"
docker compose up -d redis qdrant
python -m morgan_brain.apps.brain_api    # → http://localhost:8080/health
```

Full instructions (config, learning loop, remote access, voice/LoRA status) live in
[`docs/WIRING.md`](docs/WIRING.md).

## Documentation

- [`docs/ROADMAP.md`](docs/ROADMAP.md) — authoritative status across all phases.
- [`docs/WIRING.md`](docs/WIRING.md) — how to run, endpoints, config, the learning loop.
- [`morgan-brain/README.md`](morgan-brain/README.md) — package layout, topology, build/test/run.
- [`CLAUDE.md`](CLAUDE.md) — architecture map + non-negotiable invariants.
- Decision records (under `docs/superpowers/specs/`):
  [self-learning](docs/superpowers/specs/2026-06-08-self-learning-decision.md) ·
  [platform architecture](docs/superpowers/specs/2026-06-08-platform-architecture-decision.md) ·
  [PersonaPlex voice](docs/superpowers/specs/2026-06-09-personaplex-voice-decision.md) ·
  the five Personal Agent OS specs linked above.

> The previous monolithic implementation is archived in the git branch/tag
> **`legacy/v0.0.3-monolith`** and is the source for any selectively ported code.

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE).
