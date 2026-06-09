# Morgan

A self-hosted, **self-learning, provider-agnostic, privacy-first personal agent platform**. Morgan
is a personal assistant that measurably learns from its owner — every interaction makes it know you
better, and that knowledge changes the next response — built on the same primitives (memory, skills,
tools, MCP, permissions, event bus) that let it host and orchestrate future agents. It runs great
fully local (Ollama / llama.cpp / vLLM / LM Studio) or against any OpenAI-compatible remote provider.

> **Status: complete platform.** Phases 0–5 + Wave 0.5 are built, green, and shippable
> (820 tests). Deferred by design: the voice GPU serving service and LoRA fine-tuning (see below).

## What it does

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
  [PersonaPlex voice](docs/superpowers/specs/2026-06-09-personaplex-voice-decision.md).

> The previous monolithic implementation is archived in the git branch/tag
> **`legacy/v0.0.3-monolith`** and is the source for any selectively ported code.

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE).
