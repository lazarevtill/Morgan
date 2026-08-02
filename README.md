# Morgan

A self-hosted, **self-learning, provider-agnostic personal agent platform** — the kernel that owns
your identity, memory, learning, and policy. The chat assistant is one app on top of it, not the
product. It runs great fully local (Ollama / llama.cpp / vLLM / LM Studio) or against any
OpenAI-compatible remote provider.

> **Status: platform built; local-first reshape in design.** Phases 0–5 + Wave 0.5 + the
> self-learning engine are built, green, and shippable (633 tests). Morgan is mid-reshape toward a
> local-first, durable, project-scoped memory kernel — see
> [the reshape design spec](docs/superpowers/specs/2026-08-02-morgan-reshape-local-first-design.md).
> None of the reshape milestones are implemented yet.

## What exists today

- **Learns you, safely.** A signal→consolidation→personalization loop: every turn logs training
  signals (edits > retries > thumbs); a nightly worker consolidates episodic memory into durable
  **valid-time facts** (knowledge evolves, never overwrites); an `AdaptivePersonalizer` injects
  your compact profile + turn-relevant traits on every turn.
- **Optimizes itself, gated.** A **GEPA champion-preprompt optimizer** mines high-value signals,
  proposes an improved system prompt, and promotes it **only if it beats the current champion** on a
  3-layer held-out eval (retrieval recall → golden preference probes → A/B with a cross-family LLM
  judge). No learned update ever ships that regresses the assistant.
- **Agentic.** Permission-gated, SSRF/DoS-hardened built-in **tools** (calculator, clock,
  memory-search, fetch-url) plus your own `BaseTool`s; default-deny for side effects.
- **Skills.** Markdown + frontmatter skills, trigger-matched and champion-versioned (they
  participate in the optimizer loop).
- **Remote gateway.** API-key auth on `/api/*` and SSE streaming (`/api/chat/stream`).
- **Provider-agnostic.** Role router (fast/strong/vision/reflection/judge) behind typed seams; no
  provider SDK is imported above the adapter layer.
- **Deploys two-process or single-process.** `brain-api` (request path) + `learning-worker` (async
  learning), or everything in one process for local dev.

## Status (capabilities)

| Phase / Wave | Outcome | State |
|---|---|---|
| 0 Foundation | config, events, protocols, MemoryGate, data model | done |
| 0.5 Provider seam | provider Protocols + role router + capability descriptors + structured-output ladder | done |
| 1 Memory + Reasoning | text assistant w/ cross-turn recall (vector+BM25+entity, valid-time facts) | done |
| 2 Learning + Personalization | signal capture + consolidation worker + adaptive personalization + 3-layer eval gate | done |
| 3 Skills + Tools | permission-gated tools + champion-versioned skills + **GEPA optimizer** | done |
| 5 Remote gateway | API-key auth + SSE streaming | done |
| Self-learning engine | GEPA preprompt optimizer loop, eval-gated | done (LoRA deferred by design) |

**Deferred by design:** LoRA fine-tuning (only built if the 4-condition escalation test fires —
RAG + the GEPA champion preprompt cover the vast majority of gains).

## Quick start

```bash
cd morgan-brain
cp .env.example .env                 # set MORGAN_API_KEY before any remote exposure
pip install -e ".[dev]"
docker compose up -d redis qdrant
python -m morgan_brain.apps.brain_api    # → http://localhost:8080/health
```

Full instructions (config, learning loop, remote access) live in
[`docs/WIRING.md`](docs/WIRING.md).

## Documentation

- [`docs/ROADMAP.md`](docs/ROADMAP.md) — authoritative status across all phases.
- [`docs/WIRING.md`](docs/WIRING.md) — how to run, endpoints, config, the learning loop.
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md) — at-rest/transport protection, backups, the stack.
- [`morgan-brain/README.md`](morgan-brain/README.md) — package layout, topology, build/test/run.
- [`CLAUDE.md`](CLAUDE.md) — architecture map + non-negotiable invariants.
- [The local-first reshape design](docs/superpowers/specs/2026-08-02-morgan-reshape-local-first-design.md)
  — current direction: diagnosis, target architecture, milestone plan.
- Decision records (under `docs/superpowers/specs/`):
  [self-learning](docs/superpowers/specs/2026-06-08-self-learning-decision.md) ·
  [platform architecture](docs/superpowers/specs/2026-06-08-platform-architecture-decision.md).

> The previous monolithic implementation is archived in the git tag
> **`legacy-v0.0.3-monolith`** (branch `origin/legacy/v0.0.3-monolith`) and is the source for any
> selectively ported code.

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE).
