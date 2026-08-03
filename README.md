# Morgan

A self-hosted, **self-learning, provider-agnostic personal agent kernel** — it owns your identity,
memory, learning, and policy. The chat assistant is one app on top of it, not the product. Memory
is **local-first**: one SQLite database holds everything, project-scoped, reachable from any
project on any of your machines. The default provider is **llama.cpp** (`llama-server`),
remote-first — a GPU box on the homelab reached over an overlay network — with any other
OpenAI-compatible endpoint (Ollama included) supported as a non-default provider key.

## What exists today

- **Two usage surfaces.** The `morgan` **CLI** (`remember`/`recall`/`facts`/`forget`/`ask`/
  `doctor`; project auto-detected from the current git repository) and **`morgan-mcp`**, an MCP
  server exposing the same five operations to any MCP client (Claude Code, Claude Desktop, …) over
  stdio or streamable-HTTP with a bearer token. Both are thin adapters over the same `MemoryGate`
  — no memory logic is duplicated between them.
- **A remote gateway.** `brain-api` (`/api/chat`, `/api/chat/stream` SSE, `/api/feedback`,
  `/api/tools`, `/api/skills`, `/api/profile`), API-key auth on every route but `/health`.
- **Durable, project-scoped memory.** Recall fuses three signals — vector (sqlite-vec), FTS5
  keyword (Cyrillic-aware), and entity overlap — all surviving a restart. Every read and write is
  scoped to a `project`; `--all-projects` is the explicit cross-project escape hatch.
- **Learns you, safely.** A signal→consolidation→personalization loop: every turn logs training
  signals (edits > retries > thumbs); a nightly worker consolidates episodic memory into durable
  **valid-time facts** (knowledge evolves, never overwrites); an `AdaptivePersonalizer` injects
  your compact profile + turn-relevant traits on every turn.
- **Optimizes itself, gated — and disarmed by default.** A champion-preprompt optimizer mines
  high-value signals, proposes an improved system prompt, and would promote it only if it beats
  the current champion on a 3-layer held-out eval. `MORGAN_ENABLE_CHAMPION_PROMOTION` defaults to
  `false`: the promotion gate itself isn't statistically sound yet (see [`docs/ROADMAP.md`](docs/ROADMAP.md)).
- **Cascading erasure.** `forget()` removes a project's memories, facts, vectors, signals, and
  session history in one transaction (vectors under `vector_backend=qdrant` are the one gap —
  they must be removed from Qdrant separately).
- **Agentic.** Permission-gated, SSRF/DoS-hardened built-in **tools** (calculator, clock,
  memory-search, fetch-url) plus your own `BaseTool`s; default-deny for side effects.
- **Skills.** Markdown + frontmatter skills, trigger-matched and champion-versioned.
- **Provider-agnostic.** Role router (strong/fast/judge/reflection) behind typed seams; no
  provider SDK is imported above the adapter layer.

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the milestone-by-milestone status and known
limitations.

## Quick start

```bash
cd morgan-brain
cp .env.example .env                 # point MORGAN_LLM_ENDPOINT at your llama-server
pip install -e ".[dev]"
morgan doctor                        # diagnose the install — no Redis/Qdrant/Docker required
morgan remember "prefers terse, code-first answers"
morgan recall "how do I like answers"
```

Full instructions (llama-server setup, all four roles, the MCP server, `brain-api`, the learning
loop) live in [`docs/WIRING.md`](docs/WIRING.md).

## Documentation

- [`docs/ROADMAP.md`](docs/ROADMAP.md) — status, milestones, known limitations.
- [`docs/WIRING.md`](docs/WIRING.md) — how to run, endpoints, config, the learning loop.
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md) — at-rest/transport protection, backups, the stack.
- [`morgan-brain/README.md`](morgan-brain/README.md) — package layout, topology, build/test/run.
- [`CLAUDE.md`](CLAUDE.md) — architecture map + non-negotiable invariants.
- [The local-first reshape design](docs/superpowers/specs/2026-08-02-morgan-reshape-local-first-design.md)
  — diagnosis, target architecture, milestone plan.
- Decision records (under `docs/superpowers/specs/`):
  [self-learning](docs/superpowers/specs/2026-06-08-self-learning-decision.md) ·
  [platform architecture](docs/superpowers/specs/2026-06-08-platform-architecture-decision.md).

> Earlier builds are archived in the git tags **`legacy-v0.0.4-full`** (the platform build this
> reshape narrowed) and **`legacy-v0.0.3-monolith`** (branch `origin/legacy/v0.0.3-monolith`, the
> pre-platform monolith) — sources for any selectively ported code, not the current design.

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE).
