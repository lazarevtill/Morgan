# Morgan as a Personal Agent OS — Vision & Reframing (2026-06-09)

> **Superseded** by [the local-first reshape](2026-08-02-morgan-reshape-local-first-design.md)
> (2026-08-02). Device sync, the memory replica, the phone client, deployment profiles, and the
> `/v1` facade are out of scope.

**Status:** DRAFT — under owner review
**Decides:** what Morgan *is* for the next 6–12 months, and the architecture frame every
subsequent design plugs into.
**Companion specs (this set):**
- [Ports design](2026-06-09-ports-design.md) — the standardized surface (MCP server, /v1 facade,
  SKILL.md, Memory Passport, A2A card)
- [Deployment profiles & device sync](2026-06-09-deployment-profiles-and-sync-design.md) —
  homelab / desktop-lite / phone / hybrid burst + memory replica
- [Horizons roadmap](2026-06-09-horizons-roadmap.md) — phased H1/H2/H3 plan with standards bets
- [Ecosystem research, 2026 H1](2026-06-09-ecosystem-research-2026H1.md) — the ground truth this
  vision is built on

---

## 1. The reframe (north star)

> **Morgan is the operating system for a person's agent life** — the self-hosted kernel that owns
> their identity, memory, learning, and policy; that any agent can plug into; and that provably
> gets smarter the more it is used.

The chat assistant stops being "the product." It becomes **the first app on the OS**. The product
is the kernel: a private, auditable, provider-agnostic brain that accumulates value across every
agent, channel, and device the person uses.

This is a reframe, not a rewrite. Every layer below maps onto modules that exist and are green
today (820 tests, mypy-strict). The reframe *adds ports around* the kernel; it does not touch the
cognitive loop.

## 2. Why this, why now (foresight summary)

Grounded in the June-2026 research set (see companion doc for sources):

1. **The gap is unclaimed — but narrowing.** No product — big-lab or self-hosted — ships a
   *measurable, eval-gated, per-user learning loop*. OpenClaw (~347K GitHub stars) proved
   mainstream demand for self-hosted personal agents but has memory files, not learning. Caveat
   from the foresight review: OpenAI's "Dreaming" memory now consolidates offline too, so the
   loop's *existence* is no longer the differentiator — **ownership, auditability, portability,
   and the eval gate are** (OpenAI reduced its memory audit trail in 2026; we go the other way).
2. **MCP won the tool layer** and goes stateless on 2026-07-28. Building Morgan's agent-facing
   port natively on that revision puts us ahead of the migration wave instead of behind it.
3. **Skills became the unit of agent learning.** Agent Skills (SKILL.md) + AGENTS.md are durable
   Linux-Foundation-governed standards; the ecosystem (Letta's pivot, ACE) converged on
   versioned, trainable context — Morgan's SkillOpt thesis, independently validated.
4. **No memory-portability standard exists or is coming.** Memory is the moat nobody wants to
   standardize. A self-hosted platform can *define* the open format (auditability + portability)
   exactly where the labs are retreating.
5. **The hardware arrived.** Unified-memory mini-PCs are a mainstream "home brain" category;
   9–27B-class local models (Qwen 3.5 / Gemma 4) do real tool-calling; phones expose on-device
   models to third-party apps (Apple Foundation Models, Android ML Kit).
6. **Regulation favors this architecture.** EU AI Act GPAI enforcement (Aug 2026), expected
   EDPB guidance on AI memory, and academic pressure for memory-graph portability all land on
   cloud profile-holders, not on systems where the user owns the store. An exportable,
   inspectable bi-temporal fact store is becoming a compliance asset.

**What is actually defensible** (per strategy red-team — "moat" is the wrong word; every
component here is copyable): (a) **data gravity** — the owner's accumulated, provenance-rich
brain, whose value compounds daily; (b) **category authorship** — being the reference
implementation if the memory format earns traction; (c) **rigor** — invariant discipline a
mass-contributor project structurally can't keep. Note the deliberate trade-off: the Memory
Passport (§5) *weakens* data gravity by design. That is a values choice — personal sovereignty
over lock-in — and this platform makes it knowingly.

## 3. The layered architecture

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

Vocabulary mapping (docs adopt these role names; package names do not churn):

| OS term | What it is in `morgan_brain/` today |
|---------|--------------------------------------|
| Kernel  | `core/`, `security/`, `privacy/`, `models/`, `modules/memory`, `learning/`, `modules/personalization`, `providers/` |
| Drivers | `providers/adapters/`, vector/temporal stores, `bus/` backends, `modules/mcp` (client), `voice/` |
| Ports   | new `ports/` surface: MCP server, `/v1` facade, passport, SKILL.md I/O (A2A card static) |
| Shells  | `clients/cli`, `channels/`, any OpenAI-compatible UI via Port 2 |
| Apps    | the orchestrated assistant; external MCP clients; `proactivity/` routines |

## 4. The key inversion: agents as apps (Port 1)

Today only Morgan's own assistant benefits from the brain. Exposing the kernel as an **MCP
server** (`memory_search`, `memory_store`, `memory_forget`, `profile_get`, `skills_list/get`,
`ask_morgan`) means a Claude Code session, an IDE agent, or any future MCP client **contributes
to and draws from the same brain**. This is a **single-user flywheel**, not a network effect
(strategy review's correction — there are no other participants): every agent enriches one
brain, on the person's own hardware. The loop that must become indispensable is **cross-surface
recall**: Claude Code knows what you told Morgan on Telegram last night, and what Claude Code
learns shows up in tomorrow's chat.

Guardrails (non-negotiable):
- All port access goes through `MemoryGate` + `PermissionGate`. No port exposes a store.
- Every external client gets a **scoped client identity**; everything it writes is stamped
  `MemorySource` = `tool_observed` with the client id in provenance. An external agent can never
  create a `user_stated` fact (and `agent_inferred` stays reserved for Morgan's own inferences).
- Port writes land as **episodic/candidate** input to the cold-path consolidator — external
  agents feed the learning loop; they do not mutate durable facts synchronously. Hot path reads,
  cold path writes — globally, not just in-process.

## 5. Defining the future: the Memory Passport (Port 4) — importers first

A documented, versioned export/import format bundling: bi-temporal facts with full provenance,
the compact profile, learned skills (SKILL.md) + champion-preprompt version history, and the
audit trail. Export, import, diff, inspect — owner-only, optionally encrypted.

**The strategy review inverted the original bet, and it's right: importers before export
evangelism.** A published format with zero second implementations is a schema with a press
release. The higher-leverage wedge is **ingesting ChatGPT / Claude data exports** through
Morgan's own consolidation pipeline to seed the bi-temporal store — it solves the worst
experiential problem (a learning system is at its dumbest on day 1) and is the only credible
lab-exodus path while OpenAI makes memory background-synthesized and less auditable.

The passport format itself ships as the **internal backup / replica wire format** (it is the
prerequisite for device replicas), with a round-trip guarantee. Publishing it as an open spec
happens only on pull — regulation trends (vision §2.6) suggest the pull may come to us.

## 6. What stays sacred (kernel discipline)

Every existing invariant survives the reframe unchanged:
- Hot path reads, cold path writes (now extended across ports and devices).
- One `MemoryGate`; one permission model; one config; one bus interface; one logger.
- Facts evolve, they don't overwrite (bi-temporal supersession).
- Actor attribution on every memory; inference never becomes user-stated fact.
- Self-learning is eval-gated: candidate-only optimization, promote on strict beats-current win.
- Provider SDKs isolated to `providers/adapters/`; no provider hardcoded.
- Everything persisted is `user_id`-keyed; single-owner stays a config choice, not a code path.

New invariants introduced by the OS frame:
- **Ports never bypass the kernel.** A port is a translation layer; policy and gating live below it.
- **Every port action is auditable.** Client id + scope + what was read/written, queryable by the owner.
- **The phone/desktop replica is read-only.** Devices contribute signals/episodics upward; only
  the brain writes durable facts (single-writer; no CRDT until federation v1 justifies it).

## 7. Success criteria (12 months)

1. **The owner uses Morgan daily** (morning brief opened, assistant consulted) — without this
   the signal pipeline starves and the learning thesis is unfalsifiable (strategy review's
   central point: experiential value is what feeds the learning loop).
2. An external MCP client (e.g. Claude Code) demonstrably improves Morgan's profile/facts, and
   the improvement shows up in the next assistant turn — measured by the memory-quality harness.
3. Open WebUI / LibreChat / Home Assistant connect to Morgan via `/v1` with zero Morgan-specific
   code, and get memory-aware, learning-backed responses.
4. A ChatGPT or Claude data export imports through consolidation and measurably improves day-1
   recall; a full Memory Passport export round-trips (export → wipe → import → memory-quality
   harness within tolerance).
5. Desktop-lite runs the whole kernel in one process on a laptop with local models only.
6. A phone client works offline against the replica and its episodics sync up when reconnected.
7. The learning win-rate dashboard shows promoted champions beating predecessors on the golden
   eval — the "provably gets smarter" claim, kept honest.

## 8. Explicit non-goals (12-month horizon)

- Multi-tenant SaaS hosting (the `user_id` keying keeps the door open; we do not walk through it).
- A2A *endpoint* (we publish a signed Agent Card only; enterprise peer-to-peer traction does not
  justify the surface yet).
- Per-user LoRA by default (unchanged: only if the 4-condition escalation test fires).
- Payments / agent-commerce protocols; A2UI; OpenAI Apps SDK.
- Building our own phone runtime (use platform on-device APIs; never bundle a model).
