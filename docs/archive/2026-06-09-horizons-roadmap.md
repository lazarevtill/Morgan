# Horizons Roadmap — Personal Agent OS (2026-06-09)

> **Superseded** by [the local-first reshape](../design/local-first-reshape.md)
> (2026-08-02). Device sync, the memory replica, the phone client, deployment profiles, and the
> `/v1` facade are out of scope.

**Status:** DRAFT — under owner review
**Parent:** [Personal Agent OS vision](2026-06-09-personal-agent-os-vision.md)
**Scope:** sequencing and standards-bet timing for the next 12 months. Each horizon ships
independently valuable, fully tested increments; `main` stays green throughout. Dates are
planning anchors, not promises.

---

## Horizon 1 — "Daily value + the first ports" (now → ~3 months, to 2026-09)

Goal: Morgan becomes pluggable **and indispensable daily**. Re-sequenced per strategy review:
the learning loop eats usage signals — if daily value waits, the signal pipeline starves and
"provably gets smarter" is unfalsifiable. Everything here is host-side software.

| # | Deliverable | Spec | Why this order |
|---|-------------|------|----------------|
| 0 | **Kernel prerequisites** (review-mandated): structured provenance on `Memory`/`TemporalFact` + store migration; `system_override` threaded through `stream_turn` (streaming currently drops the champion); persistent `SessionHistoryStore` + redis-bus append path | Ports §1/§2 | Everything below builds on these; two are latent bugs today |
| 1 | **Port audit log** + `GET /api/audit` | Ports cross-cutting | Needed by every port that follows — lands first, not last |
| 2 | **/v1 OpenAI-compatible facade** (client tool passthrough descoped → H2) | Ports §2 | Instantly makes every self-hosted UI a Morgan shell |
| 3 | **MCP server port** (memory/profile/skills/ask + client registry with identity classes; static-bearer auth; quarantined consolidation tier) | Ports §1 | The single-user flywheel; for a developer-owner this *is* a daily-value channel; riskiest item — start early |
| 4 | **Morning brief + named routines** (scheduling + proactivity modules wired to channels; digest-first, interrupt-rarely) | Profiles/vision | Pulled forward from H2: the retention surface that feeds the signal pipeline |
| 5 | **Memory Passport v1 — internal format + lab importers** (`import --from chatgpt\|claude`; restore path = direct bi-temporal merge; export/diff CLI; round-trip test) | Ports §4 | Importers kill the day-1 cold start (feeds №4); format is the replica prerequisite; **publication deferred until pulled** |

Moved out (→ H2): desktop profile, SKILL.md conformance — real, but neither retains the one
user Morgan has today. Cross-cutting H1 discipline (review finding): every deliverable states
its schema migration and a performance budget (e.g., `ask_morgan` is a documented 5–10 s call).

Standards actions: track the MCP 2026-07-28 final release (July) and bump when the Python SDK
V2 ships; adopt OTel GenAI attribute names inside the `[tracing]` extra (experimental-flagged).

Exit criteria: vision success criteria 1, 2, 3, 4 (import half); an external MCP client
demonstrably writes an episodic that survives consolidation into a fact; the owner's morning
brief has run for 2+ weeks without being disabled.

## Horizon 2 — "The brain leaves the house" (~3 → 6 months, to 2026-12)

Goal: multi-device presence and the proactive/voice experience.

| # | Deliverable | Spec |
|---|-------------|------|
| 1 | **Memory Replica v0** (`/api/replica` + ETag snapshots + nightly regen) | Profiles sync |
| 2 | **Device uplink** (`/api/replica/uplink`, device provenance, consolidation integration) | Profiles sync |
| 3 | **Reference phone-thin-client contract** + minimal reference client in `clients/` | Profiles §3 |
| 4 | **`hybrid-burst` hardening** (per-role burst flags, egress audit entries, sensitive-fact never-bursts test) | Profiles §4 |
| 5 | **`desktop` profile** (standalone entrypoint, qdrant-local backend, CI matrix leg) — moved from H1 | Profiles §2 |
| 6 | **SKILL.md conformance** (description field, full frontmatter parser, serializer, GET-as-SKILL.md, metadata persistence; AGENTS.md in repo) — moved from H1; scoped honestly per review | Ports §3 |
| 6b | **/v1 client tool passthrough** (`client_tools` + yield-don't-execute mode + `tool`-role ingestion) — descoped from H1 | Ports §2 |
| 7 | **MCP Apps memory+audit inspector** (read-only `ui://` app; MCP Apps is production-rendered since Jan 2026) — moved up from H3 | Ports cross-cutting |
| 8 | **Voice serving** (`perception-gpu` with PersonaPlex behind the existing seam + persona_bridge) — *hardware-gated*: starts when a CUDA box is available | 2026-06-09 voice decision |
| 9 | **A2A Agent Card** at `/.well-known/agent-card.json` (signed, endpoint deferred) | Ports §5 |

(Passport spec *publication* is no longer a scheduled deliverable — it ships only on external
pull; see Ports §4.)

Exit criteria: vision success criteria 5, 6; an offline phone answer cites replica facts; the
owner can answer "what do you know about me and why" from the inspector.

## Horizon 3 — "The OS thesis pays off" (~6 → 12 months, to 2027-06)

Goal: compounding learning across agents/devices; selective frontier bets. Each item gets its own
spec when its horizon opens — listed here for direction, not commitment.

- **Learning observability**: win-rate dashboard over champion promotions + memory-quality trend
  — the public face of "provably gets smarter" (vision success criterion 7).
- **ACE-style playbook tier**: a curated playbook layer between episodics and the champion
  preprompt (Generator/Reflector/Curator deltas instead of monolithic preprompt rewrites) —
  the foresight review's strongest "double down" (ACE + GEPA both ICLR 2026).
- **Consolidation science**: forgetting-curve-scheduled decay and replay (FOREVER/FSFM-style)
  in the nightly worker — principled forgetting instead of flat confidence decay.
- **Learned proactivity policy**: train when/what/how-often-to-interrupt on the owner's
  dismiss/engage signals — research says naive proactivity gets disabled; the signal pipeline
  is uniquely positioned to learn this.
- **Federation v1** (separate spec): partial-write devices and/or a second brain node, built on
  the passport wire format; CRDTs only if single-writer provably no longer suffices. Design
  `MemoryGate` so the store can replicate without contract change (foresight hedge).
- **On-device fallback agents v2**: platform-model tool calling against local replica tools.
- **LoRA / test-time-training**: unchanged gate — only if the 4-condition escalation test
  fires. Keep the gate alive: context-into-weights (TTT-E2E-class) may mature 2027–28.
- **Web Bot Auth**: adopt signed-agent identity for Morgan's outbound web/tool calls if the IETF
  draft stabilizes and tool targets start requiring it.

## Standards bets summary (what we adopt, watch, ignore)

| Bet | Action | Timing |
|-----|--------|--------|
| MCP (stateless 2026-07-28) | **Adopt hard** — server port + client already in tree | H1, bump at SDK V2 |
| OpenAI-compat /v1 | **Adopt** — facade | H1 |
| Memory Passport | **Define for ourselves** — internal format + lab importers; publish only on pull | H1 build |
| Agent Skills SKILL.md / AGENTS.md | **Adopt** — conformant storage + trainable bodies | H2 |
| MCP Apps | **Adopt** — memory/audit inspector (production-rendered since Jan 2026) | H2 |
| OTel GenAI semconv | Adopt names behind `[tracing]`, expect churn | H1 |
| A2A | Card only; endpoint deferred until concrete need | H2 (card) |
| Web Bot Auth | Watch IETF WG | H3 |
| A2UI, OpenAI Apps SDK, payment protocols (x402/AP2/ACP) | **Ignore** | — |

## Kill criteria (from the strategy red-team — every bet gets a failure signal)

| Bet | Failure signal (cheap, early) | Fallback |
|-----|-------------------------------|----------|
| MCP port | After ~6 weeks of real use: external-agent writes surviving consolidation ≈ 0, or owner disables it for memory pollution | Keep read-only `memory_search`/`profile_get`; drop the write path |
| Morning brief / routines | Owner disables or stops opening within a month | Rework trigger curation before adding any new proactive surface |
| Passport importers | Imported history measurably doesn't improve recall quality (harness) | Keep export-as-backup only; stop investing in importers |
| Passport publication | (Only if published) 2 quarters: zero external implementations or import attempts | Unpublish the ambition; format stays internal |
| Desktop profile | No second real installation; CI leg breaks more than it's used | Replace code path with a documented single-node compose |
| Phone replica | Offline replica answers rarely consulted (phone is online ~99% of the time) | Thin remote client + cached profile only; kill snapshot machinery |

## Sequencing rationale

H1 pairs the two highest-leverage ports with the daily-value surface, because the two depend on
each other: ports without daily use produce an OS nobody boots; daily use without ports keeps
the brain siloed. The strategy review's starvation argument decided the tie — the morning brief
moved up, and the desktop profile + SKILL.md (real, but retention-neutral for the one current
user) moved down. H2 spends the passport format on devices and turns on the remaining
experiential differentiators (phone, inspector, voice). H3 commits only to direction because
6–12-month foresight in this field degrades fast — the roadmap's job is to make those options
cheap, not to pick them today. Every bet now carries a kill criterion so failure is cheap too.
