# Deployment Profiles & Device Sync (2026-06-09)

> **Superseded** by [the local-first reshape](2026-08-02-local-first-reshape-design.md)
> (2026-08-02). Device sync, the memory replica, the phone client, deployment profiles, and the
> `/v1` facade are out of scope.

**Status:** DRAFT — under owner review
**Parent:** [Personal Agent OS vision](2026-06-09-personal-agent-os-vision.md)
**Scope:** the four first-class deployment targets and the A+B-blended sync model (read-only
memory replica now; full federation deferred).

A **profile** is a named configuration preset + entrypoint, not a fork. One package, one
`Settings` object; a profile is `MORGAN_PROFILE={homelab|desktop|phone-backend|...}` selecting
defaults that any env var can still override. No `if profile == ...` in domain logic — profiles
only choose which implementations `composition.py` wires (which is exactly what the Protocol
seams exist for).

---

## Profile 1 — `homelab` (the reference brain; exists today)

Docker compose: `brain-api` + `learning-worker` + Redis + Qdrant; `MORGAN_EVENT_BUS=redis`,
`MORGAN_VECTOR_BACKEND=qdrant`; optional `perception-gpu` when voice serving lands.

Target hardware (from 2026 research): unified-memory mini-PC (Mac Studio M4 Max / Ryzen AI Max
class, 32 GB+) running **Qwen 3.5 9B/27B or Gemma 4 26B-A4B at Q4** via Ollama/llama.cpp with
multi-token-prediction speculative decoding; Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B locally.
Reflection/judge roles may bind to a stronger remote model via the hybrid-burst policy below —
the role router already supports the split.

Hardening to reach "reference deployment" status: documented backup = nightly passport export;
a readiness endpoint (only `/health` liveness exists today — `apps/brain_api/app.py:46`); a
`docs/WIRING.md` section per profile.

## Profile 2 — `desktop` ("Morgan in a box"; new)

Everything in **one process** on a laptop/desktop, no Docker, no Redis, no Qdrant server:

- Entrypoint: `python -m morgan_brain.apps.standalone` — brain-api app + the worker's consumers
  and scheduler in the same asyncio loop (`MORGAN_EVENT_BUS=inproc`,
  `MORGAN_ENABLE_SCHEDULING=true`). The hot/cold separation is preserved *logically* (the bus
  seam stays; cold-path work runs on background tasks) even though it is one process.
- Vectors: `qdrant-client` **embedded local mode** (`path=...`, no server) as a third
  `MORGAN_VECTOR_BACKEND=qdrant-local` option — persistent, zero-ops, same client API as the
  server backend so `QdrantVectorIndex` is reused nearly unchanged.
- Facts/signals/audit: SQLite (already the default).
- Models: Ollama/LM Studio on the same machine (Gemma 4 E4B / Qwen 3.5 4B–9B for 8–16 GB);
  role bindings let `fast`=4B local, `strong`=9B local or remote burst.
- Redis: not needed at all — Redis exists in Morgan only as the event-bus backend
  (`bus/redis_streams.py`), and this profile uses the in-proc bus. (Earlier draft claimed a
  `CacheProtocol` seam; the code review found none exists — there is no Redis cache to replace.)

Acceptance: full pytest suite green under the desktop profile in CI (a matrix leg), plus the
memory-quality harness; cold start to first token < 10 s on a 16 GB machine with a 4B model.

## Profile 3 — `phone` (thin client + offline fallback; new, H2)

**Decision: do not ship a model runtime.** The phone is a *shell* with graceful degradation:

1. **Online (default):** a thin client app (or any OpenAI-compat mobile UI via Port 2) talks to
   the home brain over private networking (Tailscale/WireGuard or authenticated HTTPS). Voice in
   the client (platform STT or streamed to brain).
2. **Offline fallback:** the device holds a **read-only memory replica** (below) and uses the
   platform on-device model — **Apple Foundation Models framework (iOS) / Android ML Kit GenAI
   Prompt API** — both now expose structured output + tool calling to third-party apps. The
   fallback agent answers from replica context only and clearly marks itself degraded.
3. **Contribution uplink:** everything the device-side agent learns or the user tells it offline
   is queued as **episodic events** (never fact writes) and synced to the brain when reconnected,
   entering the normal cold-path consolidation. Single-writer brain ⇒ no merge conflicts by
   construction.

The repo's deliverable in H2 is the **backend contract** for this (replica endpoint + uplink
endpoint + a reference client in `clients/`), not a published app-store app.

## Profile 4 — `hybrid-burst` (policy, not a topology; hardening)

Local-first with policy-gated escalation to remote frontier models. This is configuration over
existing seams, promoted to a documented, tested profile:

- `MORGAN_ROLE_BINDINGS` routes e.g. `reflection`/`judge` (and optionally `strong`) to a remote
  provider; everything else stays local.
- The **egress policy** decides per-request what may burst: data classification (privacy module)
  + `MORGAN_REDACT_EGRESS` PII redaction on anything leaving the box; memory content classified
  sensitive never appears in remote prompts.
- New: a per-role `burst` flag in settings + an audit-log entry for every remote call (provider,
  role, redaction applied) so "what left the box" is always answerable.
- **Mechanism (review finding — this needs plumbing, not just a test):** today memories are
  assembled into the prompt unconditionally and the model is resolved *after* message assembly
  (`modules/reasoning/reasoner.py` context build); egress redaction wraps remote clients
  only. The sensitive-fact guarantee requires (a) resolving the role binding (local vs remote)
  *before* context build, and (b) `MemoryGate.recall` filtering facts classified sensitive when
  the turn is remote-bound. Both are named work items.

Acceptance: a test proving a sensitive-classified fact cannot reach a remote-bound prompt, and
that turning bursting off yields a fully-local trace.

---

## Device sync: Memory Replica v0 (the B-blend)

**Goal now:** phone/desktop work offline with current knowledge; the brain stays the single
writer. Full multi-brain federation is explicitly H3+.

- **Replica format:** a *subset passport* (Port 4): compact profile + top-K currently-valid
  facts (relevance- and recency-ranked, sensitive-classified facts excluded by default) + the
  active champion preprompt. Small (≤ a few MB), embedding-free.
- **Distribution:** `GET /api/replica` (owner/device-scoped key) returns the latest snapshot +
  `ETag`; devices poll or fetch on app-foreground. The learning-worker regenerates the snapshot
  nightly (after consolidation) and on profile/champion promotion.
- **Uplink:** `POST /api/replica/uplink` accepts batched episodic events with device + timestamp
  provenance. The device authenticates with an **`owner-device`-class key** from the same client
  registry as the ports (see Ports §1 identity classes) — that class, and only that class, may
  assert `user_stated`, always carrying `via=device` + client id in the structured provenance
  field so consolidation can weigh it. `external-agent` keys hitting this endpoint are rejected.
  (This identity-class split is what reconciles the uplink with the vision-doc rule that an
  external agent can never create a `user_stated` fact.)
- **Conflict story:** none needed — replicas are read-only, uplink is append-only episodics,
  the consolidator already handles contradiction by bi-temporal supersession.
- **Deferred to federation v1 (H3, separate spec):** peer brains, partial-write devices, CRDTs,
  multi-node Qdrant. The passport format is deliberately the wire format so federation builds on
  this rather than replacing it. Foresight hedge: the local-first/CRDT community is converging
  on AI memory sync (git-backed memory trees, agents as CRDT peers) — keep `MemoryGate`'s
  contract replication-agnostic so the store can later replicate without interface change.

## Settings & composition impact

- `MORGAN_PROFILE` preset layer in `config.py` (defaults only; explicit env always wins).
- `composition.py` gains the standalone wiring path and `qdrant-local` backend selection.
- New extras: none required (qdrant-client already a core dep); `[desktop]` alias may pin
  optional niceties later.

## Testing strategy

- CI matrix leg per profile config (homelab-sim with services, desktop standalone, hybrid-burst
  policy tests with a fake remote provider).
- Replica: snapshot determinism test — note (review finding) "relevance-ranked" and
  "deterministic hash" conflict unless ranking inputs are pinned: the snapshot builder ranks by
  recency + confidence only (no query-relevance, no embedding scores), so the same store state
  yields the same snapshot. Plus staleness/ETag test and uplink → consolidation integration test.
- Profile invariance: the memory-quality harness must pass under both `homelab` and `desktop`
  profiles — same brain quality regardless of topology.
