# Ports Design — Morgan's Standardized Surface (2026-06-09)

**Status:** DRAFT — under owner review
**Parent:** [Personal Agent OS vision](2026-06-09-personal-agent-os-vision.md)
**Scope:** the five ports that turn the kernel into a platform. Each port is a translation layer
over existing seams — policy, gating, and attribution always live below it.

Proposed package home: `morgan_brain/ports/` (new top-level package: `mcp_server/`,
`openai_facade/`, `passport/`, `skills_io/`). Ports are hosted by `brain-api` (mounted routes),
except passport import/export which is also a CLI verb. The `mcp` Python SDK is a port
dependency, not a provider SDK — the providers/adapters isolation rule is about LLM vendors and
is unaffected.

---

## Port 1 — MCP server (agents become apps)

**What:** Morgan exposes a Streamable-HTTP MCP server at `/mcp`, designed stateless-native for
the **2026-07-28 revision** (no session handshake; OAuth-hardened; Tasks as extension).

**Implementation (per the June-2026 deep-dive):** official `mcp` Python SDK pinned
`>=1.27.2,<2`, built with `FastMCP(stateless_http=True, json_response=True)` and mounted into
brain-api via `app.mount("/mcp", mcp.streamable_http_app())`. Known trap: the MCP app's
lifespan **must be merged into brain-api's FastAPI lifespan** (nested lifespans don't run —
python-sdk #1367). Zero per-session state in tools; handles passed as arguments. SDK V2 (the
2026-07-28-native rewrite) has no alpha yet — expected H2 2026; the 12-month deprecation policy
makes v1.27 safe to ship on now, and statelessness makes the later bump near-trivial. Use no
deprecated feature (Roots/Sampling/Logging).

**Tools (initial set):**

| Tool | Maps to | Notes |
|------|---------|-------|
| `memory_search(query, kind?, limit?)` | `MemoryGate.recall` | read scope; returns facts/episodics with provenance + validity interval |
| `memory_store(content, kind, context?)` | episodic ingest → cold-path consolidator | write scope; stamped `tool_observed` + client id; never a synchronous durable-fact write |
| `memory_forget(memory_id)` | supersession via `invalid_at` | write scope; single item only, confirm-gated; **no bulk-delete tool exists** (mem0's `delete_all_memories` footgun) |
| `profile_get(section?)` | `AdaptivePersonalizer` compact profile | read scope |
| `skills_list()` / `skills_get(name)` | skills registry | read scope; returns SKILL.md content |
| `ask_morgan(question)` | full orchestrator turn | the whole cognitive loop as a tool — lets a coding agent ask "what does my owner prefer here?" |

Tool design rules from prior art: snake_case action-noun names (SEP-986); strict input schemas
with `additionalProperties: false`; declared `outputSchema` with exactly-matching structured
returns (mem0/OpenMemory's dict-vs-string mismatches are the top client-compat failure).
Attribution: MCP-originated writes are stamped `MemorySource.tool_observed` — from the kernel's
view an external client is a tool surface; `agent_inferred` stays reserved for Morgan's own
inferences and `user_stated` can never be asserted through this port.

**Resources:** `morgan://profile` (compact profile doc), `morgan://facts/recent` (recently
valid facts). Read-only.

**Client identity & scopes:** a small client registry (in `Settings` + SQLite): each client
gets a named key with an **identity class** — `external-agent` or `owner-device` — and scopes ⊆
{`memory.read`, `memory.write`, `profile.read`, `skills.read`, `ask`}. Default for a new
client: `external-agent`, read-only. The class is what licenses provenance: `external-agent`
writes are always `tool_observed`; `owner-device` (the person's own phone/desktop client) may
assert `user_stated` carrying `via=device` — this resolves the apparent contradiction between
this port and the replica uplink. Every call is written to the audit log (client, scope, tool,
summary of what was returned/stored).

**Cross-cutting schema work item (review finding — currently in no module):** `Memory` /
`TemporalFact` provenance today is a bare 3-value enum (`models/memory.py:22-25`). Structured
provenance (`source`, `client_id`, `via`) must be added to the models, both stores, and the
passport format, with a SQLite migration. This is a prerequisite for Ports 1/4 and the uplink.

**Auth (deep-dive verdict): static Bearer per client now; OAuth stubbed, not built.** Claude
Code connects with static headers today; claude.ai's connector currently has a known bug
dropping Bearer tokens to self-hosted servers post-OAuth, and its UI lacks static-bearer
config — so full OAuth buys nothing yet. When exposure beyond Claude Code is wanted: add
`.well-known/oauth-protected-resource` + a CIMD allowlist (pin known client_id URLs; SSRF-guard
the metadata fetch) and validate `iss`/audience per SEP-2468/2352 from day one. Per-key rate
limits and quotas regardless of auth mode.

**Invariant enforcement:** the server calls the same `MemoryGate`/`PermissionGate` the assistant
uses. `memory_store` publishes the existing-but-unused **`MEMORY_STORED`** event
(`interfaces/events.py:19`) consumed by a dedicated worker handler — *not* `RESPONSE_GENERATED`
(review finding: that event is consumed as a user/assistant conversation and records a training
signal; riding it would poison signal mining and session history). External contributions still
ride the same consolidation pipeline (dedup, contradiction → supersede, confidence decay).

**Learning-loop hygiene:** port-originated turns and writes are **signal-excluded by default** —
`ask_morgan` runs a full cognitive turn but records no base training signal and never feeds
champion optimization (only the owner's own conversations train the champion). Document
`ask_morgan` latency honestly: it is a 5–10 s tool call by design (quality-over-speed).

**Security model (a write-capable memory server is a target — Zep patched a Cypher injection
reachable via prompt injection):**
- **Memory poisoning is the #1 attack class**: stored text replayed as trusted context later.
  Sanitize instruction-like patterns (`<system>`, `<IMPORTANT>`, role markers) at **store and
  recall**; recalled port-written content is rendered as quoted data, never as instructions.
- All tool arguments are treated as untrusted model output; strict schemas
  (`additionalProperties: false`) everywhere.
- The **eval-item firewall extends to this port**: MCP-written memories are excluded from
  golden-set construction so external agents cannot contaminate the self-learning gate.
- **Quarantined consolidation tier** (review finding: bearer + audit is detection, not
  prevention): port-sourced episodics consolidate with provenance-weighted confidence — facts
  derived solely from `external-agent` writes are capped below the confidence of owner-stated
  facts, never win contradictions against them, and are visibly badged in recall and the
  inspector. Per-client write quotas bound the blast radius.
- Bi-temporal invariant holds: an MCP "update" is a supersession, never an overwrite — poisoning
  is therefore always reversible and attributable via the audit log.
- Per-key rate limits; destructive ops (`memory_forget`) require elevated scope + confirmation.
- **Network-exposure threat model:** `/mcp` is loopback/VPN-only by default; remote exposure
  requires TLS + a real `MORGAN_API_KEY` + rate limits, stated in the run guide.

## Port 2 — OpenAI-compatible /v1 facade (any UI becomes a shell)

**What:** `GET /v1/models` (returns `morgan` + role aliases `morgan-fast`, `morgan-strong`) and
`POST /v1/chat/completions` (streaming + non-streaming, OpenAI chunk format), mounted on
brain-api beside the native API.

**Semantics:** the facade presents Morgan as *a model that happens to have memory and learning*:
- `messages` → a turn: last user message is the input; prior messages are client-provided
  history. Conversation id comes from an `x-conversation-id` header when present, else a content
  hash of the first client message (adversarial review: key+`user` alone collapses every UI chat
  into one session). Dedup rule: **client history wins**; server-stored history is consulted
  only when the client sends a lone message.
- Client `system` messages are treated as *app instructions* — appended after Morgan's champion
  preprompt + personalization, never replacing them. Requires a new `client_instructions` slot
  on `ReasoningRequest` (the single `system_override` slot is already occupied by the champion).
- **Client tool passthrough is descoped from H1** (adversarial finding: the reasoning loop
  executes tools locally and `ReasoningResult` has no `tool_calls` wire field; OpenAI semantics
  need *unexecuted* tool_calls returned to the client). H1 facade rejects requests with `tools`
  with a clear error; an H2 work item adds `ReasoningRequest.client_tools` + a
  yield-don't-execute mode + `tool`-role ingestion.
- Internally executed Morgan tools/memory are invisible to the client beyond the final message —
  the facade does not leak kernel internals into the wire format.
- Feedback: OpenAI-format requests cannot carry Morgan feedback; the native `/api/feedback`
  remains the channel for edit/retry/thumb signals. (Facade turns still record base signals.)

**Prerequisite fixes surfaced by the code review (must land before/with the facade):**
1. `stream_turn` has no `system_override` parameter and `/api/chat/stream` never passes the
   champion — **the streaming path silently loses the learned champion today**
   (`core/orchestrator.py:184-191`, `apps/brain_api/app.py:77-82`). Thread it through.
2. `SessionHistoryStore` is constructed `:memory:` and its append subscriber registers only on
   the in-proc bus — under `MORGAN_EVENT_BUS=redis` the store is read but never written
   (`composition.py:311,229-230`). Persist it and append from the worker's redis handler.

**Auth:** the same `MORGAN_API_KEY` / client-registry keys as Port 1.

**Why:** Open WebUI, LibreChat, Home Assistant Voice, IDE plugins — the entire self-hosted UI
ecosystem connects with zero Morgan-specific code. Morgan rides every shell anyone builds.

## Port 3 — SKILL.md conformance (skills speak the standard)

**What:** Morgan's skills become spec-conformant **Agent Skills** packages (frontmatter
`name`/`description` + markdown body), per agentskills.io (Apache-2.0, AAIF-governed).

- **Storage/export:** the skills registry persists each skill as a SKILL.md package; `GET
  /api/skills/{name}` and Port 1 `skills_get` return standard SKILL.md text usable by Claude
  Code, Codex, Cursor, etc. — Morgan-learned skills become portable artifacts.
- **Import:** drop ecosystem SKILL.md packages into `skills/` (or POST them); they enter the
  selection pool like native ones.
- **Learning hook (the differentiator):** SkillOpt treats the SKILL.md *body* as trainable,
  validation-gated state — the GEPA optimizer may propose revisions; promotion only on a golden-
  eval win; full version history per skill. Nobody else ships *learned* standard skills.
- Repo root gains an `AGENTS.md` so coding agents working *on* Morgan follow the same standard
  we implement.

**Honest scoping (review finding — this is more greenfield than "conformance" implies):**
`Skill` has no `description` field; the loader discards unknown frontmatter; the parser is a
YAML subset that can't express multi-line descriptions; no SKILL.md serializer exists; the
current endpoint is `POST /api/skills/{name}` returning JSON (not `GET` returning SKILL.md);
and runtime-deployed skills persist only their body, losing metadata on restart
(`modules/skills/registry.py:56-72`, `routes.py:117-122`). Each of these is a named work item
in the implementation plan, not polish.

## Port 4 — Memory Passport (own your brain)

**What:** a versioned, documented export/import bundle. Layout (a `.tar.zst`, optionally
encrypted with the existing crypto module — AES-256-GCM with Argon2id key derivation,
`privacy/crypto.py`):

```
passport/
  manifest.json        # schema_version, exported_at, user_id, source build, content hashes
  facts.jsonl          # TemporalFacts: full bi-temporal fields + MemorySource provenance
  episodics.jsonl      # optional (size guard; default: last 90 days)
  profile.json         # compact profile + trait history
  skills/*.skill.md    # learned + imported skills (SKILL.md), with version metadata
  champion.jsonl       # champion-preprompt version history + eval scores at promotion
  audit.jsonl          # port/client audit trail
  embeddings.note      # embeddings are NOT exported — re-embed on import (model-agnostic by design)
```

**Operations:** `morgan passport export|import|diff|inspect` (CLI) + owner-only
`POST /api/passport/export`. Two distinct import paths (review finding — consolidation is
LLM-based, O(N) calls, non-deterministic; wrong tool for a restore):
- **Restore path (passport → same/new Morgan):** direct bi-temporal merge — facts land verbatim
  with original provenance and intervals; deterministic, no LLM. `--replace` wipes first.
- **Foreign-import path (lab exports, below):** conversations land as episodics and go through
  the normal *asynchronous* consolidation pipeline (budgeted, nightly).
Round-trip acceptance test (restore path): export → fresh instance → import → memory-quality
harness within tolerance — deterministic by construction.

**Importers (the wedge — per strategy review, this outranks export evangelism):**
`morgan passport import --from chatgpt <export.zip>` and `--from claude <export>` ingest the
labs' official data exports (conversation history) through Morgan's consolidation pipeline:
conversations land as episodics (provenance `imported:chatgpt`, original timestamps preserved
for the bi-temporal axes), then the normal consolidator distills facts. This solves the day-1
cold-start problem — a learning system is at its dumbest on day 1 — and is the credible
lab-exodus path while cloud memory gets less auditable and less exportable.

**Spec publication:** demoted from goal to option. The passport is first the **internal
backup / replica wire format**. Publish `docs/passport/SPEC.md` + JSON Schema only on external
pull (second implementer interest, or the regulatory portability pressure arriving — see vision
§2.6). Format deliberately excludes embeddings and store internals so any memory system *could*
implement it.

**Role in sync:** the replica feed for phones/desktops is a *subset passport* (profile +
top-K currently-valid facts) — one format for backup, migration, and sync.

## Port 5 — A2A Agent Card (thin, deferred endpoint)

**What:** publish a signed Agent Card at `/.well-known/agent-card.json` describing Morgan
(identity, capabilities, endpoint placeholder). **No A2A endpoint yet** — research shows v1.0
traction is enterprise cross-vendor, not personal-agent. The card costs ~nothing, makes Morgan
discoverable, and reserves the slot; an A2A server adapter can be bolted onto the same kernel
seams if a concrete peer-agent need appears.

---

## Cross-cutting: the port audit log

One append-only audit store (SQLite next to the temporal DB): `(ts, client_id, port, action,
scope, summary, turn_id?)`. Surfaced via `GET /api/audit` (owner-only) and exported in the
passport. This is the auditability bet: the owner can always answer "which agent read/wrote what,
when" — exactly what the big labs are removing.

**Owner-facing inspector:** MCP Apps (SEP-1865) went production-ready in January 2026 and is
rendered today by Claude.ai/Desktop, VS Code, Goose. A read-only `ui://` memory+audit inspector
("what do you know about me, and why") is therefore feasible in H2, served through the standard
instead of a custom frontend.

## Testing strategy

- Each port: contract tests against fakes (existing pattern) + golden wire-format fixtures
  (OpenAI chunk shapes; MCP tool-call envelopes).
- Port 1: a real `mcp` Python client in tests exercises list/call against the mounted server.
- Port 2: run an unmodified `openai` client against brain-api in tests (SDK already a dep).
- Port 4: round-trip property tests (hypothesis) on fact supersession chains; harness tolerance test.
- Invariants: tests asserting a port write can never produce `user_stated` provenance and never
  touches a store without `MemoryGate`.
