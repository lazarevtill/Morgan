# Platform Foundation Baseline (2026-06-10)

**Status:** DRAFT — under owner review
**Parent:** [Personal Agent OS vision](2026-06-09-personal-agent-os-vision.md) ·
[Horizons roadmap](2026-06-09-horizons-roadmap.md) (this spec *is* H1 item 0, expanded)
**Scope:** the cross-cutting substrate every port, profile, and horizon builds on. Ships as its
own wave **before** any port. Explicitly *not* in scope: the ports themselves, the replica,
importers — those consume this baseline.

Everything here follows one rule: **the simplest mechanism that preserves the invariants.**
Single-owner, single-node, SQLite-first; nothing enterprise-shaped without a concrete need.

---

## 1. Provenance v2 (the schema the whole OS hangs on)

Today `Memory.source` / `TemporalFact.source` is a bare 3-value enum
(`models/memory.py:22-25`) — it can say *what kind* of actor asserted a memory, but not *which*
client, over *which* surface. Ports, uplink, importers, and the audit story all need more.

**Model change (additive):** a `Provenance` value object on both `Memory` and `TemporalFact`:

```python
class Provenance(BaseModel):
    client_id: str | None = None      # registry id of the writing client (None = core assistant)
    via: Literal["chat", "port", "device", "import", "consolidation"] | None = None
    channel: str | None = None        # e.g. "telegram", "cli", "v1", "mcp"
    imported_from: str | None = None  # e.g. "chatgpt", "claude", "passport"

class Memory(UserScoped):
    ...
    source: MemorySource = MemorySource.USER_STATED   # unchanged — the trust class
    provenance: Provenance | None = None              # new — the audit trail detail
```

`MemorySource` stays the **trust class** (drives consolidation weighting and recall badging);
`Provenance` is the **accountability detail**. Rules binding them (enforced in `MemoryGate`,
tested in the invariant suite §9):
- `source=user_stated` requires `provenance.via in (None, "chat", "device")` — and `"device"`
  only when the client's identity class is `owner-device` (§2).
- Port writes (`via="port"`) are always `tool_observed`.
- Import writes carry `imported_from` and never exceed the confidence cap for non-owner sources.

**Storage:** one additive `provenance_json TEXT NULL` column on the SQLite fact/memory tables;
one `provenance` payload field on Qdrant points. `NULL` means legacy row — semantics identical
to today. No backfill needed.

## 2. Migration machinery (one-time cost, pays forever)

The temporal DB has no schema versioning today; provenance v2 is the first real migration, so
the machinery lands with it (mechanism per the 2026 foresight review — the no-ORM consensus):
- **`PRAGMA user_version`** as the version store (SQLite-native, zero extra tables); numbered,
  forward-only migration steps (`0001_provenance.sql`-style, with a Python escape hatch for
  data backfills), each wrapped in a transaction so the version bumps only on success.
- Runner in `morgan_brain/storage/migrations.py`, applied automatically at store startup (the
  runner takes a write-lock so brain-api and the worker don't race). **A file copy of the DB is
  taken before any migration runs** (pre-migration backup, pruned with audit retention).
- **Policy: additive-first.** New columns nullable with legacy-compatible semantics; shipped
  migrations are never edited; destructive changes are forbidden in-place — they require an
  export/import cycle through the passport, and any such change must carry a passport
  round-trip test. No Alembic/ORM — the codebase is raw-SQL + pydantic; `user_version` is the
  right-sized tool.
- Tests: each migration step gets a fixture DB at version N−1 → assert N applies cleanly and
  twice-idempotently; CI runs the full chain from an empty DB and from a seeded v1 DB.

## 3. Client identity registry

The single `MORGAN_API_KEY` becomes the **owner key**; everything else gets a registered
identity. New `security/clients.py` + SQLite table:

```python
class ClientIdentity(BaseModel):
    client_id: str                    # slug, e.g. "claude-code", "pixel-9"
    display_name: str
    klass: str                        # "owner" | "owner-device" | "external-agent" — str, not
                                      # Literal: foresight flags per-agent identities (OWASP
                                      # ASI10 "rogue agents") as the likely next class
    scopes: frozenset[str]            # from the fixed vocabulary below
    key_hash: str                     # argon2id (reuses the [privacy] hasher; sha256 fallback)
    created_at: datetime
    expires_at: datetime | None = None  # external-agent keys default to 90-day expiry
    disabled: bool = False
```

Key hygiene (2026 non-human-identity norms): scoped, expiring, rotatable. `morgan clients
rotate <id>` issues a new key and revokes the old; expiry warnings surface in the morning brief.

- **Scope vocabulary (fixed, complete for H1–H2):** `memory.read`, `memory.write`,
  `profile.read`, `skills.read`, `ask`, `replica.read`, `replica.uplink`, `audit.read`.
- **Identity classes license provenance** (the rule from Ports §1): `external-agent` →
  `tool_observed` only; `owner-device` → may assert `user_stated` with `via="device"`;
  `owner` → full trust (the assistant's own surfaces).
- Resolution: one auth middleware on brain-api resolves `Authorization: Bearer` / `X-API-Key`
  against (a) `MORGAN_API_KEY` → implicit `owner` identity (full back-compat, zero config
  change for existing deployments), then (b) the registry. The resolved `ClientIdentity` rides
  the request context; ports and routes read it, never re-parse headers.
- Management: `morgan clients add|list|revoke` CLI verbs + owner-only `/api/clients` CRUD.
  Keys are shown once at creation, stored only as hashes.

## 4. Audit log

One append-only store, written by the kernel, readable by the owner. New
`security/audit.py`. Envelope uses **OCSF-inspired field names** (small fixed schema now, an
export mapping later if ever needed — full OCSF is enterprise overkill):

```
audit_event(id, ts, actor, surface, action, resource, outcome, summary,
            turn_id NULL, prev_hash, hash)
```

- `actor` = client_id (or `owner`); `surface ∈ {api, mcp, v1, cli, scheduler, egress,
  learning}`; `outcome ∈ {ok, denied, error}`. Every boundary writes through the same
  `AuditLog.record()` call, injected via composition (one logger rule, applied to audit).
- **Tamper evidence, right-sized:** `hash = HMAC-SHA256(audit_key, prev_hash ‖ canonical-JSON
  row)` — HMAC (key derived from the owner passphrase/`MORGAN_API_KEY`) rather than a bare
  hash, so an attacker with DB access can't silently re-chain. `morgan audit verify` walks the
  chain. No WORM/external anchoring — single-owner box.
- What gets audited from day one: every port read/write (client, tool, summary), every remote
  egress (provider, role, redaction applied — the hybrid-burst requirement), champion
  promotions, consolidation runs, passport export/import, client registry + **permission/scope
  changes**.
- Surfacing: `GET /api/audit` (owner scope `audit.read`, filterable by client/surface/time);
  included in passport export. Retention (foresight norm): `MORGAN_AUDIT_RETENTION_DAYS`
  default **90** for operational events, **permanent for grant/revoke and promotion events**
  (the two-tier rule), pruned by the nightly worker.

## 5. Settings profiles (presets, not forks)

`MORGAN_PROFILE: Literal["homelab", "desktop", "dev"] | None = None` in `Settings`.
- Applied in a `model_validator` *before* `_fill_provider_defaults`: a profile assigns defaults
  **only to fields still at their class default** — any explicit `MORGAN_*` env var always
  wins. (Phone is not a profile; it's a client of a brain running one of these.)

| Field | `homelab` | `desktop` | `dev` (today's defaults) |
|-------|-----------|-----------|--------------------------|
| event_bus | redis | inproc | inproc |
| vector_backend | qdrant | qdrant-local *(new literal, §6)* | memory |
| enable_scheduling | true (worker) | true (in-proc) | false |
| temporal_db_url | sqlite (volume path) | sqlite (user-data dir) | sqlite ./data |

- The profile name is logged at startup and stamped into audit events — "what topology was
  this?" is always answerable.

## 6. Vector backend: `qdrant-local`

Third `vector_backend` literal (`config.py:42`): qdrant-client embedded mode
(`QdrantClient(path=...)`). `QdrantVectorIndex` already accepts an injected client
(`modules/memory/stores/vector.py:82-87`), so this is a composition branch + a settings field
(`MORGAN_QDRANT_PATH`, default under the data dir) — no new index code. This unblocks the
desktop profile (H2) but lands now because it is two dozen lines riding this baseline.

## 7. `ports/` package conventions

`morgan_brain/ports/` (new top-level package) — rules every port follows:
- A port is a **translation layer**: wire format ↔ kernel interfaces. It imports from
  `interfaces/`, `security/`, `models/` — never from `modules/` concretions or stores.
  (Enforced mechanically, §9.)
- Shape: `ports/<name>/` exposes `build_router(ctx) -> APIRouter` (or an ASGI app for `/mcp`),
  wired only in `composition.py`/`apps/brain_api`. Shared port middleware (auth resolution →
  scope check → rate limit → audit hook) lives in `ports/common.py` and is identical across
  ports.
- Rate limiting: a small in-proc token bucket per client key (settings:
  `MORGAN_PORT_RATE_LIMIT_PER_MIN`, default 120). No external dependency.
- Tests: `tests/ports/<name>/` with contract tests against fakes + golden wire-format fixtures
  (recorded OpenAI chunk shapes, MCP envelopes) so wire compat breaks loudly.

## 8. Observability baseline

- `structlog` stays the single logger; `turn_id` becomes the universal correlation key — it
  already exists on turns; the baseline threads it into audit events, bus event envelopes, and
  port logs.
- The bus `Event` model gains an optional `traceparent: str | None` field (W3C Trace Context,
  carried as envelope metadata and extracted worker-side — the Celery-instrumentation pattern)
  so a turn can be traced across brain-api → redis → worker. Populated only when `[tracing]`
  is installed; ignored otherwise.
- OTel GenAI semantic-convention names adopted inside the `[tracing]` extra: root spans carry
  `user.id` / `session.id` / `gen_ai.conversation.id`; agent/tool spans use
  `gen_ai.operation.name` (`invoke_agent`/`execute_tool`); MCP spans follow the `mcp.*`
  conventions. The semconv is still Development-status — **all attribute constants live in one
  module** and emitted telemetry stamps `telemetry.schema_url`, so upstream renames are a
  one-file change.

## 9. Conformance: the invariant test suite

`tests/invariants/` — the OS contract, executable. Mechanical checks via
[import-linter](https://import-linter.readthedocs.io) contracts (added to CI beside ruff/mypy):
1. Provider SDKs (`openai`, …) importable only under `providers/adapters/`.
2. `ports/*` may not import `modules/*` concretions or store classes.
3. Nothing outside `modules/memory` + `security` imports a `MemoryStore` implementation
   (the MemoryGate rule, made mechanical).

Plus pytest invariants:
4. No `external-agent` write can ever produce `user_stated` (property test over the gate).
5. Provenance rules of §1 hold for every write path (chat, port, uplink, import).
6. Hot path performs no learning calls (request-path module list is import-audited).
7. Every audit `surface` boundary writes an event (each port's contract test asserts it).
8. Migration chain: empty→latest and seeded-v1→latest both green, idempotent.

## 10. Versioning & deprecation

- **Platform:** semver from `0.2.0` (the baseline is the 0.2 line; 0.1 was the pre-OS platform).
- **DB:** integer `schema_version` (§2). **Passport:** `schema_version` in its manifest, may
  lag/lead the DB version (the restore path translates).
- **Ports:** wire formats follow their upstream standards (OpenAI wire, MCP revision); Morgan's
  own additions get a 12-month deprecation window, mirroring MCP's policy.

## 11. Security baseline (defaults, not options)

- Bind `127.0.0.1` by default (`MORGAN_BIND_HOST`; current uvicorn entrypoint binds wider —
  verify and fix as part of this wave). Remote exposure = explicit host + real `MORGAN_API_KEY`
  + reverse-proxy TLS; the documented remote path is **tailnet/VPN, never raw internet**.
- One sanitizer module `security/sanitize.py` (instruction-pattern stripping/neutralizing for
  `<system>`-style and role-marker payloads) applied at **store and recall** for all
  port/import-sourced content — the memory-poisoning mitigation, in exactly one place.
- Keys: hashed at rest, shown once, revocable, expiring (§3); rate limits are **per identity
  class** (generous for `owner`, tight for `external-agent`), not one global number (§7).
- **MCP tool-definition hash pinning** (client side, Morgan consuming external servers):
  **already built** — tool descriptions are sanitized, fingerprint-pinned, allowlisted, and
  default-deny (`modules/mcp`, documented in WIRING §5). The baseline's job is only to declare
  it an invariant and route grant/re-approval events through the audit log.
- **Default-deny egress posture**: tool execution reaches the network only through declared
  tool surfaces; `redact_egress` PII redaction becomes baseline-recommended-on in profile
  presets (`homelab`/`desktop` set it true by default; explicit env still wins).
- `passphrase`-derived encryption (`[privacy]`) unchanged; passport encryption reuses it;
  passport exports **null out secrets** (keys, tokens — the Letta `.af` precedent).

**Extension model (formalizing an existing rule):** Morgan extensions are **MCP servers,
out-of-process, capability-declared** — never in-proc plugin Python (Open WebUI's taxonomy
sprawl and ClawHub's 1,184 malicious skills are the cautionary tales). The existing
`mcp_servers` config grows a manifest shape: name, version, declared tools, required scopes,
pinned hashes. One extension taxonomy, forever.

## 12. Acceptance (definition of done for the baseline wave)

1. All migrations green from empty and seeded DBs; existing dev DB upgrades in place (with
   pre-migration backup file present).
2. Existing 820 tests still green; new invariant suite green; import-linter in CI.
3. A registered `external-agent` key can hit a stub port route, gets scope-checked,
   rate-limited, audited, and its writes land `tool_observed` with full provenance.
4. `morgan audit verify` passes; tampering with a row makes it fail.
5. `MORGAN_PROFILE=desktop` boots single-process with qdrant-local and passes the smoke suite.
6. Zero behavior change for an untouched existing deployment (no profile, owner key only).
7. **Threat-model traceability:** each baseline control maps to a named OWASP Agentic Top-10 /
   MCP Top-10 item in a short appendix table — every control exists for a stated reason.

## 13. Durability check (from the 2026-06-10 foresight wave)

The baseline's four core choices were independently validated for the 12–24 month window:
SQLite-first single-writer (strengthened by the "SQLite renaissance" — Litestream/libSQL; keep
the existing Postgres escape hatch), identity classes (aligned with non-human-identity
least-privilege trends; extensible for per-agent identities), ports as thin translation layers
(rewarded by MCP's stateless RC; note the MCP port must tolerate multi-version negotiation),
and profile presets over forks (Home Assistant's model; Open WebUI's sprawl is the
counter-example). The single watch item: OTel GenAI attribute churn — already isolated (§8).
