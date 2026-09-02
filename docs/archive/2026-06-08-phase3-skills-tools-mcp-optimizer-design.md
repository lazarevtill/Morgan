# Phase 3 (Wave 2) — Skills + Tools + MCP + GEPA Optimizer — Design

**Date:** 2026-06-08 · **Status:** Approved (from decisions) · **Branch:** `feat/phase3-skills-tools`
**Decisions:** [platform ADR](../decisions/platform-architecture.md) · [self-learning ADR](../decisions/self-learning.md) · [ROADMAP](../ROADMAP.md)

## Goal
Turn Morgan into a **platform**: pluggable permission-gated **tools**, external integrations via a
**hardened MCP host/client**, trigger-matched **skills**, and — the second half of self-learning —
the **GEPA champion-preprompt optimizer loop** that improves the system prompt/skill docs on the
eval gate built in Phase 2. Everything stays provider-agnostic and behind the existing seams.

Increments, each green & shippable:
- **A. Tools** — `BaseTool` registry + capability-token `PermissionGate` + built-in tools.
- **B. Skills** — markdown+frontmatter skills, trigger match, injected into context; versioned via the `PromptRegistry`.
- **C. MCP host/client (hardened)** — fill the empty `modules/mcp/` stub; OAuth2.1+PKCE+RFC8707, server pinning, tool-description sanitization, per-server isolation.
- **D. GEPA optimizer loop** — `Optimizer` impl that proposes champion-preprompt candidates from trajectories + the gate, promotes only on a held-out win (`EvalGate`), behind a flag with rollback.

## A. Tools (request path, permission-gated)
The `ToolExecutor` Protocol already exists (`interfaces/tools.py`); the `PermissionGate` exists
(`security/permissions.py`, AUTO/ASK/DENY). Build:
- `morgan_brain/modules/tools/executor.py`: `ToolRegistry` + `Executor` implementing `ToolExecutor`
  — register `BaseTool`s, list schemas, `execute(name, *, user_id, **kwargs)` gated by the
  PermissionGate, audited via an event.
- **Extend** `PermissionGate` into capability-token grants (platform ADR): `Grant{tool, scope
  (read|write|execute), allowed_params, egress_allowlist, memory_namespaces, ttl}` resolved at
  admission, enforced at execute. Default-deny; coarse for the single owner (AUTO for safe built-ins,
  ASK for side-effecting). Keep the existing enum working (back-compat).
- Built-in tools (`modules/tools/builtin/`): `calculator`, `memory_search` (via MemoryGate),
  `web_search` (behind a flag; goes through the privacy egress gate for remote), `fetch_url`,
  `current_time` (clock-injected). Each a small `BaseTool` with a JSON-schema, unit-tested with fakes.
- Reasoning integration: the reasoner can request tools by exposing their `ToolSpec`s to the
  `strong` role (capability-gated — only if the model supports tools), execute returned tool_calls,
  feed results back. Bounded tool-call loop.

## B. Skills (trigger-matched, versioned, trainable)
The `SkillEngine` Protocol exists; Phase 1 has `NoopSkillEngine`. Build:
- `morgan_brain/modules/skills/registry.py`: load markdown skills with YAML frontmatter
  (`name, triggers, tools, model, version`) from a skills dir + bundled ones; `select(perception)`
  matches triggers (keyword/intent) → returns `Skill`s injected into context (already wired in the
  orchestrator). `get`, `deploy` (install a validated version).
- Bundled skills (`modules/skills/bundled/*.md`): conversation, empathy, research, coding, planning.
- **Versioning:** skill bodies are champion documents in the `PromptRegistry` (Wave 0.5 seam) so they
  participate in the GEPA loop (D). The active skill body = `registry.champion(skill_name)`.

## C. MCP host/client (hardened) — fills `modules/mcp/`
External integrations (calendar/email/search) as MCP servers; config not code. Build behind the
`[mcp]` optional extra (lazy import the `mcp` SDK):
- `modules/mcp/hub.py`: connect to configured MCP servers (stdio/SSE/Streamable-HTTP), discover
  tools/resources, expose them through the same `ToolExecutor`/`PermissionGate` as built-in tools
  (each MCP tool is capability-granted + isolated per server).
- **Hardening (non-negotiable, platform ADR):** OAuth 2.1 + PKCE + Resource Indicators (RFC 8707)
  binding tokens to one server; server **pin/allowlist**; **sanitize + hash-verify tool
  descriptions** (tool-poisoning defense); treat all MCP content as untrusted (provenance-gate before
  any memory write); per-server isolation; egress through the privacy gate for remote servers.
- `config.MCP_SERVERS` (yaml/dict). Default empty (no servers) → no behavior change. Unit-tested with
  a fake MCP server (no network); live tested separately.

## D. GEPA optimizer loop (the self-learning second half)
On the Phase-2 `EvalGate` + Wave-0.5 `Optimizer`/`PromptRegistry` seams:
- `morgan_brain/learning/optimizer_gepa.py`: `GepaOptimizer` implementing the `Optimizer` Protocol.
  When the `[learning]` extra is present, wraps `mlflow.genai.optimize_prompts(predict_fn, train_data,
  prompt_uris, optimizer=GepaPromptOptimizer(reflection_model=<biggest local role>, max_metric_calls),
  scorers=[...from eval harness...])`; forces `MLFLOW_DISABLE_TELEMETRY`/`DO_NOT_TRACK`. Without the
  extra, falls back to a built-in lightweight reflective loop (propose bounded edits via the `strong`
  role → score with the eval harness → keep best) so the loop works dependency-light too.
- **Champion flow:** mine ~20–50 `(context, query, good_output)` examples from high-value signals
  (Phase 2A); optimize a **candidate** champion document; run the Phase-2 `EvalHarness` on the
  candidate; `EvalGate.promote_if_better` registers + sets champion ONLY on a held-out win; ship
  behind a flag; keep N versioned champions for rollback. Hard char-budget on the proposer
  (anti-bloat). Reflection model = the largest local model (a distinct `reflection` role).
- Runs offline in the `learning-worker` (Cron/idle), never the hot path. Zero inference-time cost
  (the deployed champion is just a better prompt).
- This realizes "skills are the trainable state": both the system preprompt AND skill bodies are
  GEPA-optimized champions, gated.

## Interfaces / wiring
- Tools + MCP plug into the existing `ToolExecutor` seam; the orchestrator gains a bounded tool loop
  in the reasoner (capability-gated). Skills replace `NoopSkillEngine` in composition. The optimizer
  runs in the worker, reading signals + writing champion versions via the registry — never the request
  path. All LLM calls via roles (`strong`, `reflection`, `judge`).
- Permissions/audit: every tool/MCP call is capability-gated + audited (hash-chained log seam from the
  platform ADR — minimal now). MCP content is provenance-gated before memory writes.

## Testing
Unit with fakes (no network): tool registry + permission grants + each built-in tool; skill
frontmatter parsing + trigger match + champion-body selection; MCP hub against a fake server +
description-sanitization + allowlist enforcement; GEPA optimizer with a fake reflection LLM + the
eval harness (candidate beats/loses → promote/reject). Integration: a tool-using turn end-to-end;
the optimizer promotes a better champion in a `LocalPromptRegistry`. Phase-1/2 suites stay green.

## Non-goals (deferred)
A2A serving surface (manifest only), WASI/Firecracker isolation (in-process + per-server logical
isolation now; sandbox tiers when hosting untrusted code), Cedar engine (coarse grants now), real
OAuth flows against live providers (seam + fake now), LoRA. Multi-user.

## Increment order & DoD
A → B → C → D, each on the wave branch, each green (pytest + ruff + mypy-strict). After A+B Morgan is
tool- and skill-capable; after C it integrates external MCP servers safely; after D the champion
preprompt/skills **self-improve on the gate**. Merge to main when green.
