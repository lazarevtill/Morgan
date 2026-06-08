# Decision Record — Platform Architecture (provider-agnostic, agent-hosting, privacy, future-proofing)

**Date:** 2026-06-08 · **Status:** Accepted · **Source:** Wave 0 research workflow `morgan-platform-research-wave0b` (5 agents, web-researched, 2025–2026 sources)

## Context

Morgan must be the best-possible **future-proof platform for agents** *and* a **privacy-first**
personal assistant, **provider-agnostic** (any local or remote LLM backend; Ollama is one example).
The research found a concrete gap: ROADMAP principle #9 names `LLMClient`/`Embedder` seams that **do
not exist** — Phase 1's `ReasoningModule` uses a concrete `OllamaLLMClient`. The provider seam +
privacy foundation are therefore inserted as **Wave 0.5**, before Phase 2.

## Decisions

### 1. Provider-agnostic model layer — BUILD THIN, no in-process gateway
- **Do NOT import LiteLLM (or any 100+-provider gateway) in-process.** Rationale is decisive: the
  **March 2026 LiteLLM PyPI compromise** (1.82.7/1.82.8, `litellm_init.pth` cred-harvesting at
  interpreter start) gives a single key-holding privacy process maximal blast radius; plus ~8 ms
  overhead and FastAPI/aiohttp memory leaks. If exotic remote providers are ever needed, run
  LiteLLM/TensorZero/Bifrost as a **pinned-digest sidecar** with locked egress; one
  `OpenAICompatAdapter` points at it. Core never imports it.
- **Internal wire format = OpenAI Chat Completions + `/v1/embeddings`**, via the official `openai`
  SDK with a per-provider configurable `base_url`+`api_key`. Natively covers Ollama `/v1`, vLLM,
  llama.cpp llama-server, LM Studio, OpenRouter, any OpenAI-compatible remote. Anthropic-native = one
  adapter behind the same Protocol.
- **Three new seams** (alongside `interfaces/reasoning.py`): `interfaces/llm.py` (`ChatClient`:
  `agenerate`, `astream`), `interfaces/embedding.py` (`Embedder`: `aembed`), `interfaces/rerank.py`
  (`Reranker`: `arerank`). The `Reasoner` depends only on `ChatClient` via a **role string**, never a
  provider SDK or model name.
- **CapabilityDescriptor** per (provider, model): `{context_window, max_output, supports_tools,
  supports_parallel_tools, json_mode(none|json_object|json_schema), supports_grammar,
  supports_vision, embedding_dim, cost_in, cost_out, observed_latency}`. Seed from a **vendored**
  copy of LiteLLM's `model_prices_and_context_window.json` (DATA, not code), YAML override, **probe
  at startup** (Ollama silently caps context ~4K; Claude strict-mode rejects much JSON Schema —
  capability is not inferable from the wire shape).
- **Role router:** logical roles `{fast, strong, vision, long_context, embedding, rerank}` → ordered
  config bindings; business code asks for a ROLE. Static/declarative + capability-gated. Defer
  semantic routing.
- **Structured-output ladder** (capability-gated, per call): native constrained decoding
  (vLLM guided_json / llama.cpp GBNF / Ollama `format` / XGrammar) → tool-call-as-schema →
  prompted-JSON → **always** Pydantic validate + bounded re-ask (feed the error back).
- **Resilience:** jittered backoff on 429/5xx/timeout with SDK-internal retries **disabled**;
  role-level ordered fallback; emit `LLM_FALLBACK` EventBus event; fail over only before first token.
  Normalize all streaming into one internal event type (`text_delta|tool_call_delta|usage|finish`).
- Embeddings/rerank are separate Protocols; embedding model+dim live in the descriptor (Qdrant
  collections are dimension-locked → named-vectors/model-migration on change). Keep the existing
  rerank fallback ladder (remote → CrossEncoder → cosine → BM25).

### 2. Agent platform — thin, standards-native, on existing seams
- **Manifest:** adopt the **A2A Agent Card** (JSON Schema 2020-12, `/.well-known/agent-card.json`)
  as the internal registry record, extended with a private `x-morgan` block of capability grants
  `{tools, memory_scopes, egress_allowlist, isolation_tier}`. Compile markdown+frontmatter agent/skill
  defs into the Card. Keep OASF-mappable.
- **Runtime:** small `AgentSupervisor` (defined → admitted → running → paused/awaiting-approval →
  completed/failed → evicted), **event-sourced on the existing EventBus** (`agent.*` events) + a DB
  task log so restarts resume. Defer Temporal.
- **Permissions:** **extend** the existing `PermissionGate` (don't replace) from AUTO/ASK/DENY into
  capability-based grants `{tool, allowed-params, scope, egress_allowlist, memory_namespaces, TTL,
  signature}`, default-deny, resolved at admission, enforced at `ToolExecutor` + `MemoryGate`,
  audited. Use RFC 8693 token exchange + RFC 9396 RAR to down-scope the owner token. Start COARSE
  (default-deny + audit) for the single owner; add signed-token ceremony when untrusted agents arrive.
- **Isolation tiers** (pluggable behind ToolExecutor/SkillEngine): in-process (first-party) →
  **WASI/Wasm default** for third-party skills → Firecracker microVM for arbitrary code. Ship
  in-process + Wasm first.
- **Interop ADOPT NOW:** be an **MCP host/client with full hardening** (OAuth 2.1 + PKCE + Resource
  Indicators RFC 8707, server pin/allowlist, tool-description sanitization + hash-verify — 5.5% of
  public MCP servers carry tool-poisoning, per a 1,899-server study; isolate each server). This fills
  the empty `modules/mcp/` stub (Phase 3). Expose the A2A Card shape now; defer full A2A serving.
  WATCH (don't build): AGNTCY/OASF, ANP/DID (~2027), MCP↔A2A bridge.
- **Orchestration:** Supervisor/Orchestrator-Worker (Planner → Agents → shared tooling) as typed
  state-graph routing over the EventBus, Reasoner as the thin router; Planner-Executor-with-replanner
  for long autonomous tasks (Phase 4).

### 3. Privacy model — layered, single-owner default
1. **Envelope encryption at rest:** per-domain DEKs (AES-256-GCM/XChaCha20-Poly1305) wrapped by one
   KEK from the owner passphrase via **Argon2id (RFC 9106)**. **SQLCipher** for SQLite; LUKS/dm-crypt
   for the disk; app-side encrypt sensitive Qdrant payload fields (vectors + opaque ciphertext);
   Redis ephemeral/TTL only. KEK in memory only after unlock.
2. **Data classification** (public/personal/sensitive/secret) + **single egress chokepoint** (the new
   provider layer): local providers get full context; **remote providers get only reversibly-redacted
   context**; secret-tier hard-blocked from prompts and the vector store.
3. **Reversible PII redaction at egress:** regex → Presidio/spaCy NER → deterministic session-stable
   `«TOKEN_NNN»` placeholders → streaming-aware rehydration of the provider's response.
4. **Two-gate consent, policy-as-code:** `MemoryGate` (write consent) + `PermissionGate` (read/execute)
   backed by **AWS Cedar** (readable, formally verifiable, 42–60× faster than OPA Rego). Coarse
   initially for the single owner.
5. **Forget/export/audit:** `delete_subject(user_id, selector)` fan-out across SQLite/Qdrant/Redis/logs;
   single JSON export + re-import (GDPR Art.17/Art.20); **append-only hash-chained audit log**
   (HMAC-SHA256 chain). This is *why* RAG-first/discardable-LoRA is also the privacy decision —
   personal data stays in editable rows, never baked irreversibly into weights.
6. **Threat model:** treat all tool/RAG/web/email content as untrusted (prompt injection LLM01:2025,
   memory-poisoning EchoLeak CVE-2025-32711, MCP tool-poisoning). Gate provenance before writing to
   long-term memory; firewall eval/golden items from consolidation.

### 4. Future-proofing principles (enforce everywhere)
Hexagonal core + edge adapters + **Anti-Corruption Layer** (vendor SDK types never cross into
`morgan_brain/models/`). Version every Protocol (add optional methods / new versions, never break
signatures). **Schema-as-code:** add `schema_version` to `interfaces/events.py` Event model **now**,
before events are persisted for learning/replay; additive-optional only; CI schema-drift check.
Capability negotiation (every plugin/skill/tool/agent/provider declares requires/provides). Hybrid
execution: synchronous Protocol calls on the hot chat path, versioned domain events on the bus for
all side-effects. Feature flags (`MORGAN_ENABLE_*`) per provider/adapter/protocol/plugin. State in
the DB, never process memory.

## Roadmap impact
- **Insert Wave 0.5 — "Provider Seam + Privacy Foundation"** before Wave 1: the three provider
  Protocols + role router + CapabilityDescriptor + structured-output ladder + first adapters
  (Ollama/OpenAI-compat), refactor `ReasoningModule` onto `ChatClient`, add `schema_version` to
  Event, and land the encryption + classification + egress-redaction foundation.
- **Cross-cutting Platform Layer track** spans Phase 2–5 (provider layer, agent platform, privacy,
  future-proofing) rather than living in one phase.
- Phase 3 = hardened MCP host/client + capability-token permissions + WASI isolation + A2A Card +
  AgentSupervisor. Phase 5 reuses the **same role router** for fast/strong/vision; remote access via
  **Tailscale tailnet (no public ports)** + app-level JWT/API-key + SSE heartbeat.

## Key citations
LiteLLM compromise (Datadog Security Labs; Trend Micro; docs.litellm.ai/blog/security-update-march-2026)
· TensorZero / Bifrost gateway benchmarks · vLLM/XGrammar structured outputs (Red Hat) · Pydantic-AI
output modes + FallbackModel · MCP+A2A under Linux Foundation (a2a-protocol.org; ACP merged Sept 2025)
· MCP attack surface (arXiv:2509.18787; OWASP MCP Cheat Sheet; modelcontextprotocol.io security) ·
isolation ladder WASI/gVisor/Firecracker (Northflank) · checkpoints≠durable-execution (Diagrid) ·
SQLCipher + Argon2id (RFC 9106) · reversible PII redaction at egress · OAuth2.1+PKCE / RFC 8693 / RFC
9396 · Cedar vs OPA (Oso; Natoma) · machine-unlearning leaves latent influence (IAPP; arXiv:2412.06966)
· OWASP LLM01:2025 + EchoLeak CVE-2025-32711 · hash-chained audit log · GDPR Art.17/Art.20.
