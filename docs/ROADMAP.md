# Morgan Brain — Master Roadmap

> **Living document.** The end goal: a brain-like, modular, **self-learning platform** — a
> provider-agnostic foundation for **future agents** *and* a personal, **privacy-first** assistant
> that measurably learns from its owner over time. It runs great fully-local (Ollama is one example
> backend among many: llama.cpp/vLLM/LM Studio locally, or any OpenAI-compatible / remote provider),
> exposes a remote interface, and is built to host and orchestrate future agents safely.
> Single-owner first, multi-tenant-ready. Quality over speed.
>
> Authoritative design: [`docs/superpowers/specs/2026-06-07-morgan-brain-design.md`](superpowers/specs/2026-06-07-morgan-brain-design.md).
> Background: [`docs/ARCHITECTURE_V2.md`](ARCHITECTURE_V2.md). Old monolith archived in `legacy/v0.0.3-monolith`.

## North Star

Every interaction makes Morgan know the owner better, and that knowledge measurably changes the
next response. "Knows me" = **stable traits + evolving facts + learned procedures + emotional
baseline**, all owner-scoped. The owner wires their own Ollama models + a remote interface; Morgan
learns from them continuously and safely.

## Principles (carried from the design spec)

1. **Memory ≠ Learning ≠ Personalization** (MAPLE) — three subsystems, three timescales.
2. **Knowledge evolves, never overwrites** — bi-temporal facts.
3. **Skills are trainable state** (SkillOpt) — validation-gated, zero inference cost.
4. **The seam is the contract** — modules reachable only via typed Protocols + events.
5. **Proactive but consent-gated.**
6. **Privacy-first, single owner, multi-tenant-ready** — `user_id`-keyed, one MemoryGate.
7. **Always shippable** — every wave leaves a working assistant.
8. **Self-improvement is gated** — no learned update (prompt or weights) ships unless it beats the
   current version on a held-out check. Learning never degrades the assistant.
9. **Provider-agnostic** — the LLM/embedding/rerank layers are seams (`LLMClient`, `Embedder`,
   …). Any backend (local Ollama/llama.cpp/vLLM, or remote OpenAI-compatible) plugs in without
   touching the brain. Model *routing* (fast/strong/vision) and capability detection live behind
   the seam. No provider is hardcoded anywhere above the adapter.
10. **A platform for agents, not just an app** — the same primitives that power Morgan (memory,
    skills, tools, MCP, permissions, event bus, agent spawning) are exposed so future agents can be
    hosted, composed, and orchestrated on top, each capability-scoped and permission-gated. Morgan
    is the first resident of its own platform.

## Self-learning mechanism — DECISION (Wave 0 research, 2026-06-08)

Full rationale + citations: [`docs/superpowers/specs/2026-06-08-self-learning-decision.md`](superpowers/specs/2026-06-08-self-learning-decision.md).

**Decision: Memory/RAG-first + an auto-optimized "champion" preprompt (DSPy GEPA). No LoRA by
default.** Evidence: on the LaMP personalization benchmark, RAG gives **+14.92%** vs LoRA/PEFT
**+1.07%** (hybrid +15.98% — LoRA adds only ~0.44pp). A single user never produces enough clean
data early for LoRA to win, LoRA causes catastrophic forgetting, and it bakes personal data
*irreversibly* into weights (destroys "forget me"). So:

1. **Substrate 1 — Memory as the primary learning lever.** Every preference/fact/correction is an
   editable, retrievable **bi-temporal row** (Qdrant + SQLite, already built). Instant update,
   fully reversible, zero training compute, best privacy hygiene. Consolidated by an async nightly
   "sleep" worker (Mem0 ADD/UPDATE/DELETE/NOOP, contradiction → `invalid_at` not delete, bounded
   forgetting).
2. **Substrate 2 — A versioned "champion" preprompt/skill document** (SOUL.md/USER.md + per-skill
   `.md`) for stable, always-on behavior, re-optimized offline by **`dspy.GEPA`** (reflective,
   sample-efficient — beats MIPROv2 ~13% with ~35× fewer rollouts; runs against local models at $0).
   The reflection model = the **largest local model** we can load (small models fail at reflection).
   > **Correction:** `pip install skillopt` does **not exist** (SkillOpt is a 2026 MS Research
   > paper). We adopt SkillOpt's *loop design* (bounded add/delete/replace edits on one champion doc,
   > accept only on a strict held-out win) implemented on the real `gepa-ai/gepa` / `dspy.GEPA`.
3. **Validation gate — "beats-current-or-nothing."** A 3-layer offline eval harness (retrieval
   recall@k → per-user golden set of preference probes → held-out A/B + trait-incorporation trend)
   with a **calibrated cross-family LLM judge**. Optimize a *candidate*, never mutate the live
   champion; promote only on a full-valset win, A/B behind a flag, auto-rollback on regression,
   keep N versioned champions for instant rollback. **Build this eval harness FIRST.**
4. **LoRA = conditional escalation only**, when ALL four hold: 500–1,000+ clean curated pairs in a
   stable domain; a golden-eval-proven gap prompt+RAG can't close; preprompt token/latency
   pressure; and acceptance of an offline pipeline + loss of clean deletion. Then: Unsloth QLoRA →
   **merge** to self-contained GGUF (never the Ollama `ADAPTER` directive) → versioned `morgan:vN`
   tag → eval-gated. Never online/continuous. We **log the signal now**, build the pipeline only if
   the test fires.

**Training signal** (logged from day one on the async post-response path): owner **edits** of
replies (highest value — free correction pairs) > retries/"no, I meant…" > explicit thumbs (down
reliable; **up is the least reliable** — sycophancy). Eval items are firewalled from what the
assistant may consolidate.

**Roadmap impact:** the memory learning-substrate + **eval harness** are pulled forward to the
front of Phase 2 (they deliver the +14.92% with no GPU); the GEPA optimizer is the second half of
Phase 2 / start of Phase 3, gated on the eval harness existing. Anti-sycophancy /
over-personalization guardrails become first-class in Personalization & Proactivity.

## Platform decisions (Wave 0 research)

Full rationale + citations: [`docs/superpowers/specs/2026-06-08-platform-architecture-decision.md`](superpowers/specs/2026-06-08-platform-architecture-decision.md).

- **Provider-agnostic, thin layer.** Build our own Protocol seams (`interfaces/llm.py` `ChatClient`,
  `interfaces/embedding.py` `Embedder`, `interfaces/rerank.py` `Reranker`) over the **OpenAI
  Chat-Completions wire format** (official `openai` SDK + per-provider `base_url`). **Do NOT import
  LiteLLM in-process** (March-2026 PyPI supply-chain compromise) — if a 100+-provider gateway is ever
  needed, run it as a pinned-digest sidecar behind one adapter. Role router (fast/strong/vision/…),
  per-model **CapabilityDescriptor** (seed from vendored pricing JSON + startup probe),
  **structured-output ladder** (native constrained decoding → tool-as-schema → prompted-JSON →
  Pydantic validate + re-ask), fallback as an `LLM_FALLBACK` event.
- **Agent platform, thin & standards-native.** A2A **Agent Card** manifest (+ `x-morgan` grants);
  `AgentSupervisor` lifecycle event-sourced on the bus; **extend** `PermissionGate` into
  capability-token grants (RFC 8693/9396); isolation tiers in-process → **WASI** → Firecracker;
  **MCP host/client hardened now** (OAuth 2.1+PKCE+RFC 8707, server pinning, tool-description
  sanitization). Adopt-now-as-adapters; watch AGNTCY/ANP.
- **Privacy.** Envelope encryption (Argon2id KEK + **SQLCipher**), data classification, **single
  egress chokepoint with reversible PII redaction** (local models full context; remote redacted;
  secret-tier hard-blocked), **Cedar**-backed two-gate consent, `delete_subject()` fan-out, JSON
  export, hash-chained audit log. RAG-first/discardable-LoRA *is* the privacy decision (real erasure).
- **Future-proofing.** Hexagonal core + edge adapters + anti-corruption layer; **add `schema_version`
  to the Event model before events are persisted**; version Protocols additively; capability
  negotiation; hybrid sync-hot-path + event side-effects; feature flags; state in DB not memory.
- **Learning-lifecycle substrate = MLflow 3, local.** GEPA via `mlflow.genai.optimize_prompts` +
  `GepaPromptOptimizer`; champion preprompt = Prompt Registry aliases; validation gate =
  `mlflow.genai.evaluate` + custom scorers + `make_judge`; LoRA tracking via Model Registry (later);
  tracing via the `mlflow-tracing` slim package only. Privacy hard rules:
  `MLFLOW_DISABLE_TELEMETRY=true` + `DO_NOT_TRACK=true`, all judge/reflection models local.

## Status

| Wave | Phase | Outcome | State |
|------|-------|---------|-------|
| — | 0 Foundation | skeleton, config, events, protocols, MemoryGate, data model | ✅ done |
| — | 1 Memory + Reasoning | working text assistant w/ cross-turn recall (vector+BM25+entity, temporal facts) | ✅ done (44 tests, mypy-strict clean) |
| 0 | Research + roadmap | self-learning + platform decisions + roadmap | ✅ done |
| 0.5 | Provider Seam + Privacy Foundation | provider Protocols + role router + capability descriptors + structured-output ladder; encryption + classification + egress redaction; `schema_version` on events; learning-lifecycle seam + MLflow scaffold | ✅ done (204 tests, mypy-strict clean) |
| 1 | 2 Learning + Personalization | async worker: trait/preference extraction → UserModel → measurable adaptation; chosen learning strategy | ⏳ next |
| 2 | 3 Skills + Tools + MCP | platform extensibility + self-evolving skills / auto-preprompt optimizer | ⏳ planned |
| 3 | 4 Proactivity | heartbeat + cron + pattern-triggered, consent-gated | ⏳ planned |
| 4 | 5 Perception/voice + remote | voice (Whisper+emotion) behind the seam; remote gateway (auth, streaming, channels) | ⏳ planned |
| 5 | Self-learning engine | the training loop (preprompt optimizer and/or LoRA pipeline) with held-out gates | ⏳ planned |
| 6 | Integration + wiring | E2E, personalization eval harness, hardening, Ollama+remote wiring guide | ⏳ planned |

## Wave plans

Each wave: **research delta → design spec → implementation plan → subagent-driven execution
(TDD, two-stage review) → verification → finish-branch**. Independent research/audits use
multi-agent workflows. Every wave ends green (tests + ruff + mypy-strict) and shippable.

### Wave 0.5 — Provider Seam + Privacy Foundation (NEW, precedes Wave 1)
Foundational to every later wave; highest single risk-reduction. Closes the gap that the
`LLMClient`/`Embedder` seams named in principle #9 don't yet exist (Phase 1 uses a concrete
`OllamaLLMClient`).
- **Provider seams:** `interfaces/llm.py` (`ChatClient`), `interfaces/embedding.py` (`Embedder`),
  `interfaces/rerank.py` (`Reranker`); refactor `ReasoningModule` onto `ChatClient` via a **role**
  string. First adapters: `OllamaAdapter` + `OpenAICompatAdapter` (official `openai` SDK,
  configurable `base_url`); `AnthropicAdapter` optional.
- **Capability + routing:** `CapabilityDescriptor` (vendored pricing JSON + startup probe + YAML
  override); role router (fast/strong/vision/long_context/embedding/rerank), capability-gated.
- **Structured-output ladder:** native constrained decoding → tool-as-schema → prompted-JSON →
  Pydantic validate + bounded re-ask. Streaming normalized to one internal delta type; jittered
  retry (SDK retries off) + role-level fallback emitting `LLM_FALLBACK`.
- **Future-proofing:** add `schema_version` to `interfaces/events.py` Event + CI schema-drift check.
- **Privacy foundation:** SQLCipher on the SQLite store; Argon2id KEK + envelope DEKs; data
  classification tags; the **egress redaction gateway** (regex → Presidio → deterministic
  placeholders → streaming rehydrator) co-located with the provider layer; secret-tier hard-block.
- **MLflow scaffold:** local tracking (`sqlite:///`), telemetry disabled, behind an `Optimizer`/
  `Registry` seam; `mlflow-tracing` slim wired into the service layer.
- Ends green (tests + ruff + mypy-strict), provider-swappable, with a redaction completeness test.

### Wave 1 — Phase 2: Learning + Personalization
The heart of "learns from me." Async `learning-worker` (off the request path):
- **Extraction** — LLM reads completed sessions → `{facts, preferences, behaviors}` w/ confidence.
- **UserModel maintenance** — stable traits, comm prefs, topics, behavioral patterns, emotional
  baseline, relationship_stage, confidence.
- **Consolidation** — dedup, importance decay, reflection, MEMORY.md curation.
- **Applied personalization** — real trait *selection* (not full-dump) → injected signals; this is
  where the learning becomes visible.
- **`LearningStrategy` seam** — the chosen self-learning mechanism (preprompt optimizer / LoRA)
  plugs in here without touching the request path.
- **Fact writing** — close Phase 1's deferral: extract facts during/after a turn (the read side
  already merges current facts into recall).
- **Eval** — extend the memory-quality harness with personalization metrics (trait-incorporation
  rate, A/B vs stateless).

### Wave 2 — Phase 3: Skills + Tools + MCP + prompt optimizer
- **Tools** — `BaseTool` registry + single `PermissionGate` (built-ins: calc, file, web, memory).
- **MCP Hub** — external integrations (calendar/email/search) as MCP servers; config, not code.
- **Skills** — markdown + frontmatter, trigger-matched, injected into context.
- **Self-evolving skills / auto-preprompt optimizer** — the SkillOpt/DSPy-style loop that improves
  skill docs + the personal preprompt from trajectories, behind a validation gate (per Wave 0).

### Wave 3 — Phase 4: Proactivity
HeartbeatManager (jittered), CronService, pattern-triggered (from mined behavioral patterns),
consent gate keyed on relationship_stage, delivery via channels.

### Wave 4 — Phase 5: Perception/voice + remote interface
- **Perception-gpu** — Whisper ASR + Wav2Vec2 emotion + prosody sarcasm, implementing the existing
  `Perception` Protocol (zero downstream change).
- **Remote gateway** — auth (API key→JWT), streaming (SSE/WebSocket), channel adapters
  (Telegram/Discord), model routing (fast/strong/vision) over Ollama, reliable structured output.

### Wave 5 — Self-learning engine
Implement the chosen mechanism end to end with held-out validation:
- If **preprompt optimizer**: trajectory collection → optimizer (propose bounded edits) → validate
  on held-out conversations → deploy `best_preprompt.md` / `best_skill.md`. Zero inference cost.
- If **LoRA** (escalation): personal dataset curation → QLoRA train (unsloth/axolotl) → eval gate →
  serve via Ollama `ADAPTER` (or merged) → hot-swap. Owner-triggered or scheduled.
- Likely **both**, with preprompt as always-on default and LoRA as opt-in escalation.

### Wave 6 — Integration, eval, wiring guide
End-to-end flow across all services; a **personalization eval harness** (golden set, A/B,
trait-incorporation, LLM-as-judge with mitigations); hardening; and a concrete **wiring guide**:
how the owner points Morgan at their Ollama models, exposes the remote interface securely, and
kicks off the learning loop.

## Definition of done (the goal)

- All phases (0–5) + self-learning engine implemented, each green and shippable.
- A clear, research-backed self-learning mechanism that measurably improves personalization on a
  held-out eval, with a safety gate that prevents regressions.
- A documented path to wire the owner's Ollama models + remote interface and start learning.
- The owner can run it, talk to it, and watch it get more "them" over time.

## Working agreement

- Use multi-agent **workflows** for research/audits and parallelizable work; **subagent-driven
  development** (TDD + two-stage review) for implementation plans.
- Reuse from `legacy/v0.0.3-monolith` where it accelerates (llm/embeddings/tools/channels).
- Never ship a learned update that fails its validation gate.
- Keep `main` green; each wave on its own branch, merged when verified.
