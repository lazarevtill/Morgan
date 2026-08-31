# Morgan Brain — Master Roadmap

> **Living document.** The end goal: a brain-like, **self-learning**, provider-agnostic personal
> agent kernel that measurably learns from its owner over time, exposed through a terminal client
> and an MCP server. It runs great fully local (llama.cpp is the default provider; Ollama and any
> other OpenAI-compatible endpoint remain supported non-default provider keys) and never requires
> a third-party model endpoint. Single-owner first, multi-tenant-ready. Quality over speed.
>
> Authoritative design: [`docs/superpowers/specs/2026-06-07-morgan-brain-design.md`](superpowers/specs/2026-06-07-morgan-brain-design.md)
> (kernel semantics — memory, learning, personalization). Current direction:
> [the local-first reshape design](superpowers/specs/2026-08-02-morgan-reshape-local-first-design.md)
> (diagnosis, target architecture, milestone plan; supersedes the earlier Personal Agent OS
> horizon plan). Run guide: [`docs/WIRING.md`](WIRING.md).
>
> The original platform build (phases 0–5 + the self-learning engine, built against Ollama with a
> proactivity engine, an MCP host stub, and a voice seam) is archived at tag `legacy-v0.0.4-full`.
> Its subsystems with no production importers — the MCP host stub, proactivity/heartbeat, the
> voice seam, and the SQLCipher/Cedar privacy layer — were deleted in the local-first reshape; see
> §6 of the reshape design for the verified evidence. The pre-reshape monolith predating that build
> is archived separately at tag `legacy-v0.0.3-monolith`.

## North Star

Every interaction makes Morgan know the owner better, and that knowledge measurably changes the
next response. "Knows me" = **stable traits + evolving facts + learned procedures + emotional
baseline**, all owner-scoped, all project-scoped. The owner wires their own llama.cpp models;
Morgan learns from them continuously and safely.

## Principles

1. **Memory ≠ Learning ≠ Personalization** (MAPLE) — three subsystems, three timescales.
2. **Knowledge evolves, never overwrites** — valid-time facts (`valid_from`/`valid_to`/`superseded_by`).
3. **Skills are trainable state** (SkillOpt) — validation-gated, zero inference cost.
4. **The seam is the contract** — modules reachable only via typed Protocols + events.
5. **Privacy-first, single owner, multi-tenant-ready** — `user_id`-keyed and `project`-keyed,
   one `MemoryGate`.
6. **Self-improvement is gated** — no learned update (prompt or weights) ships unless it beats
   the current version on a held-out check, and the promotion gate is disarmed until that check
   is statistically sound. Learning never degrades the assistant.
7. **Provider-agnostic** — the LLM/embedding layers are seams (`ChatClient`, `Embedder`). Any
   OpenAI-compatible backend plugs in without touching the brain. Model routing (strong/fast/
   judge/reflection) lives behind the seam. No provider is hardcoded anywhere above the adapter.
8. **One brain, homelab-hosted.** A single always-on deployment that every owned device and AI
   tool connects to — one store, one truth. Device sync and a read-only replica are out of scope.

## Self-learning mechanism — DECISION (research, 2026-06-08)

Full rationale + citations: [`docs/superpowers/specs/2026-06-08-self-learning-decision.md`](superpowers/specs/2026-06-08-self-learning-decision.md).

**Decision: memory/RAG-first + an auto-optimized "champion" preprompt (GEPA-style reflection). No
LoRA by default.** Evidence: on the LaMP personalization benchmark, RAG gives **+14.92%** vs
LoRA/PEFT **+1.07%** (hybrid +15.98% — LoRA adds only ~0.44pp). A single user never produces
enough clean data early for LoRA to win, LoRA causes catastrophic forgetting, and it bakes
personal data *irreversibly* into weights (destroys "forget me"). So:

1. **Substrate 1 — Memory as the primary learning lever.** Every preference/fact/correction is an
   editable, retrievable **valid-time row** in the shared SQLite database. Instant update, fully
   reversible via `forget()`, zero training compute, best privacy hygiene. Consolidated by an
   async nightly worker (ADD/UPDATE/DELETE/NOOP, contradiction → close the interval, not delete;
   bounded forgetting).
2. **Substrate 2 — A versioned "champion" preprompt**, re-optimized offline by a GEPA-style
   reflective loop (sample-efficient, runs against local models at $0). The reflection model is
   the largest local model bound to the `reflection` role — small models fail at reflection.
3. **Validation gate — "beats-current-or-nothing," and disarmed until it is sound.** A 3-layer
   offline eval harness (retrieval recall@k → per-user golden set of preference probes → A/B with
   a calibrated cross-family LLM judge) gates every promotion. `MORGAN_ENABLE_CHAMPION_PROMOTION`
   defaults to `false`: the current gate is a bare `>` on one scored run over a 12-item golden
   set, too statistically weak to trust unattended. Optimize a *candidate*, never mutate the live
   champion; keep versions for rollback. Growing the golden set and replacing the bare `>` with a
   paired-bootstrap win test is the open work that flips this flag — see the reshape design's
   milestone 3.
4. **LoRA = conditional escalation only**, when ALL four hold: 500–1,000+ clean curated pairs in a
   stable domain; a golden-eval-proven gap prompt+RAG can't close; preprompt token/latency
   pressure; and acceptance of an offline pipeline + loss of clean deletion. Never online/
   continuous. We log the signal now, build the pipeline only if the test fires.

**Training signal** (logged from day one on the async post-response path): owner **edits** of
replies (highest value — free correction pairs) > retries/"no, I meant…" > explicit thumbs (down
reliable; **up is the least reliable** — sycophancy). Eval items are firewalled from what the
assistant may consolidate.

## Platform decisions (research, 2026-06-08)

Full rationale + citations: [`docs/superpowers/specs/2026-06-08-platform-architecture-decision.md`](superpowers/specs/2026-06-08-platform-architecture-decision.md).
The reshape narrowed the scope this research covers — the two points below are the parts that
still hold; the multi-agent-platform and envelope-encryption ambitions the same document explored
did not survive the reshape (§6, §10 of the reshape design) and are not part of the current plan.

- **Provider-agnostic, thin layer.** Our own Protocol seams (`interfaces/llm.py` `ChatClient`,
  the memory-layer `Embedder`) over the **OpenAI Chat-Completions wire format** (official `openai`
  SDK + per-provider `base_url`). Do not import a 100+-provider gateway library in-process. Role
  router (strong/fast/judge/reflection), per-model **CapabilityDescriptor**, **structured-output
  ladder** (native constrained decoding → tool-as-schema → prompted-JSON → Pydantic validate +
  re-ask), fallback as an `LLM_FALLBACK` event.
- **At-rest protection is a host property, not a code feature.** Field-level encryption is
  incompatible with the FTS5 keyword index and would not cover vectors anyway. Volume-level
  encryption (LUKS or equivalent) on the homelab host covers the entire database; transport
  protection is a bearer token over the NetBird overlay network or TLS at a reverse proxy. See
  [`docs/OPERATIONS.md`](OPERATIONS.md).
- **Learning-lifecycle substrate.** A `PromptRegistry`/`Optimizer` seam with a local-SQLite
  backend by default and an MLflow-backed backend (`MORGAN_LEARNING_BACKEND=mlflow`) as the
  scale-up path — GEPA via `mlflow.genai.optimize_prompts`, champion preprompt = Prompt Registry
  aliases, validation gate = `mlflow.genai.evaluate` + custom scorers. Privacy hard rules when
  MLflow is enabled: `MLFLOW_DISABLE_TELEMETRY=true` + `DO_NOT_TRACK=true`, all judge/reflection
  models local.

## Status

The current plan is the [local-first reshape](superpowers/specs/2026-08-02-morgan-reshape-local-first-design.md),
sequenced as milestones M0–M3.

| Milestone | Deliverable | State |
|---|---|---|
| M0 | Repo hygiene: end CRLF churn, tag `legacy-v0.0.4-full`, delete subsystems with no production importers, fix the non-editable-install wheel bug, docs truth pass | done |
| M1 | One SQLite database (WAL + `sqlite-vec` + FTS5, Cyrillic-aware) behind an extended `MemoryGate` covering the cold path too; `project` scoping on every read/write; `forget()` cascading erasure; the cold path genuinely off the request (bounded async queue, bus lifespan wired into `brain-api`); llama.cpp defaults with all four roles bound and the promotion flag off; the `morgan` CLI (`remember`/`recall`/`facts`/`forget`/`ask`/`doctor`/`receipts`) | done |
| M2 (partial) | MCP server (`morgan-mcp`, stdio + streamable-HTTP with a bearer token) — done, ahead of its original milestone. ChatGPT import with an import-time eval holdout, and `SKILL.md` conformance — not started | MCP server done; import not started |
| M3 | Noise-floor measurement → 100+ labeled golden items from the import holdout → a paired-bootstrap promotion gate → flip `MORGAN_ENABLE_CHAMPION_PROMOTION` to `true` | not started |

Also delivered, ahead of or alongside the milestone plan above: calibration scoring (Brier score +
ECE) added to the eval gate, report-only for now — groundwork for M3's promotion gate, not yet
wired to a pass/fail decision.

And the **dual-brain memory + governance graft**
([design](superpowers/specs/2026-08-31-dual-brain-memory-and-pattern-register-design.md)), from
VoiceMem (arXiv:2608.26005) and Ouroboros:

| Graft | What it does | State |
|---|---|---|
| Entity write path | Fixed a defect it surfaced: `Memory.entities` was never populated by any write path, so the entity-overlap ranking — one of `recall`'s three fused signals — was empty in production. One script-aware extractor now serves both paths. | done |
| Semantic upper index | Schema → entity routing above the store; `recall` narrows every signal to the candidate pool before searching. Routing returns "search everything" rather than an empty pool, so it can only cost precision, never recall. | done |
| Persona graph | Attitudes stay anchored to what they concern; promotion to a stable trait needs recurrence across several anchors *and* sessions. Short horizon per turn (cold path), long horizon nightly. | done |
| Cluster emergence | Coherent subgroups earn their own slot from co-retrieval statistics, gated by an LLM judge on relevance/importance/completeness. Promotes nothing without a judge. | done |
| Pattern register | Corrections are grouped into *classes*, counted, and fed back to the optimizer — including whether a class recurred after its fix, which says the fix was at the wrong depth. | done |
| Gate integrity | The eval gate is fingerprinted; a candidate scored on a different or weaker gate is refused, as is one whose body addresses the evaluator. | done |
| Decision receipts | Every promotion decision — and every rejection, with its reason — is recorded and surfaced by `morgan receipts`. | done |

The accuracy numbers behind the first four are the **paper's**, on LoCoMo/LongMemEval/Memora.
They have not been reproduced here and cannot be until `tests/memory_quality/` runs against a real
embedder — see M3's noise-floor work.

### Known limitations (see [`CLAUDE.md`](../CLAUDE.md) and [`WIRING.md`](WIRING.md) for detail)
- `recall` has no relevance floor — a query against a non-empty project always returns something.
- `forget()` does not erase vectors under `vector_backend=qdrant` (the `sqlite` default is fully covered).
- The MCP HTTP transport serves openly whenever `MORGAN_API_KEY` is unset or left at `change-me`.
- Retrieval quality is unmeasured — `tests/memory_quality/` runs over a hash embedder, so the
  upper index's benefit is imported from the paper, not observed here.
- Entity extraction is deterministic and cased-script only; Chinese, Japanese, Arabic and Hebrew
  yield nothing rather than a guess.
- Persona attribution and cluster emergence record/promote nothing without a reachable model,
  by design.

## Working agreement

- Keep `main` green: `pytest` + `ruff check` + `ruff format --check` + `mypy --strict`.
- No learned update ships without passing its validation gate.
- Root-cause fixes only — a workaround gets flagged, not silently applied.
- Every memory read and write goes through `MemoryGate`, and is `user_id`- and `project`-scoped.
