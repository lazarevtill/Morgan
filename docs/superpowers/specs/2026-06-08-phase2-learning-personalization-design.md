# Phase 2 (Wave 1) — Learning + Personalization — Design

**Date:** 2026-06-08 · **Status:** Approved (from decisions) · **Branch:** `feat/phase2-learning`
**Decisions:** [self-learning ADR](2026-06-08-self-learning-decision.md) · [platform ADR](2026-06-08-platform-architecture-decision.md) · [ROADMAP](../../ROADMAP.md)

## Goal
Make the assistant **measurably learn the owner**: capture signal from each turn, consolidate it into
durable memory + a compact user profile, inject that profile every turn so responses adapt, and
**gate every learned change on a held-out eval** so learning never degrades quality. This realizes
MAPLE's Memory/Learning/Personalization split on the foundation built in Phases 0–0.5.

Built in increments, each green & shippable:
- **A. Signal capture** — typed edit/retry/thumb records on the async post-response path.
- **B. Consolidation worker** — bi-temporal fact consolidation (Mem0 ADD/UPDATE/DELETE/NOOP) + decay/forgetting, off the hot path.
- **C. User profile + applied personalization** — compact stable/dynamic `UserModel`/`USER.md`, real trait *selection* injected every turn, CIPHER learn-from-edits.
- **D. Eval harness (the gate)** — 3-layer golden eval + calibrated cross-family judge, on the MLflow seam. **Prereq for any self-learned promotion** — but built here so it's ready.

> The **GEPA champion-preprompt optimizer loop** (the second half of self-learning) is wired in
> Wave 2/Phase 3 on top of the eval gate + `PromptRegistry`/`Optimizer` seams shipped in Wave 0.5.
> Phase 2 delivers the *memory* learning substrate (the +14.92% lever) + the gate.

## A. Signal capture
Learning needs training signal. On the post-response async path (the existing `RESPONSE_GENERATED`
subscriber, off the hot path), log a typed record:
```
InteractionSignal(user_id, session_id, turn_id, context_summary, query, original_reply,
                  user_edit: str | None, retried: bool, thumb: Literal["up","down"] | None, created_at)
```
- New `morgan_brain/learning/signals.py` + a `SignalStore` (SQLite, deterministic clock) keyed by
  user_id. Value order (for later consumers): **edit > retry/"no, I meant…" > thumb-down**;
  thumb-up is logged but flagged low-trust (sycophancy).
- API/CLI affordances to record an edit/retry/thumb against the last turn (thin; full UI later).
- This is the substrate for BOTH consolidation (B) and the GEPA optimizer (later). **Log from day one.**

## B. Consolidation worker (bi-temporal, async)
Runs in `learning-worker` (off the request path), triggered by CronService (nightly) + idle-gap +
an importance-accumulation threshold (generative-agents style). Replaces Phase 1's `MinimalLearner`
with a real `Learner`:
- **Extract-as-filter on the hot path** stays cheap (Phase 1 already stores episodics); the heavy
  work is here.
- **Fact consolidation (Mem0 ADD/UPDATE/DELETE/NOOP):** an LLM (role `strong`) reads recent
  episodics + signals and proposes fact operations; deterministic dedup pre-filter; contradiction →
  close old interval (`valid_to`/`superseded_by`) — never hard-delete (uses the Phase-1
  `SqliteTemporalStore`). Pin episode timestamps (don't let the local LLM hallucinate dates).
- **Decay/forgetting:** importance + recency + frequency scoring; enforce a hard memory budget with
  global downscaling so recall quality doesn't collapse at scale.
- **Reflection (light):** periodically synthesize higher-level insights ("owner plans the week
  Sundays") as memories with evidence citations (memory IDs) + a confidence/decay tier.
- Provider-agnostic: calls roles (`strong`), never a model name. Off-path: never blocks chat.

## C. User profile + applied personalization
The piece that makes learning *visible* each turn.
- **`UserModel` maintenance:** the Learner maintains the Phase-1 `UserModel` (traits, comm_prefs,
  topics, behavioral_patterns, emotional_baseline, relationship_stage, confidence) from signals +
  consolidated facts. Persist to SQLite + a human-readable `USER.md` (stable vs dynamic split: stable
  traits/prefs/interests; dynamic emotional state/session context). Cap to a few hundred tokens.
- **CIPHER learn-from-edits:** when the owner edits a reply, distill a natural-language preference
  delta ("prefers concise, code-first, no hedging") and merge into comm_prefs/traits — weight-free,
  data-efficient.
- **Applied personalization (request path):** replace Phase-1 `PassthroughPersonalizer` with a real
  `Personalizer` that *selects* the traits relevant to THIS turn (budget-aware, ~15% of context),
  injects them as signals, and tunes tone/length. Inject the compact profile **every turn** (defeats
  PrefEval context decay). Reads only; writes nothing.
- **Guardrails (first-class):** anti-sycophancy + over-personalization — keep factual correctness
  independent of the user model; never inject stale/irrelevant prefs (OP-Bench negatives); downweight
  thumb-ups.

## D. Eval harness (the gate) — `tests/eval/` + MLflow seam
"Beats-current-or-nothing." 3 layers producing a JSON scorecard (also CI regression protection),
built on the Wave-0.5 `learning_lifecycle` seam (local backend now; MLflow `genai.evaluate` when the
`[learning]` extra is enabled):
- **L1 (no judge):** retrieval recall@k / F1 over vector+BM25+entity+RRF (LongMemEval taxonomy).
- **L2 (did it learn ME):** a hand-authored 20–30 item per-user golden set tagged by probe type —
  explicit recall, implicit-trait inference, **preference UPDATE** (value after the owner changed
  their mind), long-gap decay, **over-personalization NEGATIVES**, abstention. Binary
  preference-following accuracy.
- **L3:** held-out A/B (memory ON vs OFF) + trait-incorporation-rate **time series**.
- **Calibrated cross-family judge:** different model family than the assistant; both answer orderings
  (order-invariance); rubric+CoT+length-normalization; calibrate once (~50 hand-labeled, Cohen's
  κ ≥ ~0.6). Run at short (~3K) and long (~30K) context.
- **Eval items firewalled** from what the assistant may consolidate (no leakage).

## Interfaces / wiring
- `Learner` (Phase-1 Protocol) gets a real implementation in `morgan_brain/learning/`; `Personalizer`
  gets a real implementation in `morgan_brain/modules/personalization/`. Orchestrator unchanged
  (depends on the Protocols). The consolidation worker subscribes to events on the bus.
- All LLM calls go through the role router (`strong` for extraction/consolidation/judge — judge must
  be a **different family**, configurable). Profile/signals are owner data → under the privacy
  classification + (opt-in) encryption + `delete_subject` fan-out.
- Close the Phase-1 deferrals: fact *writing* during/after a turn (the read side already merges
  current facts), and thread perception entities into stored memories.

## Testing
Unit per module with fakes (no network): signal store, consolidation ops (ADD/UPDATE/DELETE/NOOP +
dedup + contradiction→supersede + decay), UserModel maintenance, trait selection/injection,
learn-from-edits delta, guardrail negatives. The eval harness L1/L2 run on fakes deterministically;
L3/judge are exercised with a fake judge in CI and real local models in a marked live test. Phase-1
suite stays green.

## Non-goals (deferred)
GEPA optimizer loop (Phase 3), LoRA (escalation only), full eval UI, real Presidio/Cedar wiring,
multi-user. Keep the `LearningStrategy` seam so the GEPA loop slots in without touching the request path.

## Increment order & DoD
A → B → C → D, each on the wave branch, each green (pytest + ruff + mypy-strict) and leaving a working
assistant. After C, the assistant **measurably adapts** (profile injected, learn-from-edits). After D,
every future self-learned change is gate-able. Merge to main when green.
