# Memory & Self-Evolving — Voice B: "The Predictive Self" (2026-06-14)

**Status:** EXPLORATORY — a deliberate *second voice*, written to contrast the current design.
Not a decision, not yet reconciled with the in-flight deep-research (that reconciliation is §9).
**Purpose:** give the owner a genuinely different architecture to weigh against the built one, so
the choice is made between two real options rather than defended into one.

> Written from first principles *before* the deep-research workflow returned, on purpose — so
> this is an independent design line, not a summary of the literature. §9 is the merge point.

---

## 1. The two voices, in one sentence each

- **Voice A (built today):** *Memory is a bi-temporal archive of truths you retrieve from;
  self-evolution is an offline, eval-gated optimizer (GEPA) that occasionally promotes a better
  global champion preprompt.* Conservative, auditable, retrieval-first.
- **Voice B (this doc):** *Memory is a continually-corrected generative model of the person, and
  self-evolution is the ongoing minimization of surprise about them.* The atomic unit is not a
  fact but a **belief that makes a falsifiable prediction**; the only thing worth writing is what
  **surprised** the model; and the evolving "self" is a structured belief-state plus a **library
  of context-scoped policies**, not one string.

Voice B is not "more RAG." It is a different claim about what memory *is for*: not to recall the
past, but to **predict the person** — and to get less wrong over time.

## 2. Why a second voice at all (Voice A's structural tensions)

Voice A is good and shipped. But four tensions are baked into its shape, not its bugs:

1. **The self is split.** Knowledge lives in a fact DB; behavior lives in a separate champion
   string. They learn by different mechanisms (consolidation vs GEPA) and barely talk.
2. **Batch consolidation is undirected.** Nightly it reprocesses everything, paying LLM cost and
   ingestion lag, with no notion of *what mattered most*. Salience is flat.
3. **One global champion is lowest-common-denominator.** A single preprompt cannot be terse in
   code review and warm in a family chat at once. Context-dependent behavior is averaged away.
4. **Memory is reactive.** It surfaces only when a query pulls it. Nothing in the architecture
   holds an *active* model of the person that anticipates, or that knows what it doesn't know.

Voice B is the smallest single idea that dissolves all four: **make prediction the substrate.**

## 3. The core inversion: prediction error is the only signal

Predictive processing (brains) and active inference give the frame: an agent maintains a
generative model and learns by minimizing the gap between what it predicted and what happened.
Port that to a personal agent:

- Before responding, Morgan forms a cheap, explicit **prediction** of what the owner wants /
  prefers / will do (from the belief-state).
- After the turn — and *especially* on a correction (edit / retry / thumb) — it measures
  **surprise** = divergence between prediction and reality.
- **Surprise is the universal learning signal.** Low surprise → the model was right → write
  nothing, just reinforce. High surprise → *this* is what's worth consolidating.

This single move fixes tension #2 (writes are salience-gated, not flat-batch), unifies tension #1
(knowledge and behavior are both just "the predictive model," improved by the same signal), and
seeds the fix for #4 (a model that predicts can also quantify its own uncertainty).

## 4. The unit: a Belief (which subsumes a TemporalFact)

```python
class Belief:
    claim: str               # what it holds true — a fact OR a behavioral preference
    prediction: str          # the concretely falsifiable expectation this belief generates
    confidence: float        # CALIBRATED by outcomes, not a decay constant
    track_record: list       # (predicted, observed, surprise) — the audit + calibration source
    scope: frozenset[str]    # contexts where it applies: {code}, {family}, {health}, {*}
    provenance: Provenance   # reuses the foundation-baseline provenance v2 verbatim
```

Key property: **a Voice-A `TemporalFact` is just a Belief whose prediction is "this stays true
until contradicted."** So Voice B does not discard the bi-temporal store — it *generalizes* it.
"User lives in Berlin" is a belief predicting that location-dependent answers should assume
Berlin; if a turn reveals Lisbon, that's surprise, and the supersession Voice A already does is
exactly the confidence/interval update Voice B wants. The whole existing memory layer becomes the
high-confidence, low-surprise tail of the belief distribution.

**Auditability gets *stronger*, not weaker** (this matters — it's the platform's moat). Every
belief carries the predictions it made and whether they came true. The owner-facing answer to
"what do you believe about me and why" becomes "Morgan believes X (confidence 0.82, right 9/11
times); here are the moments it was wrong." That is a calibration record, not just a provenance
stamp — a better version of the auditability bet, not a retreat from it.

## 5. The loop: online surprise, offline replay (true CLS)

Complementary Learning Systems, made literal:

- **Hot path (fast, hippocampal):** form the prediction; serve the turn; compute surprise from
  the outcome/feedback. The *durable* write still happens off-path (the hot-path-reads invariant
  holds — only the surprise *score* is computed inline, which is cheap and structured). High-
  surprise turns are tagged for replay.
- **Sleep (slow, neocortical):** the worker **replays high-surprise episodes** (not everything —
  prioritized replay, the neuro-inspired part), consolidates them into belief updates, recomputes
  calibration, and **decays beliefs that stopped predicting well**. Forgetting becomes principled:
  a belief that never re-surprises is *compiled* into the stable policy layer (§6) and stops
  costing recall; a belief that keeps being wrong decays or splits by scope.

This replaces undirected nightly consolidation with **prioritized replay**, which is both cheaper
(you process surprises, not the whole day) and better-targeted.

## 6. The self: a policy library, not a champion (quality-diversity)

Replace the single GEPA champion with a small **library of context-scoped behavioral policies**,
each a short preprompt fragment indexed by `scope`. Each turn composes the active policy from the
scopes that match the moment (`{code, terse}` + `{user:lazarev}`), instead of averaging into one
string.

Evolution becomes **quality-diversity (MAP-Elites-style) over scope × style**, not single-champion
hill-climbing: GEPA's reflective proposal still generates candidates, the **eval gate still
governs every promotion** (no change to the beats-current discipline), but the population is kept
*diverse and scoped* rather than collapsed to one winner. Degrades exactly to Voice A when there
is only one scope. Fixes tension #3 directly, and keeps Morgan's non-negotiable: nothing promotes
without beating current on the held-out eval.

Coherence guard (the risk this introduces): policies share an invariant core ("who Morgan is")
that no scope may override, so the owner never meets a contradictory Morgan. The gate scores
cross-scope consistency as a first-class metric.

## 7. Proactivity falls out of the memory model (active inference)

Each belief has an **expected information gain**: how much resolving its uncertainty would improve
predictions. When a belief is both *high-importance* and *high-uncertainty*, the system schedules a
cheap **epistemic action** — ask a well-timed clarifying question, or make a small reversible
suggestion and watch the reaction. Proactivity stops being a separate `proactivity/` module bolted
on; it **emerges from the memory wanting to reduce its own uncertainty**.

This is the unify-everything payoff: recall, learning, and proactivity are three faces of one loop
(predict → measure surprise → act to reduce future surprise). It is also the one part most likely
to annoy (CHI research is clear), so it is hard-budgeted: epistemic actions are rate-limited, always
skippable, and themselves learned against dismiss/engage feedback.

## 8. Honest costs and failure modes (a second voice must indict itself)

- **Calibration is the whole ballgame and it is hard.** A miscalibrated confidence is worse than a
  dumb decay constant. Voice B is only better if the track-record calibration actually works;
  needs explicit Brier/ECE monitoring or it silently rots.
- **Prediction is extra inference.** Forming a prediction every turn costs tokens. Mitigation:
  predictions are small/structured and only *consolidated* on surprise — but it is not free.
- **Policy-library fragmentation.** Quality-diversity can yield an incoherent agent; the coherence
  guard (§6) is load-bearing and unproven.
- **Bigger eval surface.** Per-scope policies multiply what the gate must cover; the golden set has
  to grow scope-aware or the gate gives false confidence.
- **Complexity tax for a single maintainer.** Voice A's conservatism is a feature when one person
  maintains the system. Voice B is more moving parts to keep green.

## 9. Reconciliation with the deep-research (done — see the [research report](2026-06-14-memory-self-evolving-research-report.md))

The deep-research workflow has returned (adversarially verified). Verdict on Voice B's
mechanisms:

**Independently confirmed (3-0 verified evidence) — Voice B was right where it counts:**
- **§3 surprise / prediction-error-gated writes** — directly neuro-grounded (human hippocampus
  fMRI, Nature Comms 2022: encode the surprising, predict the stable). This is Voice B's central
  inversion, and it is the single best-evidenced new technique in the whole report.
- **§5 sleep / prioritized replay** — matches "sleep-time compute" (~5× test-time, directional),
  a natural cold-path extension.
- **§6 policy library / evolving the behavioral unit** — matches the field's convergence on
  "context/playbook as the trainable unit" (ACE). The research sharpens it: the right mechanism
  is **ACE-style incremental *delta updates*** (append-then-curate) to avoid *brevity bias* and
  *context collapse* — a concrete improvement over both Voice A's monolithic champion *and* my
  vaguer "library of fragments." Adopt ACE's delta mechanism as the *how* for §6.

**Tempered / kept speculative (no direct 2026 evidence):**
- The full **calibrated-belief substrate** (§4) and **active-inference proactivity** (§7) are
  coherent and attractive but unproven in the literature. They stay in the "diverge and let the
  gate decide" experiment (§10 path 3), not the near-term roadmap.

**What the research added that neither voice emphasized:**
- The **hoarding-vs-amnesia open problem**: consolidation/forgetting is *unsolved*, and rare
  high-importance facts vanish under naive compression. Both voices need an explicit
  **importance-weighted retention + pinned-facts + reversible-only forgetting** guard. This is
  now a hard requirement, not a nuance (research report §5).
- **Defer harder than I implied:** parametric/latent memory and TTRL are not just "deferred" —
  parametric memory is *disqualified for now* because weights can't be targeted-deleted or
  audited (privacy moat), and the headline RL-memory result was *refuted*. The belief-substrate
  must stay non-parametric/inspectable, which it is.

## 10. Three ways to act on this (with a recommendation)

1. **Replace** — commit to the predictive-self wholesale. Highest risk; throws away conservatism
   that currently serves a single maintainer. *Not recommended.*
2. **Augment (recommended near-term)** — keep Voice A's store + GEPA, but graft the two cheapest,
   highest-leverage Voice-B ideas: **(a) surprise as the consolidation prioritizer** (prioritized
   replay instead of flat nightly batch) and **(b) context-scoped policies** instead of one
   champion. Both fit existing seams; both are individually reversible. Captures ~80% of the upside
   at a fraction of the risk.
3. **Diverge and let the gate decide (the most *Morgan* answer)** — Voice A and Voice B already sit
   behind the same `MemoryProtocol` / `LearningProtocol` seams. Implement Voice B as an alternative
   implementation and run **both brains against the golden eval + memory-quality harness**, letting
   the platform's own eval-gated promotion principle choose the winner — *applying Morgan's central
   thesis to Morgan's own design.* The cleanest experiment the architecture makes possible; the
   honest way to settle A-vs-B with evidence instead of taste.

**Recommendation (now evidence-backed):** ship #2 as the near-term direction, and the
deep-research makes the #2 payload concrete — it is the research report's **Voice A+** graft set:
(A) champion → **ACE delta-updated, scope-aware playbook**, (B) **sleep-time** cold-path jobs,
(C) **surprise-gated** consolidation prioritizer, (D) **streaming self-evolving benchmarks** in
the gate, all behind an (anti-amnesia) **importance-weighted retention + pinned-facts** guard.
Frame #3 as the standing experiment: the calibrated-belief substrate and active-inference
proactivity become candidate implementations that must *beat current on the gate* to ship — the
rule the whole platform already lives by. The two voices converge: **Voice A+ is the roadmap;
the full Predictive Self is the eval-gated bet placed against it.**

## 11. What stays sacred regardless of voice

Whatever is chosen, the invariants do not move: hot path reads / cold path writes; one MemoryGate;
actor attribution and provenance v2; **eval-gated promotion, beats-current-or-nothing**; provider
SDKs isolated; everything `user_id`-keyed; auditability *increased*, never reduced. Voice B is a
different shape *inside* those rails, not a way around them.
