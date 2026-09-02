# Self-Evolving Memory — Fusion + Curiosity (reconciliation & design) — 2026-06-14

**Status:** DESIGN (approved in brainstorm; pending owner review of this written spec).
**Supersedes/merges:** completes the §9 "merge point" of
[`2026-06-14-memory-self-evolving-voice-b-design.md`](2026-06-14-memory-self-evolving-voice-b-design.md)
("Voice B: The Predictive Self") by reconciling it with deep-research evidence.
**Evidence base (cited):**
`Neural-Interface/docs/research/2026-06-14-self-evolving-memory-landscape.md` (a sibling repository, not part of this one)
(HippoRAG 2, Generative Agents, ExpeL, Voyager, SEAL, JitRL, 2025–2026 surveys).
**Scope:** an extension of Morgan's brain. Local-only, single machine. Respects every Morgan invariant.

> **One-line thesis:** Keep Voice A's eval-gated conservatism and Voice B's prediction-as-substrate,
> add the two things neither voice had — **graph recall** (the strongest evidence-backed retrieval
> upgrade) and an **opt-in, gated parametric "sleep" tier** — and let the platform's own
> beats-current gate decide which risky pieces actually ship.

---

## 1. The three inputs, reconciled

| Source | Core claim | What this design takes |
|---|---|---|
| **Voice A** (built) | Bi-temporal fact archive + offline eval-gated GEPA champion preprompt | The invariants, the eval gate, the store as the high-confidence tail |
| **Voice B** (Predictive Self doc) | Memory = a corrected generative model of the person; surprise is the only learning signal; belief unit; prioritized replay; scoped policy library; active-inference proactivity | The **belief substrate**, **surprise-gated replay (sleep)**, **scoped policies**, **curiosity** |
| **Deep-research** | Non-parametric memory + reflection online; parametric offline/gated. HippoRAG-2 best recall upgrade; SEAL/JitRL for weights | **Graph recall (PPR)**, **insight tier**, **gated sleep-LoRA**, sequencing discipline |

**§9 reconciliation verdict (which Voice-B mechanisms the research backs):**
- ✅ **Strong evidence:** prioritized replay / sleep-time consolidation (Generative Agents reflection;
  CLS); scoped policies ≈ skills-as-learning-unit (Voyager) and context-engineering; reflection→memory
  write-back (ExpeL insights). These graduate from speculative to **recommended**.
- ⚠️ **Plausible but unproven:** the calibrated-belief substrate and active-inference proactivity have
  no direct 2026 benchmark — keep them, but **behind the gate** as candidate implementations.
- 🆕 **Neither voice anticipated:** **HippoRAG-2 PPR graph recall** (research's #1 recall upgrade) and a
  **gradient-free/gated parametric tier** (JitRL → SEAL-style LoRA). Added here as new components.

## 2. What we are building (component map)

All five live behind Protocols in `morgan_brain/interfaces/` (**contract changes land there first**),
reachable only through `MemoryGate` / the learning seams, honoring hot-path-reads / cold-path-writes.

**C1 — Graph recall (PPR over the belief/fact graph)** · *non-parametric · Phase 1 · highest evidence*
A HippoRAG-2-style Personalized-PageRank recall behind the existing `MemoryProtocol`. Query entities
seed PPR over a graph built from beliefs/facts + episodics; return top nodes/passages. Unifies flat
vector RAG + the entity graph into one recall (multi-hop/associative). Hot-path read — PPR over a
*personal-scale* graph is CPU-cheap and capped per `user_id`; the graph is (re)built offline during
consolidation. Small local extractor (not 70B) → offline batched re-extraction mitigates quality loss.

**C2 — Belief substrate (subsumes `TemporalFact`)** · *non-parametric · Phase 2* (adopt Voice B §4)
The atomic unit becomes a `Belief{claim, prediction, confidence(calibrated), track_record, scope,
provenance}`. A Voice-A `TemporalFact` is the degenerate belief "true until contradicted," so the
bi-temporal store is **generalized, not discarded** — it's the high-confidence/low-surprise tail.
**Auditability increases:** "what do you believe about me, and how often were you right?" + the moments
it was wrong (calibration record). Provenance = foundation-baseline provenance v2, verbatim.

**C3 — Surprise-gated sleep (prioritized replay)** · *non-parametric · Phase 2* (adopt Voice B §3,§5)
Hot path computes only a cheap **surprise score** (prediction vs outcome/feedback) inline and tags
high-surprise turns; the durable write stays off-path. The worker **replays high-surprise episodes**
(not the whole day), updates beliefs, recomputes calibration (Brier/ECE), decays beliefs that stopped
predicting, and **compiles** stable low-surprise beliefs into the policy layer so they stop costing
recall. Replaces undirected nightly batch with cheaper, better-targeted consolidation.

**C4 — Curiosity / active-inference proactivity** · *non-parametric · Phase 3 · the "+curiosity"*
(adopt Voice B §7) Each belief has an **expected information gain**. When a belief is both
high-importance and high-uncertainty, the system schedules a **budgeted epistemic action** (a well-timed
question, or a small reversible suggestion + watch the reaction) via Morgan's `proactivity` triggers.
Recall, learning, and proactivity become three faces of one loop: predict → measure surprise → act to
reduce future surprise. **Hard-budgeted**, always skippable, itself learned against dismiss/engage.

**C5 — Gated sleep-LoRA (parametric)** · *Phase 4 · opt-in · highest risk · NEW*
The genuinely weight-level tier, **offline only**, behind Morgan's deferred-LoRA 4-condition escalation
test. Nightly: distill the consolidated narrative-view + high-value/high-surprise signals into a
**candidate LoRA** → evaluate on the 3-layer golden gate → promote only on beats-current → version for
rollback. **Hard constraint:** requires a LoRA-serving backend (vLLM / llama.cpp adapter hot-swap) + GPU
— **Ollama adapter support is insufficient**, so this tier implies a vLLM serving path. **Cheaper
predecessor:** evaluate **JitRL** (gradient-free kNN-over-(state,action,return) logit re-weighting) as a
Phase-3.5 experiment *before* committing to LoRA.

### Reconciling "narrative self-model" vs "belief + policy library"
The brainstorm approved a *narrative self-model*; Voice B proposes *beliefs + scoped policies*. **Decision:**
the **belief substrate + scoped policy library is the source of truth** (rigorous, calibrated, Morgan-native);
the **narrative self-model becomes a generated, read-only VIEW** over beliefs/policies ("read the agent's
whole mind") — keeping the interpretability win without a second drifting prose store. The policy library
(Voice B §6, quality-diversity / MAP-Elites over scope×style, coherence-guarded) **replaces the single
GEPA champion**, degrading exactly to Voice A when there is one scope.

## 3. The unified loop (data flow)

**Online (hot path, per turn — reads + cheap surprise score only):**
`perceive → personalize (compose active scoped policy + inject compact belief slices) →
recall = vector + PPR(graph) + relevance-gated insights → reason → answer →
async: store raw turn, compute & tag surprise, update gap/uncertainty register`

**Offline (cold path, nightly, learning-worker — all writes):**
`prioritized replay of high-surprise episodes → belief updates + calibration (Brier/ECE) →
rebuild PPR graph → synthesize insights + regenerate narrative view → recompute curiosity gaps →
[Phase 3.5] JitRL bank update → [Phase 4] distill candidate LoRA → eval-gate (beats-current) →
promote/rollback → scoped-policy quality-diversity optimization (GEPA, coherence-guarded)`

## 4. Morgan seams touched (Protocols first)
- `interfaces/memory` — add PPR graph-recall capability + the belief/uncertainty register.
- `interfaces/learning` — belief consolidation via prioritized replay; policy-library optimizer; LoRA distiller (cold path).
- `interfaces/personalization` — compose scoped policy + inject belief slices (extends `AdaptivePersonalizer`).
- `interfaces/skills`/`modules/proactivity` — curiosity (epistemic-action) generation, budgeted.
- `eval/` — extend golden set: multi-hop **recall**, **calibration** (Brier/ECE), **cross-scope coherence**, **curiosity-appropriateness**, self-model-view faithfulness. Gates every promotion.
- `providers/` + serving — LoRA-capable backend (vLLM) for C5 only; no provider hardcoded above the adapter.

## 5. The meta-strategy: let the gate decide (Voice B §10 option 3)
Because C2–C5 sit behind the same `MemoryProtocol`/`LearningProtocol` seams, the risky tiers ship as
**candidate implementations that must beat current on the golden + memory-quality harness** — applying
Morgan's central thesis to its own design. Nothing is "defended into" production by taste.

## 6. Sequencing (YAGNI — prove gains before escalating)
1. **Phase 1 — Graph recall + insight tier.** Lowest risk, highest evidence, immediate recall win.
2. **Phase 2 — Belief substrate + surprise-gated replay + scoped policy library.** The core inversion.
3. **Phase 3 — Curiosity / active-inference proactivity (budgeted).**
4. **Phase 3.5 — JitRL experiment** (gradient-free) as the cheap parametric probe.
5. **Phase 4 — Gated sleep-LoRA** (opt-in, vLLM, only if the escalation test fires and gains justify).

Each phase is independently shippable and reversible; each must keep `main` green (pytest + ruff + mypy strict).

## 7. Risks & mitigations (honest)
- **Calibration is the whole ballgame** (Voice B §8) → explicit Brier/ECE monitoring or beliefs silently rot.
- **Prediction every turn costs tokens** → predictions small/structured; consolidate only on surprise.
- **Policy-library fragmentation / incoherent agent** → invariant core no scope may override; gate scores cross-scope coherence (load-bearing, unproven).
- **Reflection persists hallucinated insights / self-model drift** → confidence + importance thresholds, decay, auditable narrative view, eval-gate + rollback.
- **Curiosity annoyance** → strict budget, always skippable, learned against dismiss/engage.
- **Graph quality with a small extractor** → offline batched re-extraction; PPR degrades gracefully.
- **LoRA: infra lift + catastrophic forgetting** → opt-in behind escalation test; LoRA (not full FT) keeps base intact; eval-gate + versioned rollback; vLLM serving flagged as prerequisite.
- **Hot-path latency** (PPR + extra recall) → cap graph per user, precompute offline, async store-after.
- **Complexity tax for a single maintainer** → phased; Phases 1–2 deliver most value; 3.5/4 strictly optional.

## 8. What stays sacred (all voices)
Hot path reads / cold path writes; one `MemoryGate`; actor attribution + provenance v2; **eval-gated,
beats-current-or-nothing** promotion; provider SDKs isolated to adapters; everything `user_id`-keyed;
**auditability increased, never reduced.** This design is a different shape *inside* those rails.

## 9. Open questions (carry into the plan)
1. Real **recall latency** of PPR+vector on a modest machine inside a voice turn budget (<~100–200 ms)?
2. Quality drop from a **7–8B quantized extractor** vs the 70B HippoRAG-2 reference?
3. Is **JitRL** robust on *conversational* data (vs WebArena/Jericho), enough to defer/avoid LoRA?
4. Best **forgetting/decay + insight-confidence** policy to balance retention vs drift?
5. Does the **scoped policy** coherence guard hold under quality-diversity, and how to keep the eval scope-aware without exploding the golden set?
