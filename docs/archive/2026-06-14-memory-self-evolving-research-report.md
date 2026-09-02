# Memory & Self-Evolving — Deep-Research Report (2026-06-14)

**Status:** reference (adversarially-verified research snapshot)
**Method:** deep-research workflow — 5 search angles → 20 sources fetched → 87 claims →
25 verified by 3-vote adversarial panels (21 confirmed, 4 killed) → 10 synthesized findings.
**Companion:** [Voice B design](2026-06-14-memory-self-evolving-voice-b-design.md) (the creative
alternative this report reconciles with) · current design = the 2026-06-08 phase-2 learning spec.

> Time-sensitivity: most primary sources are 2025–early-2026 arXiv preprints (several
> single-author / pre-peer-review). Benchmark figures are author-reported, often best-case, on
> constructed benchmarks — treat as **directional**, not guaranteed for a personal-assistant
> workload. Four notable refutations bound the conclusions (§4).

---

## 1. Bottom line

**Morgan's design is on the published mainline, not idiosyncratic — the next 12 months should be
evolution, not rewrite.** Two convergent signals dominate:

1. **GEPA-style reflective evolution is the 2026 SOTA optimizer** (ICLR 2026 Oral): beats the
   GRPO RL baseline ~6% avg / up to 20% with **up to 35× fewer rollouts**, and beats MIPROv2
   >10%. This directly validates Morgan's eval-gated champion-preprompt bet — and means the RL
   alternatives don't justify their cost for a single-owner system. (3-0,
   [arXiv:2507.19457](https://arxiv.org/pdf/2507.19457))
2. **The field crystallized around "context/playbook as the trainable unit"** (ACE) and a
   **Storage→Reflection→Experience** taxonomy that mirrors Morgan's episodic→semantic
   consolidation. Morgan's champion preprompt *is* an instance of the converged direction.
   (3-0, [ACE arXiv:2510.04618](https://arxiv.org/abs/2510.04618);
   [survey arXiv:2605.06716](https://arxiv.org/html/2605.06716v1))

## 2. Adopt (high-confidence, high-fit upgrades)

| # | Upgrade | Why / evidence | Fit |
|---|---------|----------------|-----|
| A | **ACE-style delta-updated playbook** — replace monolithic champion *rewrites* with structured incremental delta updates (append-then-curate) | ACE diagnoses **brevity bias** (dropping domain insight for concise summaries) and **context collapse** (iterative rewriting eroding detail). Morgan's monolithic champion is exactly the artifact that suffers this. Keeps the zero-inference-time-cost property. (3-0) | drop-in upgrade to the GEPA optimizer's output format |
| B | **Sleep-time compute** — cold-path jobs that pre-derive likely-needed facts/answers for the owner's recurring contexts *before* queries arrive | ~5× test-time / ~2.5× cost amortization (directional; constructed benchmarks). The learning-worker already owns the cold path — this **extends** the hot-reads/cold-writes invariant, doesn't break it. (3-0, [arXiv:2504.13171](https://arxiv.org/abs/2504.13171)) | new worker job class |
| C | **Surprise / prediction-error-gated writes** — gate episodic-write strength and consolidation *priority* on novelty; lower write weight for inputs already predicted by stable facts | Neuro-grounded: human hippocampus fMRI (Nature Comms 2022, N=24×2) shows it preferentially encodes *surprising* stimuli while learning, then switches to *predicted* representation once stable. (3-0, [Nature s41467-022-31040-w](https://www.nature.com/articles/s41467-022-31040-w)) | prioritizer in front of consolidation |
| D | **Streaming self-evolving-memory benchmarks** in the eval gate (Evo-Memory style, + LoCoMo/LongMemEval/selective-forgetting) | Self-evolving memory is now a *measured* capability gap; validating consolidation/champion on test-time-learning (not just static golden sets) reinforces Morgan's eval-gated invariant. (3-0, [arXiv:2511.20857](https://arxiv.org/pdf/2511.20857)) | extend `tests/eval` + `tests/memory_quality` |

## 3. Keep unchanged (validated)

Bi-temporal fact store; episodic→semantic consolidation; the **eval gate** (beats-current-or-
nothing); hot-path-reads / cold-path-writes; GEPA as the core optimizer; provider-agnostic
isolation; everything `user_id`-keyed and inspectable.

## 4. Defer / watch-list (real but immature, or refuted)

- **Parametric / latent memory** (G-MemLLM and the explicit-vs-parametric axis): real, but
  weight-resident memory **"fails at targeted deletion and auditing"** — *disqualifying* for a
  privacy-first system that must support owner-facing deletion + audit. Keep the inspectable
  store as source of truth. Consistent with Morgan's existing LoRA deferral. (3-0,
  [arXiv:2602.00015](https://arxiv.org/pdf/2602.00015), survey 2603.07670)
- **Test-Time RL / RL-trained memory controllers:** TTRL is correctly defined and promising
  (self-reward via consensus) but carries reward-hacking / mode-collapse risk; and the headline
  claim that an RL-trained memory updater (MemoPilot) beats frontier models **did not survive
  verification (REFUTED 1-2)**. GEPA already wins at 35× fewer rollouts → not justified.
- **Forgetting-curve replay (FOREVER):** its "model-time = optimizer-update magnitude" principle
  is interesting, but the supporting claim that **LLM forgetting follows the Ebbinghaus curve was
  unanimously REFUTED (0-3)**. Prefer the surprise-gated schedule (C), which has direct
  neuroscience support, over forgetting-curve replay, which does not.
- Also refuted: "no memory system masters all four competencies / only MemoryAgentBench tests
  forgetting" (1-2). Do not cite these as settled.

## 5. The open problem to engineer around: hoarding vs amnesia

The 2026 survey is blunt that **continual consolidation and learned forgetting remain OPEN** —
systems "oscillate between hoarding (store everything, drown in noise) and amnesia (compress
aggressively, lose rare but vital facts)," and **low-frequency high-importance instructions
(e.g. 'never call the production DB directly') tend to vanish after a few compression passes.**
(3-0, [arXiv:2603.07670](https://arxiv.org/html/2603.07670v1))

**Engineering response (a Morgan invariant, not an experiment):** importance-weighted retention;
**pinned facts** that are never auto-compressed/auto-deleted; never silently drop a `user_stated`
high-importance fact; all forgetting conservative and **reversible** (bi-temporal supersession,
which Morgan already has, is the right substrate — closing an interval is reversible; deletion is
not). Validate retention-of-rare-facts as an explicit eval-gate scenario.

## 6. Synthesized target architecture (12-month)

Keep the spine; graft four things; defer the rest:

```
KEEP   bi-temporal store · episodic→semantic consolidation · eval gate · hot/cold · GEPA
GRAFT  (A) champion → ACE delta-updated, scope-aware playbook   [evolves the self-model]
       (B) sleep-time cold-path jobs (pre-derive recurring-context answers)
       (C) surprise/prediction-error gate in front of consolidation (priority + write strength)
       (D) streaming self-evolving benchmarks in the eval gate
GUARD  importance-weighted retention + pinned facts (anti-amnesia, reversible-only forgetting)
DEFER  parametric/latent memory · TTRL/RL-memory · LoRA · forgetting-curve replay   (watch-list)
```

This is **Voice A+**: the conservative architecture, upgraded exactly where the evidence is
strong, with the privacy/auditability constraints doing the deferral filtering.

### Shipped status (2026-06-14)
Four of the five GRAFT/GUARD items are implemented, tested, and on `main`; the fifth is
evidence-deferred:
- ✅ **GUARD — anti-amnesia** (`ca96183`): user-stated facts protected from inferred DELETE +
  decay floor (importance = source). + `test_anti_amnesia`.
- ✅ **C — surprise/prediction-error gate** (`ca96183`): deterministic novelty pre-filter before
  the consolidation LLM call. + `test_surprise_gating`.
- ✅ **A — ACE delta-playbook** (`e79bf7d`, `01b0cf1`): `curate_playbook` + `ReflectiveOptimizer`
  now grows the champion by curated deltas (no context-collapse). + `test_curate_playbook`,
  updated optimizer/promotion tests.
- ✅ **D — streaming self-evolving benchmark** (`8714d3d`): Evo-Memory-style stream measures
  distance-independent recall (1.0, past the history window) + mid-stream update propagation
  (1.0). `tests/e2e/streaming.py`.
- ⏸️ **B — sleep-time compute**: **deferred on evidence**, not skipped. The benefit is
  amortising precompute across *related-query reuse*, and Open Question #1 (below) is precisely
  whether a single owner generates enough reuse to amortise it. Shipping speculative precompute
  infra before that payoff is demonstrated would violate the "only best solutions / YAGNI" bar.
  Revisit once usage data shows recurring-context density; the cold-path (learning-worker) seam
  is ready to host it when justified.

## 7. Open questions the research could not close

1. Do sleep-time / GEPA efficiency gains transfer from constructed math benchmarks to a *single
   owner's* sparse, idiosyncratic recurring contexts (enough related-query reuse to amortize)?
2. The exact anti-amnesia mechanism (importance-weighted vs pinned vs surprise-gated vs hybrid)
   and how it's validated on the gate.
3. When, if ever, the 4-condition LoRA/parametric escalation fires — and whether machine
   unlearning matures enough by then to satisfy deletion requirements.
4. How a delta-update playbook stays bounded under strict beats-current promotion — what
   curation/forgetting policy governs the *playbook itself*.

## 8. Primary sources

GEPA [2507.19457](https://arxiv.org/pdf/2507.19457) · ACE [2510.04618](https://arxiv.org/abs/2510.04618)
· Storage→Experience survey [2605.06716](https://arxiv.org/html/2605.06716v1) · Sleep-time compute
[2504.13171](https://arxiv.org/abs/2504.13171) · Hippocampus prediction-error
[Nature 2022](https://www.nature.com/articles/s41467-022-31040-w) · Memory-for-Agents survey
[2603.07670](https://arxiv.org/html/2603.07670v1) · Parametric memory G-MemLLM
[2602.00015](https://arxiv.org/pdf/2602.00015) · TTRL [2504.16084](https://arxiv.org/abs/2504.16084)
· MemoPilot (refuted) [2606.08656](https://arxiv.org/abs/2606.08656) · FOREVER
[2601.03938](https://arxiv.org/abs/2601.03938) · Evo-Memory [2511.20857](https://arxiv.org/pdf/2511.20857)
