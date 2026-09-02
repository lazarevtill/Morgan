# Calibration Scoring in the Eval Gate (2026-06-16)

**Status:** DRAFT — design only, pending owner review before any implementation.
**Motivation sources:** [Inbrain](https://github.com/inbrainfun/inbrain)'s Brier-score calibration
tracking (it scores how well its prediction confidences match outcomes) + this repo's Voice B
"calibrated belief" idea ([2026-06-14-memory-self-evolving-voice-b-design.md](../archive/2026-06-14-memory-self-evolving-voice-b-design.md) §4),
which the deep-research report left as *speculative*. Inbrain is working evidence that
calibration tracking is practical; this spec makes it concrete and cheap for Morgan.
**Parent:** the eval gate (`morgan_brain/eval/harness.py`) + the self-learning ADR
(beats-current-or-nothing).

---

## 1. Why

Today the gate measures **accuracy** only: `Scorecard.layer2` holds per-probe pass-rates and an
overall `overall_preference_following_accuracy`, and `beats_current()` promotes a champion when
accuracy doesn't regress (`harness.py:78-112`). Accuracy answers *"is it right more often?"* — it
does **not** answer *"does it know when it's right?"*

A champion that is more accurate but **overconfident** (high confidence on wrong answers) is a
worse personal agent — it asserts stale preferences and wrong facts with false certainty. The
platform's headline is *"provably gets smarter."* **Calibration** is the missing half of "smarter":
a well-calibrated agent's stated confidence matches its actual hit-rate, so it hedges when it
should and commits when it should. This is exactly what Voice B's calibrated-belief substrate
wanted, and what Inbrain ships for its market predictions.

## 2. What we measure

Standard, well-understood calibration metrics over the golden set, paired per item as
`(p_i, y_i)` where `p_i ∈ [0,1]` is the system's confidence in its answer and `y_i ∈ {0,1}` is
whether the answer was judged correct:

- **Brier score** = `mean((p_i − y_i)²)` — lower is better (0 = perfect, 0.25 = always-0.5,
  1 = confidently wrong). The single headline number (Inbrain's choice).
- **ECE (Expected Calibration Error)** = `Σ_b (n_b/N) · |acc_b − conf_b|` over confidence bins
  `b` — the average gap between confidence and accuracy across the reliability curve.
- **Reliability bins** = per-bin `(mean_confidence, accuracy, count)` — the reliability diagram in
  table form, for inspection (and an MCP-Apps inspector later).

These are pure functions of `(p_i, y_i)` lists — deterministic, no LLM, trivially unit-testable.

## 3. Where the per-item confidence comes from

`y_i` already exists — it's `verdict.passed` from the judge (`harness.py:171`). The new piece is
`p_i`. Confidence sources, in the order this spec adopts them:

**v1 (this spec) — recalled-belief confidence.** Morgan's beliefs already carry confidence:
`TemporalFact.confidence` and `UserModel.confidence`. The system's confidence in an answer is the
confidence of the knowledge it relied on. The eval runner (`eval/runner.py`) builds a per-item
scratch `MemoryGate`, seeds `item.setup` facts, and drives the orchestrator; it can return, beside
the answer, the **max confidence among the currently-valid facts that fed recall** for that item
(0.5 neutral prior when the item used no facts). This is the Voice-B-native signal: *how sure was
the agent of the beliefs behind this answer?*

**v2 (future) — self-rated confidence.** Have the assistant emit a confidence token with its
answer (structured output). More direct, but costs a contract change on the reasoning path.

**v3 (future) — judge score as a soft target.** `JudgeVerdict.score` (continuous, currently
computed but unused in aggregation) can serve as a secondary calibration reference. Noted only;
not a v1 confidence source (the judge evaluates, it is not the predictor).

The confidence is supplied to the harness via an injected, optional `ConfidenceFn` so the harness
stays decoupled from where confidence originates (and v2/v3 are drop-in).

## 4. Data flow (firewall preserved)

```
golden item ─▶ predict_fn ─▶ (answer, confidence p_i)        # runner; scratch-gate firewalled
                   │
                   ├─▶ judge.judge(answer, expected) ─▶ verdict.passed = y_i
                   │
                   └─▶ calibration collector: append (p_i, y_i)
aggregate ─▶ brier, ece, reliability_bins ─▶ Scorecard.layer3
```

The FIREWALL is unchanged: the harness still only *reads* `predict_fn` output and never writes to
any store (`harness.py:138-139`). Confidence is read-only metadata about the answer.

## 5. Scorecard + gate changes

**Scorecard** (`harness.py:56`): add
```python
layer3: dict[str, float] = Field(default_factory=dict)   # {"brier": .., "ece": ..}
reliability: list[dict[str, float]] = Field(default_factory=list)  # per-bin rows
```
`layer3`/`reliability` default empty, so existing scorecards and stored champion metrics remain
valid (back-compat).

**Gate** (`beats_current`, `harness.py:78`): introduce a `CalibrationMode`:
- **`report` (default, phase 1):** compute + store calibration; **do not gate on it.** This is the
  conservative, eval-gated way to introduce a new gate dimension — observe the metric on real
  promotions before letting it block anything (no unvalidated metric vetoes a more-accurate
  champion).
- **`guard` (phase 2, after the signal is trusted):** a candidate must also not worsen Brier
  beyond `_CALIBRATION_EPSILON` (default 0.05). Promotion then requires **accuracy-not-worse AND
  calibration-not-worse** — an accurate-but-overconfident candidate is rejected.

The mode is a setting (`MORGAN_CALIBRATION_GATE=report|guard`, default `report`), threaded into
`beats_current`. `EvalGate.promote_if_better` stores `layer2 ∪ layer3` in champion metrics so the
next comparison has the champion's Brier (`harness.py:248-253`).

## 6. Components / files

- **New `morgan_brain/eval/calibration.py`** — pure functions: `brier_score(pairs)`,
  `expected_calibration_error(pairs, n_bins=10)`, `reliability_bins(pairs, n_bins=10)`. Lives
  beside `eval/scorers.py` (where `cohen_kappa` already is).
- **`eval/harness.py`** — `Scorecard.layer3`/`reliability`; `run_l2` accepts an optional
  `confidence_fn`/collects `p_i`, populates `layer3`; `beats_current` gains the `mode` guard.
- **`eval/runner.py`** — `make_predict_fn` optionally returns `(answer, confidence)`; default
  `ConfidenceFn` derives confidence from the scratch gate's recalled-fact confidence.
- **`config.py`** — `calibration_gate: Literal["report","guard"] = "report"`.
- **`PredictFn` type** — widen to allow returning either `str` (back-compat) or
  `tuple[str, float]`; the harness normalises.

No change to the orchestrator, providers, or the hot path. All additive.

## 7. Testing strategy (PoW)

- **`tests/unit/eval/test_calibration.py`** — `brier_score`/`ece` against hand-computed values
  (e.g. all-correct@conf-1.0 → Brier 0, ECE 0; confidently-wrong → Brier 1; always-0.5 → 0.25);
  reliability bins partition correctly; property test `0 ≤ brier ≤ 1`.
- **`tests/unit/eval/test_harness_calibration.py`** — `run_l2` with a scripted judge + confidence
  fn populates `layer3` with the expected Brier/ECE; empty/streaming cases are safe.
- **Gate tests** — `report` mode never blocks on calibration; `guard` mode rejects a candidate
  that is accuracy-equal but Brier-worse-by >epsilon, and promotes one that improves both.
- **Firewall test** — adding confidence does not cause any store write (assert via a spy gate).
- **No regression** — the existing 88 eval-gate tests stay green (back-compat defaults).

## 8. Honest limitations + sequencing

- The **v1 confidence source is a heuristic** (recalled-fact confidence), not the assistant's
  introspective certainty. It is the cheapest faithful signal and the Voice-B-native one; v2
  self-rating is the upgrade if v1's calibration tracks poorly.
- **Phase 1 ships `report` only.** Calibration becomes a *gate* (`guard`) only after we have
  watched it on real champion promotions and confirmed it's a trustworthy, non-noisy signal —
  same discipline the platform applies to every learned change (don't let an unproven metric gate
  promotions).
- Golden-set size bounds ECE resolution (few items per bin → noisy bins); report the bin counts so
  low-evidence bins are visible, and consider widening the golden set (overlaps with the
  deep-research "Evo-Memory streaming benchmarks" work already shipped).

## 9. Why this is worth doing now

It is small (one pure module + ~3 touch points, all additive, firewall-safe), it retroactively
**de-risks the Voice B calibrated-belief idea** with a concrete, tested mechanism, and it makes
the "provably gets smarter" claim measure *calibration*, not just accuracy — the dimension Inbrain
found worth tracking. Recommended as the next eval-gate increment after owner approval.
