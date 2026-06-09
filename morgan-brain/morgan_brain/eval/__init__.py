"""Eval harness — 3-layer golden eval + calibrated cross-family judge.

Layer 1 (L1): deterministic retrieval scorers (no judge, no network).
Layer 2 (L2): LLM judge over a hand-authored golden set.
Layer 3 (L3): full held-out A/B + time-series — the richer online layer, not yet wired
(the ``beats_current`` champion gate below already enforces the offline beats-current rule).

The validation gate ("beats-current-or-nothing") gates every self-learned
change: a candidate prompt must equal or beat the current champion on all
probe types before it is promoted.

Eval items are FIREWALLED from what the assistant may consolidate:
the harness only reads ``predict_fn`` output; it never writes to memory.
"""
