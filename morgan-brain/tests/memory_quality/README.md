# Memory-quality regression suite

A LoCoMo / LongMemEval-style harness that measures recall quality so memory changes are
**measured, not vibed** (design spec §13). Question categories:

- **single-hop** — direct fact recall
- **multi-hop** — recall requiring composition across memories
- **temporal** — "where did I *used* to live?" (bi-temporal correctness)
- **knowledge-update** — a fact changed; the latest must win, the old must not leak

Each case is `(seed memories/facts, question, expected)`. Scored as recall@k. Wired in Phase 1
once the memory module exists; this directory holds the fixtures and scorer.
