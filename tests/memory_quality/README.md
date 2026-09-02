# Memory-quality regression suite

A LoCoMo / LongMemEval-style harness that measures recall quality so memory changes are
**measured, not vibed** (design spec §13). Question categories:

- **single-hop** — direct fact recall
- **multi-hop** — recall requiring composition across memories
- **temporal** — "where did I *used* to live?" (bi-temporal correctness)
- **knowledge-update** — a fact changed; the latest must win, the old must not leak

Each case is `(seed memories/facts, question, expected)`, scored as recall@k.

## What this measures today: the plumbing, not relevance

**Read this before quoting a number from it.** The harness runs against `FakeEmbedder`, a
sha256-based stub. It is deterministic across processes, which is what makes restart and
cross-process recall testable — and it means the *similarity* it produces is meaningless. The
suite proves the retrieval path is wired and survives a restart. It does not measure whether
recall returns the right memory.

That gap is why the accuracy claims behind the semantic upper index and the persona graph are
cited as **the paper's** ([VoiceMem, arXiv:2608.26005](https://arxiv.org/abs/2608.26005), on
LoCoMo / LongMemEval / Memora) rather than as observed here. Reproducing them needs this harness
pointed at a real embedding model with a labelled question set — roadmap milestone M3, which
starts with a noise-floor measurement so a difference can be told from a fluctuation.

Until then: any memory change must keep this suite green (it catches wiring regressions), and no
memory change should be described as improving recall quality on the strength of it.
