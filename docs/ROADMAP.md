# Roadmap

## Where this is

Morgan is a project-scoped memory for the owner's AI tools: one SQLite database, a CLI and an
MCP server over one gate, three-signal recall routed by a semantic index, and on-demand
consolidation of memories into valid-time facts by a local model. About 4,500 lines, one
process, no services beyond a model server.

## Where it came from

Until September 2026 this repository was a **self-learning personal agent kernel**: the same
memory underneath, plus a cognitive loop with perception, personalization, skills and tools; a
persona graph; a signal recorder; an eval-gated champion-prompt optimizer; a REST/SSE gateway;
a learning worker on an event bus with a nightly scheduler; Redis, Qdrant and MLflow backends.
About 15,600 lines, of which the memory was the part in daily use.

It was cut to the core for three reasons:

1. **The learning loop was switched off.** Champion promotion shipped disarmed because its
   gate was a bare comparison over a 12-item golden set. A loop that cannot be trusted to run
   is cost without benefit.
2. **Its quality was unmeasured.** Retrieval quality and the persona graph's accuracy were
   the papers' numbers, not this system's; the harness ran over a hash embedder.
3. **The premise had changed.** The chat assistant was to be the product; in practice the
   owner's AI tools are the assistant and Morgan is their memory. Skills, tools, streaming and
   a REST gateway serve an assistant, not a memory.

The full build is at the tag **`legacy-v0.1.0-kernel`**, its designs and decision records under
[`archive/`](archive/). Earlier: `legacy-v0.0.4-full` (the platform build), `legacy-v0.0.3-monolith`.

## Kept from the kernel, on purpose

- The one-database, one-gate, project-scoped memory with cascading `forget()`.
- The semantic upper index (from VoiceMem, arXiv:2608.26005): routing that can cost precision,
  never recall.
- Bi-temporal facts with actor attribution, and the consolidation that produces them.
- The reachability contract: a model server that is down is reported by name on every surface.

## What retrieval actually measures

`tests/memory_quality/` holds 60 labelled probes over an 86-memory corpus, half of it
Russian, with every query written to share few or no words with its target so the keyword
signal cannot carry it. `pytest --live` runs them against a real embedding endpoint.

| | recall@8 | MRR | stale-first | abstain |
|---|---|---|---|---|
| overall | 0.85 | 0.48 | 0.50 | 0.00 |
| single-hop | 0.95 | 0.57 | — | — |
| temporal | 1.00 | 0.65 | — | — |
| knowledge-update | 0.90 | 0.49 | 0.50 | — |
| multi-hop | 0.38 | 0.08 | — | — |
| unanswerable | — | — | — | 0.00 |

Stale-first measures order, not presence: the same memory is forbidden by "where do I live
now" and expected by "where did I live before", so a metric counting presence could only be
satisfied by suppressing it, which breaks the question that asks for it.

Semantic retrieval works: a query and its answer sharing no token find each other, in both
languages. Two categories do not, and both are open work rather than regressions.

## Next

- **Fit the relevance floor to a real corpus.** Recall can now decline to answer, and the
  mechanism is validated: fitted on half the probes and reported on the sealed half, it
  silenced every unanswerable question there for 11 points of recall. It is off by default
  because that fitted number did not transfer -- against 184 real imported memories it let
  two of three plainly unanswerable questions through. The probes are short single
  sentences and real memories are long conversation turns, so the margin spreads
  differently. Finishing the import and fitting against that corpus is the work. Reciprocal rank fusion keeps ranks and discards scores, so
  the floor belongs on each signal before fusion, and its thresholds have to come from the
  measurement above rather than from a guess.
- **A superseded memory outranks the current one**, three times in four: the old answer
  comes back at rank 1 and the current one below it. Fusion is rank-only and carries no
  recency term, so nothing orders new above old. The probes store episodics, which have no
  validity interval at all -- supersession lives on facts, and nothing has consolidated
  these.
- **Multi-hop composition does not happen.** Recall ranks memories; it has no mechanism to
  combine two of them into one answer, and the numbers say so.
- **Model-backed entity extraction** for scripts without letter case.
- **Bring learning back only against a sound gate.** Anything from the archived kernel returns
  designed against this core, gated by an evaluation with enough items and a real statistical
  test, not before.

## Working agreement

Keep `main` green: `pytest`, `ruff check`, `ruff format --check`, `mypy --strict`, `bandit`.
Every memory read and write goes through `MemoryGate`, and is user- and project-scoped.
Root-cause fixes only.
