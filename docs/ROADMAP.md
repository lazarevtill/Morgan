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

The bundled probes are built so the keyword signal cannot carry them, and as a result they
cannot see the entity signal or routing either: the entity list fires on 1 of 60 and routing
on none. A real archive measures both.

### On a real archive

~2,000 imported conversation turns, 90 questions a chat model wrote from sampled turns (each
labelled with its source) and 38 authored questions the keyword index confirms the archive
never mentions. Qwen3-Embedding-0.6B.

| recall signals | recall@8 | MRR |
|---|---|---|
| vector only | 0.92 | 0.77 |
| vector + keyword | 0.90 | 0.72 |
| vector + keyword + entity, routed (as shipped) | 0.83 | 0.56 |

The relevance floor separates the two kinds of question: a random answerable question has
the wider margin 97% of the time. At 0.11, on the half of the questions held out of the fit,
it kept 24 of 25 answers and silenced 12 of 13 unanswerable questions.

## Next

- **The entity signal and routing cost recall on real memories.** The extractor stores
  sentence openers and code words as names (13,677 of them; the most common are "на", "для",
  "for", "true"), so the entity list is non-empty for 88 of 90 questions and holds the answer
  for 24. Routing narrowed two questions and cut the answer out of both. A cleaner extractor
  alone makes it worse -- recall@8 0.76 -- because a sparser index lets routing narrow more
  often. Lookups by exact name, host or part number are where these signals should pay, and
  that is unmeasured; measuring it decides what changes.
- **A superseded memory outranks the current one** in half the probes: the old answer
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
