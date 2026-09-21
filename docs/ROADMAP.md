# Roadmap

## Where this is

Morgan is a project-scoped memory for the owner's AI tools: one SQLite database, a CLI and an
MCP server over one gate, recall that fuses vector and keyword search, and on-demand
consolidation of memories into valid-time facts by a local model. About 10,300 lines, one
process, no services beyond a model server.

Its core:

- The one-database, one-gate, project-scoped memory with cascading `forget()`.
- Bi-temporal facts with actor attribution, and the consolidation that produces them.
- The reachability contract: a model server that is down, slow or refusing is reported by
  name on every surface.

## What is not in the code, and why

- **No learning loop.** Nothing tunes prompts, promotes a champion or learns from signals. A
  loop that changes behaviour needs an evaluation gate sound enough to trust, with enough items
  and a real statistical test, and there is none yet.
- **No assistant.** No cognitive loop, persona graph, skills, tools, streaming or REST
  gateway. The owner's AI tools are the assistant; Morgan is their memory, and its surfaces are
  a CLI and an MCP server.
- **No services.** No worker, event bus, scheduler, Redis, Qdrant or MLflow. One process and
  one SQLite file hold everything, and consolidation runs when asked.

The archived agent kernel that has all of the above is at the git tag
**`legacy-v0.1.0-kernel`**, its designs and decision records under [`archive/`](archive/);
`legacy-v0.0.4-full` and `legacy-v0.0.3-monolith` are older builds.

## What retrieval actually measures

`tests/memory_quality/` holds 60 labelled probes over an 86-memory corpus, half of it
Russian, with every query written to share few or no words with its target so the keyword
signal cannot carry it. `pytest --live` runs them against a real embedding endpoint. With
`qwen3-embedding:8b` at 4,096 dimensions and no floor
([`measurements/2026-09-phase0-baseline.md`](measurements/2026-09-phase0-baseline.md)):

| | recall@8 | MRR | stale-first | abstain |
|---|---|---|---|---|
| overall | 0.89 | 0.51 | 0.70 | 0.00 |
| single-hop | 0.90 | 0.58 | — | — |
| temporal | 1.00 | 0.66 | — | — |
| knowledge-update | 1.00 | 0.50 | 0.70 | — |
| multi-hop | 0.62 | 0.16 | — | — |
| unanswerable | — | — | — | 0.00 |

Stale-first measures order, not presence: the same memory is forbidden by "where do I live
now" and expected by "where did I live before", so a metric counting presence could only be
satisfied by suppressing it, which breaks the question that asks for it.

Semantic retrieval works: a query and its answer sharing no token find each other, in both
languages. Two categories do not, and both are open work rather than regressions.

The bundled probes are built so the keyword signal cannot carry them, so they cannot show
what keyword search or stored names add. A real archive does.

### On a real archive

~2,000 imported conversation turns and three sets of questions: 90 a chat model wrote from
sampled turns, each labelled with its source; 60 lookups, each naming a term the keyword index
finds in at most three memories; and 38 authored questions the keyword index confirms the
archive never mentions. Qwen3-Embedding-0.6B.

| recall signals | paraphrase recall@8 | MRR | lookup recall@8 | MRR |
|---|---|---|---|---|
| vector only | 0.92 | 0.77 | 0.88 | 0.65 |
| **vector + keyword (recall as it is)** | **0.90** | **0.72** | **1.00** | **0.88** |
| + the entity ranking fused | 0.83 | 0.59 | 1.00 | 0.94 |
| + the entity ranking, narrowed by a semantic index | 0.76 | 0.53 | 0.97 | 0.91 |

Lookups were chosen by what the keyword index finds, so keyword search finding all of them is
partly by construction; what they show is that an embedding alone misses one name in eight.
A stored name is in the memory's text, so fusing the entity ranking counts the keyword match
twice: it ranks lookups a little higher and paraphrases much lower. Narrowing the search to
the memories that share a name with the question (VoiceMem's semantic index) cut the answer
out of questions and put none in.

The relevance floor separates the two kinds of question: a random answerable question has
the wider margin 97% of the time. At 0.11, on the half of the questions held out of the fit,
it kept 24 of 25 answers and silenced 12 of 13 unanswerable questions; over all of them, 141
of 150 answers and 34 of 38 silences.

With `qwen3-embedding:8b`, over 3,010 imported memories and 128 labelled questions, the floor
is 0.08. On the 38 questions sealed from the fit, recall@8 is 0.84 against 0.88 with no floor,
and 85% of the unanswerable questions among them are silenced; on the 90 fitted ones, 0.88
against 0.89, and 84%. A higher margin silences no more of the sealed questions and costs
recall ([`measurements/2026-09-phase0-baseline.md`](measurements/2026-09-phase0-baseline.md)).

## Next

- **A superseded memory outranks the current one** in 7 of the 10 knowledge-update probes:
  the old answer comes back above the current one. Fusion is rank-only and carries no
  recency term, so nothing orders new above old. The probes store episodics, which have no
  validity interval at all -- supersession lives on facts, and nothing has consolidated
  these.
- **Multi-hop composition does not happen.** Recall ranks memories; it has no mechanism to
  combine two of them into one answer, and the numbers say so.
- **Model-backed entity extraction** for scripts without letter case.
- **Learning, only against a sound gate.** Nothing from the archived kernel is built on this
  core until an evaluation with enough items and a real statistical test can gate it.

## Working agreement

Keep `main` green: `pytest`, `ruff check`, `ruff format --check`, `mypy --strict`, `bandit`.
Every memory read and write goes through `MemoryGate`, and is user- and project-scoped.
Root-cause fixes only.
