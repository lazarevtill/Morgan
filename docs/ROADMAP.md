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

## Next

- **Measure retrieval.** The one live test proves an embedding model bridges a query and a
  memory that share no token. A real harness (LoCoMo/LongMemEval-style, over a real embedder)
  is what would let the semantic index's benefit be observed rather than assumed.
- **A relevance floor for recall.** Today a non-empty project always answers.
- **Model-backed entity extraction** for scripts without letter case.
- **Bring learning back only against a sound gate.** Anything from the archived kernel returns
  designed against this core, gated by an evaluation with enough items and a real statistical
  test, not before.

## Working agreement

Keep `main` green: `pytest`, `ruff check`, `ruff format --check`, `mypy --strict`, `bandit`.
Every memory read and write goes through `MemoryGate`, and is user- and project-scoped.
Root-cause fixes only.
