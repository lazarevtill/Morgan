# Morgan

A **project-scoped memory for your AI tools**, consolidated into facts by a local model.

You tell Morgan things from any repository, on any of your machines. Claude Code, Claude
Desktop or any other MCP client recalls them, scoped to the repository it is working in. A
local model turns what was said into durable, dated facts that evolve instead of being
overwritten. Everything lives in one SQLite file on hardware you own.

## What it does

- **Remembers per project.** Every memory belongs to a project, which the CLI takes from the
  current git repository's directory name. Recall is scoped to it; `--all-projects` is the
  explicit escape hatch.
- **Recalls by three signals at once.** Vector search (sqlite-vec), full-text search (FTS5,
  Cyrillic-aware) and entity overlap, fused by reciprocal rank. A semantic index above them
  routes a query to the memories that share its entities and topics, so a small top-k is
  dense rather than merely small. When the index has nothing useful to say, recall searches
  everything: routing can cost precision, never recall.
- **Consolidates into facts.** `morgan consolidate` asks your model to turn recent memories
  into subject-predicate-object facts with validity intervals. An update closes the old
  interval and opens a new one; nothing is overwritten and history stays queryable. Every
  fact records who asserted it: you, or the model's inference.
- **Answers with what it knows.** `morgan ask` recalls first, answers, and remembers the
  exchange.
- **Forgets completely.** `morgan forget` erases a project from every table in one
  transaction, including vectors and the derived index, and reports exactly what it touched.
- **Talks to any model server.** Any OpenAI-compatible endpoint: llama-server by default,
  Ollama's `/v1`, vLLM. The model server is the only thing Morgan needs that it does not ship.
- **Two surfaces, one gate.** The `morgan` CLI and the `morgan-mcp` server (stdio, or HTTP
  with a bearer token for other machines) call the same handlers through the same
  `MemoryGate`. No memory logic is duplicated.

## Quick start

```bash
pip install -e .
mkdir -p ~/.config/morgan && cp .env.example ~/.config/morgan/.env
#   ↑ point MORGAN_LLM_ENDPOINT at your model server; read from every working directory
morgan doctor
cd ~/src/any-repo                      # the brain is the same from every repository
morgan remember "prefers terse, code-first answers"
morgan recall "how do I like answers"  # needs only the embedding model
morgan ask "what do you know about me" # needs the chat model
morgan consolidate                      # recent memories → dated facts
```

Give Claude Code the same memory:

```bash
claude mcp add morgan -- morgan-mcp --transport stdio
```

The database is `~/.local/share/morgan/morgan.db` (`MORGAN_DATA_DIR`). The memory
commands work with no model server at all under `MORGAN_EMBEDDING_BACKEND=hash`.

## Documentation

- [`docs/WIRING.md`](docs/WIRING.md) — configuration, the model server, the CLI, the MCP server, Docker.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the package, the write path, recall, consolidation, erasure.
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md) — at-rest and transport protection, backups, the stack.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — where this came from, what was cut, what is next.
- [`CLAUDE.md`](CLAUDE.md) — the invariants, for anyone (or anything) changing the code.

## History

Morgan began as a self-learning personal agent kernel: a cognitive loop, a persona graph,
an eval-gated prompt optimizer, a REST gateway, a learning worker, skills and tools. That
build is archived at the git tag `legacy-v0.1.0-kernel` and its documents under
[`docs/archive/`](docs/archive/). It was cut to this core in September 2026 because the
memory was the part that was used, and the learning loop was switched off pending an
evaluation gate sound enough to trust it. Earlier builds: `legacy-v0.0.4-full`,
`legacy-v0.0.3-monolith`. All are tags; `main` is the only branch.

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE).
