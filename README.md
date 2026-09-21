# Morgan

A **project-scoped memory for your AI tools**, consolidated into facts by a local model.

You tell Morgan things from any repository, on any of your machines. Claude Code, Claude
Desktop or any other MCP client recalls them, scoped to the repository it is working in. A
local model turns what was said into durable, dated facts that evolve instead of being
overwritten. Everything lives in one SQLite file on hardware you own.

## What it does

- **Remembers per project.** Every memory belongs to a project, which the CLI takes from the
  current git repository's name, worktrees included. Recall is scoped to it;
  `--all-projects` is the explicit escape hatch.
- **Recalls by meaning and by keyword.** Vector search (sqlite-vec) and full-text search
  (FTS5, Cyrillic-aware) over every memory in the project, fused by reciprocal rank. With a
  relevance floor set, recall can decline a question the project cannot answer.
- **Consolidates into facts.** `morgan consolidate` asks your model to turn up to 50 of a
  project's memories, the ones nearest an empty query rather than the most recent, into
  subject-predicate-object facts with validity intervals. An update closes the old interval
  and opens a new one; nothing is overwritten, and closed intervals stay in the database.
  Every fact records who asserted it; the facts consolidation writes are marked as the
  model's inference, never as something you said.
- **Answers with what it knows.** `morgan ask` recalls first, answers, and remembers the
  exchange.
- **Starts from your history, not from nothing.** `morgan import` seeds memory from a ChatGPT
  export. A fifth of the conversations are held back in a separate project, so the memory can
  later be evaluated against conversations nothing has learned from.
- **Forgets a project.** `morgan forget` erases it from every table in one transaction,
  including vectors and the entity index, and reports exactly what it touched. The snapshot
  it takes first is the undo, and keeps a full copy until you delete it.
- **Talks to any model server.** Any OpenAI-compatible endpoint: llama-server by default,
  Ollama's `/v1`, vLLM. The model server is the only thing Morgan needs that it does not ship.
- **Two surfaces, one gate.** The `morgan` CLI and the `morgan-mcp` server (stdio, or HTTP
  with a bearer token for other machines) call the same handlers through the same
  `MemoryGate`. No memory logic is duplicated.

## Quick start

```bash
uv tool install --editable .           # morgan and morgan-mcp on PATH, for every agent
mkdir -p ~/.config/morgan && cp .env.example ~/.config/morgan/.env
#   ↑ point MORGAN_LLM_ENDPOINT at your model server, and MORGAN_EMBEDDING_ENDPOINT at the
#     embedding server if it is a separate one; read from every working directory
morgan doctor                          # the database, the chat server, the embedding server
cd ~/src/any-repo                      # the brain is the same from every repository
morgan remember "prefers terse, code-first answers"
morgan recall "how do I like answers"  # needs only the embedding model
morgan ask "what do you know about me" # needs the chat model
morgan consolidate                      # memories nearest an empty query → dated facts
morgan import ~/Downloads/conversations.json   # optional: seed from a ChatGPT export
```

Give Claude Code the same memory, and teach every coding agent here when to use it:

```bash
claude mcp add -s user morgan -- morgan-mcp --transport stdio   # in every project
morgan install-skill                   # lists what it writes, then asks
```

The database is `~/.local/share/morgan/morgan.db` (`MORGAN_DATA_DIR`). The memory
commands work with no model server at all under `MORGAN_EMBEDDING_BACKEND=hash`. After an
upgrade, `morgan doctor` says whether the database waits for `morgan migrate`, which upgrades
it behind a snapshot; `morgan snapshot` and `morgan restore` are the backup and its undo.

## Is the recall any good?

Measured, not assumed. `tests/memory_quality/` holds labelled probes — half of them Russian,
every query written to share few or no words with its answer — and `pytest --live` scores them
against a real embedding endpoint: recall@k, where in the ranking the answer landed, whether a
superseded memory came back with it, and whether an unanswerable question was correctly met
with silence. [`docs/ROADMAP.md`](docs/ROADMAP.md) carries the current numbers, including the
two categories that do not yet work.

## Documentation

- [`docs/WIRING.md`](docs/WIRING.md) — configuration, the model server, the CLI (snapshot,
  restore and migrate included), the MCP server, Docker.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the package, recall, the embedding space,
  migrations and snapshots, consolidation, erasure.
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md) — at-rest and transport protection, backups, the stack.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — what is not in the code and why, what retrieval
  measures, what is next.
- [`docs/decisions/`](docs/decisions/) — decisions, and what the code does about each.
- [`CLAUDE.md`](CLAUDE.md) — the invariants, for anyone (or anything) changing the code;
  [`AGENTS.md`](AGENTS.md) carries the same text for Codex and the other agents that read it.
- [`docs/archive/`](docs/archive/) — the designs of the archived agent kernel, whose code is
  at the git tag `legacy-v0.1.0-kernel`.

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE).
