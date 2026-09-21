# AGENTS.md

Guidance for Codex when working in this repository.

## What Morgan is

A **project-scoped memory for the owner's AI tools**, consolidated into facts by a local
model. One SQLite database under `MORGAN_DATA_DIR` holds memories, facts, vectors
(sqlite-vec), the FTS5 index, the entity index and session history. Two surfaces, the `morgan` CLI and the `morgan-mcp` server, are thin adapters over
one `MemoryGate`. The model server is any OpenAI-compatible endpoint (llama-server by
default). One process, no queue, no worker, no scheduler.

Read first: `docs/ARCHITECTURE.md` (the package), `docs/WIRING.md` (running it),
`docs/ROADMAP.md` (what was cut and why). The archived kernel this was cut from is at the
tag `legacy-v0.1.0-kernel` with its designs under `docs/archive/`.

## Package map (`morgan_brain/`, ~5,000 lines)

The tree is grouped by what a file does, so "where does a write go" and "where does a request
come in" are answered by the directory names.

- `config.py` — the single `MORGAN_`-prefixed settings source (`settings_for(surface)`). The
  CLI reads `~/.config/morgan/.env`, then `./.env`, then the environment; `morgan-mcp` reads
  the user file and the environment only, because its working directory is the client's.
  `doctor` lists the files read. The database defaults to `~/.local/share/morgan/`.
  `MORGAN_EMBEDDING_ENDPOINT` addresses embeddings separately when the chat server does not
  serve them.
- `models.py` — the domain models. Everything that persists is `user_id`- and
  `project`-keyed. `Memory` carries a `MemorySource`; `TemporalFact` carries
  `valid_from`/`valid_to`/`superseded_by`.
- `logging_setup.py` — process output. stdout is UTF-8 because the protocols on it are; logs
  go to stderr. One call per entrypoint.
- `composition.py` — opens the database and wires the above. `build_memory_context` needs no
  chat model and sends no request: it registers the settings' embedding model and width on a
  writable database that has no embedding space, refusing instead when the vector table was
  created at another width (the hash backend registers none), and it refuses a space of
  another width than `MORGAN_EMBEDDING_DIM`. `build_app_context` adds the chat model.
- `memory/` — the core. `gate.py` is the only door and `module.py` is the one write path and
  the fused recall; `embedder.py` is the embedding seam; `fingerprint.py` is the five frozen
  strings that identify an embedding space and the pure arithmetic (`cosine`, `compare`,
  `pack`/`unpack`) over their vectors, with no I/O of its own; `checked_embedder.py` wraps an
  embedder so that a process's first request also carries those strings, and refuses with
  `EmbeddingSpaceMismatch` when the model answering is not the one that wrote the active
  space; `migrations.py` upgrades a database written by an older version, its light steps
  when it is opened and its heavy ones under `morgan migrate`; `snapshot.py` writes and lists
  verified `VACUUM INTO` copies of the whole database, and restores one behind a safety
  snapshot of its own. Below them:
  - `store/` — persistence only: `db`, `episodic`, `temporal`, `vectors`, `fts`, `entities`,
    `history`, `spaces` (the `embedding_spaces` table and its one-active partial index),
    `projects` (the `projects` table, keyed by name: classification, remote, root and the
    per-project capture/consolidate switches; `get`, `all` and `seed`, migration step 7's seed
    of one row per project already named in `memories`, `facts` or `session_history`),
    `tables` (`PROJECT_TABLES`, the one list of project-keyed tables `forget` reaches). Each
    owns its schema and its queries; none of them ranks anything. Every write goes through
    `db.write_transaction`.
  - `recall/` — `fusion` (reciprocal rank over vector and keyword search, rank-only),
    `floor` (the relevance floor, judged on vector scores) and `language` (a query's language
    by script alone, no model call, logged on every recall).
  - `knowledge/` — `extract`, `surprise`, `fact_ops`, `consolidation`.
    The work that costs a model call or a full pass, and never runs inside a recall.
- `eval/` — measuring what recall returns: labelled probes, recall@k, MRR and leak rate,
  scored per probe kind, printed beside the run's configuration. Run with `pytest --live`
  against a real embedding endpoint.
- `providers/` — the only place a model SDK is imported: `openai_compat.py` (chat),
  `embeddings.py`, `structured.py` (JSON-validated output), `factory.py` (settings → adapters,
  where embeddings are sent, the `CheckedEmbedder` around the embedding model, and which
  retry budget an embedding call gets), `wire.py` (message types, `ChatClient`,
  `ProviderUnreachable`, `ProviderRefused`, `EmbeddingSpaceMismatch`).
- `app/chat.py` — one turn: recall, answer, remember. Not a surface: the one use-case both
  surfaces call.
- `surfaces/` — where requests come in. `cli/` (`__main__` parses and dispatches, `commands`
  answers, `maintenance` answers `morgan snapshot`, `morgan restore` and `morgan migrate`,
  `payloads` shapes the result, `render` prints it, `doctor` diagnoses and probes the chat
  and embedding servers separately, `install_skill` writes the packaged `skill/SKILL.md` into
  the coding agents installed here), `mcp_server.py` (five MCP tools over stdio or
  streamable-HTTP, calling those same command handlers), and `network.py`, the bind guard
  that protects the HTTP one.

## Invariants

- **All memory access goes through `MemoryGate`.** No caller holds the `MemoryModule`.
- **Every read and write is project-scoped.** `Memory` and `TemporalFact` carry a required
  `project`; the gate rejects an empty one. A write that names no project lands in `personal`,
  and the result says so; there is no silent default. `all_projects=True` is the explicit
  cross-project escape hatch, never the default.
- **One write path.** `MemoryModule.store` writes every index in one transaction: episodic
  row, vector, FTS5, entity index. Entities are extracted there when the caller gave none.
  Every row carries its provenance: origin, client, session, working directory, author and
  scope, with defaults; every writer names its origin. A memory visible to one index and not
  another is found by one search and missed by the next.
- **`forget` reaches every project-keyed table.** `store/tables.py::PROJECT_TABLES` is the one
  list; a store that adds a table registers it there, and a test fails on any table with a
  `project` column missing from it.
- **Every write holds the lock from its first statement.** Other processes share the database
  file, so a write that reads before acting takes the lock before the read:
  `store/db.py::write_transaction` opens `BEGIN IMMEDIATE`, and a write inside another one
  joins it as a savepoint. Nothing awaits while the lock is held -- the embedding happens
  before it is taken -- and no store method commits on its own.
- **Recall ranks the whole scope.** Vector and keyword search each rank every memory in the
  project (or every project, with `all_projects`), fused by rank. Nothing narrows the
  candidates first: on a real archive a narrowing pool cut answers out and found none. The
  entity index is the relevance floor's evidence, not a third ranking.
- **Light steps run on open; heavy steps run only under `morgan migrate`, behind a snapshot.**
  Derived data is re-derived by numbered steps in `memory/migrations.py`, counted in SQLite's
  `user_version`. A step that only adds a table or a defaulted column runs when the database
  is opened; a step that rewrites, moves or deletes rows runs only under `morgan migrate`,
  which takes a `VACUUM INTO` snapshot first and runs every pending step in one write
  transaction. Until then the database opens read-only and every write says so by name.
  `morgan restore` puts a snapshot back.
- **Nothing destructive runs without a snapshot.** A migration wave, a project-grain `forget`
  and any re-embed take a `VACUUM INTO` snapshot first, into `MORGAN_SNAPSHOT_DIR`, which
  Morgan never prunes.
- **Facts evolve, they don't overwrite.** Update = close the old interval, open a new one. A
  key has at most one current fact, and a unique index on `facts` enforces it.
- **Facts are surfaced alongside episodics, never instead of them.** Recall budgets the
  fact block so a matching memory cannot be pushed out of the window by fact volume.
- **Actor attribution.** Every memory records its `MemorySource`. The reply to `ask` is
  stored as `agent_inferred`; never treat an inference as a user statement.
- **One embedding space is active, and it is fingerprinted.** `embedding_spaces` records
  model, width, prefixes and the vectors of five fixed strings; a model that answers outside
  the measured tolerance of that fingerprint is a named error, never a silently wrong search. A
  same-width swap is caught like a width change.
- **A model server that is down is reported by name.** Adapters raise
  `providers.wire.ProviderUnreachable` carrying the endpoint and the setting that addresses
  it; the CLI and MCP tools print its message. A transient failure is retried first, within a
  budget: short when the host refuses connections; longer when it answers slowly or drops one,
  as a cold host loading its model does; longer still for an import. The message says which. A
  4xx is `ProviderRefused`, naming the key setting. A bare `HTTPStatusError` is a regression.
  The factory hands each adapter the setting that addresses its endpoint, because embeddings
  go to the chat endpoint unless `MORGAN_EMBEDDING_ENDPOINT` is set. A bare traceback is a
  regression, and so is sending the owner to check a server that works.
- **stdout is a protocol on both surfaces.** `--json` output and the MCP stdio transport are
  parsed by machines; all logs go to stderr. Never `print()` diagnostics from library code.
- **Nothing runs a model unasked.** `ask` and `consolidate` call the chat model; nothing else
  does, and nothing runs on a schedule. Consolidation is on demand (or the owner's own cron).
  Nothing embeds at open, in a hook or in a sweep. The active embedding space is fingerprinted
  at a process's first embedding call, riding on that call, or by `morgan migrate` once its
  wave has committed; `doctor` embeds when it is run, and `doctor --vectors` re-embeds a
  sample when asked.
- **Every MCP tool declares what it does.** `TOOL_ANNOTATIONS` states all four hints for every
  tool, and only a tool that changes nothing claims read-only: a client may run those
  unprompted, and `install-skill` allows exactly those in Claude Code. `ask_morgan` stores the
  exchange, so it is a write.
- **A project is a repository.** The CLI names it after the enclosing git repository; a linked
  worktree belongs to the repository it was created from, a submodule is its own. Outside one,
  the project is `personal`.
- **No listener beyond loopback without a key.** `network.assert_safe_bind` refuses to start
  `morgan-mcp --transport http` on a non-loopback host while `MORGAN_API_KEY` is unset or the
  placeholder.
- **One of each.** One settings object, one database, one way to write to it, one gate, one
  entity extractor, one logging configuration.
- **Never hardcode the owner.** Everything is keyed by `user_id`; single-owner is a config fact.

## Known limitations

- `recall` declines to answer only when `MORGAN_RECALL_FLOOR_MARGIN` is set. The value belongs
  to the embedding model: 0.11 for Qwen3-Embedding-0.6B, measured on the bundled probes and on
  a real archive. Other models are unmeasured.
- Entity extraction is deterministic and cased-script only; scripts without letter case
  (Chinese, Japanese, Arabic, Hebrew) yield nothing rather than a guess. A capitalised word is
  a name only where the text capitalises it away from a sentence, clause or line start, so a
  name that only ever opens sentences in a memory is not indexed. Code in a memory still
  yields words like `true` and `error`.
- A superseded memory outranks the current one in half the knowledge-update probes. Fusion
  is rank-only and carries no recency term. Supersession lives on facts; the probes store
  episodics, which carry none.
- Multi-hop questions are not answered. Recall ranks memories and has no mechanism to
  compose two of them; measured recall@8 is 0.38 against 0.95 for single-hop.

## Build, test, run

```bash
pip install -e ".[dev]"
mkdir -p ~/.config/morgan && cp .env.example ~/.config/morgan/.env   # MORGAN_LLM_ENDPOINT
morgan doctor
pytest -q                     # 560 passed, 4 skipped (the live ones)
ruff check . && ruff format --check . && mypy morgan_brain && bandit -c pyproject.toml -r morgan_brain
```

Python 3.12+, line length 100, `ruff` is the linter and formatter, `mypy --strict`. Keep
`main` green. Root-cause fixes only; a workaround gets flagged, not silently applied.
