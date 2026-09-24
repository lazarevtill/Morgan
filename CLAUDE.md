# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What Morgan is

A **project-scoped memory for the owner's AI tools**, consolidated into facts by a local
model. One SQLite database under `MORGAN_DATA_DIR` holds memories, facts, vectors
(sqlite-vec), the FTS5 index, the entity index and session history. Two surfaces, the `morgan` CLI and the `morgan-mcp` server, are thin adapters over
one `MemoryGate`. The model server is any OpenAI-compatible endpoint (llama-server by
default). One process, no queue, no worker, no scheduler.

Read first: `docs/ARCHITECTURE.md` (the package), `docs/WIRING.md` (running it),
`docs/ROADMAP.md` (what is not in the code and why, what retrieval measures, what is next),
`docs/decisions/` (decisions, and what the code does about each). The archived agent kernel is
at the tag `legacy-v0.1.0-kernel`, its designs under `docs/archive/`.

## Package map (`morgan_brain/`, ~11,200 lines)

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
  another width than `MORGAN_EMBEDDING_DIM`. While a heavy migration step waits, it opens the
  gate read-only and registers nothing. `build_app_context` adds the chat model.
- `memory/` — the core. `gate.py` is the only door and `module.py` is the one write path and
  the fused recall; `embedder.py` is the embedding seam; `fingerprint.py` is the five frozen
  strings that identify an embedding space and the pure arithmetic (`cosine`, `compare`,
  `pack`/`unpack`) over their vectors, with no I/O of its own; `checked_embedder.py` wraps an
  embedder so that a process's first request also carries those strings, and refuses with
  `EmbeddingSpaceMismatch` when the model answering is not the one that wrote the active
  space; its `check()` re-sends them on demand, never short-circuited by that first-call
  cache, for `morgan import`'s canary; `migrations.py` upgrades a database written by an
  older version, its light steps when it is opened and its heavy ones under `morgan migrate`;
  `snapshot.py` writes and lists verified `VACUUM INTO` copies of the whole database, and
  restores one behind a safety
  snapshot of its own.
  `secrets/` is the secret gate: `rules.py`, the 31 rules (21 provider, 4 generic, 6 Russian
  identifiers) with their checksums, keyword gates and runtime-assembled fixtures, every
  threshold a setting; `scan.py`, the scanner -- two verdicts, windows with overlap, a tool
  call's decoded JSON values, positions on the stored text, `[redacted:<rule>]` and
  `SecretRefused` by rule and offset, never a value; CPU only, no model, no network. Below them:
  - `store/` — persistence only: `db` (refuses an SQLite older than 3.42.0 by name, and opens every
    connection with `secure_delete` on), `episodic`, `temporal`, `vectors`, `fts`, `entities`,
    `history`, `sessions` (the archive's `sessions` and `turns` tables, `turns_fts` as the
    turns' keyword index, and their writers and readers: `upsert_session`, `mark_trigger_first`,
    `add_session_counts`, `insert_turns`, `get_session`, `find_session`, `turns_of`,
    `list_sessions` and `search_turns`; `corrections_fts`, the keyword index over corrections'
    normalised text, and `turn_links`, a turn's links to earlier similar turns by its columns,
    are also created here, with no writer or reader of either yet; the capture cursors with
    their lease, the exclusions, the capture state and the pause intervals are also created
    here, each with its own writers and readers), `calls` (the `call_log`
    table and its `ts` index, and its writer and readers: `insert_call`, `calls_between` and
    `call_counts_since`), `digests` (`digests`, `digest_refs`, `digest_ratings` and
    `link_ratings` and their indexes, and their writers and readers: `insert_digest`,
    `has_first_for_session`, `get_digest`, `last_digest`, `newest_unrated_first`,
    `digests_between`, `ratings_of`, `rate_line`, `rate_link`, `link_ratings_between`,
    `digests_quoting` and `backfill_entrypoint`), `spaces` (the `embedding_spaces`
    table and its one-active partial index), `projects` (the `projects`
    table, keyed by name: classification, remote, root and the per-project capture/consolidate
    switches; `get` and `list_all`, `seed` for migration step 7's one row per project already
    named in `memories`, `facts` or `session_history`, `register` for every project-keyed write
    after it, and `record` for the classification, remote and root a CLI write from inside a
    repository fills in), `tables` (`PROJECT_TABLES`, the one list of project-keyed tables
    `forget` reaches). Each owns its schema and its queries; none of them ranks anything. Every
    write goes through `db.write_transaction`.
  - `recall/` — `fusion` (reciprocal rank over vector and keyword search, rank-only),
    `floor` (the relevance floor, judged on vector scores), `language` (a query's language
    by script alone, no model call, logged on every recall) and `render` (`delimit`, the
    `<<<morgan-<kind> <token>>>>` block a model reads stored text inside, every line marked;
    `neutralise_links`; pure).
  - `knowledge/` — `extract`, `surprise`, `fact_ops`, `consolidation`.
    The work that costs a model call or a full pass, and never runs inside a recall.
- `eval/` — measuring what recall returns: labelled probes, recall@k, MRR and leak rate,
  scored per probe kind, printed beside the run's configuration. Run with `pytest --live`
  against a real embedding endpoint.
- `providers/` — the only place a model SDK is imported: `openai_compat.py` (chat),
  `embeddings.py`, `structured.py` (JSON-validated output), `factory.py` (settings → adapters,
  where embeddings are sent, the `CheckedEmbedder` around the embedding model, and which
  retry budget an embedding call gets), `wire.py` (message types, `ChatClient`,
  `ProviderUnreachable`, `ProviderRefused`, `EmbeddingSpaceMismatch`, and `is_refusal`,
  which statuses refuse a request, for the embedder and `doctor` alike).
- `app/chat.py` — one turn: recall, answer, remember. Not a surface: the one use-case both
  surfaces call. `app/chatgpt_import.py` seeds memory from a ChatGPT export, routes a fifth of
  the conversations to a holdout project, and runs the import canary.
- `surfaces/` — where requests come in. `cli/` (`__main__` parses and dispatches, `commands`
  answers, `maintenance` answers `morgan snapshot`, `morgan restore` and `morgan migrate`,
  `payloads` shapes the result, `render` prints it, `doctor` diagnoses by reading only -- it
  opens the file read-only, builds no store, creates no table and runs no migration step --
  and probes the chat and embedding servers separately, telling reachable, slow, refused and
  unreachable apart, `project` resolves the enclosing git repository once -- its name, its
  root and its remote, read by parsing the repository's own git config rather than spawning
  `git` -- and holds `classify`, `install_skill` writes the packaged `skill/SKILL.md`
  into the coding agents installed here), `mcp_server.py` (five MCP tools over stdio or
  streamable-HTTP, calling those same command handlers), and `network.py`, the bind guard
  that protects the HTTP one.

## Invariants

- **All memory access goes through `MemoryGate`.** No caller holds the `MemoryModule`.
- **Every read and write is project-scoped.** `Memory` and `TemporalFact` carry a required
  `project`; the gate rejects an empty one. A write that names no project lands in `personal`,
  and the result says so; there is no silent default. `all_projects=True` is the explicit
  cross-project escape hatch, never the default.
- **One write path.** `MemoryModule.store` writes every index in one transaction: episodic
  row, vector, FTS5, entity index, and the project's `projects` row. Entities are extracted
  there when the caller gave none. Every row carries its provenance: origin, client, session,
  working directory, author and scope, with defaults; every writer names its origin. A memory
  visible to one index and not another is found by one search and missed by the next.
- **The secret gate sits on every write.** `memory/secrets` scans a memory's content, a
  fact's three fields, a history row, an imported message and a recorded remote before it is
  stored, and `ask`'s question before it is embedded or sent. A provider token refuses text
  the caller can rephrase — a `remember`, a question, a fact — and is redacted and counted in
  history nobody can: an imported message, a reply already produced. A recorded remote loses
  its userinfo and the rest is scanned under redact, so a token in its path or query is stored
  as its placeholder; its classification is computed from the remote as read, before that
  scan, and its root, a local path, is stored as given. A generic or high-entropy match and a
  Russian identifier with a passing checksum and its keyword are redacted and flagged; a
  pattern without a checksum is flagged only beside its keyword. No error, log line, result or
  report ever carries the value: a refusal names the rule, the offset and the length, and a
  question that goes on goes on redacted.
- **`forget` reaches every project-keyed table.** `store/tables.py::PROJECT_TABLES` is the one
  list; a store that adds a table registers it there, and a test fails on any table with a
  `project` column missing from it. `forget` walks the registry and erases each table through
  the deleter its store owns, by the memory or turn ids it holds, and by the owner's
  `user_id` and `project` columns where that deleter uses them; `turns_fts` and
  `corrections_fts` are erased at the rowids of the turns they index, never by their own
  `user_id`/`project` columns, because a filter on an FTS5 table's UNINDEXED column would scan
  it inside the lock. A space's vec0 table other than `vec_items` is erased by those owner
  columns alone. A registered table it cannot erase that way stops it by name before
  anything is erased. The forgotten words leave the database files: FTS5 is optimized, the
  database vacuumed and the write-ahead log truncated; while another connection's read blocks
  that checkpoint they stay until a later one, and a warning says so. The snapshot `morgan
  forget` takes first is the undo and keeps them until the owner deletes it.
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
  key has at most one current fact, and a unique index on `facts` enforces it. The key is
  (`user_id`, `project`, `subject`, `predicate`); it gains `author_id` and `scope` once
  supersession is assembled deterministically
  (`docs/decisions/0001-fact-key-and-forget-reach.md`).
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
  the project is `personal`. One resolution answers the name and the repository together, so
  the two can never come from different checkouts.
- **Every project written to has a row, and a CLI write from inside a repository says what
  that repository is.** A memory, a fact and a turn of session history each register their
  project in `projects` in their own write transaction (`INSERT OR IGNORE`, `unclassified`),
  below the gate, so both surfaces do it and a failed write leaves no row. `remember`, `ask`
  and `consolidate` then record `classify(remote, MORGAN_WORK_REMOTE_GLOBS)`, the remote and
  the root through the gate -- only when the project was named by the repository they ran in
  (not `--project`, not the `personal` default, not `--all-projects`), recomputed every time,
  and refused like any other write while a migration step waits. A read records nothing. Only
  those three columns are written: the owner's capture, retention and consolidate switches are
  theirs. A config Morgan cannot read is not a repository without a remote: the reader says
  which of the two it found, and only the second is recorded -- the first leaves the row as it
  is. A recording that fails never fails the command whose write already committed; it warns
  on stderr, naming the project. The remote and the root are the owner's data -- never logged,
  and `doctor` prints neither.
- **No listener beyond loopback without a key.** `network.assert_safe_bind` refuses to start
  `morgan-mcp --transport http` on a non-loopback host while `MORGAN_API_KEY` is unset or the
  placeholder.
- **One of each.** One settings object, one database, one way to write to it, one gate, one
  entity extractor, one logging configuration.
- **Never hardcode the owner.** Everything is keyed by `user_id`; single-owner is a config fact.

## Known limitations

- `recall` declines to answer only when `MORGAN_RECALL_FLOOR_MARGIN` is set. The value belongs
  to the embedding model: 0.08 for `qwen3-embedding:8b`, measured on a 3,010-memory archive
  with 128 labelled questions (`docs/measurements/2026-09-phase0-baseline.md`); 0.11 for
  Qwen3-Embedding-0.6B, measured on the bundled probes and on a real archive. Other models are
  unmeasured.
- `consolidate` reads a project's episodics with `recall(text="", top_k=50)`: the memories
  nearest an empty query's embedding, not the most recent, and fewer than 50 when the project
  holds facts, which take up to half of those places. With `MORGAN_RECALL_FLOOR_MARGIN` set,
  the floor can decline that query, and consolidate then has nothing to read.
- The import canary catches a model that is still answering wrong when a check runs; a single
  transient wrong vector between two checks passes it. Suspect memories stay stored, a re-run
  of the import skips them, and `doctor --vectors` samples rather than checks every row.
  Nothing re-embeds named memory ids.
- A database keeps the embedding model its vectors were written with. Once its space is
  fingerprinted, a different model -- same width or not -- is refused on every path that
  stores or searches; nothing retires a space or re-embeds, and a snapshot holds the same
  model's vectors. To change the model on purpose, point `MORGAN_DATA_DIR` at a new database, with
  `MORGAN_EMBEDDING_DIM` at the new model's width; the old one keeps working with its own.
- The remote is read from the repository's own config file: `url.<base>.insteadOf` rewriting
  is not applied, and `include`/`includeIf` files are not followed. A work host reached only
  through a rewrite is classified by the URL as written, and a remote declared in an included
  file is not seen at all, leaving the project `unclassified`.
- `projects` is keyed by the project's name, and a project is named after its folder, so two
  checkouts called the same thing share one row: each CLI write from either rewrites the
  other's classification, remote and root. Nothing acts on the label yet, and recall
  and consolidation are unaffected -- they are keyed by that same name, so the two checkouts
  share their memories as well.
- A project written to only through `morgan-mcp` has a row and stays `unclassified`. The
  server may run on another machine than the client, so it never sees the repository a call
  came from, and its `project` argument is a name, not a checkout. The classification, remote
  and root are recorded by a CLI write from inside the repository; the walk over
  `MORGAN_CODE_ROOTS` that would classify the rest is not built.
- `morgan-mcp` builds a FastMCP server, and FastMCP's own settings read a `./.env` in the
  folder the client starts the server in. FastMCP passes every one of its settings
  explicitly, so the `FASTMCP_*` values in that file change nothing, and Morgan's settings
  never read it; but a `./.env` there that is not UTF-8 stops `morgan-mcp` at start with a
  `UnicodeDecodeError`.
- Entity extraction is deterministic and cased-script only; scripts without letter case
  (Chinese, Japanese, Arabic, Hebrew) yield nothing rather than a guess. A capitalised word is
  a name only where the text capitalises it away from a sentence, clause or line start, so a
  name that only ever opens sentences in a memory is not indexed. Code in a memory still
  yields words like `true` and `error`.
- A superseded memory outranks the current one in 7 of the 10 knowledge-update probes
  (`qwen3-embedding:8b`). Fusion is rank-only and carries no recency term. Supersession lives
  on facts; the probes store episodics, which carry none.
- Multi-hop questions are not answered. Recall ranks memories and has no mechanism to
  compose two of them; measured recall@8 is 0.62 against 0.90 for single-hop
  (`qwen3-embedding:8b`).
- A hex secret is caught only through an assignment's context: uniform hex tops out at 4.0
  bits per character, below the entropy rule's threshold, so a hex key in free text with no
  `password=` or `token:` beside it is not redacted.
- A Russian identifier without its keyword is not redacted: an INN, SNILS, OGRN, card number,
  passport or phone number is read only beside its keyword (`ИНН`, `СНИЛС`, `ОГРН`, `карт`,
  `паспорт`, `тел`, or their Latin forms) on the same line, because a tenth of epoch timestamps
  pass the checksums.

## Build, test, run

```bash
pip install -e ".[dev]"
mkdir -p ~/.config/morgan && cp .env.example ~/.config/morgan/.env   # MORGAN_LLM_ENDPOINT
morgan doctor
pytest -q                     # 690 passed, 4 skipped (the live ones)
ruff check . && ruff format --check . && mypy morgan_brain && bandit -c pyproject.toml -r morgan_brain
```

Python 3.12+, line length 100, `ruff` is the linter and formatter, `mypy --strict`. Keep
`main` green. Root-cause fixes only; a workaround gets flagged, not silently applied.
