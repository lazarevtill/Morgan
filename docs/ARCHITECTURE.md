# Architecture

One package, `morgan_brain`, one process, one SQLite file. Two surfaces over one gate.

```text
morgan CLI ──┐                          ┌─ episodic rows
             ├─▶ MemoryGate ─▶ MemoryModule ─┼─ sqlite-vec vectors
morgan-mcp ──┘        │                 ├─ FTS5 keyword index      one morgan.db
                      │                 └─ entity index
   Chat (ask) ────────┤
   Consolidator ──────┘   ──▶ valid-time facts
         │
         └──▶ model server (any OpenAI-compatible endpoint)
```

## The package (`morgan_brain/`)

| Module | Responsibility |
| --- | --- |
| `config.py` | The single `MORGAN_`-prefixed settings source, `settings_for(surface)`. The CLI reads `~/.config/morgan/.env`, then `./.env`, then the environment; `morgan-mcp` reads the user file and the environment only, because its working directory is the client's. `doctor` lists the files read. The database defaults to `~/.local/share/morgan/`, its snapshots to `snapshots/` beside it. Chat and embeddings are addressed separately when one server does not serve both. |
| `models.py` | `Memory`, `TemporalFact`, `MemoryQuery`, `Message`. Everything that persists is `user_id`- and `project`-keyed; a write that names no project lands in `personal`. A `Memory` carries its provenance: origin, client, session, working directory, author and scope. |
| `logging_setup.py` | Process output: stdout is UTF-8 because the protocols on it are; every log line goes to stderr. |
| `composition.py` | Opens the database and wires everything. `build_memory_context` for the memory commands: it stamps a database it creates at the last migration step, runs the pending light steps, opens the gate read-only while a heavy step waits, and on a writable database with no embedding space registers the settings' model and width (the hash backend registers none). It sends no request. `build_app_context` adds the chat client. |
| `memory/gate.py` | `MemoryGate`: the only door to memory. Refuses an empty user or project, and every write with `DatabaseNeedsMigration` while a heavy step waits. `RecallOutcome`, `ForgetReport`. |
| `memory/module.py` | `MemoryModule`: the one write path (every index in one transaction, entities extracted if absent, the project registered in `projects`) and the fused recall, which logs one `recall.done` line per call. `forget()` in one transaction. |
| `memory/embedder.py` | The `Embedder` protocol and the deterministic hash stub that stands in for a model server. |
| `memory/fingerprint.py` | The five fixed strings that identify an embedding space, and the arithmetic over their vectors (`cosine`, `compare`, `pack`/`unpack`). No I/O. |
| `memory/checked_embedder.py` | Wraps the live embedder. A process's first request also carries the five strings, and, while the active space has no fingerprint, up to `MORGAN_EMBEDDING_FINGERPRINT_SAMPLE_ROWS` stored memories: the fingerprint is recorded only when their fresh vectors match the stored ones, and compared once it is. Answers below `MORGAN_EMBEDDING_FINGERPRINT_TOLERANCE` raise `EmbeddingSpaceMismatch`. `check()` re-sends the strings on demand, for the import canary. |
| `memory/migrations.py` | Numbered steps, counted in `user_version`. A light step only adds a table or a defaulted column and runs when the database is opened; a heavy step rewrites, moves or deletes rows and runs only under `morgan migrate`. Steps run strictly in order, so a light step behind a heavy one waits with it. Steps 1 and 2 rewrite and drop, yet stay light: they predate the split. See [Migrations and snapshots](#migrations-and-snapshots). |
| `memory/snapshot.py` | `VACUUM INTO` copies of the whole database, each `quick_check`ed before it is returned, and `restore`, which puts one back behind a snapshot of its own. |
| `memory/store/` | Persistence, one file per table family, all over the one connection: `db`, `episodic`, `temporal`, `vectors`, `fts`, `entities`, `history`, `spaces` (`embedding_spaces`, with a partial unique index that keeps one space active), `projects` (one row per project: classification, remote, root, and the capture and consolidate switches; every project-keyed write registers its row, a CLI write from inside a repository records what that repository is, and neither ever touches the owner's switches) and `tables` (the registry of project-keyed tables that `forget` and `distinct_projects` read). Every write goes through `db.write_transaction`, which holds the write lock from the first statement and nests as a savepoint, so a store method is atomic alone and inside a larger write. A fact key has at most one current fact, enforced by a unique index; opening a database that a race left with two keeps the newest and closes the rest. Vectors are scoped *inside* the KNN via vec0 metadata columns, not filtered afterwards. |
| `memory/recall/` | `fusion` merges the vector and keyword rankings by reciprocal rank. Rank-only, so the relevance threshold, `floor`, judges the vector scores before fusion. `language` names a query's language by script alone, for the log line. |
| `memory/knowledge/` | `extract` (the one entity extractor: words the text capitalises away from a sentence start, acronyms, CamelCase; Latin and Cyrillic), `surprise` (drops episodics the facts already predict), `fact_ops` (the operation schema the model is constrained to), `consolidation` (applies them: supersede, never overwrite). |
| `providers/` | `openai_compat.py` (chat over the `openai` SDK), `embeddings.py` (`/embeddings` over httpx, retried within a budget), `structured.py` (JSON-schema, JSON-object or prompted, validated, re-asked), `factory.py` (settings → adapters; the one place that decides where embeddings are sent, and so which setting an unreachable server's error names; wraps the embedder in `CheckedEmbedder`; picks the interactive or the import retry budget), `wire.py` (`ChatClient`, `ProviderUnreachable`, `ProviderRefused`, `EmbeddingSpaceMismatch`, and `is_refusal`, the one rule for which HTTP statuses refuse a request). Nothing above imports a model SDK. |
| `eval/retrieval.py` | Labelled probes, and the recall@k / MRR / leak-rate scorecard they produce, printed beside a `RunConfig`: embedding model and width, k, floor, probe file and its digest, corpus size, database upgrade step, commit. Never the endpoint. The measurement that turns retrieval quality from an assumption into a number. |
| `app/chatgpt_import.py` | Seeds memory from a ChatGPT export. Splits a turn too long for the embedding context, routes a fifth of conversations to a holdout project the optimizer can never mine, and runs the import canary. |
| `app/chat.py` | One turn: recall → prompt → answer → remember both halves, attributed. The one use-case both surfaces share. |
| `surfaces/cli/` | `morgan`: `__main__` parses and dispatches, `commands` answers, `maintenance` answers `snapshot`, `restore` and `migrate`, `payloads` shapes the result, `render` prints it, `doctor` diagnoses the install by reading only -- the file opened read-only, no store built, no table created, no migration step run -- and probes the chat and embedding servers separately, telling reachable, slow, refused and unreachable apart, `project` names the project and reads its repository's remote, `install_skill` teaches the coding agents installed here when to use Morgan. Project = the enclosing git repository's name; a linked worktree counts as the repository it came from; outside a repository it is `personal`. One resolution answers both the name and the repository, and the remote comes from parsing that repository's git config -- `origin`, else the sole remote, else none -- never from spawning `git`. The answer is three-valued: a remote, no remote, or a config that could not be read, which records nothing rather than relabelling the project. `project.classify` labels it against `MORGAN_WORK_REMOTE_GLOBS`, and `remember`, `ask` and `consolidate` record that label, the remote and the root when the project was named by the repository they ran in. |
| `surfaces/mcp_server.py` | `morgan-mcp`: `remember`, `recall`, `facts`, `forget`, `ask_morgan` over stdio or streamable-HTTP with a bearer token. Calls the CLI's command handlers; `project` is a tool argument, and a call without one works in `personal` (`remember` says so, with `project_defaulted`). Every tool declares MCP's read-only, destructive, idempotent and open-world hints; only `recall` and `facts` claim read-only. |
| `surfaces/network.py` | The bind guard: no listener beyond loopback without a real key. |

## Recall

1. The query is embedded, and vector search returns its top 2k over the whole project, or
   every project with `all_projects`.
2. With `MORGAN_RECALL_FLOOR_MARGIN` set, the floor judges the vector hits alone
   (`recall/floor.py`), before anything else is gathered: recall declines unless the best one
   stands that far above the background the same query pulled up, or an exact entity match
   lands on a memory the vector search ranked in its top k. A decline returns nothing, facts
   included. Fewer than five vector hits are no background to judge against, so they go on
   unjudged. Unset, a non-empty project always answers.
3. FTS5 search returns its top 2k over the same scope, and reciprocal rank fusion merges the
   two rankings. The entity index is not a third: a stored name is in the memory's text,
   which the keyword search already matches.
4. Currently-valid facts for the project are placed first, but budgeted: episodics keep half
   the window whenever they have hits, and the facts that survive a narrow budget are the
   ones the query mentions. Facts fill the whole window only when little else came back.

Recall returns a `RecallOutcome`: the memories, `abstained`, and a `reason`, which the CLI's
`--json`, the MCP result and the `recall.done` log line all carry. An empty result is
abstained, as `empty` (nothing in scope, not even a fact) or `declined` (the floor). Results
come back with `too_few_to_judge`, `no_floor`, or `null` when the floor judged them and they
answered. `keyword_only` is reserved for a keyword-only fallback when embeddings are down,
which does not exist yet; nothing emits it. The `recall.done` line also carries the
embedding's latency and outcome and the query's language.

Facts never suppress episodics: prepended in full and then truncated, a project holding
`top_k` facts could return no memory however exactly it matched, and the fact count only
grows as consolidation runs.

## The embedding space

`embedding_spaces` records which model wrote the stored vectors: model, width, prefixes, the
vec0 table that holds them, and the fingerprint, the vectors of five fixed strings
(`memory/fingerprint.py`). One space is active. Under the provider backend, opening a writable
database that has none registers the settings' model and width, refusing when the vector table
was created at another width; the hash backend, which no model answers, registers none. An
active space of another width than `MORGAN_EMBEDDING_DIM` is refused on open.

Nothing embeds at open. A process's first embedding call carries the check
(`CheckedEmbedder`): the five strings ride on the request that was being sent anyway, and the
answer is compared with the recorded fingerprint. A model answering outside the tolerance is
`EmbeddingSpaceMismatch`, named, never a silently wrong search, so a model of the same width
is caught like one of another width. While no fingerprint is recorded, the same request
re-embeds a sample of stored memories, and the fingerprint is recorded only when their fresh
vectors match the stored ones. `morgan migrate` makes the same check once its wave has
committed.

## Migrations and snapshots

A database written by an older Morgan is upgraded by numbered steps:

| Step | Kind | What |
| --- | --- | --- |
| 1 | light | re-extract every memory's entities |
| 2 | light | drop the semantic index's tables |
| 3 | light | create `embedding_spaces` and `projects` |
| 4 | heavy | provenance columns on `memories` and `facts`, backfilled |
| 5 | heavy | rename the project `default` to `personal` in every project-keyed table |
| 6 | heavy | rebuild `vec_items` and `fts_memories` with `status`, `scope` and `author_id` |
| 7 | light | seed `projects` with one row per project already named (every write after this registers its own) |
| 8 | light | create the session archive's fourteen tables (`sessions`, `turns`, their two FTS5 indexes, `turn_links`, the capture cursors, exclusions, pauses and state, `call_log`, `digests`, `digest_refs`, `digest_ratings`, `link_ratings`) and add `retention_confirmed` to `projects` and `redactions`/`flags` to `memories` and `facts` where missing |

Opening the database runs the pending light steps up to the first heavy one. While a heavy
step waits, the gate refuses every write with `DatabaseNeedsMigration`, whose message names the
pending steps and `morgan migrate`; reads work on the schema the database already has.
`morgan migrate` takes a `migrate` snapshot, runs every pending step in one write transaction,
runs `PRAGMA quick_check`, and reports the row counts before and after. A step that fails
rolls the whole wave back, and the snapshot stays.

A snapshot is a `VACUUM INTO` copy of the whole file in `MORGAN_SNAPSHOT_DIR`, named by UTC
time and reason, and `quick_check`ed before it is returned; one that fails the check is
deleted, and Morgan deletes none for any other reason. `morgan snapshot` takes one on demand,
`morgan migrate` and `morgan forget` take one before they change anything, and `morgan restore`
takes a `before-restore` one before it puts a snapshot back. A snapshot newer than this code's
steps is refused.

## Consolidation (`morgan consolidate`)

Episodics read through recall, minus those current facts already cover → the model proposes
fact operations as JSON validated against `FactOpBatch` → applied through the gate, in one
write transaction that re-reads the current facts first, so two runs at once see each other's
result. UPDATE closes the old interval and opens a new one (`valid_to`, `superseded_by`);
DELETE closes it; confidence decays with age since last confirmation. `--all-projects` skips a
project whose `projects` row switches consolidation off. It runs when asked, never on a
schedule of its own.

## Erasure (`morgan forget`)

First a `forget` snapshot, the undo. Then one write transaction, holding the lock before the
memory ids are read. It erases:

- memories, FTS rows and entity rows;
- vectors: `vec_meta` and every embedding space's vec0 table, `vec_items` among them;
- facts and session history;
- the project's `projects` row, once no owner has a row left in the project.

It walks `store/tables.py`, the registry of the tables this has to reach, and erases each table
with the deleter its store owns:

- Rows are matched by the memory or turn ids the deleter holds, and by the owner's
  `user_id` and `project` columns where the deleter uses them, so an index row whose memory is
  gone goes too; `turns_fts` and `corrections_fts` are matched at the rowids of the turns they
  index, never by their own `user_id`/`project` columns, because a filter on an FTS5 table's
  UNINDEXED column would scan it inside the lock.
- A space's vec0 table other than `vec_items` is matched by those columns alone.
- A registered table with no deleter, or a space's table without those columns, stops it by
  name before anything is erased.
- FTS5's `optimize` then merges `fts_memories`, so the deleted words leave its segments.
- The name-keyed tables go last, so the `projects` row's check sees what the erasure left.

A memory being stored by another process is either entirely erased or entirely kept, because
storing is one transaction too. Tables that were never created on this database are named in
`tables_skipped` rather than counted as zero.

After the commit, it runs `VACUUM`, then `PRAGMA wal_checkpoint(TRUNCATE)`, so the forgotten
words are in neither the database file nor its write-ahead log. A connection in the middle of a
read blocks that checkpoint: the forgotten words then stay in the database file and its log
until a later checkpoint completes, and `forget` logs `forget.wal-not-truncated`, never raises.

[`decisions/0001`](decisions/0001-fact-key-and-forget-reach.md) says what it guarantees.

## Tests (`tests/`)

`unit/` per module; `integration/` runs the CLI as a subprocess, the MCP server over raw stdio
pipes and in-process, cross-process durability, two processes upserting the same vectors or
superseding the same facts at once, a vector delete racing a reinsert, a project erased while
a memory is being stored, two consolidation runs applying the same facts, erasure atomicity and
completeness, migrate and restore through the CLI, the wheel build. Four live tests
(`pytest --live`) need a real embedding model. `pip install -e ".[dev]"` installs exactly what
the suite needs. `tests/fakes.py` holds the scripted chat client and a loopback
OpenAI-compatible server for the probes that need one to answer; nothing in the package exists
only for tests.
