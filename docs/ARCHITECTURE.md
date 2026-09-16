# Architecture

One package, `morgan_brain`, one process, one SQLite file. Two surfaces over one gate.

```
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
|---|---|
| `config.py` | The single `MORGAN_`-prefixed settings source. Reads `~/.config/morgan/.env`, then `./.env`, then the environment; the database defaults to `~/.local/share/morgan/`. Chat and embeddings are addressed separately when one server does not serve both. |
| `models.py` | `Memory`, `TemporalFact`, `MemoryQuery`, `Message`. Everything that persists is `user_id`- and `project`-keyed. |
| `logging_setup.py` | Process output: stdout is UTF-8 because the protocols on it are; every log line goes to stderr. |
| `composition.py` | Opens the database and wires everything. `build_memory_context` for the memory commands; `build_app_context` adds the chat client. |
| `memory/gate.py` | `MemoryGate`: the only door to memory. Refuses an empty user or project. `ForgetReport`. |
| `memory/module.py` | `MemoryModule`: the one write path (every index in one transaction, entities extracted if absent) and the fused recall. `forget()` in one transaction. |
| `memory/embedder.py` | The `Embedder` protocol and the deterministic hash stub that stands in for a model server. |
| `memory/migrations.py` | Upgrades a database written by an older version when it is opened: each step re-derives stored data whose rule changed (entities) or drops what is no longer used. Steps are counted in `user_version` and run in one write transaction, so two processes run each once. |
| `memory/store/` | Persistence, one file per table family, all over the one connection: `db`, `episodic`, `temporal`, `vectors`, `fts`, `entities`, `history`. Every write goes through `db.write_transaction`, which holds the write lock from the first statement and nests as a savepoint, so a store method is atomic alone and inside a larger write. A fact key has at most one current fact, enforced by a unique index; opening a database that a race left with two keeps the newest and closes the rest. Vectors are scoped *inside* the KNN via vec0 metadata columns, not filtered afterwards. |
| `memory/recall/` | `fusion` merges the vector and keyword rankings by reciprocal rank. Rank-only, so the relevance threshold, `floor`, judges the vector scores before fusion. |
| `memory/knowledge/` | `extract` (the one entity extractor: words the text capitalises away from a sentence start, acronyms, CamelCase; Latin and Cyrillic), `surprise` (drops episodics the facts already predict), `fact_ops` (the operation schema the model is constrained to), `consolidation` (applies them: supersede, never overwrite). |
| `providers/` | `openai_compat.py` (chat over the `openai` SDK), `embeddings.py` (`/embeddings` over httpx), `structured.py` (JSON-schema, JSON-object or prompted, validated, re-asked), `factory.py`, `wire.py` (`ChatClient`, `ProviderUnreachable`). Nothing above imports a model SDK. |
| `eval/retrieval.py` | Labelled probes, and the recall@k / MRR / leak-rate scorecard they produce, printed beside a `RunConfig`: embedding model and width, k, floor, probe file and its digest, corpus size, database upgrade step, commit. Never the endpoint. The measurement that turns retrieval quality from an assumption into a number. |
| `app/chatgpt_import.py` | Seeds memory from a ChatGPT export. Splits a turn too long for the embedding context, and routes a fifth of conversations to a holdout project the optimizer can never mine. |
| `app/chat.py` | One turn: recall → prompt → answer → remember both halves, attributed. The one use-case both surfaces share. |
| `surfaces/cli/` | `morgan`: `__main__` parses and dispatches, `commands` answers, `payloads` shapes the result, `render` prints it, `doctor` diagnoses the install, `install_skill` teaches the coding agents installed here when to use Morgan. Project = the enclosing git repository's name; a linked worktree counts as the repository it came from. |
| `surfaces/mcp_server.py` | `morgan-mcp`: `remember`, `recall`, `facts`, `forget`, `ask_morgan` over stdio or streamable-HTTP with a bearer token. Calls the CLI's command handlers; `project` is a tool argument. Every tool declares MCP's read-only, destructive, idempotent and open-world hints; only `recall` and `facts` claim read-only. |
| `surfaces/network.py` | The bind guard: no listener beyond loopback without a real key. |

## Recall

1. Vector and FTS5 search each return their top 2k over the whole project, or every project
   with `all_projects`.
2. Reciprocal rank fusion merges the two rankings. The entity index is not a third: a stored
   name is in the memory's text, which the keyword search already matches.
3. Currently-valid facts for the project are placed first, but budgeted: episodics keep half
   the window whenever they have hits, and the facts that survive a narrow budget are the
   ones the query mentions. Facts fill the whole window only when little else came back.
4. With `MORGAN_RECALL_FLOOR_MARGIN` set, recall returns nothing unless the best vector hit
   stands that far above the background the same query pulled up (`recall/floor.py`), or an
   exact entity match lands on a memory the vector search also ranked. Unset, a non-empty
   project always answers.

Facts never suppress episodics. Prepending every fact and then truncating meant that once
a project held `top_k` facts, no memory could be returned however exactly it matched --
silently, since the fact count only grows as consolidation runs.

## Consolidation (`morgan consolidate`)

Recent episodics minus those current facts already cover → the model proposes fact operations
as JSON validated against `FactOpBatch` → applied through the gate, in one write transaction
that re-reads the current facts first, so two runs at once see each other's result. UPDATE
closes the old interval and opens a new one (`valid_to`, `superseded_by`); DELETE closes it;
confidence decays with age since last confirmation. It runs when asked, never on a schedule of
its own.

## Erasure (`morgan forget`)

One write transaction, holding the lock before the memory ids are read, then: memories, FTS
rows, entity rows, vectors (`vec_items` + `vec_meta`), facts, session history. A memory being stored by another process is either entirely erased
or entirely kept, because storing is one transaction too. Tables that were never created on
this database are named in `tables_skipped` rather than counted as zero. Vacuum afterwards.

## Tests (`tests/`)

`unit/` per module; `integration/` runs the CLI as a subprocess, the MCP server over raw stdio
pipes and in-process, cross-process durability, two processes upserting the same vectors or
superseding the same facts at once, a vector delete racing a reinsert, a project erased while
a memory is being stored, two consolidation runs applying the same facts, erasure atomicity and
completeness, the wheel build. One live test (`pytest --live`) needs a real
embedding model.
`pip install -e ".[dev]"` installs exactly what the suite needs. `tests/fakes.py` holds the
scripted chat client; nothing in the package exists only for tests.
