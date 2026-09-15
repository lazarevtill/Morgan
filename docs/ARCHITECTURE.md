# Architecture

One package, `morgan_brain`, one process, one SQLite file. Two surfaces over one gate.

```
morgan CLI ──┐                          ┌─ episodic rows
             ├─▶ MemoryGate ─▶ MemoryModule ─┼─ sqlite-vec vectors
morgan-mcp ──┘        │                 ├─ FTS5 keyword index      one morgan.db
                      │                 ├─ entity index
   Chat (ask) ────────┤                 └─ semantic upper index
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
| `memory/store/` | Persistence, one file per table family, all over the one connection: `db`, `episodic`, `temporal`, `vectors`, `fts`, `entities`, `history`. Every write goes through `db.write_transaction`, which holds the write lock from the first statement and nests as a savepoint, so a store method is atomic alone and inside a larger write. Vectors are scoped *inside* the KNN via vec0 metadata columns, not filtered afterwards. |
| `memory/recall/` | `semantic_index` routes a query to a candidate pool and returns `None`, never an empty pool, when it has nothing useful to say. `fusion` merges the three rankings by reciprocal rank. Rank-only, so a relevance threshold cannot live downstream of it. |
| `memory/knowledge/` | `extract` (the one entity extractor: cased words, acronyms, CamelCase, Latin and Cyrillic), `schema_classifier` (files entities into slots by keyword cue, once), `surprise` (drops episodics the facts already predict), `fact_ops` (the operation schema the model is constrained to), `consolidation` (applies them: supersede, never overwrite). |
| `providers/` | `openai_compat.py` (chat over the `openai` SDK), `embeddings.py` (`/embeddings` over httpx), `structured.py` (JSON-schema, JSON-object or prompted, validated, re-asked), `factory.py`, `wire.py` (`ChatClient`, `ProviderUnreachable`). Nothing above imports a model SDK. |
| `eval/retrieval.py` | Labelled probes, and the recall@k / MRR / leak-rate scorecard they produce. The measurement that turns retrieval quality from an assumption into a number. |
| `app/chatgpt_import.py` | Seeds memory from a ChatGPT export. Splits a turn too long for the embedding context, and routes a fifth of conversations to a holdout project the optimizer can never mine. |
| `app/chat.py` | One turn: recall → prompt → answer → remember both halves, attributed. The one use-case both surfaces share. |
| `surfaces/cli/` | `morgan`: `__main__` parses and dispatches, `commands` answers, `payloads` shapes the result, `render` prints it, `doctor` diagnoses the install. Project = the current git repository's directory name. |
| `surfaces/mcp_server.py` | `morgan-mcp`: `remember`, `recall`, `facts`, `forget`, `ask_morgan` over stdio or streamable-HTTP with a bearer token. Calls the CLI's command handlers; `project` is a tool argument. |
| `surfaces/network.py` | The bind guard: no listener beyond loopback without a real key. |

## Recall

1. The semantic index is asked for a candidate pool from the query's terms. `None` means
   search everything, and cross-project queries are never routed.
2. Vector, FTS5 and entity search each return their top-k *inside* that pool.
3. Reciprocal rank fusion merges the three rankings.
4. Currently-valid facts for the project are placed first, but budgeted: episodics keep half
   the window whenever they have hits, and the facts that survive a narrow budget are the
   ones the query mentions. Facts fill the whole window only when little else came back.

5. With `MORGAN_RECALL_FLOOR_MARGIN` set, recall returns nothing unless the best vector hit
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
rows, entity rows, vectors (`vec_items` + `vec_meta`), facts, the semantic index (nodes, edges,
schemas), session history. A memory being stored by another process is either entirely erased
or entirely kept, because storing is one transaction too. Tables that were never created on
this database are named in `tables_skipped` rather than counted as zero. Vacuum afterwards.

## Tests (`tests/`)

`unit/` per module; `integration/` runs the CLI as a subprocess, the MCP server over raw stdio
pipes and in-process, cross-process durability, two processes upserting the same vectors or
superseding the same facts at once, a vector delete racing a reinsert, a project erased while
a memory is being stored, two consolidation runs applying the same facts, erasure atomicity and
completeness, routing end to end, the wheel build. One live test (`pytest --live`) needs a real embedding model.
`pip install -e ".[dev]"` installs exactly what the suite needs. `tests/fakes.py` holds the
scripted chat client; nothing in the package exists only for tests.
