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
| `config.py` | The single `MORGAN_`-prefixed settings source. Reads `~/.config/morgan/.env`, then `./.env`, then the environment; the database defaults to `~/.local/share/morgan/`. |
| `models.py` | `Memory`, `TemporalFact`, `MemoryQuery`, `Message`. Everything that persists is `user_id`- and `project`-keyed. |
| `memory/gate.py` | `MemoryGate`: the only door to memory. Refuses an empty user or project. `ForgetReport`. |
| `memory/module.py` | `MemoryModule`: the one write path (every index in one call, entities extracted if absent) and the fused recall (three signals, reciprocal rank fusion, current facts first). `forget()` in one transaction. |
| `memory/episodic.py`, `temporal.py`, `vectors.py`, `fts.py`, `entities.py`, `history.py` | The stores, all over the one connection. Vectors are scoped *inside* the KNN via vec0 metadata columns, not filtered afterwards. |
| `memory/semantic_index.py` | The upper index: schemas route coarsely, entities locate concretely, one-hop co-occurrence expands. `route()` returns `None`, never an empty pool, when it has nothing useful to say. |
| `memory/schema_classifier.py` | Files entities into schema slots (keyword cues, deterministic) and records co-occurrence. An entity is classified once. |
| `memory/extract.py` | The one entity extractor: cased words, acronyms, CamelCase, Latin and Cyrillic. |
| `memory/consolidation.py` | Episodics → facts. Asks the model for ADD/UPDATE/DELETE/NOOP operations as validated JSON, applies them through the gate: supersede, never overwrite. Skips episodics current facts already cover. |
| `providers/` | `openai_compat.py` (chat over the `openai` SDK), `embeddings.py` (`/embeddings` over httpx), `structured.py` (JSON-schema, JSON-object or prompted, validated, re-asked), `factory.py`, `wire.py` (`ChatClient`, `ProviderUnreachable`). Nothing above imports a model SDK. |
| `chat.py` | One turn: recall → prompt → answer → remember both halves, attributed. |
| `composition.py` | Opens the database and wires everything. `build_memory_context` for the memory commands; `build_app_context` adds the chat client. |
| `cli/` | `morgan` with `remember`, `recall`, `facts`, `forget`, `ask`, `consolidate`, `doctor`. Project = the current git repository's directory name. |
| `mcp_server.py` | `morgan-mcp`: `remember`, `recall`, `facts`, `forget`, `ask_morgan` over stdio or streamable-HTTP with a bearer token. Calls the CLI's handlers; `project` is a tool argument. |
| `network.py` | The bind guard. |
| `logging_setup.py` | All logs to stderr; stdout is `--json` and JSON-RPC. |

## Recall

1. The semantic index is asked for a candidate pool from the query's terms. `None` means
   search everything, and cross-project queries are never routed.
2. Vector, FTS5 and entity search each return their top-k *inside* that pool.
3. Reciprocal rank fusion merges the three rankings.
4. Currently-valid facts for the project are placed first; the fused episodics follow, cut
   to `top_k`.

There is no relevance floor: a non-empty project always answers.

## Consolidation (`morgan consolidate`)

Recent episodics minus those current facts already cover → the model proposes fact operations
as JSON validated against `FactOpBatch` → applied through the gate. UPDATE closes the old
interval and opens a new one (`valid_to`, `superseded_by`); DELETE closes it; confidence decays
with age since last confirmation. It runs when asked, never on a schedule of its own.

## Erasure (`morgan forget`)

One `BEGIN IMMEDIATE`, then: memories, FTS rows, entity rows, vectors (`vec_items` +
`vec_meta`), facts, the semantic index (nodes, edges, schemas), session history. Tables that
were never created on this database are named in `tables_skipped` rather than counted as
zero. Vacuum afterwards.

## Tests (`tests/`)

`unit/` per module; `integration/` runs the CLI as a subprocess, the MCP server over raw stdio
pipes and in-process, cross-process durability, erasure atomicity and completeness, routing end
to end, the wheel build. One live test (`pytest --live`) needs a real embedding model.
`pip install -e ".[dev]"` installs exactly what the suite needs. `tests/fakes.py` holds the
scripted chat client; nothing in the package exists only for tests.
