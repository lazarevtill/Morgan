# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What Morgan is

A **project-scoped memory for the owner's AI tools**, consolidated into facts by a local
model. One SQLite database under `MORGAN_DATA_DIR` holds memories, facts, vectors
(sqlite-vec), the FTS5 index, the entity index, the semantic upper index and session
history. Two surfaces, the `morgan` CLI and the `morgan-mcp` server, are thin adapters over
one `MemoryGate`. The model server is any OpenAI-compatible endpoint (llama-server by
default). One process, no queue, no worker, no scheduler.

Read first: `docs/ARCHITECTURE.md` (the package), `docs/WIRING.md` (running it),
`docs/ROADMAP.md` (what was cut and why). The archived kernel this was cut from is at the
tag `legacy-v0.1.0-kernel` with its designs under `docs/archive/`.

## Package map (`morgan_brain/`, ~4,700 lines)

The tree is grouped by what a file does, so "where does a write go" and "where does a request
come in" are answered by the directory names.

- `config.py` — the single `MORGAN_`-prefixed settings source (`get_settings()`). Reads
  `~/.config/morgan/.env`, then `./.env`, then the environment. The database defaults to
  `~/.local/share/morgan/`. `MORGAN_EMBEDDING_ENDPOINT` addresses embeddings separately when
  the chat server does not serve them.
- `models.py` — the domain models. Everything that persists is `user_id`- and
  `project`-keyed. `Memory` carries a `MemorySource`; `TemporalFact` carries
  `valid_from`/`valid_to`/`superseded_by`.
- `logging_setup.py` — process output. stdout is UTF-8 because the protocols on it are; logs
  go to stderr. One call per entrypoint.
- `composition.py` — opens the database and wires the above. `build_memory_context` needs no
  chat model; `build_app_context` adds it.
- `memory/` — the core. `gate.py` is the only door and `module.py` is the one write path and
  the fused recall; `embedder.py` is the embedding seam. Below them:
  - `store/` — persistence only: `db`, `episodic`, `temporal`, `vectors`, `fts`, `entities`,
    `history`. Each owns its schema and its queries; none of them ranks anything.
  - `recall/` — `semantic_index` (routing, which may cost precision but never recall) and
    `fusion` (reciprocal rank, rank-only).
  - `knowledge/` — `extract`, `schema_classifier`, `surprise`, `fact_ops`, `consolidation`.
    The work that costs a model call or a full pass, and never runs inside a recall.
- `eval/` — measuring what recall returns: labelled probes, recall@k, MRR and leak rate,
  scored per probe kind. Run with `pytest --live` against a real embedding endpoint.
- `providers/` — the only place a model SDK is imported: `openai_compat.py` (chat),
  `embeddings.py`, `structured.py` (JSON-validated output), `factory.py` (settings → adapters),
  `wire.py` (message types, `ChatClient`, `ProviderUnreachable`).
- `app/chat.py` — one turn: recall, answer, remember. Not a surface: the one use-case both
  surfaces call.
- `surfaces/` — where requests come in. `cli/` (`__main__` parses and dispatches, `commands`
  answers, `payloads` shapes the result, `render` prints it, `doctor` diagnoses),
  `mcp_server.py` (five MCP tools over stdio or streamable-HTTP, calling those same command
  handlers), and `network.py`, the bind guard that protects the HTTP one.

## Invariants

- **All memory access goes through `MemoryGate`.** No caller holds the `MemoryModule`.
- **Every read and write is project-scoped.** `Memory` and `TemporalFact` carry a required
  `project`; the gate rejects an empty one. `all_projects=True` is the explicit cross-project
  escape hatch, never the default.
- **One write path.** `MemoryModule.store` writes every index at once: episodic row, vector,
  FTS5, entity index, semantic upper index. Entities are extracted there when the caller gave
  none. A memory visible to one signal and not another is the failure routing turns into lost
  recall.
- **Routing never costs recall.** `SemanticIndex.route()` returns `None` ("search
  everything") whenever it has nothing useful to say, never an empty pool. The pool is pushed
  into each signal's query, never applied to its output. Cross-project recall is never routed.
- **Facts evolve, they don't overwrite.** Update = close the old interval, open a new one.
- **Actor attribution.** Every memory records its `MemorySource`. The reply to `ask` is
  stored as `agent_inferred`; never treat an inference as a user statement.
- **A model server that is down is reported by name.** Adapters raise
  `providers.wire.ProviderUnreachable` carrying the endpoint; the CLI and MCP tools print its
  message. A bare traceback is a regression.
- **stdout is a protocol on both surfaces.** `--json` output and the MCP stdio transport are
  parsed by machines; all logs go to stderr. Never `print()` diagnostics from library code.
- **Nothing runs a model unasked.** `ask` and `consolidate` call the model; nothing else does,
  and nothing runs on a schedule. Consolidation is on demand (or the owner's own cron).
- **No listener beyond loopback without a key.** `network.assert_safe_bind` refuses to start
  `morgan-mcp --transport http` on a non-loopback host while `MORGAN_API_KEY` is unset or the
  placeholder.
- **One of each.** One settings object, one database, one gate, one entity extractor, one
  `SemanticIndex` per assembly, one logging configuration.
- **Never hardcode the owner.** Everything is keyed by `user_id`; single-owner is a config fact.

## Known limitations

- `recall` has no relevance floor: once a project holds anything, a query returns something.
- Schema classification for the upper index is keyword-based and an entity is classified once.
- Entity extraction is deterministic and cased-script only; scripts without letter case
  (Chinese, Japanese, Arabic, Hebrew) yield nothing rather than a guess.
- Recall returns a superseded memory alongside the current one: every knowledge-update
  probe leaks, and the current answer is not reliably first. Supersession lives on facts;
  the probes store episodics, which carry none.
- Multi-hop questions are not answered. Recall ranks memories and has no mechanism to
  compose two of them; measured recall@8 is 0.50 against 1.00 for single-hop.
- Surprise gating tokenises with `[a-z0-9]+`, so Cyrillic episodics yield an empty token
  set and are dropped before consolidation rather than scored.

## Build, test, run

```bash
pip install -e ".[dev]"
mkdir -p ~/.config/morgan && cp .env.example ~/.config/morgan/.env   # MORGAN_LLM_ENDPOINT
morgan doctor
pytest -q                     # 258 passed, 2 skipped (the live ones)
ruff check . && ruff format --check . && mypy morgan_brain && bandit -c pyproject.toml -r morgan_brain
```

Python 3.12+, line length 100, `ruff` is the linter and formatter, `mypy --strict`. Keep
`main` green. Root-cause fixes only; a workaround gets flagged, not silently applied.
