# 0001: The fact key, and how far `forget` reaches

**Status:** accepted, 2026-09-22. Decision 1 is built when supersession is assembled
deterministically; decision 2 holds now.

## Decision 1: the fact key includes the author and the scope

A fact's key becomes (`user_id`, `project`, `subject`, `predicate`, `author_id`, `scope`), and
a key has at most one current fact: the one with `valid_to IS NULL`.

**Now.** The key is (`user_id`, `project`, `subject`, `predicate`). The unique partial index
`idx_facts_one_current` in `memory/store/temporal.py` enforces one current fact per key, and
`SqliteTemporalStore.upsert_fact` closes the current fact of the same key before it inserts the
new one. `facts` carries `author_id` and `scope` columns (migration step 4), and neither the
index nor the upsert reads them.

**When.** The index and the upsert change together when supersession is assembled
deterministically.

**Why not now.** Nothing writes a second author or a second scope. Consolidation is the only
writer of facts, and every fact it writes carries `author_id` equal to the owner's `user_id`
and `scope` `private`; step 4 gave existing facts the same values. With one owner, one author
and one scope, the four-column key and the six-column key select the same rows. Changing the
index is a heavy migration step, and until something writes a second author or scope it
changes nothing.

**Until then.** A writer that stores a second author's or a second scope's fact under an existing
key closes the other one's fact instead of keeping both. The first writer that sets another
author or scope depends on this change.

## Decision 2: `forget` reaches every project-keyed table through one registry

`memory/store/tables.py` is the one list of tables that hold a project's data:

- `PROJECT_TABLES`: the tables with a `project` column. `project_tables(conn)` adds each
  embedding space's vec0 table named in `embedding_spaces`.
- `NAME_KEYED_PROJECT_TABLES`: the tables keyed by the project's own name. Today that is
  `projects`.

`MemoryModule.forget` and `EpisodicStore.distinct_projects` read the registry. A store that adds
a table registers it there. `tests/unit/memory/test_forget_reaches_every_project_keyed_table.py`
opens a database with every store, walks `sqlite_master`, and fails on any table with a
`project` column that the registry does not list. It also fails on any registered table that
does not exist.

**A real `project` column is required.** A table that keeps its project inside a JSON column
has no `project` column for that test to find, so `forget` would never reach it. Every table
that holds a project's data has a `project` column, or is keyed by the project's name and is
listed in `NAME_KEYED_PROJECT_TABLES`.

**How `forget` erases a registered table.** `forget()` walks `project_tables(conn)` and
`NAME_KEYED_PROJECT_TABLES`. Each table is erased by a deleter its store owns: a function in the
store's module, found by the table's name in `_DELETERS` in `memory/module.py`. Each deleter is
handed one `Erasure` (`store/tables.py`):

- the owner and the project;
- the ids of the project's memories;
- the rowids of the `vec_meta` rows erased with them, which is where
  `SqliteVectorIndex.upsert` wrote their vectors in `vec_items`.

The deleters erase as follows:

- A table with a `user_id` and a `project` column is erased by those two columns as well as by
  the ids, so an index row whose memory is gone goes too. No deleter matches a `project`
  column without its `user_id`, so another owner's rows are never touched.
- `vec_items` is erased at those rowids and by its own `user_id` and `project` columns.
- Another embedding space's vec0 table, named in `embedding_spaces`, is erased by its own
  `user_id` and `project` columns alone (`vectors.space_deleter`). No writer puts its vectors
  at `vec_meta`'s rowids, so a rowid there says nothing about whose row it is.
- The `projects` row is erased by its name, once no registered project-keyed table holds a
  row of the project from any owner. It has no owner of its own: its remote, root and
  switches belong to everyone with data in the project. So one owner's `forget` keeps it
  while another's rows remain, and the last one's removes it. The name-keyed tables are
  erased after every project-keyed one, so the check sees what the erasure left.
- The FTS5 deleter then runs `optimize` on `fts_memories`. FTS5 answers a DELETE with a
  tombstone and keeps the row's words in its segment b-tree until a merge; `optimize` merges
  now, so the words leave `fts_memories_data`.

Every table is resolved before any row is deleted. `forget()` raises, naming the table, with
nothing erased, when a registered table exists and has no deleter, or when a space's table has
no `user_id` and `project` columns. A table in `project_tables(conn)` that the database does not
have is named in `ForgetReport.tables_skipped`. The ids and rowids are selected, and every
table is erased, in one write transaction. Once it commits, the database is vacuumed and the
write-ahead log truncated (`PRAGMA wal_checkpoint(TRUNCATE)`). A connection in the middle of a
read blocks that checkpoint; the forgotten words then stay in the database file and its log
until a later checkpoint completes, and `forget()` logs the warning `forget.wal-not-truncated`,
naming the log, and still succeeds.

A store that adds a table therefore registers it in `tables.py` and maps its deleter in
`_DELETERS`. The tests in `tests/unit/memory/test_forget_reaches_every_project_keyed_table.py`
cover the following:

- **Every registered table.** It writes rows for two projects into every registered table
  through the stores' own write paths and forgets one. Then it checks each table the registry
  names: the forgotten project's rows are gone and the other's are unchanged.
- **A second embedding space.** It registers the space and creates its vec0 table directly,
  since no code writes a second space yet. It writes that table's rows at rowids that differ
  from `vec_meta`'s, and checks that only the forgotten project's vector goes.
- **Orphans.** It checks that an index row whose memory or `vec_meta` row is gone is erased,
  and that another owner's rows in the same project are kept.
- **The `projects` row.** With two owners in a project, the first owner's `forget` keeps the
  row and the second's removes it. A single owner's `forget` removes it, with every registered
  table holding that owner's rows beforehand.
- **Refusals.** It checks that each refusal above stops `forget()` with every row of the
  project still present.

`tests/unit/memory/test_forget_leaves_no_trace.py` reads the raw bytes of the database file
and its `-wal` file after `forget()`, alone and beside a second open connection. It checks that
a word of the forgotten project is in neither.

**Later.** Erasing by session or by date, once `forget` can, cascades through the same
registry.
