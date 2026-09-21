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
`NAME_KEYED_PROJECT_TABLES`. Each table is erased by a deleter its store owns, a function in the
store's module found by the table's name in `_DELETERS` in `memory/module.py`. Each deleter is
handed one `Erasure` (`store/tables.py`): the owner, the project, the ids of the project's
memories, and those memories' rowids in `vec_meta`. An embedding space's vec0 table is erased by
`vectors.vector_deleter` for its name. A memory's vector has the rowid `vec_meta` gives it in
every space's table, the rowid `vectors.stored_sample` and `vectors.audit_sample` read any
space's table by. Every table is resolved before any row is deleted. A registered table that
exists and has no deleter makes `forget()` raise, naming it, with nothing erased. A table in
`project_tables(conn)` that the database does not have is named in
`ForgetReport.tables_skipped`. The ids and rowids are selected, and every table is erased, in
one write transaction; the database is vacuumed once it commits.

A store that adds a table therefore registers it in `tables.py` and maps its deleter in
`_DELETERS`. `tests/unit/memory/test_forget_reaches_every_project_keyed_table.py` writes rows
for two projects into every registered table through the stores' own write paths, forgets one,
and checks each table the registry names: the forgotten project's rows are gone and the other's
are unchanged. It checks a second embedding space's table the same way. It also checks that a
registered table with no deleter stops `forget()` before it erases anything.

**Later.** Erasing by session or by date, once `forget` can, cascades through the same
registry.
