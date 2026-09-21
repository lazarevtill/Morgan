# 0001: The fact key, and how far `forget` reaches

**Status:** accepted, 2026-09-22. Decision 1 is built in phase 2; decision 2 holds now.

## Decision 1: the fact key includes the author and the scope

A fact's key becomes (`user_id`, `project`, `subject`, `predicate`, `author_id`, `scope`), and
a key has at most one current fact: the one with `valid_to IS NULL`.

**Now.** The key is (`user_id`, `project`, `subject`, `predicate`). The unique partial index
`idx_facts_one_current` in `memory/store/temporal.py` enforces one current fact per key, and
`SqliteTemporalStore.upsert_fact` closes the current fact of the same key before it inserts the
new one. `facts` carries `author_id` and `scope` columns (migration step 4), and neither the
index nor the upsert reads them.

**When.** The index and the upsert change together in phase 2, with R1, when supersession is
assembled deterministically.

**Why not now.** Nothing writes a second author or a second scope. Consolidation is the only
writer of facts, and every fact it writes carries `author_id` equal to the owner's `user_id`
and `scope` `private`; step 4 gave existing facts the same values. With one owner, one author
and one scope, the four-column key and the six-column key select the same rows. Changing the
index is a heavy migration step, and this phase gains nothing from it.

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

**What the registry test does not check.** `forget()` erases the tables with a `project` column
by one statement each, written out in `MemoryModule.forget`; `project_tables` decides only
which of them it reports as absent. (It deletes the project's row from each table in
`NAME_KEYED_PROJECT_TABLES` by walking that list.)
`tests/unit/memory/test_forget.py::test_forget_empties_every_underlying_table` checks each
table against a list of its own. A store that adds a table therefore also adds its delete to
`forget()` and adds the table to that test's list. A second embedding space's vec0 table, once
phase 2 creates one, is in `project_tables(conn)` but has no delete in `forget()`: `forget()`
deletes vectors from `vec_items` only.

**Later.** Phase 1a's `forget --session` and `forget --since` cascade through the same registry.
