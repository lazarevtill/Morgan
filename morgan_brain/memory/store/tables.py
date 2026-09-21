"""The one registry of project-keyed tables. `forget()` and `distinct_projects()` both read it,
so a table a store adds is reachable by both the moment it is registered here.

``embedding_spaces`` does not exist yet -- Task 9 creates it, and Task 16 gives each embedding
space its own ``vec0`` table named there. That is why the registry is a function over the
connection rather than a plain constant: a caller that read ``PROJECT_TABLES`` alone would stop
reaching a space's table the day it stopped being the only one.
"""

from __future__ import annotations

import sqlite3

#: The statically known tables that carry a `project` column. `vec_items` is also the active
#: embedding space's table once `embedding_spaces` exists -- `project_tables()` below dedupes it
#: rather than listing it twice.
PROJECT_TABLES: tuple[str, ...] = (
    "memories",
    "facts",
    "memory_entities",
    "vec_meta",
    "vec_items",
    "fts_memories",
    "session_history",
)


#: Tables that hold one project's own data but are keyed by `name`, not `project` --
#: `projects` itself, whose primary key *is* the project's name. `forget()` deletes the
#: project's own row from each of these by name, alongside the id-based deletes it runs
#: against `project_tables()`. Kept as its own registry rather than folded into
#: `PROJECT_TABLES`: a caller that walks `project_tables()` to build an id-scoped `WHERE
#: project = ?` delete (`forget`'s own id-based statements, `EpisodicStore.distinct_projects`)
#: would run that same statement against `projects` and find no `project` column at all.
NAME_KEYED_PROJECT_TABLES: tuple[str, ...] = ("projects",)


def project_tables(conn: sqlite3.Connection) -> tuple[str, ...]:
    """`PROJECT_TABLES` plus every `embedding_spaces.table_name` row, once that table exists.

    Order is preserved -- `PROJECT_TABLES` first, then each space's table in the order
    `embedding_spaces` returns them -- and a name already present (`vec_items`, the active
    space, is both static and registered) is not repeated.
    """
    tables = list(PROJECT_TABLES)
    has_embedding_spaces = (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'embedding_spaces'"
        ).fetchone()
        is not None
    )
    if has_embedding_spaces:
        for row in conn.execute("SELECT table_name FROM embedding_spaces"):
            name = row["table_name"]
            if name not in tables:
                tables.append(name)
    return tuple(tables)
