"""The one registry of project-keyed tables. `forget()` and `distinct_projects()` both read it,
so a table a store adds is reachable by both the moment it is registered here.

Each embedding space names its own vec0 table in ``embedding_spaces.table_name``. That is why
the registry is a function over the connection rather than a plain constant: a caller that read
``PROJECT_TABLES`` alone would stop reaching a space's table the day it stopped being the only
one.

`forget()` erases each registered table through a deleter owned by the table's store: a
`Deleter`, handed the `Erasure` it carries out. A registered table with no deleter stops
`forget()` by name before it erases anything.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

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
#: project's own row from each of these by name, alongside the tables `project_tables()`
#: names. Kept as its own registry rather than folded into `PROJECT_TABLES`: a caller that
#: walks `project_tables()` to build a `WHERE project = ?` statement
#: (`EpisodicStore.distinct_projects`, migration step 5) would run it against `projects` and
#: find no `project` column at all.
NAME_KEYED_PROJECT_TABLES: tuple[str, ...] = ("projects",)


@dataclass(frozen=True)
class Erasure:
    """What one `forget()` erases, selected under its write lock before any row is deleted.

    Both id lists are JSON arrays, bound as one parameter and expanded by ``json_each``, so
    no deleter binds a parameter per id, and a project with more memories than
    ``SQLITE_MAX_VARIABLE_NUMBER`` is still erased by one statement per table.
    """

    user_id: str
    project: str
    #: The ids of every memory *user_id* stored under *project*.
    memory_ids: str
    #: Those memories' rowids in ``vec_meta``: the rowid each one's vector has in every
    #: embedding space's vec0 table. Selected before ``vec_meta`` loses its rows, so no vec0
    #: table depends on the order the tables are erased in.
    vector_rowids: str


#: A store's delete for one of its tables: erases the `Erasure` from that table inside
#: `forget()`'s write transaction, commits nothing itself, and returns the rows it deleted.
Deleter = Callable[[sqlite3.Connection, Erasure], int]


def space_tables(conn: sqlite3.Connection) -> tuple[str, ...]:
    """Every embedding space's vec0 table, as ``embedding_spaces`` names it; none before that
    table exists."""
    has_embedding_spaces = (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'embedding_spaces'"
        ).fetchone()
        is not None
    )
    if not has_embedding_spaces:
        return ()
    return tuple(
        str(r["table_name"]) for r in conn.execute("SELECT table_name FROM embedding_spaces")
    )


def project_tables(conn: sqlite3.Connection) -> tuple[str, ...]:
    """`PROJECT_TABLES` plus every `embedding_spaces.table_name` row, once that table exists.

    Order is preserved -- `PROJECT_TABLES` first, then each space's table in the order
    `embedding_spaces` returns them -- and a name already present (`vec_items`, the active
    space, is both static and registered) is not repeated.
    """
    tables = list(PROJECT_TABLES)
    for name in space_tables(conn):
        if name not in tables:
            tables.append(name)
    return tuple(tables)
