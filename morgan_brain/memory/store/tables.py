"""The one registry of project-keyed tables. `forget()` and `distinct_projects()` both read it,
so a table a store adds is reachable by both the moment it is registered here.

Each embedding space names its own vec0 table in ``embedding_spaces.table_name``. That is why
the registry is a function over the connection rather than a plain constant: a caller that read
``PROJECT_TABLES`` alone would stop reaching a space's table the day it stopped being the only
one.

`forget()` erases each registered table through a deleter owned by the table's store: a
`Deleter`, handed the `Erasure` it carries out. A registered table with no deleter stops
`forget()` by name before it erases anything.

Two grains: a project-grain erasure names an owner and a project; a session-grain one names
sessions, and every registered table's store either erases its rows of those sessions or
declares that it holds nothing per session.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

#: The two grains an erasure runs at.
Grain = Literal["project", "sessions"]

#: The statically known tables that carry a `project` column. `vec_items` is also the active
#: embedding space's table once `embedding_spaces` exists -- `project_tables()` below dedupes it
#: rather than listing it twice. The memory tables come first, then the session archive's
#: tables, in the order every erasure walks them: the archive's index tables (`link_ratings`,
#: `turn_links`, the two FTS mirrors) before the rows they index (`turns`), `turns` before
#: `sessions`, and the archive's own tables last.
PROJECT_TABLES: tuple[str, ...] = (
    "memories",
    "facts",
    "memory_entities",
    "vec_meta",
    "vec_items",
    "fts_memories",
    "session_history",
    "link_ratings",
    "turn_links",
    "corrections_fts",
    "turns_fts",
    "turns",
    "sessions",
    "capture_pauses",
    "digests",
    "call_log",
)

#: A registered table another table answers for: `distinct_projects` and `_holds_rows_of` skip
#: the FTS tables, whose rows are the `turns` rows' by rowid, because a filter on an FTS5
#: table's UNINDEXED column is a full scan inside the lock. The deleters still erase them, by
#: rowid.
ANSWERED_BY: Mapping[str, str] = {"turns_fts": "turns", "corrections_fts": "turns"}


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
    """What one erasure -- `forget()` or `forget_sessions()` -- erases, selected under its
    write lock before any row is deleted.

    At the project grain, a table with a ``user_id`` and a ``project`` column is erased by
    those columns as well as by these ids, so an index row whose memory is gone goes too; no
    deleter matches a ``project`` column without its ``user_id``. `turns_fts` is the exception:
    it is erased only at the turns' rowids, because a filter on an FTS5 table's UNINDEXED
    column is a full scan inside the lock. Every list is a JSON array, bound as one parameter
    and expanded by ``json_each``, so no deleter binds a parameter per id, and a project with
    more memories than ``SQLITE_MAX_VARIABLE_NUMBER`` is still erased by one statement per
    table.

    At the session grain ``memory_ids``, ``vector_rowids`` and ``fact_ids`` are empty arrays;
    ``session_ids``, ``native_ids`` and ``turn_ids`` name the sessions erased and their turns.
    At the project grain every list is filled from the project's own rows.
    """

    #: The owner whose rows are erased; no other owner's row is touched.
    user_id: str
    project: str
    #: The ids of every memory *user_id* stored under *project*.
    memory_ids: str
    #: The rowids of the ``vec_meta`` rows erased with them, where ``upsert`` wrote their
    #: vectors in ``vec_items``. Selected before ``vec_meta`` loses its rows, so ``vec_items``
    #: does not depend on the order the tables are erased in. No other space's table is erased
    #: by rowid.
    vector_rowids: str
    #: Which grain this erasure runs at.
    grain: Grain = "project"
    #: The ``sessions.id``s erased whole, and their harnesses' native ids, as JSON arrays.
    session_ids: str = "[]"
    native_ids: str = "[]"
    #: The ``turns.id``s erased, and the ``facts.id``s, as JSON arrays.
    turn_ids: str = "[]"
    fact_ids: str = "[]"
    #: Where a partial erasure would cut off which turns to keep. No caller sets it: every
    #: erasure erases named sessions whole or a whole project, so this stays ``None``.
    erased_since: str | None = None
    #: The reason written to ``capture_exclusions`` for every session erased whole.
    exclusion_reason: str = "forget"
    #: When this erasure ran, ``utc_iso`` of the module's clock -- stamped on every exclusion
    #: it writes, never the row's own ``updated_at``, so an exclusion always says when it was
    #: made rather than when its session last changed.
    erased_at: str = ""


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
