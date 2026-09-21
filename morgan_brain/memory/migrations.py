"""Bring a database written by an older Morgan up to what this one writes.

Some stored data is derived by code rather than given by the caller -- a memory's entities
are extracted from its content when it is stored. When that code changes, nothing already
stored changes with it, so each such change adds a numbered step here that re-derives what
is stored.

A step is light or heavy. A light step only adds a table or a defaulted column, and runs when
the database is opened (``upgrade``). A heavy step rewrites, moves or deletes rows, and runs
only under ``morgan migrate`` (``migrate``), behind a snapshot: a client that merely opened
the file has no snapshot behind it and no way back. Steps run strictly in order, so a light
step queued behind a heavy one waits for ``migrate`` with it; until then the database opens
read-only and every write raises ``DatabaseNeedsMigration``.

A database this code creates starts at the code's version (``stamp_if_new``): the stores
write the schema the last step leaves, so no step has anything to do in it.

The number of steps a database has been through is SQLite's own ``user_version`` header
field. It is read and advanced inside the same write transaction as the steps, so two
processes opening one file at once run each step once, and a step that fails leaves both
the data and the counter as they were, for the next open or ``morgan migrate`` to retry.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from itertools import takewhile
from typing import NamedTuple

from morgan_brain.memory.knowledge.extract import extract_entity_names
from morgan_brain.memory.store import projects, spaces
from morgan_brain.memory.store.db import write_transaction
from morgan_brain.memory.store.entities import EntityIndex
from morgan_brain.memory.store.episodic import EpisodicStore
from morgan_brain.memory.store.tables import PROJECT_TABLES
from morgan_brain.models import Entity


class Stores(NamedTuple):
    """The stores a step reads and writes, already open, so no step creates a table mid-upgrade."""

    episodics: EpisodicStore
    entities: EntityIndex


class Step(NamedTuple):
    """One numbered change to what is stored.

    *run* returns the rows it touched, per table, or ``None`` when it counts nothing;
    ``migrate`` reports them. *heavy* is true for any step that rewrites, moves or deletes
    rows -- those run only under ``morgan migrate``.
    """

    number: int
    name: str
    heavy: bool
    run: Callable[[sqlite3.Connection, Stores], dict[str, int] | None]


class DatabaseNeedsMigration(Exception):
    """A write reached a database whose next step is heavy and has not been run.

    *reason* names the pending steps and the command that runs them. It is the whole message,
    and both surfaces show it to the owner: the CLI as its error, an MCP client as the tool's.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _reextract_entities(conn: sqlite3.Connection, stores: Stores) -> None:
    """Re-extract every memory's entities under the current rule.

    Both copies are rewritten: the entity index recall searches, and the list stored on the
    memory itself, which is what a memory read back carries. Rewriting one would leave the two
    disagreeing about the same memory.
    """
    episodics, entities = stores
    for memory_id in episodics.ids():
        memory = episodics.get(memory_id)
        if memory is None:
            continue
        memory.entities = [Entity(name=n) for n in extract_entity_names(memory.content)]
        episodics.put(memory)
        entities.add(
            memory.id,
            [e.name for e in memory.entities],
            user_id=memory.user_id,
            project=memory.project,
        )


def _drop_the_semantic_index(conn: sqlite3.Connection, stores: Stores) -> None:
    """Drop the tables of the semantic upper index, which recall no longer has.

    Nothing reads, writes or erases them now, and the entity names in them are the owner's
    words: left in place they would outlive every ``forget``.
    """
    conn.execute("DROP TABLE IF EXISTS mem_entity_edges")
    conn.execute("DROP TABLE IF EXISTS mem_schema_edges")
    conn.execute("DROP TABLE IF EXISTS mem_entity_nodes")
    conn.execute("DROP TABLE IF EXISTS mem_schemas")


def _create_embedding_spaces_and_projects(conn: sqlite3.Connection, stores: Stores) -> None:
    """Add ``embedding_spaces`` and ``projects``. Light: two ``CREATE TABLE``s, no row moved.

    *stores* is unused -- neither table is part of the episodics/entities pair every step
    receives. Calls each store's ``create_schema`` directly, not its class -- the class also
    commits, and this runs inside ``upgrade``/``migrate``'s ``write_transaction``: a commit
    here would end that transaction early, so a later step's failure could no longer roll
    this one back with it. The same two functions run at open (via ``EmbeddingSpaceStore`` and
    ``ProjectStore``, which do commit, outside any transaction), so a fresh database and an
    old one upgraded through this step end up with identical schema either way.
    """
    spaces.create_schema(conn)
    projects.create_schema(conn)


#: In order. Step *n* brings a database from ``user_version`` *n - 1* to *n*; append only.
#: Steps 1 and 2 rewrite and drop, yet stay light: they predate the split, and every Morgan
#: that shipped them already ran them on open.
_STEPS: tuple[Step, ...] = (
    # A capital counts as a name only where its position does not explain it.
    Step(1, "reextract entities", False, _reextract_entities),
    Step(2, "drop the semantic index", False, _drop_the_semantic_index),
    Step(3, "create embedding_spaces and projects", False, _create_embedding_spaces_and_projects),
)


def _version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _advance(conn: sqlite3.Connection, step: Step) -> None:
    # PRAGMA takes no bound parameters; the value is an int a Step carries.
    conn.execute(f"PRAGMA user_version = {int(step.number)}")


def pending(conn: sqlite3.Connection, steps: Sequence[Step] | None = None) -> tuple[Step, ...]:
    """The steps the database behind *conn* has not been through, in order.

    *steps* defaults to ``_STEPS``, looked up when called rather than bound when defined, so
    opening, ``morgan migrate`` and ``restore``'s version check all read the one list.
    """
    done = _version(conn)
    return tuple(s for s in (_STEPS if steps is None else steps) if s.number > done)


def _holds_morgan_tables(conn: sqlite3.Connection) -> bool:
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    return not names.isdisjoint(PROJECT_TABLES)


def stamp_if_new(conn: sqlite3.Connection, steps: Sequence[Step] | None = None) -> bool:
    """Record a database this code is about to create as through every step; ``True`` if so.

    A file holding none of Morgan's tables has nothing a step could re-derive, and the stores
    about to open it create the schema the last step leaves. Started at ``user_version`` 0
    instead, its first heavy step would open it read-only on its first day and then fail on
    the columns its own ``CREATE TABLE`` had just made. ``user_version`` 0 alone does not say
    "new": a database written before step 1 existed reads 0 too, and holds tables. Call it
    before any store opens the connection.
    """
    if _holds_morgan_tables(conn):
        return False
    resolved = _STEPS if steps is None else steps
    with write_transaction(conn):
        # Again under the lock: another process may have created the tables since.
        if _holds_morgan_tables(conn):
            return False
        # PRAGMA takes no bound parameters; the value is a length this function took.
        conn.execute(f"PRAGMA user_version = {len(resolved)}")
    return True


def _light_prefix(steps: Sequence[Step]) -> tuple[Step, ...]:
    """The steps before the first heavy one: all that may run on open."""
    return tuple(takewhile(lambda s: not s.heavy, steps))


def upgrade(
    conn: sqlite3.Connection, stores: Stores, *, steps: Sequence[Step] | None = None
) -> None:
    """Run the consecutive pending light steps, stopping at the first heavy one.

    Nothing to run means no write lock: a database waiting for ``morgan migrate`` is opened
    by every client, and none of them should queue behind another process's write to find
    that out.
    """
    if not _light_prefix(pending(conn, steps)):
        return
    with write_transaction(conn):
        # Read again under the lock: another process may have upgraded since the check above.
        for step in _light_prefix(pending(conn, steps)):
            step.run(conn, stores)
            _advance(conn, step)


def migrate(
    conn: sqlite3.Connection, stores: Stores, *, steps: Sequence[Step] | None = None
) -> list[tuple[Step, dict[str, int]]]:
    """Run every pending step, light and heavy, in one write transaction.

    ``user_version`` advances after each step, inside the transaction, so a step that raises
    rolls back the whole wave: the data and the counter stay where they were before this call.
    Returns each step run with the rows it reported touching (``{}`` when it counts nothing).
    The caller takes the snapshot first -- ``VACUUM INTO`` cannot run inside a transaction.
    """
    applied: list[tuple[Step, dict[str, int]]] = []
    with write_transaction(conn):
        # Read under the lock, for the same reason as ``upgrade``.
        for step in pending(conn, steps):
            counts = step.run(conn, stores)
            _advance(conn, step)
            applied.append((step, dict(counts or {})))
    return applied
