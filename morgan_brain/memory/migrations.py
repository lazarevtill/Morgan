"""Bring a database written by an older Morgan up to what this one writes, once, on open.

Some stored data is derived by code rather than given by the caller -- a memory's entities
are extracted from its content when it is stored. When that code changes, nothing already
stored changes with it, so each such change adds a step here that re-derives what is stored.

The number of steps a database has been through is SQLite's own ``user_version`` header
field. It is read and advanced inside the same write transaction as the steps, so two
processes opening one file at once run each step once, and a step that fails leaves both
the data and the counter as they were for the next open to retry.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import NamedTuple

from morgan_brain.memory.knowledge.extract import extract_entity_names
from morgan_brain.memory.store.db import write_transaction
from morgan_brain.memory.store.entities import EntityIndex
from morgan_brain.memory.store.episodic import EpisodicStore
from morgan_brain.models import Entity


class Stores(NamedTuple):
    """The stores a step reads and writes, already open, so no step creates a table mid-upgrade."""

    episodics: EpisodicStore
    entities: EntityIndex


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


#: In order. Step *n* brings a database from ``user_version`` *n - 1* to *n*; append only.
_STEPS: tuple[Callable[[sqlite3.Connection, Stores], None], ...] = (
    # A capital counts as a name only where its position does not explain it.
    _reextract_entities,
    _drop_the_semantic_index,
)


def _version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def upgrade(conn: sqlite3.Connection, stores: Stores) -> None:
    """Run every step the database behind *conn* has not been through."""
    if _version(conn) >= len(_STEPS):
        return
    with write_transaction(conn):
        # Read again under the lock: another process may have upgraded since the check above.
        done = _version(conn)
        for number, step in enumerate(_STEPS[done:], start=done + 1):
            step(conn, stores)
            # PRAGMA takes no bound parameters; the value is an int this function counted.
            conn.execute(f"PRAGMA user_version = {int(number)}")
