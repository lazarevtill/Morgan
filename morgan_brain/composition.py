"""Composition root: open the one database and wire the core over it.

``build_memory_context`` is enough for every memory operation (``remember``, ``recall``,
``facts``, ``forget``, ``doctor``) and needs no chat model. ``build_app_context`` adds the
chat client for ``ask`` and ``consolidate``. Both share one connection: every store below
lives in the same SQLite file, which is what makes ``forget()`` one transaction.
"""

from __future__ import annotations

import pathlib
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import structlog

from morgan_brain.app.chat import Chat
from morgan_brain.config import Settings
from morgan_brain.memory.embedder import Embedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.knowledge.consolidation import MemoryConsolidator
from morgan_brain.memory.migrations import Step, Stores, pending, stamp_if_new, upgrade
from morgan_brain.memory.module import MemoryModule
from morgan_brain.memory.store import spaces, vectors
from morgan_brain.memory.store.db import open_db, write_transaction
from morgan_brain.memory.store.entities import EntityIndex
from morgan_brain.memory.store.episodic import EpisodicStore
from morgan_brain.memory.store.fts import FtsIndex
from morgan_brain.memory.store.history import SessionHistoryStore
from morgan_brain.memory.store.projects import ProjectStore
from morgan_brain.memory.store.spaces import EmbeddingSpaceStore
from morgan_brain.memory.store.temporal import SqliteTemporalStore
from morgan_brain.memory.store.vectors import SqliteVectorIndex
from morgan_brain.providers.factory import Budget, build_chat_client, build_embedder
from morgan_brain.providers.openai_compat import OpenAICompatAdapter

log = structlog.get_logger("composition")

#: The vec0 table ``store/vectors.py`` keeps every vector in: the table of the one space a
#: database has before any second one exists.
_VECTOR_TABLE = "vec_items"


def utcnow() -> datetime:
    return datetime.now(UTC)


def sqlite_path(url: str) -> str:
    """Turn a sqlite:/// URL into a filesystem path; pass through ':memory:'."""
    return url.removeprefix("sqlite:///")


@dataclass
class MemoryContext:
    """Handles for callers that only read and write memory -- no chat model involved."""

    gate: MemoryGate
    conn: sqlite3.Connection
    history: SessionHistoryStore
    embedder: Embedder
    settings: Settings


@dataclass
class AppContext(MemoryContext):
    """Everything in ``MemoryContext`` plus the chat model: ``ask`` and ``consolidate``."""

    chat: Chat
    consolidator: MemoryConsolidator
    client: OpenAICompatAdapter


def migration_stores(conn: sqlite3.Connection) -> Stores:
    """The stores a migration step reads and writes, opened over *conn*.

    One place builds them, for the light steps an open runs and for ``morgan migrate``.
    """
    return Stores(episodics=EpisodicStore(conn), entities=EntityIndex(conn))


def build_memory_module(
    conn: sqlite3.Connection,
    *,
    embedder: Embedder,
    dim: int,
    clock: Any = utcnow,
    floor_margin: float | None = None,
) -> MemoryModule:
    """Every store over one connection, with the pending light migration steps run.

    A database with none of Morgan's tables is stamped at the code's version first, before
    any store creates one: this code writes it at the latest schema. A heavy step is left for
    ``morgan migrate``; ``build_memory_context`` then opens the gate read-only. Also the seam
    tests use with a small fake embedder.
    """
    stamp_if_new(conn)
    stores = migration_stores(conn)
    EmbeddingSpaceStore(conn)
    ProjectStore(conn)
    module = MemoryModule(
        embedder=embedder,
        vectors=SqliteVectorIndex(conn, dim=dim),
        temporal=SqliteTemporalStore(conn=conn),
        clock=clock,
        fts=FtsIndex(conn),
        entities=stores.entities,
        episodics=stores.episodics,
        floor_margin=floor_margin,
    )
    upgrade(conn, stores)
    return module


def _read_only_reason(remaining: Sequence[Step]) -> str | None:
    """What every write says while steps are left after the light ones ran, or ``None``.

    Steps are left only when the next one is heavy; each is named, because ``morgan migrate``
    runs all of them, light ones queued behind a heavy step included.
    """
    if not remaining:
        return None
    count = f"{len(remaining)} step{'' if len(remaining) == 1 else 's'} pending"
    names = ", ".join(f"{s.number} {s.name}" for s in remaining)
    return f"writes are blocked until `morgan migrate` runs: {count} ({names})"


def register_the_settings_space(conn: sqlite3.Connection, settings: Settings) -> None:
    """Record the settings' model and width as the active space of a database that has none.

    A database write, no embedding: the fingerprint is left ``NULL`` for the first embedding
    call to record, once a sample of the stored vectors shows the model wrote them. The check
    is read again under the write lock, because another process may open the same fresh file
    at once, and a second active space fails on the partial unique index.

    The width recorded must be the vector table's own. The table keeps the width it was created
    at -- ``morgan doctor`` builds every store at the width set then, and registers nothing --
    so a setting that disagrees with it is refused, and no space is registered: recorded, it
    would pass the space-width check below while every write failed on the table.

    Every open of a writable database runs it, and so does ``morgan migrate`` once its wave
    has committed: migration step 6 rebuilds the vector table without registering a space,
    because a step is not given the settings that name the model.
    """
    if spaces.active(conn) is not None:
        return
    with write_transaction(conn):
        if spaces.active(conn) is not None:
            return
        width = vectors.declared_width(conn, table_name=_VECTOR_TABLE)
        if width is not None and width != settings.embedding_dim:
            raise RuntimeError(
                f"the vector table {_VECTOR_TABLE} was created {width} wide but "
                f"MORGAN_EMBEDDING_DIM is {settings.embedding_dim}, so no embedding space was "
                f"registered; set MORGAN_EMBEDDING_DIM={width} if that is the model's width, or "
                f"point MORGAN_DATA_DIR at a database created at {settings.embedding_dim}"
            )
        space = spaces.register(
            conn,
            model=settings.embedding_model,
            dims=settings.embedding_dim,
            table_name=_VECTOR_TABLE,
            clock=utcnow,
        )
    log.info("embedding-space.registered", space_id=space.id, model=space.model, dims=space.dims)


def _require_the_space_width(conn: sqlite3.Connection, settings: Settings) -> None:
    """Refuse a database whose active space is not ``MORGAN_EMBEDDING_DIM`` wide.

    Read from the database, never asked of the model: every vector the space holds is that
    wide, and a query or a write at the settings' width would fail against them.
    """
    space = spaces.active(conn)
    if space is not None and space.dims != settings.embedding_dim:
        raise RuntimeError(
            f"embedding space {space.id} ({space.model}) holds {space.dims}-dimensional "
            f"vectors but MORGAN_EMBEDDING_DIM is {settings.embedding_dim}; the two must "
            f"agree (set MORGAN_EMBEDDING_DIM={space.dims}, the width this database was "
            "written at)"
        )


def build_memory_context(settings: Settings, *, budget: Budget = "interactive") -> MemoryContext:
    """Open the database and wire the memory core over it. Nothing is embedded.

    A writable database with no active embedding space is given the settings' model and
    width; a database waiting for ``morgan migrate`` registers nothing, and neither does the
    hash backend, which no model answers. A space whose width disagrees with the settings is
    refused here, from the database alone. The model is asked only when something is
    embedded, and its first request carries the space's check (``CheckedEmbedder``). *budget*
    is how long a failing embedding call keeps retrying: ``import`` for ``morgan import``,
    ``interactive`` for everything else.
    """
    path = sqlite_path(settings.temporal_db_url)
    if path != ":memory:":
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = open_db(path)
    try:
        embedder = build_embedder(settings, conn=conn, budget=budget)
        module = build_memory_module(
            conn,
            embedder=embedder,
            dim=settings.embedding_dim,
            floor_margin=settings.recall_floor_margin,
        )
        read_only_reason = _read_only_reason(pending(conn))
        if read_only_reason is None and settings.embedding_backend == "provider":
            register_the_settings_space(conn, settings)
        _require_the_space_width(conn, settings)
    except BaseException:
        # A refused open lets go of the file it opened.
        conn.close()
        raise
    return MemoryContext(
        gate=MemoryGate(module, read_only_reason=read_only_reason),
        conn=conn,
        history=SessionHistoryStore(conn, clock=utcnow),
        embedder=embedder,
        settings=settings,
    )


def build_app_context(settings: Settings) -> AppContext:
    memory = build_memory_context(settings)
    client = build_chat_client(settings)
    return AppContext(
        gate=memory.gate,
        conn=memory.conn,
        history=memory.history,
        embedder=memory.embedder,
        settings=settings,
        client=client,
        chat=Chat(
            gate=memory.gate,
            history=memory.history,
            client=client,
            model=settings.llm_model,
            clock=utcnow,
        ),
        consolidator=MemoryConsolidator(
            gate=memory.gate,
            client=client,
            model=settings.llm_model,
            clock=utcnow,
            json_mode=settings.llm_json_mode,
        ),
    )
