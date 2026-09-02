"""Composition root: open the one database and wire the core over it.

``build_memory_context`` is enough for every memory operation (``remember``, ``recall``,
``facts``, ``forget``, ``doctor``) and needs no chat model. ``build_app_context`` adds the
chat client for ``ask`` and ``consolidate``. Both share one connection: every store below
lives in the same SQLite file, which is what makes ``forget()`` one transaction.
"""

from __future__ import annotations

import asyncio
import pathlib
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import structlog

from morgan_brain.chat import Chat
from morgan_brain.config import Settings, get_settings
from morgan_brain.memory.consolidation import MemoryConsolidator
from morgan_brain.memory.db import open_db
from morgan_brain.memory.embedder import Embedder
from morgan_brain.memory.entities import EntityIndex
from morgan_brain.memory.episodic import EpisodicStore
from morgan_brain.memory.fts import FtsIndex
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.history import SessionHistoryStore
from morgan_brain.memory.module import MemoryModule
from morgan_brain.memory.schema_classifier import KeywordSchemaClassifier, SemanticIndexBuilder
from morgan_brain.memory.semantic_index import SemanticIndex
from morgan_brain.memory.temporal import SqliteTemporalStore
from morgan_brain.memory.vectors import SqliteVectorIndex
from morgan_brain.providers.factory import build_chat_client, build_embedder
from morgan_brain.providers.openai_compat import OpenAICompatAdapter
from morgan_brain.providers.wire import ProviderUnreachable

log = structlog.get_logger("composition")


def utcnow() -> datetime:
    return datetime.now(UTC)


def sqlite_path(url: str) -> str:
    """Turn a sqlite:/// URL into a filesystem path; pass through ':memory:'."""
    return url.removeprefix("sqlite:///")


def _run_coro_isolated(coro: Any) -> Any:
    """Run *coro* to completion whether or not an event loop is already running here."""
    result: list[Any] = []
    error: list[BaseException] = []

    def _runner() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001 — re-raised on the calling thread below
            error.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


def _probe_embedding_dim(embedder: Embedder, settings: Settings) -> None:
    """Verify the live embedder's output dimension matches ``settings.embedding_dim``.

    Catches the class of bug where ``embedding_model`` and ``embedding_dim`` disagree: the
    vector table is created with one width and every insert would then fail. Skipped for the
    hash stub, whose width *is* the setting. An unreachable endpoint is logged, not raised:
    the memory commands that never embed anything must still work.
    """
    if settings.embedding_backend == "hash":
        return
    try:
        vector = _run_coro_isolated(embedder.embed("probe"))
    except ProviderUnreachable as exc:
        log.warning(
            "embedding-dim-probe.unreachable", endpoint=settings.llm_endpoint, error=str(exc)
        )
        return
    if len(vector) != settings.embedding_dim:
        raise RuntimeError(
            f"embedding model {settings.embedding_model!r} at {settings.llm_endpoint} "
            f"returned a {len(vector)}-dimensional vector but settings.embedding_dim="
            f"{settings.embedding_dim}; the two must agree "
            "(set MORGAN_EMBEDDING_DIM to the model's real output size)"
        )


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


def build_memory_module(
    conn: sqlite3.Connection, *, embedder: Embedder, dim: int, clock: Any = utcnow
) -> MemoryModule:
    """Every store over one connection. Also the seam tests use with a small fake embedder."""
    semantic = SemanticIndex(conn)
    return MemoryModule(
        embedder=embedder,
        vectors=SqliteVectorIndex(conn, dim=dim),
        temporal=SqliteTemporalStore(conn=conn),
        clock=clock,
        fts=FtsIndex(conn),
        entities=EntityIndex(conn),
        episodics=EpisodicStore(conn),
        semantic=semantic,
        index_builder=SemanticIndexBuilder(semantic=semantic, classifier=KeywordSchemaClassifier()),
    )


def build_memory_context(settings: Settings | None = None) -> MemoryContext:
    settings = settings or get_settings()
    path = sqlite_path(settings.temporal_db_url)
    if path != ":memory:":
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = open_db(path)
    embedder = build_embedder(settings)
    _probe_embedding_dim(embedder, settings)
    module = build_memory_module(conn, embedder=embedder, dim=settings.embedding_dim)
    return MemoryContext(
        gate=MemoryGate(module),
        conn=conn,
        history=SessionHistoryStore(conn, clock=utcnow),
        embedder=embedder,
        settings=settings,
    )


def build_app_context(settings: Settings | None = None) -> AppContext:
    settings = settings or get_settings()
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
