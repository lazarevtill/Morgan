"""Phase 2A — Typed interaction-signal capture.

Design refs:
  * Design doc §A (signal capture) — edit > retry > thumb-down; thumb-up low-trust (sycophancy).
  * self-learning ADR — value order: edit(3) > retry/thumb-down(2) > thumb-up(1) > nothing(0).
  * Deterministic clock pattern from SqliteTemporalStore.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from enum import Enum
from typing import Callable

from morgan_brain.models.base import UserScoped
from morgan_brain.models.memory import DEFAULT_PROJECT

# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


class Thumb(str, Enum):
    UP = "up"
    DOWN = "down"


class InteractionSignal(UserScoped):
    """One logged interaction turn, accumulating feedback signals over time.

    value_rank encodes signal trustworthiness for downstream consumers:
      3 — user edited the reply  (ground-truth correction pair)
      2 — user retried OR thumbed down
      1 — user thumbed up       (low-trust; correlates with sycophancy)
      0 — no feedback
    """

    project: str = DEFAULT_PROJECT
    session_id: str
    turn_id: str
    context_summary: str = ""
    query: str
    original_reply: str
    user_edit: str | None = None
    retried: bool = False
    thumb: Thumb | None = None

    @property
    def value_rank(self) -> int:
        if self.user_edit is not None:
            return 3
        if self.retried or self.thumb is Thumb.DOWN:
            return 2
        if self.thumb is Thumb.UP:
            return 1
        return 0


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interaction_signals (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    project     TEXT NOT NULL DEFAULT 'default',
    session_id  TEXT NOT NULL,
    turn_id     TEXT NOT NULL,
    context_summary TEXT NOT NULL DEFAULT '',
    query       TEXT NOT NULL,
    original_reply  TEXT NOT NULL,
    user_edit   TEXT,
    retried     INTEGER NOT NULL DEFAULT 0,
    thumb       TEXT,
    consumed    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_signals_user
    ON interaction_signals (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_user_turn
    ON interaction_signals (user_id, turn_id);
"""

# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


class SignalStore:
    """SQLite-backed store for :class:`InteractionSignal` records.

    Parameters
    ----------
    conn:
        A shared :class:`sqlite3.Connection`, e.g. from
        :func:`morgan_brain.modules.memory.stores.db.open_db`, so signals live in the
        same database file as every other store (required for a single-transaction
        ``forget()``). Defaults to a private ``:memory:`` connection for tests.
    clock:
        Injected callable that returns the current :class:`datetime`.
        Never calls ``datetime.now()`` directly — keeps the store deterministic.
    """

    def __init__(
        self,
        conn: sqlite3.Connection | None = None,
        *,
        clock: Callable[[], datetime],
    ) -> None:
        self._clock = clock
        self._conn = (
            conn if conn is not None else sqlite3.connect(":memory:", check_same_thread=False)
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._migrate_project_column()

    def _migrate_project_column(self) -> None:
        """Idempotent upgrade for a database written before project scoping existed --
        required for forget() to filter interaction_signals by (user_id, project)."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(interaction_signals)")}
        if "project" not in cols:
            self._conn.execute(
                f"ALTER TABLE interaction_signals ADD COLUMN project TEXT NOT NULL "
                f"DEFAULT '{DEFAULT_PROJECT}'"
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_signal(self, row: sqlite3.Row) -> InteractionSignal:
        thumb_val: str | None = row["thumb"]
        return InteractionSignal(
            id=row["id"],
            user_id=row["user_id"],
            project=row["project"],
            session_id=row["session_id"],
            turn_id=row["turn_id"],
            context_summary=row["context_summary"],
            query=row["query"],
            original_reply=row["original_reply"],
            user_edit=row["user_edit"],
            retried=bool(row["retried"]),
            thumb=Thumb(thumb_val) if thumb_val is not None else None,
            created_at=_dt(row["created_at"]),
        )

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    async def record(self, signal: InteractionSignal) -> str:
        """Persist *signal*, stamping ``created_at`` from the injected clock if unset.

        Returns the signal's ``id``.
        """
        if signal.created_at is None:
            signal = signal.model_copy(update={"created_at": self._clock()})
        thumb_str = signal.thumb.value if signal.thumb is not None else None
        self._conn.execute(
            """
            INSERT INTO interaction_signals
                (id, user_id, project, session_id, turn_id, context_summary,
                 query, original_reply, user_edit, retried, thumb, consumed, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,0,?)
            """,
            (
                signal.id,
                signal.user_id,
                signal.project,
                signal.session_id,
                signal.turn_id,
                signal.context_summary,
                signal.query,
                signal.original_reply,
                signal.user_edit,
                int(signal.retried),
                thumb_str,
                _iso(signal.created_at),  # type: ignore[arg-type]
            ),
        )
        self._conn.commit()
        return signal.id

    async def mark_consumed(self, ids: list[str]) -> None:
        """Mark the given signal ids as consumed so the async worker won't re-process them."""
        if not ids:
            return
        # json_each over one bound JSON array, rather than a generated "?,?,?" list: the SQL
        # stays a literal (no query text built from data) and the id count is not capped by
        # SQLITE_MAX_VARIABLE_NUMBER.
        self._conn.execute(
            "UPDATE interaction_signals SET consumed=1 "
            "WHERE id IN (SELECT value FROM json_each(?))",
            (json.dumps(ids),),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    async def for_user(self, user_id: str, *, limit: int = 50) -> list[InteractionSignal]:
        """Return the *limit* most-recently recorded signals for *user_id* (newest first)."""
        rows = self._conn.execute(
            """
            SELECT * FROM interaction_signals
            WHERE user_id = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return [self._row_to_signal(r) for r in rows]

    async def high_value(
        self, user_id: str, *, min_rank: int = 2, limit: int = 50
    ) -> list[InteractionSignal]:
        """Return signals whose ``value_rank >= min_rank`` for *user_id* (newest first).

        ``value_rank`` is a computed property, so we filter in Python after a
        bounded fetch (``limit`` applies to the post-filter result).
        """
        # Fetch all signals for the user (bounded by a generous internal cap) then
        # filter by rank so we avoid re-implementing the rank logic in SQL.
        rows = self._conn.execute(
            """
            SELECT * FROM interaction_signals
            WHERE user_id = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
            """,
            (user_id, max(limit * 10, 500)),
        ).fetchall()
        result: list[InteractionSignal] = []
        for row in rows:
            sig = self._row_to_signal(row)
            if sig.value_rank >= min_rank:
                result.append(sig)
                if len(result) >= limit:
                    break
        return result

    async def unconsumed(self, user_id: str) -> list[InteractionSignal]:
        """Return all not-yet-consumed signals for *user_id* (oldest first for processing order)."""
        rows = self._conn.execute(
            """
            SELECT * FROM interaction_signals
            WHERE user_id = ? AND consumed = 0
            ORDER BY created_at ASC, rowid ASC
            """,
            (user_id,),
        ).fetchall()
        return [self._row_to_signal(r) for r in rows]
