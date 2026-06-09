"""Lightweight session history store — persists per-session Message lists in SQLite.

``SessionHistoryStore`` is used by the composition root to thread conversation
history into ``ReasoningRequest.history`` so multi-turn context reaches the LLM.

Design notes
------------
- SQLite-backed (same deterministic pattern as ``SignalStore`` / ``SqliteTemporalStore``).
- ``append`` is synchronous (hot path; no I/O wait on in-process bus).
- ``recent`` returns messages in chronological order (oldest first), bounded by *limit*.
- Clock is injected for deterministic ordering in tests; defaults to ``None`` (ordering
  is by insertion rowid when no explicit timestamp is needed).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Callable

from morgan_brain.models.message import Message, Role

_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_history_session
    ON session_history (session_id, id ASC);
"""


class SessionHistoryStore:
    """SQLite-backed store for per-session :class:`Message` records.

    Parameters
    ----------
    path:
        SQLite file path.  Defaults to ``:memory:`` for tests.
    clock:
        Optional injected callable returning the current :class:`datetime`.
        When ``None``, ``created_at`` is stored as ``None`` and ordering is by
        insertion rowid (still deterministic within a test run).
    """

    def __init__(
        self,
        path: str = ":memory:",
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._clock = clock
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def append(self, session_id: str, message: Message) -> None:
        """Append *message* to the history for *session_id*.

        Synchronous — intended for the cold-path turn-storage subscriber which
        runs after the reply is already sent to the caller.
        """
        created_at = self._clock().isoformat() if self._clock else None
        self._conn.execute(
            """
            INSERT INTO session_history (session_id, user_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                message.user_id,
                message.role.value,
                message.content,
                created_at,
            ),
        )
        self._conn.commit()

    def recent(self, session_id: str, *, limit: int = 10) -> list[Message]:
        """Return the *limit* most-recent messages for *session_id*, chronological order.

        Returns an empty list when no history exists (e.g. first turn).
        """
        rows = self._conn.execute(
            """
            SELECT user_id, role, content
            FROM session_history
            WHERE session_id = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        return [
            Message(user_id=row["user_id"], role=Role(row["role"]), content=row["content"])
            for row in rows
        ]
