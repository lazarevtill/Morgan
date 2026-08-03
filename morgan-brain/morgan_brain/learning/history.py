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

from morgan_brain.models.memory import DEFAULT_PROJECT
from morgan_brain.models.message import Message, Role


def session_key(user_id: str, session_id: str | None) -> str:
    """Compose the durable history key from ``(user_id, session_id)``.

    History is keyed per-user so two clients that happen to pick the same
    ``session_id`` can never see each other's turns. A missing/empty
    ``session_id`` falls back to a *per-user* ``"<user>:default"`` bucket —
    never a single global ``"default"`` (that would cross-contaminate memory and
    learning across users, the one thing the platform must never do).
    """
    return f"{user_id}:{session_id or 'default'}"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    project     TEXT NOT NULL DEFAULT 'default',
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
    conn:
        A shared :class:`sqlite3.Connection`, e.g. from
        :func:`morgan_brain.modules.memory.stores.db.open_db`, so history lives in the
        same database file as every other store (required for a single-transaction
        ``forget()``). Defaults to a private ``:memory:`` connection for tests.
    clock:
        Optional injected callable returning the current :class:`datetime`.
        When ``None``, ``created_at`` is stored as ``None`` and ordering is by
        insertion rowid (still deterministic within a test run).
    """

    def __init__(
        self,
        conn: sqlite3.Connection | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
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
        required for forget() to filter session_history by (user_id, project)."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(session_history)")}
        if "project" not in cols:
            self._conn.execute(
                f"ALTER TABLE session_history ADD COLUMN project TEXT NOT NULL "
                f"DEFAULT '{DEFAULT_PROJECT}'"
            )
            self._conn.commit()

    def append(self, session_id: str, message: Message, *, project: str = DEFAULT_PROJECT) -> None:
        """Append *message* to the history for *session_id*.

        Synchronous — intended for the cold-path turn-storage subscriber which
        runs after the reply is already sent to the caller.
        """
        created_at = self._clock().isoformat() if self._clock else None
        self._conn.execute(
            """
            INSERT INTO session_history (session_id, user_id, project, role, content, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                message.user_id,
                project,
                message.role.value,
                message.content,
                created_at,
            ),
        )
        self._conn.commit()

    def recent(self, session_id: str, *, project: str, limit: int = 10) -> list[Message]:
        """Return the *limit* most-recent messages for *session_id* **in *project***.

        Fetches the last *limit* rows by insertion order, then re-sorts ascending so
        callers receive them oldest-first (correct context order for LLM prompts).
        Returns an empty list when no history exists (e.g. first turn).

        ``project`` is required, and filtering on it is not optional decoration. Rows carried
        a project already, but this read did not use it, and ``session_key`` falls back to a
        per-user ``"<user>:default"`` bucket whenever a client sends no ``session_id`` -- which
        the CLI and every default API call do. Every project therefore shared one history key,
        so a turn in one repository injected the previous turn from another straight into the
        prompt, and erasing a project did not stop its transcript reappearing under the next
        one. Writes are already project-keyed, so this is the read catching up.
        """
        rows = self._conn.execute(
            """
            SELECT user_id, role, content FROM (
                SELECT user_id, role, content, id
                FROM session_history
                WHERE session_id = ? AND project = ?
                ORDER BY id DESC
                LIMIT ?
            ) ORDER BY id ASC
            """,
            (session_id, project, limit),
        ).fetchall()
        return [
            Message(user_id=row["user_id"], role=Role(row["role"]), content=row["content"])
            for row in rows
        ]
