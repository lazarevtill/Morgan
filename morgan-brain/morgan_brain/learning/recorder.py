"""Phase 2A — SignalRecorder: thin facade for accumulating per-turn feedback.

The API/CLI calls these methods after a turn is served.  All mutation is on
the *same* row (keyed by ``user_id`` + ``turn_id``) so one signal per turn
accumulates edit/retry/thumb feedback without duplicating the base record.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable

from morgan_brain.learning.signals import InteractionSignal, SignalStore, Thumb
from morgan_brain.models.memory import DEFAULT_PROJECT


class SignalRecorder:
    """Facade over :class:`SignalStore` for the post-response async path.

    Parameters
    ----------
    store:
        The underlying :class:`SignalStore`.
    clock:
        Injected callable for deterministic timestamps.
    """

    def __init__(self, *, store: SignalStore, clock: Callable[[], datetime]) -> None:
        self._store = store
        self._clock = clock

    # ------------------------------------------------------------------
    # Base turn logging
    # ------------------------------------------------------------------

    async def record_turn(
        self,
        *,
        user_id: str,
        session_id: str,
        turn_id: str,
        query: str,
        reply: str,
        project: str = DEFAULT_PROJECT,
        context_summary: str = "",
    ) -> str:
        """Log the base signal for a completed turn.  Returns the signal id."""
        sig = InteractionSignal(
            user_id=user_id,
            project=project,
            session_id=session_id,
            turn_id=turn_id,
            query=query,
            original_reply=reply,
            context_summary=context_summary,
            created_at=self._clock(),
        )
        return await self._store.record(sig)

    # ------------------------------------------------------------------
    # Feedback accumulation helpers
    #
    # None of add_edit/add_retry/add_thumb take a ``project`` -- they UPDATE the existing
    # row keyed by (user_id, turn_id) that ``record_turn`` already wrote (synchronously, with
    # the real project, from the orchestrator's hot path) rather than constructing a new row.
    # The row's project is therefore already correct and untouched by these calls. The one
    # exception is the stub _ensure_signal creates for a turn_id with no matching base row
    # (a forged/garbage turn_id, since a real one always has its base row by the time the
    # client can call /api/feedback) -- that stub falls back to DEFAULT_PROJECT, which is
    # the only sane choice for a turn that was never actually served (fix round 2 audit).
    # ------------------------------------------------------------------

    async def add_edit(self, *, turn_id: str, user_id: str, edited_reply: str) -> None:
        """Record that the user edited the reply for *turn_id*.

        Updates the existing signal row in-place.  If no base signal exists
        for this ``(user_id, turn_id)`` pair, a stub is created first.
        """
        await self._ensure_signal(turn_id=turn_id, user_id=user_id)
        self._store._conn.execute(  # noqa: SLF001  (internal access, same package)
            """
            UPDATE interaction_signals
            SET user_edit = ?
            WHERE user_id = ? AND turn_id = ?
            """,
            (edited_reply, user_id, turn_id),
        )
        self._store._conn.commit()  # noqa: SLF001

    async def add_retry(self, *, turn_id: str, user_id: str) -> None:
        """Record that the user retried the turn (asked again / regenerated)."""
        await self._ensure_signal(turn_id=turn_id, user_id=user_id)
        self._store._conn.execute(  # noqa: SLF001
            """
            UPDATE interaction_signals
            SET retried = 1
            WHERE user_id = ? AND turn_id = ?
            """,
            (user_id, turn_id),
        )
        self._store._conn.commit()  # noqa: SLF001

    async def add_thumb(self, *, turn_id: str, user_id: str, thumb: Thumb) -> None:
        """Record a thumb-up or thumb-down rating for *turn_id*."""
        await self._ensure_signal(turn_id=turn_id, user_id=user_id)
        self._store._conn.execute(  # noqa: SLF001
            """
            UPDATE interaction_signals
            SET thumb = ?
            WHERE user_id = ? AND turn_id = ?
            """,
            (thumb.value, user_id, turn_id),
        )
        self._store._conn.commit()  # noqa: SLF001

    async def original_reply_for(self, *, user_id: str, turn_id: str) -> str | None:
        """Return the base reply recorded for ``(user_id, turn_id)``, or ``None``.

        The edit signal only carries the *edited* text; the learn-from-edits loop needs
        the ORIGINAL reply (recorded by ``record_turn``) to distil a preference delta.
        Returns ``None`` when no base signal exists or the reply is empty (e.g. a stub
        created by feedback that arrived before the turn was recorded).
        """
        row = self._store._conn.execute(  # noqa: SLF001  (internal access, same package)
            "SELECT original_reply FROM interaction_signals WHERE user_id = ? AND turn_id = ?",
            (user_id, turn_id),
        ).fetchone()
        if row is None:
            return None
        value: str | None = row[0]
        return value or None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _ensure_signal(self, *, turn_id: str, user_id: str) -> None:
        """Create a stub signal for ``(user_id, turn_id)`` if none exists yet."""
        row = self._store._conn.execute(  # noqa: SLF001
            "SELECT id FROM interaction_signals WHERE user_id = ? AND turn_id = ?",
            (user_id, turn_id),
        ).fetchone()
        if row is None:
            stub = InteractionSignal(
                user_id=user_id,
                session_id="",
                turn_id=turn_id,
                query="",
                original_reply="",
                created_at=self._clock(),
            )
            await self._store.record(stub)
