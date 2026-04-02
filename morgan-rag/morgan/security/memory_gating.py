"""Memory gating -- controls which session types may load memory."""

from __future__ import annotations

_ALLOWED_SESSION_TYPES = frozenset({"main", "dm"})


class MemoryGate:
    """Determines whether a session type is permitted to load memory."""

    @staticmethod
    def should_load_memory(session_type: str) -> bool:
        """Return True only for 'main' and 'dm' sessions."""
        return session_type in _ALLOWED_SESSION_TYPES
