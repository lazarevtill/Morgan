"""Per-chat default-deny allowlist.

Policy (per platform ADR 2026-06-08):
* An empty ``allowed`` set **blocks all** chats (default-deny).
* Only chat_ids explicitly added are permitted.
* No wildcard matching — exact string equality only.

Example::

    al = ChatAllowlist(allowed={"chat_123", "chat_456"})
    al.is_allowed("chat_123")  # True
    al.is_allowed("chat_999")  # False
    ChatAllowlist().is_allowed("chat_123")  # False (empty → deny all)
"""

from __future__ import annotations


class ChatAllowlist:
    """Exact-match, default-deny allowlist for channel chat identifiers."""

    def __init__(self, allowed: set[str] | None = None) -> None:
        self._allowed: frozenset[str] = frozenset(allowed or set())

    def is_allowed(self, chat_id: str) -> bool:
        """Return ``True`` iff ``chat_id`` is in the explicit allow set."""
        return chat_id in self._allowed

    def __repr__(self) -> str:
        return f"ChatAllowlist(allowed={set(self._allowed)!r})"
