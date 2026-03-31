"""Channel allowlist with open, allowlist, and pairing policies."""

from __future__ import annotations

import secrets
from typing import Optional


class ChannelAllowlist:
    """Controls which peer IDs may communicate on a channel.

    Policies
    --------
    - ``"open"`` -- every peer is allowed.
    - ``"allowlist"`` -- only explicitly approved peer IDs.
    - ``"pairing"`` -- peers must present a one-time pairing code.
    """

    def __init__(
        self,
        policy: str = "open",
        allowed_ids: Optional[set[str]] = None,
    ) -> None:
        if policy not in ("open", "allowlist", "pairing"):
            raise ValueError(f"Unknown policy: {policy!r}")
        self.policy = policy
        self.allowed_ids: set[str] = set(allowed_ids) if allowed_ids else set()
        # Maps pairing code -> peer_id
        self._pending_pairings: dict[str, str] = {}

    def is_allowed(self, peer_id: str) -> bool:
        """Check whether *peer_id* is allowed under the current policy."""
        if self.policy == "open":
            return True
        return peer_id in self.allowed_ids

    def request_pairing(self, peer_id: str) -> str:
        """Generate a hex pairing code for *peer_id* (pairing policy)."""
        code = secrets.token_hex(4)  # 8-char hex string
        self._pending_pairings[code] = peer_id
        return code

    def approve_pairing(self, code: str) -> bool:
        """Approve a pairing code, adding the peer to allowed_ids.

        Returns True if the code was valid and the peer was added.
        """
        peer_id = self._pending_pairings.pop(code, None)
        if peer_id is None:
            return False
        self.allowed_ids.add(peer_id)
        return True

    def add(self, peer_id: str) -> None:
        """Manually add a peer ID to the allowlist."""
        self.allowed_ids.add(peer_id)

    def remove(self, peer_id: str) -> None:
        """Remove a peer ID from the allowlist."""
        self.allowed_ids.discard(peer_id)
