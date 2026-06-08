"""The single permission model for tool execution. One enum, one gate — no duplication.

Capability grants extend the gate with fine-grained token-style authorisation while
preserving the original AUTO/ASK/DENY enum API completely (back-compat).

Permission resolution order (most-specific wins):
1. DENY mode always blocks — even a valid grant cannot override it.
2. An explicit Grant authorises the call (within its scope / param constraints).
3. AUTO mode authorises when no grant exists (back-compat — safe built-ins).
4. ASK mode without a grant = default-deny at execute time (caller must obtain a grant).
5. Unknown tool with no grant and default=ASK → denied.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Literal

from pydantic import BaseModel


class PermissionMode(str, Enum):
    AUTO = "auto"    # execute without asking
    ASK = "ask"      # require confirmation
    DENY = "deny"    # never execute


class Grant(BaseModel):
    """A capability token that authorises one tool call pattern.

    Fields
    ------
    tool:
        Exact tool name this grant covers.
    scope:
        Coarse operation level: ``"read"``, ``"write"``, or ``"execute"`` (default).
        The gate checks that the requested scope matches or is narrower.
    allowed_params:
        If set, the *keys* of kwargs passed to execute must be a subset of this list.
        ``None`` means any params are permitted.
    egress_allowlist:
        URLs / host prefixes that the tool is allowed to contact (advisory; enforced by
        the tool itself or a proxy layer). Not checked by the gate today — stored for
        future use.
    memory_namespaces:
        Memory namespaces the tool may read/write. Not enforced by the gate today —
        stored for MemoryGate integration in a later increment.
    ttl_seconds:
        If set, the grant expires this many seconds after ``grant()`` is called.
        ``None`` means the grant never expires.
    """

    tool: str
    scope: Literal["read", "write", "execute"] = "execute"
    allowed_params: list[str] | None = None  # None = any
    egress_allowlist: list[str] = []
    memory_namespaces: list[str] = []
    ttl_seconds: int | None = None


_SCOPE_RANK: dict[str, int] = {"read": 0, "write": 1, "execute": 2}


class PermissionGate:
    """Single authority for all tool permission checks.

    Backward-compatible: ``set``, ``mode_for``, ``allowed`` work identically to before.
    New surface: ``grant``, ``revoke``, ``check``.
    """

    def __init__(self, default: PermissionMode = PermissionMode.ASK) -> None:
        self._default = default
        self._by_tool: dict[str, PermissionMode] = {}
        self._grants: dict[str, tuple[Grant, float | None]] = {}  # tool → (grant, expires_at)

    # ------------------------------------------------------------------
    # Original API (unchanged)
    # ------------------------------------------------------------------

    def set(self, tool_name: str, mode: PermissionMode) -> None:
        """Override the permission mode for a specific tool."""
        self._by_tool[tool_name] = mode

    def mode_for(self, tool_name: str) -> PermissionMode:
        """Return the effective PermissionMode for *tool_name*."""
        return self._by_tool.get(tool_name, self._default)

    def allowed(self, tool_name: str) -> bool:
        """Legacy check — True unless the tool's mode is DENY."""
        return self.mode_for(tool_name) is not PermissionMode.DENY

    # ------------------------------------------------------------------
    # Capability-grant API
    # ------------------------------------------------------------------

    def grant(self, g: Grant) -> None:
        """Install a capability grant for *g.tool*.

        If the grant carries a ``ttl_seconds``, expiry is measured from now
        (using ``time.monotonic()``).  Installing a new grant for the same tool
        replaces the previous one.
        """
        expires_at: float | None = None
        if g.ttl_seconds is not None:
            expires_at = time.monotonic() + g.ttl_seconds
        self._grants[g.tool] = (g, expires_at)

    def revoke(self, tool: str) -> None:
        """Remove any capability grant for *tool* (no-op if none exists)."""
        self._grants.pop(tool, None)

    def check(
        self,
        tool: str,
        *,
        scope: Literal["read", "write", "execute"] = "execute",
        params: list[str] | None = None,
    ) -> bool:
        """Return True if the call is authorised, False otherwise.

        Resolution order
        ----------------
        1. DENY mode → always False.
        2. Valid, in-scope grant with matching params → True.
        3. AUTO mode (no grant required) → True.
        4. Anything else → False (default-deny).
        """
        mode = self.mode_for(tool)

        # 1. DENY always wins.
        if mode is PermissionMode.DENY:
            return False

        # 2. Check for a live grant.
        grant_entry = self._grants.get(tool)
        if grant_entry is not None:
            g, expires_at = grant_entry
            if expires_at is not None and time.monotonic() >= expires_at:
                # Expired — remove and fall through.
                del self._grants[tool]
            else:
                # Scope check: requested scope must be ≤ granted scope.
                if _SCOPE_RANK[scope] <= _SCOPE_RANK[g.scope]:
                    # Param check: requested params must be subset of allowed_params.
                    if g.allowed_params is None:
                        return True
                    if params is None or set(params).issubset(set(g.allowed_params)):
                        return True
                return False  # grant exists but scope/params violated

        # 3. No valid grant — AUTO mode still authorises (back-compat).
        return mode is PermissionMode.AUTO
