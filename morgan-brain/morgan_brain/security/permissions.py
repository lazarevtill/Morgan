"""The single permission model for tool execution. One enum, one gate — no duplication."""
from __future__ import annotations

from enum import Enum


class PermissionMode(str, Enum):
    AUTO = "auto"        # execute without asking
    ASK = "ask"          # require confirmation
    DENY = "deny"        # never execute


class PermissionGate:
    def __init__(self, default: PermissionMode = PermissionMode.ASK) -> None:
        self._default = default
        self._by_tool: dict[str, PermissionMode] = {}

    def set(self, tool_name: str, mode: PermissionMode) -> None:
        self._by_tool[tool_name] = mode

    def mode_for(self, tool_name: str) -> PermissionMode:
        return self._by_tool.get(tool_name, self._default)

    def allowed(self, tool_name: str) -> bool:
        return self.mode_for(tool_name) is not PermissionMode.DENY
