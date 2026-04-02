"""
morgan.workspace — SOUL.md + Workspace Pattern.

Manages the on-disk workspace that stores Morgan's identity (SOUL.md),
user preferences (USER.md), long-term memory (MEMORY.md), tool definitions
(TOOLS.md), and heartbeat config (HEARTBEAT.md).

Ported from OpenClaw's SOUL.md / AGENTS.md / USER.md pattern with a key
security property: MEMORY.md is only surfaced in main/dm sessions, never
in group or cron sessions.
"""

from morgan.workspace.manager import WorkspaceManager
from morgan.workspace.paths import get_morgan_home, get_workspace_path, validate_workspace_path
from morgan.workspace.templates import (
    HEARTBEAT_TEMPLATE,
    MEMORY_TEMPLATE,
    SOUL_TEMPLATE,
    TOOLS_TEMPLATE,
    USER_TEMPLATE,
)

__all__ = [
    "WorkspaceManager",
    "get_morgan_home",
    "get_workspace_path",
    "validate_workspace_path",
    "SOUL_TEMPLATE",
    "USER_TEMPLATE",
    "MEMORY_TEMPLATE",
    "TOOLS_TEMPLATE",
    "HEARTBEAT_TEMPLATE",
]
