"""
Morgan Tool System.

Provides a pluggable tool execution framework ported from Claude Code's
Tool.ts pattern, adapted for Python async/await.

Architecture:
- base.py: BaseTool ABC, ToolResult, ToolContext, ToolInputSchema
- permissions.py: Permission checking with fnmatch pattern matching
- executor.py: ToolExecutor for registry, validation, and execution
- builtin/: Built-in tools (calculator, file_read, bash, web_search, memory_search)

Usage:
    from morgan.tools import ToolExecutor, ToolContext, ToolResult
    from morgan.tools.builtin import ALL_BUILTIN_TOOLS
    from morgan.tools.permissions import PermissionContext, PermissionMode

    executor = ToolExecutor()
    for tool in ALL_BUILTIN_TOOLS:
        executor.register(tool)

    result = await executor.execute(
        "calculator",
        {"expression": "2 + 2"},
        context=ToolContext(user_id="user1"),
        permission_context=PermissionContext(mode=PermissionMode.BYPASS),
    )
    print(result.output)  # "4"
"""

from morgan.tools.base import (
    BaseTool,
    SessionType,
    ToolContext,
    ToolInputSchema,
    ToolResult,
)
from morgan.tools.executor import ToolExecutor
from morgan.tools.permissions import (
    PermissionContext,
    PermissionMode,
    PermissionResult,
    PermissionRule,
    check_permission,
)

__all__ = [
    # Base
    "BaseTool",
    "ToolContext",
    "ToolInputSchema",
    "ToolResult",
    "SessionType",
    # Executor
    "ToolExecutor",
    # Permissions
    "PermissionMode",
    "PermissionRule",
    "PermissionContext",
    "PermissionResult",
    "check_permission",
]
