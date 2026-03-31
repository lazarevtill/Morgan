"""
Tool executor for the Morgan tool system.

Manages the tool registry, validates inputs, checks permissions,
and executes tools with proper error handling.

Usage:
    from morgan.tools.executor import ToolExecutor
    from morgan.tools.base import ToolContext
    from morgan.tools.permissions import PermissionContext, PermissionMode

    executor = ToolExecutor()
    executor.register(my_tool)

    result = await executor.execute(
        tool_name="calculator",
        input_data={"expression": "2 + 2"},
        context=ToolContext(user_id="user1"),
        permission_context=PermissionContext(mode=PermissionMode.BYPASS),
    )
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from morgan.tools.base import BaseTool, ToolContext, ToolResult
from morgan.tools.permissions import (
    PermissionContext,
    PermissionMode,
    check_permission,
)

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Central executor for Morgan tools.

    Manages a registry of tools (by name and aliases), validates input,
    checks permissions, and handles execution with try/except.

    Attributes:
        _tools: Mapping of tool name/alias to tool instance.
        _primary_tools: Mapping of primary tool name to tool instance.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}
        self._primary_tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool by its name and all aliases.

        Args:
            tool: The tool instance to register.

        Raises:
            ValueError: If the tool name or any alias is already registered.
        """
        if not tool.name:
            raise ValueError("Tool must have a non-empty name")

        # Check for conflicts
        all_names = tool.get_all_names()
        for name in all_names:
            if name in self._tools:
                existing = self._tools[name]
                raise ValueError(
                    f"Name '{name}' is already registered to tool '{existing.name}'"
                )

        # Register under primary name and all aliases
        self._primary_tools[tool.name] = tool
        for name in all_names:
            self._tools[name] = tool

    def unregister(self, tool_name: str) -> bool:
        """
        Unregister a tool by its primary name.

        Removes the tool and all its aliases from the registry.

        Args:
            tool_name: The primary name of the tool to remove.

        Returns:
            True if the tool was found and removed, False otherwise.
        """
        tool = self._primary_tools.pop(tool_name, None)
        if tool is None:
            return False

        for name in tool.get_all_names():
            self._tools.pop(name, None)
        return True

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Look up a tool by name or alias.

        Args:
            name: Tool name or alias.

        Returns:
            The tool instance, or None if not found.
        """
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        """Return all registered tools (deduplicated by primary name)."""
        return list(self._primary_tools.values())

    async def execute(
        self,
        tool_name: str,
        input_data: Dict[str, Any],
        context: Optional[ToolContext] = None,
        permission_context: Optional[PermissionContext] = None,
    ) -> ToolResult:
        """
        Execute a tool by name with input validation and permission checking.

        Steps:
        1. Look up tool by name/alias.
        2. Check permissions.
        3. Validate input.
        4. Execute with error handling.

        Args:
            tool_name: Name or alias of the tool to execute.
            input_data: Input dictionary for the tool.
            context: Runtime context (defaults to a default ToolContext).
            permission_context: Permission context (defaults to BYPASS for convenience).

        Returns:
            ToolResult with the execution outcome.
        """
        if context is None:
            context = ToolContext()
        if permission_context is None:
            permission_context = PermissionContext(mode=PermissionMode.BYPASS)

        # Step 1: Look up tool
        tool = self.get_tool(tool_name)
        if tool is None:
            available = ", ".join(sorted(self._primary_tools.keys()))
            return ToolResult(
                output=f"Unknown tool: '{tool_name}'. Available tools: {available}",
                is_error=True,
                error_code="TOOL_NOT_FOUND",
            )

        # Step 2: Check permissions (use the primary tool name for permission checks)
        perm_result = check_permission(tool.name, permission_context)
        if not perm_result.allowed:
            return ToolResult(
                output=f"Permission denied for tool '{tool.name}': {perm_result.reason}",
                is_error=True,
                error_code="PERMISSION_DENIED",
            )

        # Step 3: Validate input
        validation_error = tool.validate_input(input_data)
        if validation_error is not None:
            return ToolResult(
                output=f"Input validation failed for tool '{tool.name}': {validation_error}",
                is_error=True,
                error_code="VALIDATION_ERROR",
            )

        # Step 4: Execute with error handling
        start_time = time.monotonic()
        try:
            result = await tool.execute(input_data, context)
            elapsed = time.monotonic() - start_time
            result.metadata.setdefault("execution_time_ms", round(elapsed * 1000, 2))
            return result
        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.exception(
                "Tool '%s' raised an exception during execution", tool.name
            )
            return ToolResult(
                output=f"Tool '{tool.name}' execution failed: {type(e).__name__}: {e}",
                is_error=True,
                error_code="EXECUTION_ERROR",
                metadata={"execution_time_ms": round(elapsed * 1000, 2)},
            )
