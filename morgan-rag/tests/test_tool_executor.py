"""
Tests for ToolExecutor: registry, validation, permission checking, and execution.

Covers:
- Register and unregister tools by name and alias
- Reject duplicate registrations
- Look up tools by alias
- Execute with permission checking
- Execute with input validation
- Execute with error handling (exception in tool)
- Execute unknown tool returns TOOL_NOT_FOUND
- Permission denied returns PERMISSION_DENIED
- Validation error returns VALIDATION_ERROR
- Execution timing metadata
"""

import pytest

from morgan.tools.base import BaseTool, ToolContext, ToolInputSchema, ToolResult
from morgan.tools.executor import ToolExecutor
from morgan.tools.permissions import (
    PermissionContext,
    PermissionMode,
    PermissionRule,
)
from morgan.tools.builtin.calculator import CalculatorTool


# =============================================================================
# Test Fixtures
# =============================================================================


class EchoTool(BaseTool):
    """Simple tool that echoes its input for testing."""

    name = "echo"
    description = "Echoes input back"
    aliases = ("echo_tool", "parrot")
    input_schema = ToolInputSchema(
        properties={
            "message": {"type": "string", "description": "Message to echo"},
        },
        required=("message",),
    )

    async def execute(self, input_data, context):
        return ToolResult(output=input_data["message"])


class FailingTool(BaseTool):
    """Tool that always raises an exception."""

    name = "failing"
    description = "Always fails"

    async def execute(self, input_data, context):
        raise RuntimeError("Intentional failure for testing")


class SlowTool(BaseTool):
    """Tool that takes a long time (for timeout testing)."""

    name = "slow"
    description = "Takes a while"

    async def execute(self, input_data, context):
        import asyncio
        await asyncio.sleep(999)
        return ToolResult(output="done")


# =============================================================================
# Registry Tests
# =============================================================================


class TestToolExecutorRegistry:
    """Test tool registration and lookup."""

    def test_register_and_lookup(self):
        """Register a tool and look it up by name."""
        executor = ToolExecutor()
        echo = EchoTool()
        executor.register(echo)

        assert executor.get_tool("echo") is echo

    def test_lookup_by_alias(self):
        """Look up a tool by alias."""
        executor = ToolExecutor()
        echo = EchoTool()
        executor.register(echo)

        assert executor.get_tool("echo_tool") is echo
        assert executor.get_tool("parrot") is echo

    def test_lookup_unknown_returns_none(self):
        """Looking up an unknown tool returns None."""
        executor = ToolExecutor()
        assert executor.get_tool("nonexistent") is None

    def test_list_tools(self):
        """list_tools returns all registered tools."""
        executor = ToolExecutor()
        echo = EchoTool()
        calc = CalculatorTool()
        executor.register(echo)
        executor.register(calc)

        tools = executor.list_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"echo", "calculator"}

    def test_duplicate_name_raises(self):
        """Registering a tool with a conflicting name raises ValueError."""
        executor = ToolExecutor()
        executor.register(EchoTool())

        # Try to register another tool with the same name
        class AnotherEcho(BaseTool):
            name = "echo"

            async def execute(self, input_data, context):
                return ToolResult(output="other")

        with pytest.raises(ValueError, match="already registered"):
            executor.register(AnotherEcho())

    def test_duplicate_alias_raises(self):
        """Registering a tool whose alias conflicts raises ValueError."""
        executor = ToolExecutor()
        executor.register(EchoTool())

        # This tool's alias "parrot" conflicts with EchoTool's alias
        class ConflictTool(BaseTool):
            name = "conflict"
            aliases = ("parrot",)

            async def execute(self, input_data, context):
                return ToolResult(output="conflict")

        with pytest.raises(ValueError, match="already registered"):
            executor.register(ConflictTool())

    def test_empty_name_raises(self):
        """Registering a tool with empty name raises ValueError."""
        executor = ToolExecutor()

        class NoNameTool(BaseTool):
            name = ""

            async def execute(self, input_data, context):
                return ToolResult(output="no name")

        with pytest.raises(ValueError, match="non-empty name"):
            executor.register(NoNameTool())

    def test_unregister(self):
        """Unregistering a tool removes it and its aliases."""
        executor = ToolExecutor()
        echo = EchoTool()
        executor.register(echo)
        assert executor.get_tool("echo") is echo

        result = executor.unregister("echo")
        assert result is True
        assert executor.get_tool("echo") is None
        assert executor.get_tool("parrot") is None

    def test_unregister_nonexistent(self):
        """Unregistering a nonexistent tool returns False."""
        executor = ToolExecutor()
        assert executor.unregister("nonexistent") is False


# =============================================================================
# Execution Tests
# =============================================================================


class TestToolExecutorExecution:
    """Test tool execution through the executor."""

    @pytest.fixture
    def executor(self):
        ex = ToolExecutor()
        ex.register(EchoTool())
        ex.register(CalculatorTool())
        ex.register(FailingTool())
        return ex

    @pytest.fixture
    def ctx(self):
        return ToolContext()

    @pytest.fixture
    def bypass_perms(self):
        return PermissionContext(mode=PermissionMode.BYPASS)

    @pytest.mark.asyncio
    async def test_execute_success(self, executor, ctx, bypass_perms):
        """Successful execution returns output."""
        result = await executor.execute(
            "echo",
            {"message": "hello"},
            context=ctx,
            permission_context=bypass_perms,
        )
        assert result.output == "hello"
        assert result.is_error is False
        assert "execution_time_ms" in result.metadata

    @pytest.mark.asyncio
    async def test_execute_by_alias(self, executor, ctx, bypass_perms):
        """Execution works via alias."""
        result = await executor.execute(
            "parrot",
            {"message": "hello via alias"},
            context=ctx,
            permission_context=bypass_perms,
        )
        assert result.output == "hello via alias"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, executor, ctx, bypass_perms):
        """Executing unknown tool returns TOOL_NOT_FOUND."""
        result = await executor.execute(
            "nonexistent",
            {},
            context=ctx,
            permission_context=bypass_perms,
        )
        assert result.is_error is True
        assert result.error_code == "TOOL_NOT_FOUND"
        assert "nonexistent" in result.output

    @pytest.mark.asyncio
    async def test_execute_permission_denied(self, executor, ctx):
        """Executing with denied permission returns PERMISSION_DENIED."""
        denied_perms = PermissionContext(
            mode=PermissionMode.DEFAULT,
            denied_tools=["echo"],
        )
        result = await executor.execute(
            "echo",
            {"message": "blocked"},
            context=ctx,
            permission_context=denied_perms,
        )
        assert result.is_error is True
        assert result.error_code == "PERMISSION_DENIED"

    @pytest.mark.asyncio
    async def test_execute_validation_error(self, executor, ctx, bypass_perms):
        """Executing with invalid input returns VALIDATION_ERROR."""
        # EchoTool requires "message" field
        result = await executor.execute(
            "echo",
            {},
            context=ctx,
            permission_context=bypass_perms,
        )
        assert result.is_error is True
        assert result.error_code == "VALIDATION_ERROR"

    @pytest.mark.asyncio
    async def test_execute_tool_exception(self, executor, ctx, bypass_perms):
        """Tool that raises exception returns EXECUTION_ERROR."""
        result = await executor.execute(
            "failing",
            {},
            context=ctx,
            permission_context=bypass_perms,
        )
        assert result.is_error is True
        assert result.error_code == "EXECUTION_ERROR"
        assert "RuntimeError" in result.output
        assert "execution_time_ms" in result.metadata

    @pytest.mark.asyncio
    async def test_execute_default_context(self, executor):
        """Executing without explicit context uses defaults."""
        # Default permission context is BYPASS
        result = await executor.execute(
            "echo",
            {"message": "default context"},
        )
        assert result.output == "default context"
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_execution_time_metadata(self, executor, ctx, bypass_perms):
        """Execution result includes timing metadata."""
        result = await executor.execute(
            "calculator",
            {"expression": "2 + 2"},
            context=ctx,
            permission_context=bypass_perms,
        )
        assert result.output == "4"
        assert "execution_time_ms" in result.metadata
        assert isinstance(result.metadata["execution_time_ms"], float)


# =============================================================================
# Permission Integration Tests
# =============================================================================


class TestToolExecutorPermissions:
    """Test permission integration in executor."""

    @pytest.fixture
    def executor(self):
        ex = ToolExecutor()
        ex.register(EchoTool())
        ex.register(CalculatorTool())
        return ex

    @pytest.fixture
    def ctx(self):
        return ToolContext()

    @pytest.mark.asyncio
    async def test_rule_based_allow(self, executor, ctx):
        """Rule-based allow works through executor."""
        perms = PermissionContext(
            mode=PermissionMode.DEFAULT,
            rules=[
                PermissionRule(tool_pattern="calculator", allowed=True),
            ],
        )
        result = await executor.execute(
            "calculator",
            {"expression": "1 + 1"},
            context=ctx,
            permission_context=perms,
        )
        assert result.is_error is False
        assert result.output == "2"

    @pytest.mark.asyncio
    async def test_rule_based_deny(self, executor, ctx):
        """Rule-based deny works through executor."""
        perms = PermissionContext(
            mode=PermissionMode.DEFAULT,
            rules=[
                PermissionRule(tool_pattern="echo*", allowed=False),
            ],
        )
        result = await executor.execute(
            "echo",
            {"message": "blocked"},
            context=ctx,
            permission_context=perms,
        )
        assert result.is_error is True
        assert result.error_code == "PERMISSION_DENIED"

    @pytest.mark.asyncio
    async def test_alias_uses_primary_name_for_permission(self, executor, ctx):
        """Permission checks use the primary tool name, not the alias."""
        perms = PermissionContext(
            mode=PermissionMode.DEFAULT,
            rules=[
                PermissionRule(tool_pattern="echo", allowed=True),
            ],
        )
        # Execute via alias "parrot" - permission check should use "echo"
        result = await executor.execute(
            "parrot",
            {"message": "via alias"},
            context=ctx,
            permission_context=perms,
        )
        assert result.is_error is False
        assert result.output == "via alias"

    @pytest.mark.asyncio
    async def test_plan_mode_integration(self, executor, ctx):
        """PLAN mode blocks tools without explicit allow."""
        perms = PermissionContext(
            mode=PermissionMode.PLAN,
            rules=[
                PermissionRule(tool_pattern="calculator", allowed=True),
            ],
        )
        # calculator is allowed
        result1 = await executor.execute(
            "calculator",
            {"expression": "3 * 3"},
            context=ctx,
            permission_context=perms,
        )
        assert result1.is_error is False

        # echo is NOT allowed
        result2 = await executor.execute(
            "echo",
            {"message": "blocked"},
            context=ctx,
            permission_context=perms,
        )
        assert result2.is_error is True
        assert result2.error_code == "PERMISSION_DENIED"


# =============================================================================
# Built-in Tool Registration Tests
# =============================================================================


class TestBuiltinToolRegistration:
    """Test registering all built-in tools."""

    def test_register_all_builtins(self):
        """All built-in tools can be registered without conflict."""
        from morgan.tools.builtin import ALL_BUILTIN_TOOLS

        executor = ToolExecutor()
        for tool in ALL_BUILTIN_TOOLS:
            executor.register(tool)

        assert len(executor.list_tools()) == 5

    @pytest.mark.asyncio
    async def test_execute_builtin_calculator(self):
        """Calculator tool works through executor."""
        from morgan.tools.builtin import ALL_BUILTIN_TOOLS

        executor = ToolExecutor()
        for tool in ALL_BUILTIN_TOOLS:
            executor.register(tool)

        result = await executor.execute(
            "calculator",
            {"expression": "sqrt(16)"},
        )
        assert result.output == "4.0"
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_execute_builtin_via_alias(self):
        """Built-in tools work via aliases through executor."""
        from morgan.tools.builtin import ALL_BUILTIN_TOOLS

        executor = ToolExecutor()
        for tool in ALL_BUILTIN_TOOLS:
            executor.register(tool)

        result = await executor.execute(
            "calc",  # alias for calculator
            {"expression": "7 * 8"},
        )
        assert result.output == "56"
        assert result.is_error is False
