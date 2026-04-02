"""
Tests for the Morgan tool system: base classes, permissions, and built-in tools.

Covers:
- BaseTool ABC cannot be instantiated directly
- ToolInputSchema is frozen and has to_dict()
- ToolResult fields and defaults
- ToolContext fields and defaults
- PermissionMode enum values
- check_permission with fnmatch, deny-first resolution
- CalculatorTool: safe math via ast.parse
- FileReadTool: read files with line numbers
- BashTool: blocked patterns, timeout
- WebSearchTool / MemorySearchTool: handle ImportError gracefully
"""

import asyncio
import math
import os
import tempfile
from dataclasses import FrozenInstanceError

import pytest

from morgan.tools.base import (
    BaseTool,
    SessionType,
    ToolContext,
    ToolInputSchema,
    ToolResult,
)
from morgan.tools.permissions import (
    PermissionContext,
    PermissionMode,
    PermissionResult,
    PermissionRule,
    check_permission,
)
from morgan.tools.builtin.calculator import CalculatorTool, safe_eval
from morgan.tools.builtin.file_read import FileReadTool
from morgan.tools.builtin.bash_tool import BashTool, is_command_blocked
from morgan.tools.builtin.web_search import WebSearchTool
from morgan.tools.builtin.memory_search import MemorySearchTool
from morgan.tools.builtin import ALL_BUILTIN_TOOLS


# =============================================================================
# BaseTool ABC Tests
# =============================================================================


class TestBaseTool:
    """Test that BaseTool is a proper ABC."""

    def test_cannot_instantiate_directly(self):
        """BaseTool cannot be instantiated because execute is abstract."""
        with pytest.raises(TypeError):
            BaseTool()

    def test_concrete_subclass_can_be_instantiated(self):
        """A subclass that implements execute can be created."""

        class ConcreteTool(BaseTool):
            name = "test_tool"
            description = "A test tool"

            async def execute(self, input_data, context):
                return ToolResult(output="ok")

        tool = ConcreteTool()
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"

    def test_get_all_names_includes_aliases(self):
        """get_all_names returns name plus aliases."""

        class ConcreteTool(BaseTool):
            name = "my_tool"
            aliases = ("alias1", "alias2")

            async def execute(self, input_data, context):
                return ToolResult(output="ok")

        tool = ConcreteTool()
        names = tool.get_all_names()
        assert names == ["my_tool", "alias1", "alias2"]

    def test_validate_input_checks_required_fields(self):
        """Default validate_input checks required fields."""

        class ConcreteTool(BaseTool):
            name = "test"
            input_schema = ToolInputSchema(
                properties={"name": {"type": "string"}},
                required=("name",),
            )

            async def execute(self, input_data, context):
                return ToolResult(output="ok")

        tool = ConcreteTool()
        error = tool.validate_input({})
        assert error is not None
        assert "name" in error
        assert tool.validate_input({"name": "test"}) is None

    def test_validate_input_rejects_unknown_fields(self):
        """Default validate_input rejects unknown fields."""

        class ConcreteTool(BaseTool):
            name = "test"
            input_schema = ToolInputSchema(
                properties={"name": {"type": "string"}},
                required=("name",),
                additional_properties=False,
            )

            async def execute(self, input_data, context):
                return ToolResult(output="ok")

        tool = ConcreteTool()
        error = tool.validate_input({"name": "test", "extra": "field"})
        assert error is not None
        assert "extra" in error

    def test_validate_input_allows_extra_when_configured(self):
        """additional_properties=True allows unknown fields."""

        class ConcreteTool(BaseTool):
            name = "test"
            input_schema = ToolInputSchema(
                properties={"name": {"type": "string"}},
                required=("name",),
                additional_properties=True,
            )

            async def execute(self, input_data, context):
                return ToolResult(output="ok")

        tool = ConcreteTool()
        assert tool.validate_input({"name": "test", "extra": "field"}) is None


# =============================================================================
# ToolInputSchema Tests
# =============================================================================


class TestToolInputSchema:
    """Test ToolInputSchema frozen dataclass."""

    def test_is_frozen(self):
        """ToolInputSchema instances cannot be mutated."""
        schema = ToolInputSchema(
            properties={"x": {"type": "string"}},
            required=("x",),
        )
        with pytest.raises(FrozenInstanceError):
            schema.additional_properties = True

    def test_to_dict(self):
        """to_dict returns a proper JSON-serializable dict."""
        schema = ToolInputSchema(
            properties={"expr": {"type": "string", "description": "Expression"}},
            required=("expr",),
            additional_properties=False,
        )
        d = schema.to_dict()
        assert d["type"] == "object"
        assert "expr" in d["properties"]
        assert d["required"] == ["expr"]
        assert d["additionalProperties"] is False

    def test_default_values(self):
        """Defaults are empty properties, no required, no additional props."""
        schema = ToolInputSchema()
        assert schema.properties == {}
        assert schema.required == ()
        assert schema.additional_properties is False


# =============================================================================
# ToolResult Tests
# =============================================================================


class TestToolResult:
    """Test ToolResult dataclass."""

    def test_default_values(self):
        """Default ToolResult is a successful empty result."""
        result = ToolResult()
        assert result.output == ""
        assert result.is_error is False
        assert result.error_code is None
        assert result.metadata == {}

    def test_error_result(self):
        """Error results carry error info."""
        result = ToolResult(
            output="Something went wrong",
            is_error=True,
            error_code="SOME_ERROR",
            metadata={"detail": "info"},
        )
        assert result.is_error is True
        assert result.error_code == "SOME_ERROR"
        assert result.metadata["detail"] == "info"


# =============================================================================
# ToolContext Tests
# =============================================================================


class TestToolContext:
    """Test ToolContext dataclass."""

    def test_default_values(self):
        """Default ToolContext has sensible defaults."""
        ctx = ToolContext()
        assert ctx.user_id == "default"
        assert ctx.conversation_id == "default"
        assert ctx.working_directory == "."
        assert ctx.session_type == SessionType.INTERACTIVE
        assert ctx.environment == {}
        assert ctx.abort_signal is None

    def test_custom_values(self):
        """ToolContext accepts custom values."""
        event = asyncio.Event()
        ctx = ToolContext(
            user_id="user1",
            conversation_id="conv123",
            working_directory="/tmp",
            session_type=SessionType.BACKGROUND,
            environment={"KEY": "value"},
            abort_signal=event,
        )
        assert ctx.user_id == "user1"
        assert ctx.working_directory == "/tmp"
        assert ctx.abort_signal is event


# =============================================================================
# Permission Tests
# =============================================================================


class TestPermissionMode:
    """Test PermissionMode enum."""

    def test_enum_values(self):
        """All four modes exist."""
        assert PermissionMode.DEFAULT == "default"
        assert PermissionMode.PLAN == "plan"
        assert PermissionMode.BYPASS == "bypass"
        assert PermissionMode.AUTO == "auto"


class TestCheckPermission:
    """Test the check_permission function."""

    def test_bypass_always_allows(self):
        """BYPASS mode allows any tool."""
        ctx = PermissionContext(mode=PermissionMode.BYPASS)
        result = check_permission("dangerous_tool", ctx)
        assert result.allowed is True

    def test_auto_always_allows(self):
        """AUTO mode allows any tool."""
        ctx = PermissionContext(mode=PermissionMode.AUTO)
        result = check_permission("anything", ctx)
        assert result.allowed is True

    def test_explicit_deny_list(self):
        """Explicit deny list blocks tools."""
        ctx = PermissionContext(
            mode=PermissionMode.DEFAULT,
            denied_tools=["bash"],
        )
        result = check_permission("bash", ctx)
        assert result.allowed is False
        assert "deny list" in result.reason

    def test_explicit_allow_list(self):
        """Explicit allow list permits tools."""
        ctx = PermissionContext(
            mode=PermissionMode.DEFAULT,
            allowed_tools=["calculator"],
        )
        result = check_permission("calculator", ctx)
        assert result.allowed is True
        assert "allow list" in result.reason

    def test_deny_first_resolution(self):
        """Deny rules are checked before allow rules."""
        ctx = PermissionContext(
            mode=PermissionMode.DEFAULT,
            rules=[
                PermissionRule(tool_pattern="bash*", allowed=True),
                PermissionRule(tool_pattern="bash*", allowed=False, reason="No bash"),
            ],
        )
        result = check_permission("bash", ctx)
        assert result.allowed is False
        assert "No bash" in result.reason

    def test_fnmatch_pattern(self):
        """fnmatch patterns work for matching."""
        ctx = PermissionContext(
            mode=PermissionMode.DEFAULT,
            rules=[
                PermissionRule(tool_pattern="file_*", allowed=True),
            ],
        )
        result = check_permission("file_read", ctx)
        assert result.allowed is True

        result2 = check_permission("bash", ctx)
        assert result2.allowed is False

    def test_plan_mode_denies_unmatched(self):
        """PLAN mode denies tools without explicit allow rule."""
        ctx = PermissionContext(
            mode=PermissionMode.PLAN,
            rules=[
                PermissionRule(tool_pattern="calculator", allowed=True),
            ],
        )
        result = check_permission("bash", ctx)
        assert result.allowed is False
        assert "PLAN" in result.reason

    def test_default_mode_denies_unmatched(self):
        """DEFAULT mode denies when no rules match."""
        ctx = PermissionContext(mode=PermissionMode.DEFAULT)
        result = check_permission("any_tool", ctx)
        assert result.allowed is False

    def test_deny_overrides_allow(self):
        """If both deny and allow patterns match, deny wins."""
        ctx = PermissionContext(
            mode=PermissionMode.DEFAULT,
            rules=[
                PermissionRule(tool_pattern="*", allowed=True),
                PermissionRule(tool_pattern="bash*", allowed=False),
            ],
        )
        result = check_permission("bash", ctx)
        assert result.allowed is False
        result2 = check_permission("calculator", ctx)
        assert result2.allowed is True

    def test_explicit_deny_beats_explicit_allow(self):
        """Explicit denied_tools is checked before allowed_tools."""
        ctx = PermissionContext(
            mode=PermissionMode.DEFAULT,
            allowed_tools=["bash"],
            denied_tools=["bash"],
        )
        result = check_permission("bash", ctx)
        assert result.allowed is False


# =============================================================================
# Calculator Tool Tests
# =============================================================================


class TestCalculatorTool:
    """Test the safe calculator tool."""

    @pytest.fixture
    def calc(self):
        return CalculatorTool()

    @pytest.fixture
    def ctx(self):
        return ToolContext()

    def test_safe_eval_basic_math(self):
        """Basic arithmetic works."""
        assert safe_eval("2 + 2") == 4
        assert safe_eval("10 - 3") == 7
        assert safe_eval("6 * 7") == 42
        assert safe_eval("15 / 4") == 3.75
        assert safe_eval("15 // 4") == 3
        assert safe_eval("15 % 4") == 3

    def test_safe_eval_exponentiation(self):
        """Power operator works."""
        assert safe_eval("2 ** 10") == 1024

    def test_safe_eval_math_functions(self):
        """Math functions work."""
        assert safe_eval("sqrt(144)") == 12.0
        assert abs(safe_eval("sin(pi / 2)") - 1.0) < 1e-10
        assert safe_eval("factorial(5)") == 120
        assert safe_eval("abs(-42)") == 42
        assert safe_eval("max(1, 2, 3)") == 3

    def test_safe_eval_constants(self):
        """Constants are available."""
        assert safe_eval("pi") == math.pi
        assert safe_eval("e") == math.e

    def test_safe_eval_comparisons(self):
        """Comparisons work."""
        assert safe_eval("2 > 1") is True
        assert safe_eval("1 == 2") is False

    def test_safe_eval_rejects_strings(self):
        """String constants are rejected."""
        with pytest.raises(ValueError, match="Unsupported constant"):
            safe_eval("'hello'")

    def test_safe_eval_rejects_unknown_names(self):
        """Unknown variable names are rejected."""
        with pytest.raises(ValueError, match="Unknown name"):
            safe_eval("os")

    def test_safe_eval_large_exponent_blocked(self):
        """Extremely large exponents are blocked."""
        with pytest.raises(ValueError, match="Exponent too large"):
            safe_eval("2 ** 100000")

    def test_safe_eval_division_by_zero(self):
        """Division by zero raises ZeroDivisionError."""
        with pytest.raises(ZeroDivisionError):
            safe_eval("1 / 0")

    @pytest.mark.asyncio
    async def test_calculator_tool_execute(self, calc, ctx):
        """CalculatorTool.execute returns correct result."""
        result = await calc.execute({"expression": "2 + 3"}, ctx)
        assert result.output == "5"
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_calculator_tool_execute_error(self, calc, ctx):
        """CalculatorTool.execute handles errors gracefully."""
        result = await calc.execute({"expression": "1 / 0"}, ctx)
        assert result.is_error is True
        assert result.error_code == "CALCULATION_ERROR"

    def test_calculator_validation_empty(self, calc):
        """Empty expression is rejected."""
        error = calc.validate_input({"expression": ""})
        assert error is not None

    def test_calculator_validation_missing(self, calc):
        """Missing expression is rejected."""
        error = calc.validate_input({})
        assert error is not None

    def test_calculator_has_aliases(self, calc):
        """Calculator has calc and math aliases."""
        assert "calc" in calc.aliases
        assert "math" in calc.aliases

    def test_calculator_uses_ast_parse(self):
        """Verify the calculator uses ast.parse for safe evaluation."""
        import inspect

        source = inspect.getsource(safe_eval)
        assert "ast.parse" in source


# =============================================================================
# FileRead Tool Tests
# =============================================================================


class TestFileReadTool:
    """Test the file read tool."""

    @pytest.fixture
    def file_read(self):
        return FileReadTool()

    @pytest.fixture
    def ctx(self):
        return ToolContext()

    @pytest.fixture
    def sample_file(self):
        """Create a temporary file with known content."""
        content = "\n".join(f"Line {i}" for i in range(1, 11))
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(content)
            path = f.name
        yield path
        os.unlink(path)

    @pytest.mark.asyncio
    async def test_read_full_file(self, file_read, ctx, sample_file):
        """Read an entire file."""
        result = await file_read.execute({"file_path": sample_file}, ctx)
        assert result.is_error is False
        assert "Line 1" in result.output
        assert "Line 10" in result.output
        assert result.metadata["total_lines"] == 10

    @pytest.mark.asyncio
    async def test_read_with_offset_and_limit(self, file_read, ctx, sample_file):
        """Read file with offset and limit."""
        result = await file_read.execute(
            {"file_path": sample_file, "offset": 3, "limit": 2}, ctx
        )
        assert result.is_error is False
        assert "Line 3" in result.output
        assert "Line 4" in result.output
        assert "Line 5" not in result.output
        assert result.metadata["lines_shown"] == 2

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, file_read, ctx):
        """Reading a nonexistent file returns an error."""
        result = await file_read.execute(
            {"file_path": "/tmp/nonexistent_file_12345.txt"}, ctx
        )
        assert result.is_error is True
        assert result.error_code == "FILE_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_read_line_numbers(self, file_read, ctx, sample_file):
        """Output includes line numbers with tab separator."""
        result = await file_read.execute({"file_path": sample_file}, ctx)
        lines = result.output.split("\n")
        assert lines[0].startswith("1\t")

    def test_validation_missing_path(self, file_read):
        """Missing file_path is rejected."""
        error = file_read.validate_input({})
        assert error is not None

    def test_validation_invalid_offset(self, file_read):
        """Invalid offset is rejected."""
        error = file_read.validate_input({"file_path": "test.txt", "offset": 0})
        assert error is not None

    def test_has_aliases(self, file_read):
        """FileReadTool has read and cat aliases."""
        assert "read" in file_read.aliases
        assert "cat" in file_read.aliases


# =============================================================================
# Bash Tool Tests
# =============================================================================


class TestBashTool:
    """Test the bash command execution tool."""

    @pytest.fixture
    def bash(self):
        return BashTool()

    @pytest.fixture
    def ctx(self):
        return ToolContext(working_directory="/tmp")

    def test_blocked_rm_rf_root(self):
        """rm -rf / is blocked."""
        assert is_command_blocked("rm -rf /") is not None
        assert is_command_blocked("rm -rf /home") is None
        assert is_command_blocked("rm -fr /") is not None

    def test_blocked_mkfs(self):
        """mkfs is blocked."""
        assert is_command_blocked("mkfs.ext4 /dev/sda1") is not None

    def test_blocked_fork_bomb(self):
        """Fork bomb pattern is blocked."""
        assert is_command_blocked(":(){ :|:& };:") is not None

    def test_blocked_dd_dev_zero(self):
        """dd if=/dev/zero is blocked."""
        assert is_command_blocked("dd if=/dev/zero of=/dev/sda") is not None

    def test_safe_commands_not_blocked(self):
        """Normal commands are not blocked."""
        assert is_command_blocked("ls -la") is None
        assert is_command_blocked("echo hello") is None
        assert is_command_blocked("python3 --version") is None
        assert is_command_blocked("cat /etc/hostname") is None

    @pytest.mark.asyncio
    async def test_bash_simple_command(self, bash, ctx):
        """A simple command runs successfully."""
        result = await bash.execute({"command": "echo hello"}, ctx)
        assert result.is_error is False
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_bash_nonzero_exit(self, bash, ctx):
        """Non-zero exit code is flagged as error."""
        result = await bash.execute({"command": "false"}, ctx)
        assert result.is_error is True
        assert result.error_code == "EXIT_CODE_NONZERO"

    @pytest.mark.asyncio
    async def test_bash_timeout(self, bash, ctx):
        """Command that exceeds timeout is killed."""
        result = await bash.execute(
            {"command": "sleep 60", "timeout": 1}, ctx
        )
        assert result.is_error is True
        assert result.error_code == "TIMEOUT"

    def test_validation_blocked_command(self, bash):
        """Blocked commands fail validation."""
        error = bash.validate_input({"command": "rm -rf /"})
        assert error is not None

    def test_validation_empty_command(self, bash):
        """Empty command is rejected."""
        error = bash.validate_input({"command": ""})
        assert error is not None

    def test_validation_timeout_too_large(self, bash):
        """Timeout exceeding max is rejected."""
        error = bash.validate_input({"command": "ls", "timeout": 99999})
        assert error is not None

    def test_has_aliases(self, bash):
        """BashTool has shell and sh aliases."""
        assert "shell" in bash.aliases
        assert "sh" in bash.aliases


# =============================================================================
# WebSearch Tool Tests
# =============================================================================


class TestWebSearchTool:
    """Test the web search tool."""

    @pytest.fixture
    def web_search(self):
        return WebSearchTool()

    @pytest.fixture
    def ctx(self):
        return ToolContext()

    @pytest.mark.asyncio
    async def test_handles_import_error_gracefully(self, web_search, ctx):
        """WebSearchTool handles missing service gracefully."""
        result = await web_search.execute({"query": "test query"}, ctx)
        assert isinstance(result, ToolResult)
        if result.is_error:
            assert result.error_code in ("SERVICE_UNAVAILABLE", "SEARCH_ERROR")

    def test_validation_empty_query(self, web_search):
        """Empty query is rejected."""
        error = web_search.validate_input({"query": ""})
        assert error is not None

    def test_validation_missing_query(self, web_search):
        """Missing query is rejected."""
        error = web_search.validate_input({})
        assert error is not None

    def test_validation_max_results_limit(self, web_search):
        """max_results exceeding 50 is rejected."""
        error = web_search.validate_input({"query": "test", "max_results": 100})
        assert error is not None

    def test_has_aliases(self, web_search):
        """WebSearchTool has expected aliases."""
        assert "search_web" in web_search.aliases


# =============================================================================
# MemorySearch Tool Tests
# =============================================================================


class TestMemorySearchTool:
    """Test the memory search tool."""

    @pytest.fixture
    def memory_search(self):
        return MemorySearchTool()

    @pytest.fixture
    def ctx(self):
        return ToolContext()

    @pytest.mark.asyncio
    async def test_handles_import_error_gracefully(self, memory_search, ctx):
        """MemorySearchTool handles missing service gracefully."""
        result = await memory_search.execute({"query": "test"}, ctx)
        assert isinstance(result, ToolResult)
        if result.is_error:
            assert result.error_code in (
                "SERVICE_UNAVAILABLE",
                "SEARCH_ERROR",
                "SERVICE_INCOMPATIBLE",
            )

    def test_validation_empty_query(self, memory_search):
        """Empty query is rejected."""
        error = memory_search.validate_input({"query": ""})
        assert error is not None

    def test_validation_max_results_limit(self, memory_search):
        """max_results exceeding 100 is rejected."""
        error = memory_search.validate_input({"query": "test", "max_results": 200})
        assert error is not None

    def test_has_aliases(self, memory_search):
        """MemorySearchTool has expected aliases."""
        assert "recall" in memory_search.aliases


# =============================================================================
# ALL_BUILTIN_TOOLS Tests
# =============================================================================


class TestAllBuiltinTools:
    """Test the ALL_BUILTIN_TOOLS list."""

    def test_contains_all_tools(self):
        """ALL_BUILTIN_TOOLS has all 5 built-in tools."""
        assert len(ALL_BUILTIN_TOOLS) == 5
        names = {t.name for t in ALL_BUILTIN_TOOLS}
        assert names == {
            "calculator",
            "file_read",
            "bash",
            "web_search",
            "memory_search",
        }

    def test_all_are_base_tool_instances(self):
        """All entries are BaseTool instances."""
        for tool in ALL_BUILTIN_TOOLS:
            assert isinstance(tool, BaseTool)

    def test_all_have_names(self):
        """All tools have non-empty names."""
        for tool in ALL_BUILTIN_TOOLS:
            assert tool.name, f"Tool {tool} has no name"

    def test_no_duplicate_names(self):
        """No tool name or alias collisions."""
        seen = set()
        for tool in ALL_BUILTIN_TOOLS:
            for name in tool.get_all_names():
                assert name not in seen, f"Duplicate name/alias: {name}"
                seen.add(name)
