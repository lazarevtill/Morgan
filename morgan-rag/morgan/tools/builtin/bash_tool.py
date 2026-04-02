"""
Bash command execution tool for the Morgan tool system.

Runs shell commands via asyncio.create_subprocess_shell with timeout
and blocked pattern protection.

Security: Blocks known-dangerous command patterns before execution.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional

from morgan.tools.base import BaseTool, ToolContext, ToolInputSchema, ToolResult

# Dangerous command patterns to block
# Each is a regex pattern that will be checked against the command string
_BLOCKED_PATTERNS: List[str] = [
    r"rm\s+-rf\s+/(?:\s|$)",        # rm -rf /
    r"rm\s+-fr\s+/(?:\s|$)",        # rm -fr /
    r"mkfs\b",                       # mkfs (format filesystem)
    r":\(\)\s*\{",                   # fork bomb :(){ :|:& };:
    r"dd\s+if=/dev/zero",           # dd if=/dev/zero (disk wipe)
    r"dd\s+if=/dev/random",         # dd if=/dev/random
    r"dd\s+if=/dev/urandom",        # dd if=/dev/urandom
    r">\s*/dev/sd[a-z]",            # write to raw disk device
    r"chmod\s+-R\s+777\s+/(?:\s|$)", # chmod -R 777 /
    r"chown\s+-R\s+.*\s+/(?:\s|$)", # chown -R ... /
]

# Compile patterns for performance
_COMPILED_BLOCKED = [re.compile(p) for p in _BLOCKED_PATTERNS]


def is_command_blocked(command: str) -> Optional[str]:
    """
    Check if a command matches any blocked pattern.

    Args:
        command: The shell command string to check.

    Returns:
        The matched pattern string if blocked, None if safe.
    """
    for pattern, compiled in zip(_BLOCKED_PATTERNS, _COMPILED_BLOCKED):
        if compiled.search(command):
            return pattern
    return None


class BashTool(BaseTool):
    """
    Execute shell commands via asyncio subprocess.

    Runs commands with configurable timeout and blocked pattern protection.
    Captures stdout and stderr, returns combined output.

    Security:
    - Blocks known-dangerous commands (rm -rf /, mkfs, fork bombs, etc.)
    - Enforces execution timeout (default 120 seconds)
    - Respects abort signals from the tool context

    Examples:
        {"command": "ls -la"}
        {"command": "python3 --version", "timeout": 10}
    """

    name = "bash"
    description = "Execute a shell command with timeout and safety checks"
    aliases = ("shell", "sh")
    input_schema = ToolInputSchema(
        properties={
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 120, max 600)",
            },
        },
        required=("command",),
    )

    DEFAULT_TIMEOUT = 120
    MAX_TIMEOUT = 600

    def validate_input(self, input_data: Dict[str, Any]) -> Optional[str]:
        """Validate bash tool input."""
        base_error = super().validate_input(input_data)
        if base_error:
            return base_error

        command = input_data.get("command", "")
        if not isinstance(command, str) or not command.strip():
            return "command must be a non-empty string"

        timeout = input_data.get("timeout", self.DEFAULT_TIMEOUT)
        if not isinstance(timeout, (int, float)):
            return "timeout must be a number"
        if timeout <= 0:
            return "timeout must be positive"
        if timeout > self.MAX_TIMEOUT:
            return f"timeout cannot exceed {self.MAX_TIMEOUT} seconds"

        # Check for blocked patterns
        blocked = is_command_blocked(command)
        if blocked:
            return f"Command blocked by safety rule: {blocked}"

        return None

    async def execute(
        self, input_data: Dict[str, Any], context: ToolContext
    ) -> ToolResult:
        """Execute the shell command."""
        command = input_data["command"].strip()
        timeout = input_data.get("timeout", self.DEFAULT_TIMEOUT)

        # Double-check blocked patterns (defense in depth)
        blocked = is_command_blocked(command)
        if blocked:
            return ToolResult(
                output=f"Command blocked by safety rule: {blocked}",
                is_error=True,
                error_code="COMMAND_BLOCKED",
            )

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.working_directory,
            )

            # Wait with timeout, also check abort signal
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                # Kill the process on timeout
                try:
                    process.kill()
                    await process.wait()
                except ProcessLookupError:
                    pass
                return ToolResult(
                    output=f"Command timed out after {timeout} seconds",
                    is_error=True,
                    error_code="TIMEOUT",
                    metadata={"command": command, "timeout": timeout},
                )

            stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

            # Combine output
            output_parts = []
            if stdout_str:
                output_parts.append(stdout_str)
            if stderr_str:
                output_parts.append(stderr_str)

            output = "\n".join(output_parts).rstrip()

            return_code = process.returncode or 0
            is_error = return_code != 0

            return ToolResult(
                output=output if output else "(no output)",
                is_error=is_error,
                error_code="EXIT_CODE_NONZERO" if is_error else None,
                metadata={
                    "command": command,
                    "return_code": return_code,
                    "timeout": timeout,
                },
            )

        except OSError as e:
            return ToolResult(
                output=f"Failed to execute command: {e}",
                is_error=True,
                error_code="EXECUTION_ERROR",
                metadata={"command": command},
            )
