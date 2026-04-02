"""
Base tool abstractions for the Morgan tool system.

Provides the foundational ABC and data structures that all tools must implement.
Ported from Claude Code's Tool.ts pattern, adapted for Python async/await.

Usage:
    from morgan.tools.base import BaseTool, ToolResult, ToolContext, ToolInputSchema

    class MyTool(BaseTool):
        name = "my_tool"
        description = "Does something useful"

        async def execute(self, input_data: dict, context: ToolContext) -> ToolResult:
            return ToolResult(output="done")
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ToolInputSchema:
    """
    Describes the expected input schema for a tool.

    Frozen dataclass to ensure immutability after creation.
    Tools declare their input schema so the executor can validate
    before calling execute().

    Attributes:
        properties: Mapping of field name to type/description metadata.
        required: List of required field names.
        additional_properties: Whether extra fields are allowed.
    """

    properties: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    additional_properties: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to a JSON-serializable dictionary."""
        return {
            "type": "object",
            "properties": dict(self.properties),
            "required": list(self.required),
            "additionalProperties": self.additional_properties,
        }


class SessionType(str, Enum):
    """Type of session the tool is running in."""

    INTERACTIVE = "interactive"
    BACKGROUND = "background"
    API = "api"


@dataclass
class ToolContext:
    """
    Runtime context passed to every tool execution.

    Provides information about the current user, conversation, and environment
    so tools can make context-aware decisions.

    Attributes:
        user_id: Identifier of the user invoking the tool.
        conversation_id: Current conversation identifier.
        working_directory: Filesystem directory to use as CWD for file operations.
        session_type: Type of session (interactive, background, api).
        environment: Arbitrary environment key-value pairs.
        abort_signal: An asyncio.Event that, when set, signals the tool should abort.
    """

    user_id: str = "default"
    conversation_id: str = "default"
    working_directory: str = "."
    session_type: SessionType = SessionType.INTERACTIVE
    environment: Dict[str, str] = field(default_factory=dict)
    abort_signal: Optional[asyncio.Event] = None


@dataclass
class ToolResult:
    """
    The result returned by a tool execution.

    Attributes:
        output: The primary output string from the tool.
        is_error: Whether the execution resulted in an error.
        error_code: Machine-readable error code (e.g., "PERMISSION_DENIED").
        metadata: Additional structured data about the execution.
    """

    output: str = ""
    is_error: bool = False
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """
    Abstract base class for all Morgan tools.

    Every tool must define a name, description, and implement execute().
    Optionally, tools can declare aliases, an input_schema, and
    override validate_input() for custom validation.

    Cannot be instantiated directly — subclasses must implement execute().

    Example:
        class CalculatorTool(BaseTool):
            name = "calculator"
            description = "Evaluate safe math expressions"
            aliases = ("calc", "math")

            async def execute(self, input_data, context):
                ...
    """

    name: str = ""
    description: str = ""
    aliases: tuple[str, ...] = ()
    input_schema: ToolInputSchema = ToolInputSchema()

    @abstractmethod
    async def execute(
        self, input_data: Dict[str, Any], context: ToolContext
    ) -> ToolResult:
        """
        Execute the tool with the given input and context.

        Args:
            input_data: Validated input dictionary.
            context: Runtime context for the execution.

        Returns:
            ToolResult with the execution outcome.
        """
        ...

    def validate_input(self, input_data: Dict[str, Any]) -> Optional[str]:
        """
        Validate tool input against the schema.

        Returns None if valid, or an error message string if invalid.
        The default implementation checks required fields and property names.
        Subclasses can override for custom validation.

        Args:
            input_data: The input dictionary to validate.

        Returns:
            None if valid, error message string if invalid.
        """
        schema = self.input_schema

        # Check required fields
        for req in schema.required:
            if req not in input_data:
                return f"Missing required field: '{req}'"

        # Check for unknown fields if additional properties are not allowed
        if not schema.additional_properties and schema.properties:
            allowed = set(schema.properties.keys())
            unknown = set(input_data.keys()) - allowed
            if unknown:
                return f"Unknown fields: {', '.join(sorted(unknown))}"

        return None

    def get_all_names(self) -> List[str]:
        """Return the tool name plus all aliases."""
        return [self.name] + list(self.aliases)
