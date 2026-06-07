"""Tools contract — pluggable, permission-gated execution. There is exactly one permission
model in the system (see security/permissions.py): one PermissionMode enum, one gate.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    ok: bool = True
    output: Any = None
    error: str | None = None


@runtime_checkable
class BaseTool(Protocol):
    name: str
    description: str

    def schema(self) -> dict[str, Any]: ...

    async def run(self, *, user_id: str, **kwargs: Any) -> ToolResult: ...


@runtime_checkable
class ToolExecutor(Protocol):
    def register(self, tool: BaseTool) -> None: ...

    def list(self) -> list[dict[str, Any]]:
        """Tool names + schemas."""
        ...

    async def execute(self, name: str, *, user_id: str, **kwargs: Any) -> ToolResult:
        """Permission-checked execution."""
        ...
