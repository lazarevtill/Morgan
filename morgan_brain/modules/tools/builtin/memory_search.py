"""MemorySearchTool — searches the user's memory via the MemoryGate.

Requires a ``MemoryGate`` instance; does not hit any network.
"""

from __future__ import annotations

from typing import Any

from morgan_brain.interfaces.tools import ToolResult
from morgan_brain.models.memory import MemoryQuery
from morgan_brain.security.memory_gate import MemoryGate


class MemorySearchTool:
    """Search the user's memories via the MemoryGate.

    Parameters
    ----------
    gate:
        The ``MemoryGate`` instance to delegate recalls to.
    """

    name = "memory_search"
    description = "Search the user's stored memories by semantic query."

    def __init__(self, gate: MemoryGate) -> None:
        self._gate = gate

    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of memories to return.",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def run(
        self,
        *,
        user_id: str,
        query: str,
        project: str,
        top_k: int = 5,
        **_: Any,
    ) -> ToolResult:
        """Search within *project* only.

        ``project`` is a required keyword supplied by the executor from the turn, never by the
        model: it is deliberately absent from ``schema()`` above, so the assistant cannot ask
        to search a project the caller did not scope the turn to.
        """
        memories = await self._gate.recall(
            MemoryQuery(user_id=user_id, project=project, text=query, top_k=top_k)
        )
        return ToolResult(ok=True, output=[m.content for m in memories])
