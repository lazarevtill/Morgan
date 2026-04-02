"""
Memory search tool for the Morgan tool system.

Delegates to morgan.memory_consolidation.HybridMemorySearch, combining
keyword (BM25-like) search with optional vector scoring over daily logs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from morgan.tools.base import BaseTool, ToolContext, ToolInputSchema, ToolResult


class MemorySearchTool(BaseTool):
    """
    Memory search tool that delegates to HybridMemorySearch.

    Examples:
        {"query": "user's favorite programming language"}
        {"query": "previous conversation about travel", "max_results": 10}
    """

    name = "memory_search"
    description = "Search conversation memories and stored knowledge"
    aliases = ("search_memory", "recall")
    input_schema = ToolInputSchema(
        properties={
            "query": {
                "type": "string",
                "description": "Search query for memories",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5)",
            },
            "conversation_id": {
                "type": "string",
                "description": "Optional conversation id (reserved for future filtering)",
            },
        },
        required=("query",),
    )

    def validate_input(self, input_data: Dict[str, Any]) -> Optional[str]:
        """Validate memory search input."""
        base_error = super().validate_input(input_data)
        if base_error:
            return base_error

        query = input_data.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return "query must be a non-empty string"

        max_results = input_data.get("max_results", 5)
        if not isinstance(max_results, int) or max_results < 1:
            return "max_results must be a positive integer"
        if max_results > 100:
            return "max_results cannot exceed 100"

        return None

    async def execute(
        self, input_data: Dict[str, Any], context: ToolContext
    ) -> ToolResult:
        """Execute a memory search via HybridMemorySearch."""
        query = input_data["query"].strip()
        max_results = input_data.get("max_results", 5)
        conversation_id = input_data.get("conversation_id", context.conversation_id)

        try:
            from morgan.config import get_settings
            from morgan.memory_consolidation import HybridMemorySearch
            from morgan.workspace import get_workspace_path

            settings = get_settings()
            workspace_dir = (
                Path(settings.morgan_workspace_path)
                if settings.morgan_workspace_path
                else get_workspace_path()
            )
            memory_dir = workspace_dir / "memory"

            search = HybridMemorySearch(memory_dir)
            results = search.hybrid_search(query=query, limit=max_results)

            if not results:
                return ToolResult(
                    output=f"No memories found for: {query}",
                    metadata={
                        "query": query,
                        "result_count": 0,
                        "conversation_id": conversation_id,
                    },
                )

            output_lines = [f"Memory search results for: {query}\n"]
            for i, memory in enumerate(results, 1):
                content = memory.get("content", "")
                score = memory.get("score")
                path = memory.get("path", "")
                date_hint = Path(path).stem if path else "unknown-date"
                score_text = f" (score: {score:.2f})" if isinstance(score, (int, float)) else ""

                output_lines.append(f"{i}. [{date_hint}] {content.strip()}{score_text}")
                output_lines.append("")

            return ToolResult(
                output="\n".join(output_lines).rstrip(),
                metadata={
                    "query": query,
                    "result_count": len(results),
                    "conversation_id": conversation_id,
                    "workspace_memory_dir": str(memory_dir),
                },
            )

        except ImportError as e:
            return ToolResult(
                output="Memory search service is not available. "
                "The memory consolidation modules could not be imported.",
                is_error=True,
                error_code="SERVICE_UNAVAILABLE",
                metadata={"query": query, "details": str(e)},
            )
        except Exception as e:
            return ToolResult(
                output=f"Memory search failed: {type(e).__name__}: {e}",
                is_error=True,
                error_code="SEARCH_ERROR",
                metadata={"query": query},
            )
