"""
Memory search tool for the Morgan tool system.

Delegates to morgan.memory.memory_processor for searching stored memories.
Handles ImportError gracefully if the service is not available.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from morgan.tools.base import BaseTool, ToolContext, ToolInputSchema, ToolResult


class MemorySearchTool(BaseTool):
    """
    Memory search tool that delegates to Morgan's memory processor.

    Searches conversation memories and stored knowledge through the
    existing morgan.memory module. If the service is not available
    (ImportError), returns a graceful error.

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
                "description": "Optional: limit search to a specific conversation",
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
        """Execute a memory search via the memory processor."""
        query = input_data["query"].strip()
        max_results = input_data.get("max_results", 5)
        conversation_id = input_data.get("conversation_id", context.conversation_id)

        try:
            from morgan.memory import get_memory_processor

            processor = get_memory_processor()

            # The memory processor may have different search method signatures
            # depending on the version. We try the most common patterns.
            if hasattr(processor, "search_memories"):
                results = await processor.search_memories(
                    query=query,
                    conversation_id=conversation_id,
                    max_results=max_results,
                )
            elif hasattr(processor, "search"):
                results = await processor.search(
                    query=query,
                    limit=max_results,
                )
            else:
                return ToolResult(
                    output="Memory processor does not have a search method",
                    is_error=True,
                    error_code="SERVICE_INCOMPATIBLE",
                    metadata={"query": query},
                )

            if not results:
                return ToolResult(
                    output=f"No memories found for: {query}",
                    metadata={"query": query, "result_count": 0},
                )

            # Format results
            output_lines = [f"Memory search results for: {query}\n"]
            for i, memory in enumerate(results, 1):
                if hasattr(memory, "content"):
                    content = memory.content
                elif isinstance(memory, dict):
                    content = memory.get("content", str(memory))
                else:
                    content = str(memory)

                score = ""
                if hasattr(memory, "importance_score"):
                    score = f" (importance: {memory.importance_score:.2f})"
                elif isinstance(memory, dict) and "score" in memory:
                    score = f" (score: {memory['score']:.2f})"

                output_lines.append(f"{i}. {content}{score}")
                output_lines.append("")

            return ToolResult(
                output="\n".join(output_lines).rstrip(),
                metadata={
                    "query": query,
                    "result_count": len(results),
                    "conversation_id": conversation_id,
                },
            )

        except ImportError:
            return ToolResult(
                output="Memory search service is not available. "
                "The morgan.memory module could not be imported.",
                is_error=True,
                error_code="SERVICE_UNAVAILABLE",
                metadata={"query": query},
            )
        except Exception as e:
            return ToolResult(
                output=f"Memory search failed: {type(e).__name__}: {e}",
                is_error=True,
                error_code="SEARCH_ERROR",
                metadata={"query": query},
            )
