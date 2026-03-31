"""
Web search tool for the Morgan tool system.

Delegates to morgan.services.external_knowledge for actual web search.
Handles ImportError gracefully if the service is not available.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from morgan.tools.base import BaseTool, ToolContext, ToolInputSchema, ToolResult


class WebSearchTool(BaseTool):
    """
    Web search tool that delegates to Morgan's external knowledge service.

    Provides web search capabilities by calling the existing
    morgan.services.external_knowledge module. If the service is
    not available (ImportError), returns a graceful error.

    Examples:
        {"query": "Python asyncio best practices"}
        {"query": "latest FastAPI release", "max_results": 5}
    """

    name = "web_search"
    description = "Search the web for information"
    aliases = ("search_web", "websearch")
    input_schema = ToolInputSchema(
        properties={
            "query": {
                "type": "string",
                "description": "Search query string",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5)",
            },
        },
        required=("query",),
    )

    def validate_input(self, input_data: Dict[str, Any]) -> Optional[str]:
        """Validate web search input."""
        base_error = super().validate_input(input_data)
        if base_error:
            return base_error

        query = input_data.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return "query must be a non-empty string"

        max_results = input_data.get("max_results", 5)
        if not isinstance(max_results, int) or max_results < 1:
            return "max_results must be a positive integer"
        if max_results > 50:
            return "max_results cannot exceed 50"

        return None

    async def execute(
        self, input_data: Dict[str, Any], context: ToolContext
    ) -> ToolResult:
        """Execute a web search via the external knowledge service."""
        query = input_data["query"].strip()
        max_results = input_data.get("max_results", 5)

        try:
            from morgan.services.external_knowledge import get_web_search_service

            service = get_web_search_service()
            results = await service.search(query, max_results=max_results)

            if not results:
                return ToolResult(
                    output=f"No results found for: {query}",
                    metadata={"query": query, "result_count": 0},
                )

            # Format results
            output_lines = [f"Web search results for: {query}\n"]
            for i, result in enumerate(results, 1):
                output_lines.append(f"{i}. {result.title}")
                output_lines.append(f"   URL: {result.url}")
                output_lines.append(f"   {result.snippet}")
                output_lines.append("")

            return ToolResult(
                output="\n".join(output_lines).rstrip(),
                metadata={
                    "query": query,
                    "result_count": len(results),
                },
            )

        except ImportError:
            return ToolResult(
                output="Web search service is not available. "
                "The morgan.services.external_knowledge module could not be imported.",
                is_error=True,
                error_code="SERVICE_UNAVAILABLE",
                metadata={"query": query},
            )
        except Exception as e:
            return ToolResult(
                output=f"Web search failed: {type(e).__name__}: {e}",
                is_error=True,
                error_code="SEARCH_ERROR",
                metadata={"query": query},
            )
