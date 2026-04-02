"""
Local URL fetch and content extraction tool.

Downloads a web page and extracts readable text content locally.
No external APIs — uses httpx + HTML parsing. Privacy-first.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from morgan.tools.base import BaseTool, ToolContext, ToolInputSchema, ToolResult

logger = logging.getLogger(__name__)


class FetchURLTool(BaseTool):
    """Fetch a URL and extract readable text content locally.

    Downloads the page with httpx, strips HTML tags, and returns
    clean text. No external services or APIs used.
    """

    name = "fetch_url"
    description = (
        "Fetch a web page URL and extract its text content locally. "
        "Use this to read articles, documentation, blog posts, or any web page. "
        "Returns clean text without HTML tags."
    )
    aliases = ("read_url", "get_page", "web_fetch")
    input_schema = ToolInputSchema(
        properties={
            "url": {
                "type": "string",
                "description": "The URL to fetch (must start with http:// or https://)",
            },
            "max_length": {
                "type": "integer",
                "description": "Maximum characters to return (default 8000)",
            },
        },
        required=("url",),
    )

    def validate_input(self, input_data: Dict[str, Any]) -> Optional[str]:
        base_error = super().validate_input(input_data)
        if base_error:
            return base_error
        url = input_data.get("url", "")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            return "url must start with http:// or https://"
        return None

    async def execute(self, input_data: Dict[str, Any], context: ToolContext) -> ToolResult:
        url = input_data["url"]
        max_length = input_data.get("max_length", 8000)

        try:
            import httpx
        except ImportError:
            return ToolResult(
                output="httpx not installed",
                is_error=True,
                error_code="DEPENDENCY_MISSING",
            )

        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; MorganBot/1.0; "
                    "+https://github.com/anthropics/claude-code)"
                ),
                "Accept": "text/html,application/xhtml+xml,text/plain,application/json",
                "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
            }
            async with httpx.AsyncClient(
                timeout=20.0, follow_redirects=True, max_redirects=5
            ) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")
                raw = resp.text

            # JSON response — return as-is (truncated)
            if "json" in content_type:
                text = raw[:max_length]
                return ToolResult(
                    output=text,
                    metadata={"url": url, "content_type": content_type, "chars": len(text)},
                )

            # HTML — extract text
            text = self._extract_text_from_html(raw)
            if len(text) > max_length:
                text = text[:max_length] + "\n\n[... truncated]"

            if not text.strip():
                return ToolResult(
                    output=f"Page fetched but no readable text extracted from {url}",
                    metadata={"url": url, "raw_length": len(raw)},
                )

            return ToolResult(
                output=text,
                metadata={
                    "url": url,
                    "content_type": content_type,
                    "chars": len(text),
                },
            )

        except httpx.HTTPStatusError as e:
            return ToolResult(
                output=f"HTTP {e.response.status_code} fetching {url}",
                is_error=True,
                error_code="HTTP_ERROR",
            )
        except Exception as e:
            return ToolResult(
                output=f"Failed to fetch {url}: {e}",
                is_error=True,
                error_code="FETCH_ERROR",
            )

    @staticmethod
    def _extract_text_from_html(html: str) -> str:
        """Extract readable text from HTML, removing scripts/styles/tags."""
        # Remove script and style blocks
        text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<nav[^>]*>.*?</nav>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<footer[^>]*>.*?</footer>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<header[^>]*>.*?</header>", " ", text, flags=re.DOTALL | re.IGNORECASE)

        # Convert common block elements to newlines
        text = re.sub(r"<(?:p|div|br|h[1-6]|li|tr|td|th|blockquote)[^>]*>", "\n", text, flags=re.IGNORECASE)

        # Remove all remaining tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Decode common HTML entities
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        text = text.replace("&#39;", "'")
        text = text.replace("&nbsp;", " ")
        text = re.sub(r"&#\d+;", " ", text)
        text = re.sub(r"&\w+;", " ", text)

        # Collapse whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)

        return text.strip()
