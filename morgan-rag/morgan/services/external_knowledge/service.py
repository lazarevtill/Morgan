"""
Unified External Knowledge Service for Morgan.

Provides a single interface to all external knowledge sources:
- Web search for real-time information
- Context7 for library documentation
- Combined intelligent search across all sources

This service orchestrates multiple knowledge sources to provide
the most relevant and up-to-date information for any query.
"""

import asyncio
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from morgan.config import get_settings
from morgan.utils.logger import get_logger

from morgan.services.external_knowledge.web_search import (
    WebSearchService,
    get_web_search_service,
)
from morgan.services.external_knowledge.context7 import (
    Context7Service,
    DocumentationMode,
    LibraryInfo,
    get_context7_service,
)

logger = get_logger(__name__)


class KnowledgeSource(str, Enum):
    """Available external knowledge sources."""

    WEB = "web"  # Real-time web search
    CONTEXT7 = "context7"  # Library documentation
    ALL = "all"  # Search all sources


@dataclass
class ExternalKnowledgeResult:
    """Unified result from external knowledge retrieval."""

    content: str
    source: KnowledgeSource
    source_url: Optional[str] = None
    title: Optional[str] = None
    score: float = 1.0
    result_type: str = "external"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Additional context
    library_info: Optional[LibraryInfo] = None
    code_examples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "source": self.source.value,
            "source_url": self.source_url,
            "title": self.title,
            "score": self.score,
            "result_type": self.result_type,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "library_info": (
                self.library_info.to_dict() if self.library_info else None
            ),
            "code_examples": self.code_examples,
        }

    def to_search_result_format(self) -> Dict[str, Any]:
        """
        Convert to format compatible with Morgan's SearchResult.

        This allows external results to be integrated with
        the existing search infrastructure.
        """
        return {
            "content": self.content,
            "source": self.source_url or f"external:{self.source.value}",
            "score": self.score,
            "result_type": self.result_type,
            "metadata": {
                **self.metadata,
                "external_source": self.source.value,
                "title": self.title,
                "code_examples": self.code_examples,
            },
        }


class ExternalKnowledgeService:
    """
    Unified External Knowledge Service.

    Orchestrates web search and Context7 documentation services
    to provide comprehensive external knowledge retrieval.

    Features:
    - Intelligent query routing to appropriate sources
    - Parallel search across multiple sources
    - Result ranking and deduplication
    - Automatic library detection for documentation
    - Caching across all sources

    Example:
        >>> service = ExternalKnowledgeService()
        >>>
        >>> # Search all sources
        >>> results = await service.search(
        ...     "How to implement async middleware in FastAPI",
        ...     sources=KnowledgeSource.ALL
        ... )
        >>>
        >>> # Get documentation only
        >>> docs = await service.get_library_info(
        ...     "fastapi",
        ...     topic="middleware"
        ... )
    """

    # Keywords that indicate documentation lookup
    DOCUMENTATION_KEYWORDS = [
        "how to",
        "tutorial",
        "example",
        "documentation",
        "best practice",
        "pattern",
        "api",
        "reference",
        "guide",
        "setup",
        "configure",
        "implement",
    ]

    # Keywords that indicate web search
    WEB_SEARCH_KEYWORDS = [
        "latest",
        "recent",
        "news",
        "current",
        "today",
        "error",
        "bug",
        "issue",
        "problem",
        "fix",
        "release",
        "update",
        "version",
        "announcement",
    ]

    def __init__(
        self,
        enable_web_search: bool = True,
        enable_context7: bool = True,
    ):
        """
        Initialize external knowledge service.

        Args:
            enable_web_search: Enable web search functionality
            enable_context7: Enable Context7 documentation
        """
        self.settings = get_settings()

        # Initialize sub-services
        self._web_search: Optional[WebSearchService] = None
        self._context7: Optional[Context7Service] = None

        if enable_web_search:
            self._web_search = get_web_search_service()

        if enable_context7:
            self._context7 = get_context7_service()

        # Statistics
        self._stats = {
            "total_queries": 0,
            "web_searches": 0,
            "context7_lookups": 0,
            "combined_searches": 0,
        }

        logger.info(
            f"ExternalKnowledgeService initialized: "
            f"web_search={enable_web_search}, "
            f"context7={enable_context7}"
        )

    async def search(
        self,
        query: str,
        sources: KnowledgeSource = KnowledgeSource.ALL,
        max_results: int = 10,
        include_code_examples: bool = True,
    ) -> List[ExternalKnowledgeResult]:
        """
        Search external knowledge sources.

        Args:
            query: Search query
            sources: Which sources to search
            max_results: Maximum results to return
            include_code_examples: Whether to include code examples

        Returns:
            List of external knowledge results
        """
        self._stats["total_queries"] += 1

        results = []

        if sources == KnowledgeSource.WEB:
            results = await self._search_web(query, max_results)

        elif sources == KnowledgeSource.CONTEXT7:
            results = await self._search_context7(
                query, max_results, include_code_examples
            )

        elif sources == KnowledgeSource.ALL:
            self._stats["combined_searches"] += 1
            results = await self._search_all(query, max_results, include_code_examples)

        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:max_results]

    async def intelligent_search(
        self,
        query: str,
        max_results: int = 10,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExternalKnowledgeResult]:
        """
        Intelligent search that automatically routes to best sources.

        Analyzes the query to determine the best knowledge sources
        and combines results intelligently.

        Args:
            query: Search query
            max_results: Maximum results
            context: Optional context (e.g., detected libraries)

        Returns:
            List of relevant results from appropriate sources
        """
        self._stats["total_queries"] += 1

        # Analyze query
        query_lower = query.lower()

        # Detect if it's a documentation query
        is_docs_query = any(kw in query_lower for kw in self.DOCUMENTATION_KEYWORDS)

        # Detect if it's a web search query
        is_web_query = any(kw in query_lower for kw in self.WEB_SEARCH_KEYWORDS)

        # Detect mentioned libraries
        libraries = self._detect_libraries(query)
        if context and "libraries" in context:
            libraries.extend(context["libraries"])

        results = []

        # If libraries are mentioned and it looks like a docs query
        if libraries and (is_docs_query or not is_web_query):
            for lib in libraries[:3]:  # Limit to 3 libraries
                lib_results = await self._get_library_docs(
                    lib, query, max_results // len(libraries)
                )
                results.extend(lib_results)

        # If it looks like a web query or we need more results
        if is_web_query or len(results) < max_results // 2:
            web_results = await self._search_web(query, max_results - len(results))
            results.extend(web_results)

        # If no results yet, do a broader search
        if not results:
            results = await self._search_all(query, max_results, True)

        # Deduplicate and sort
        results = self._deduplicate_results(results)
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:max_results]

    async def get_library_info(
        self,
        library_name: str,
        topic: Optional[str] = None,
        include_examples: bool = True,
    ) -> Optional[ExternalKnowledgeResult]:
        """
        Get documentation for a specific library.

        Args:
            library_name: Name of the library
            topic: Optional specific topic
            include_examples: Whether to include code examples

        Returns:
            Documentation result or None if not found
        """
        if not self._context7:
            logger.warning("Context7 service not available")
            return None

        self._stats["context7_lookups"] += 1

        # Resolve library
        library = await self._context7.resolve_library(library_name)
        if not library:
            logger.warning(f"Could not resolve library: {library_name}")
            return None

        # Get documentation
        mode = DocumentationMode.CODE if include_examples else DocumentationMode.INFO
        docs = await self._context7.get_documentation(
            library.library_id,
            topic=topic,
            mode=mode,
        )

        if not docs:
            return None

        return ExternalKnowledgeResult(
            content=docs.content,
            source=KnowledgeSource.CONTEXT7,
            source_url=f"context7://{library.library_id}",
            title=f"{library_name} Documentation" + (f": {topic}" if topic else ""),
            score=0.9,
            result_type="documentation",
            library_info=library,
            code_examples=docs.code_examples,
            metadata={
                "library_id": library.library_id,
                "topic": topic,
                "mode": mode.value,
            },
        )

    async def get_best_practices(
        self,
        technology: str,
        topic: Optional[str] = None,
    ) -> List[ExternalKnowledgeResult]:
        """
        Get best practices for a technology.

        Combines Context7 documentation with web search
        for comprehensive best practices.

        Args:
            technology: Technology/library name
            topic: Optional specific topic

        Returns:
            List of best practice results
        """
        results = []

        # Get documentation best practices
        if self._context7:
            docs = await self._context7.get_best_practices(technology, topic)
            if docs:
                results.append(
                    ExternalKnowledgeResult(
                        content=docs.content,
                        source=KnowledgeSource.CONTEXT7,
                        title=f"{technology} Best Practices",
                        score=0.95,
                        result_type="best_practices",
                        code_examples=docs.code_examples,
                    )
                )

        # Supplement with web search
        if self._web_search:
            query = f"{technology} best practices {topic or ''}".strip()
            web_results = await self._web_search.search(query, max_results=5)

            for result in web_results:
                results.append(
                    ExternalKnowledgeResult(
                        content=result.snippet,
                        source=KnowledgeSource.WEB,
                        source_url=result.url,
                        title=result.title,
                        score=result.score * 0.8,  # Slightly lower than docs
                        result_type="best_practices",
                        metadata=result.metadata,
                    )
                )

        return results

    async def search_current(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[ExternalKnowledgeResult]:
        """
        Search for current/recent information only.

        Focuses on web search for real-time information.

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            List of current information results
        """
        if not self._web_search:
            logger.warning("Web search service not available")
            return []

        self._stats["web_searches"] += 1

        results = await self._web_search.search_current_events(query, max_results)

        return [
            ExternalKnowledgeResult(
                content=r.snippet,
                source=KnowledgeSource.WEB,
                source_url=r.url,
                title=r.title,
                score=r.score,
                result_type="current_info",
                metadata=r.metadata,
            )
            for r in results
        ]

    async def _search_web(
        self,
        query: str,
        max_results: int,
    ) -> List[ExternalKnowledgeResult]:
        """Search web sources."""
        if not self._web_search:
            return []

        self._stats["web_searches"] += 1

        results = await self._web_search.search(query, max_results)

        return [
            ExternalKnowledgeResult(
                content=r.content or r.snippet,
                source=KnowledgeSource.WEB,
                source_url=r.url,
                title=r.title,
                score=r.score,
                result_type="web",
                metadata=r.metadata,
            )
            for r in results
        ]

    async def _search_context7(
        self,
        query: str,
        max_results: int,
        include_examples: bool,
    ) -> List[ExternalKnowledgeResult]:
        """Search Context7 documentation."""
        if not self._context7:
            return []

        self._stats["context7_lookups"] += 1

        # Detect libraries in query
        libraries = self._detect_libraries(query)

        if not libraries:
            # Search across common libraries
            libraries = ["python", "fastapi", "docker"]

        results = []
        mode = DocumentationMode.CODE if include_examples else DocumentationMode.INFO

        for lib_name in libraries[:3]:
            lib = await self._context7.resolve_library(lib_name)
            if not lib:
                continue

            docs = await self._context7.get_documentation(
                lib.library_id,
                topic=query,
                mode=mode,
            )

            if docs:
                results.append(
                    ExternalKnowledgeResult(
                        content=docs.content,
                        source=KnowledgeSource.CONTEXT7,
                        source_url=f"context7://{lib.library_id}",
                        title=f"{lib_name} - {query}",
                        score=0.9,
                        result_type="documentation",
                        library_info=lib,
                        code_examples=docs.code_examples,
                    )
                )

        return results[:max_results]

    async def _search_all(
        self,
        query: str,
        max_results: int,
        include_examples: bool,
    ) -> List[ExternalKnowledgeResult]:
        """Search all available sources in parallel."""
        tasks = []

        # Web search
        if self._web_search:
            tasks.append(self._search_web(query, max_results // 2))

        # Context7 search
        if self._context7:
            tasks.append(
                self._search_context7(query, max_results // 2, include_examples)
            )

        # Run in parallel
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        all_results = []
        for results in results_lists:
            if isinstance(results, list):
                all_results.extend(results)
            elif isinstance(results, Exception):
                logger.warning(f"Search source failed: {results}")

        return all_results

    async def _get_library_docs(
        self,
        library_name: str,
        query: str,
        max_results: int,
    ) -> List[ExternalKnowledgeResult]:
        """Get documentation for a specific library."""
        if not self._context7:
            return []

        result = await self.get_library_info(
            library_name,
            topic=query,
            include_examples=True,
        )

        return [result] if result else []

    def _detect_libraries(self, query: str) -> List[str]:
        """Detect library names mentioned in query."""
        if not self._context7:
            return []

        query_lower = query.lower()
        detected = []

        for lib_name in self._context7.KNOWN_LIBRARIES:
            if lib_name in query_lower:
                detected.append(lib_name)

        return detected

    def _deduplicate_results(
        self,
        results: List[ExternalKnowledgeResult],
    ) -> List[ExternalKnowledgeResult]:
        """Remove duplicate results based on content similarity."""
        seen_content = set()
        unique_results = []

        for result in results:
            # Create content fingerprint
            content_key = result.content[:100].lower().strip()

            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(result)

        return unique_results

    def get_stats(self) -> Dict[str, Any]:
        """Get combined service statistics."""
        stats = {**self._stats}

        if self._web_search:
            stats["web_search"] = self._web_search.get_stats()

        if self._context7:
            stats["context7"] = self._context7.get_stats()

        return stats

    def clear_caches(self) -> None:
        """Clear all caches."""
        if self._web_search:
            self._web_search.clear_cache()

        if self._context7:
            self._context7.clear_cache()

        logger.info("All external knowledge caches cleared")


# Singleton instance
_service_instance: Optional[ExternalKnowledgeService] = None
_service_lock = threading.Lock()


def get_external_knowledge_service(
    enable_web_search: bool = True,
    enable_context7: bool = True,
) -> ExternalKnowledgeService:
    """
    Get singleton external knowledge service instance.

    Args:
        enable_web_search: Enable web search
        enable_context7: Enable Context7 documentation

    Returns:
        Shared ExternalKnowledgeService instance
    """
    global _service_instance

    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = ExternalKnowledgeService(
                    enable_web_search=enable_web_search,
                    enable_context7=enable_context7,
                )

    return _service_instance
