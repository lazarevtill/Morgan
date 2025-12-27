"""
Context7 Documentation Service for Morgan.

Provides integration with Context7 MCP server for fetching
up-to-date library documentation and best practices.

Features:
- Library ID resolution for accurate documentation lookup
- Documentation fetching with topic filtering
- Code examples and API reference retrieval
- Caching for performance optimization
- Best practices and patterns extraction
"""

import asyncio
import hashlib
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from morgan.config import get_settings
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentationMode(str, Enum):
    """Documentation retrieval mode."""

    CODE = "code"  # API references and code examples
    INFO = "info"  # Conceptual guides and architecture


@dataclass
class LibraryInfo:
    """Information about a resolved library."""

    library_id: str  # Context7-compatible ID (e.g., "/vercel/next.js")
    name: str
    description: str = ""
    version: Optional[str] = None
    code_snippets_count: int = 0
    reputation: str = "unknown"  # High, Medium, Low
    benchmark_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "library_id": self.library_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "code_snippets_count": self.code_snippets_count,
            "reputation": self.reputation,
            "benchmark_score": self.benchmark_score,
        }


@dataclass
class DocumentationResult:
    """Result from documentation retrieval."""

    library_id: str
    topic: Optional[str]
    content: str
    mode: DocumentationMode
    page: int = 1
    has_more_pages: bool = False
    code_examples: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "library_id": self.library_id,
            "topic": self.topic,
            "content": self.content,
            "mode": self.mode.value,
            "page": self.page,
            "has_more_pages": self.has_more_pages,
            "code_examples": self.code_examples,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class Context7Service:
    """
    Context7 Documentation Service.

    Provides access to up-to-date library documentation through
    the Context7 MCP server. Supports library resolution,
    documentation retrieval, and code example extraction.

    Example:
        >>> service = Context7Service()
        >>>
        >>> # Resolve library ID
        >>> library = await service.resolve_library("fastapi")
        >>> print(f"Found: {library.name} ({library.library_id})")
        >>>
        >>> # Get documentation
        >>> docs = await service.get_documentation(
        ...     library.library_id,
        ...     topic="routing",
        ...     mode=DocumentationMode.CODE
        ... )
        >>> print(docs.content)
    """

    # Common library mappings for quick resolution
    KNOWN_LIBRARIES = {
        "fastapi": "/tiangolo/fastapi",
        "react": "/facebook/react",
        "nextjs": "/vercel/next.js",
        "next.js": "/vercel/next.js",
        "python": "/python/cpython",
        "docker": "/docker/docs",
        "kubernetes": "/kubernetes/website",
        "pytorch": "/pytorch/pytorch",
        "tensorflow": "/tensorflow/tensorflow",
        "langchain": "/langchain-ai/langchain",
        "qdrant": "/qdrant/qdrant",
        "redis": "/redis/redis",
        "postgresql": "/postgres/postgres",
        "mongodb": "/mongodb/docs",
        "supabase": "/supabase/supabase",
        "prisma": "/prisma/prisma",
        "sqlalchemy": "/sqlalchemy/sqlalchemy",
        "pydantic": "/pydantic/pydantic",
        "httpx": "/encode/httpx",
        "requests": "/psf/requests",
        "numpy": "/numpy/numpy",
        "pandas": "/pandas-dev/pandas",
        "scikit-learn": "/scikit-learn/scikit-learn",
        "transformers": "/huggingface/transformers",
        "ollama": "/ollama/ollama",
    }

    def __init__(
        self,
        cache_ttl_hours: int = 24,
        max_cache_size: int = 500,
    ):
        """
        Initialize Context7 service.

        Args:
            cache_ttl_hours: Cache time-to-live in hours
            max_cache_size: Maximum number of cached entries
        """
        self.settings = get_settings()
        self._lock = threading.Lock()

        # Cache settings
        self._library_cache: Dict[str, tuple[LibraryInfo, datetime]] = {}
        self._docs_cache: Dict[str, tuple[DocumentationResult, datetime]] = {}
        self._cache_ttl = timedelta(hours=cache_ttl_hours)
        self._max_cache_size = max_cache_size

        # Statistics
        self._stats = {
            "library_resolutions": 0,
            "documentation_fetches": 0,
            "cache_hits": 0,
            "errors": 0,
        }

        logger.info(
            f"Context7Service initialized: "
            f"cache_ttl={cache_ttl_hours}h, "
            f"known_libraries={len(self.KNOWN_LIBRARIES)}"
        )

    async def resolve_library(
        self,
        library_name: str,
        use_cache: bool = True,
    ) -> Optional[LibraryInfo]:
        """
        Resolve a library name to its Context7 library ID.

        Args:
            library_name: Library name to resolve (e.g., "fastapi", "react")
            use_cache: Whether to use cached results

        Returns:
            LibraryInfo if found, None otherwise
        """
        self._stats["library_resolutions"] += 1

        # Normalize library name
        normalized = library_name.lower().strip()

        # Check if it's already a Context7 ID format
        if normalized.startswith("/"):
            return LibraryInfo(
                library_id=normalized,
                name=library_name,
            )

        # Check known libraries first
        if normalized in self.KNOWN_LIBRARIES:
            library_id = self.KNOWN_LIBRARIES[normalized]
            return LibraryInfo(
                library_id=library_id,
                name=library_name,
                description=f"Documentation for {library_name}",
            )

        # Check cache
        cache_key = f"lib:{normalized}"
        if use_cache:
            cached = self._get_cached_library(cache_key)
            if cached:
                self._stats["cache_hits"] += 1
                return cached

        try:
            # Resolve via MCP
            library = await self._resolve_library_mcp(library_name)

            # Cache result
            if library:
                self._cache_library(cache_key, library)

            return library

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Failed to resolve library '{library_name}': {e}")
            return None

    async def get_documentation(
        self,
        library_id: str,
        topic: Optional[str] = None,
        mode: DocumentationMode = DocumentationMode.CODE,
        page: int = 1,
        use_cache: bool = True,
    ) -> Optional[DocumentationResult]:
        """
        Get documentation for a library.

        Args:
            library_id: Context7-compatible library ID
            topic: Optional topic to focus on (e.g., "routing", "hooks")
            mode: Documentation mode (CODE for examples, INFO for concepts)
            page: Page number for pagination
            use_cache: Whether to use cached results

        Returns:
            DocumentationResult if successful, None otherwise
        """
        self._stats["documentation_fetches"] += 1

        # Build cache key
        cache_key = self._get_docs_cache_key(library_id, topic, mode, page)

        # Check cache
        if use_cache:
            cached = self._get_cached_docs(cache_key)
            if cached:
                self._stats["cache_hits"] += 1
                return cached

        try:
            # Fetch via MCP
            docs = await self._fetch_documentation_mcp(library_id, topic, mode, page)

            # Cache result
            if docs:
                self._cache_docs(cache_key, docs)

            return docs

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Failed to get documentation for '{library_id}': {e}")
            return None

    async def get_best_practices(
        self,
        library_name: str,
        topic: Optional[str] = None,
    ) -> Optional[DocumentationResult]:
        """
        Get best practices for a library.

        Convenience method that resolves library and fetches
        conceptual/best practices documentation.

        Args:
            library_name: Library name
            topic: Optional topic to focus on

        Returns:
            DocumentationResult with best practices
        """
        # Resolve library first
        library = await self.resolve_library(library_name)
        if not library:
            logger.warning(f"Could not resolve library: {library_name}")
            return None

        # Get conceptual documentation (best practices)
        topic_query = f"{topic} best practices" if topic else "best practices"

        return await self.get_documentation(
            library.library_id,
            topic=topic_query,
            mode=DocumentationMode.INFO,
        )

    async def get_code_examples(
        self,
        library_name: str,
        topic: str,
        max_pages: int = 3,
    ) -> List[str]:
        """
        Get code examples for a specific topic in a library.

        Args:
            library_name: Library name
            topic: Topic to get examples for
            max_pages: Maximum pages to fetch

        Returns:
            List of code example strings
        """
        # Resolve library
        library = await self.resolve_library(library_name)
        if not library:
            return []

        examples = []

        for page in range(1, max_pages + 1):
            docs = await self.get_documentation(
                library.library_id,
                topic=topic,
                mode=DocumentationMode.CODE,
                page=page,
            )

            if docs and docs.code_examples:
                examples.extend(docs.code_examples)

            if not docs or not docs.has_more_pages:
                break

        return examples

    async def search_documentation(
        self,
        query: str,
        libraries: Optional[List[str]] = None,
        mode: DocumentationMode = DocumentationMode.CODE,
    ) -> List[DocumentationResult]:
        """
        Search documentation across multiple libraries.

        Args:
            query: Search query
            libraries: List of library names to search (searches all if None)
            mode: Documentation mode

        Returns:
            List of documentation results
        """
        results = []

        # Determine libraries to search
        if libraries is None:
            # Search in commonly used libraries
            libraries = [
                "fastapi",
                "python",
                "docker",
                "kubernetes",
                "react",
                "nextjs",
                "langchain",
                "qdrant",
            ]

        # Search each library concurrently
        tasks = []
        for lib_name in libraries:
            tasks.append(self._search_library(lib_name, query, mode))

        library_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in library_results:
            if isinstance(result, DocumentationResult):
                results.append(result)
            elif isinstance(result, Exception):
                logger.warning(f"Search failed for a library: {result}")

        return results

    async def _search_library(
        self,
        library_name: str,
        query: str,
        mode: DocumentationMode,
    ) -> Optional[DocumentationResult]:
        """Search a single library."""
        library = await self.resolve_library(library_name)
        if not library:
            return None

        return await self.get_documentation(
            library.library_id,
            topic=query,
            mode=mode,
        )

    async def _resolve_library_mcp(self, library_name: str) -> Optional[LibraryInfo]:
        """
        Resolve library via MCP Context7 server or HTTP API fallback.

        Uses the Context7 API to find the correct library ID.
        """
        try:
            logger.info(f"Resolving library via Context7: {library_name}")

            # Try HTTP API first (doesn't require MCP server)
            result = await self._resolve_library_http(library_name)
            if result:
                return result

            # Try MCP client if available
            from morgan.services.external_knowledge.mcp_client import get_mcp_client

            client = get_mcp_client()
            response = await client.resolve_library(library_name)

            if response.success and response.data:
                data = response.data

                # Handle different response formats
                if isinstance(data, dict):
                    return LibraryInfo(
                        library_id=data.get(
                            "id", data.get("library_id", f"/{library_name}")
                        ),
                        name=data.get("name", library_name),
                        description=data.get("description", ""),
                        version=data.get("version"),
                        code_snippets_count=data.get("code_snippets_count", 0),
                        reputation=data.get("reputation", "unknown"),
                        benchmark_score=data.get("benchmark_score", 0.0),
                    )
                elif isinstance(data, str):
                    return LibraryInfo(
                        library_id=data,
                        name=library_name,
                    )

            return None

        except Exception as e:
            logger.error(f"MCP library resolution failed: {e}")
            # Return None instead of raising to allow fallback to known libraries
            return None

    async def _resolve_library_http(self, library_name: str) -> Optional[LibraryInfo]:
        """
        Resolve library using Context7 HTTP API.

        This is a fallback when MCP is not available.
        """
        try:
            import httpx
        except ImportError:
            return None

        try:
            # Context7 public API endpoint
            url = "https://context7.com/api/v1/resolve"

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    url,
                    json={"libraryName": library_name},
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code == 200:
                    data = response.json()
                    if data and isinstance(data, list) and len(data) > 0:
                        # Return first (best) match
                        best = data[0]
                        return LibraryInfo(
                            library_id=best.get("id", ""),
                            name=best.get("name", library_name),
                            description=best.get("description", ""),
                            version=best.get("version"),
                            code_snippets_count=best.get("codeSnippetsCount", 0),
                            reputation=best.get("reputation", "unknown"),
                            benchmark_score=best.get("benchmarkScore", 0.0),
                        )
                    elif data and isinstance(data, dict):
                        return LibraryInfo(
                            library_id=data.get("id", ""),
                            name=data.get("name", library_name),
                            description=data.get("description", ""),
                            version=data.get("version"),
                            code_snippets_count=data.get("codeSnippetsCount", 0),
                            reputation=data.get("reputation", "unknown"),
                            benchmark_score=data.get("benchmarkScore", 0.0),
                        )

            return None

        except Exception as e:
            logger.debug(f"Context7 HTTP API fallback failed: {e}")
            return None

    async def _fetch_documentation_mcp(
        self,
        library_id: str,
        topic: Optional[str],
        mode: DocumentationMode,
        page: int,
    ) -> Optional[DocumentationResult]:
        """
        Fetch documentation via MCP Context7 server or HTTP API fallback.
        """
        try:
            logger.info(
                f"Fetching documentation: {library_id}, "
                f"topic={topic}, mode={mode.value}, page={page}"
            )

            # Try HTTP API first
            result = await self._fetch_documentation_http(library_id, topic, mode, page)
            if result:
                return result

            # Try MCP client if available
            from morgan.services.external_knowledge.mcp_client import get_mcp_client

            client = get_mcp_client()
            response = await client.get_library_docs(
                library_id=library_id,
                topic=topic,
                mode=mode.value,
                page=page,
            )

            if response.success and response.data:
                data = response.data

                # Extract content based on response format
                content = ""
                code_examples = []
                has_more = False

                if isinstance(data, str):
                    content = data
                    # Extract code blocks as examples
                    code_examples = self._extract_code_examples(content)
                elif isinstance(data, dict):
                    content = data.get("content", data.get("text", str(data)))
                    code_examples = data.get("code_examples", [])
                    has_more = data.get("has_more_pages", data.get("hasMore", False))
                    if not code_examples:
                        code_examples = self._extract_code_examples(content)

                return DocumentationResult(
                    library_id=library_id,
                    topic=topic,
                    content=content,
                    mode=mode,
                    page=page,
                    has_more_pages=has_more,
                    code_examples=code_examples,
                    metadata={
                        "source": "mcp_context7",
                        "response_length": len(content),
                    },
                )

            return None

        except Exception as e:
            logger.error(f"MCP documentation fetch failed: {e}")
            return None

    async def _fetch_documentation_http(
        self,
        library_id: str,
        topic: Optional[str],
        mode: DocumentationMode,
        page: int,
    ) -> Optional[DocumentationResult]:
        """
        Fetch documentation using Context7 HTTP API.

        This is a fallback when MCP is not available.
        """
        try:
            import httpx
        except ImportError:
            return None

        try:
            # Context7 public API endpoint
            url = "https://context7.com/api/v1/docs"

            params = {
                "libraryId": library_id,
                "mode": mode.value,
                "page": page,
            }
            if topic:
                params["topic"] = topic

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    url,
                    params=params,
                    headers={"Accept": "application/json"},
                )

                if response.status_code == 200:
                    data = response.json()
                    content = data.get("content", "")

                    if content:
                        code_examples = data.get("codeExamples", [])
                        if not code_examples:
                            code_examples = self._extract_code_examples(content)

                        return DocumentationResult(
                            library_id=library_id,
                            topic=topic,
                            content=content,
                            mode=mode,
                            page=page,
                            has_more_pages=data.get("hasMore", False),
                            code_examples=code_examples,
                            metadata={
                                "source": "context7_http",
                                "response_length": len(content),
                            },
                        )

            return None

        except Exception as e:
            logger.debug(f"Context7 HTTP API documentation fetch failed: {e}")
            return None

    def _extract_code_examples(self, content: str) -> List[str]:
        """Extract code blocks from documentation content."""
        import re

        code_examples = []

        # Match fenced code blocks ```...```
        fenced_pattern = r"```[\w]*\n(.*?)```"
        fenced_matches = re.findall(fenced_pattern, content, re.DOTALL)
        code_examples.extend(fenced_matches)

        # Match indented code blocks (4 spaces)
        lines = content.split("\n")
        current_block = []
        in_block = False

        for line in lines:
            if line.startswith("    ") and line.strip():
                current_block.append(line[4:])  # Remove indentation
                in_block = True
            elif in_block and not line.strip():
                current_block.append("")  # Keep blank lines in block
            elif in_block:
                if current_block:
                    code_examples.append("\n".join(current_block).strip())
                current_block = []
                in_block = False

        # Don't forget last block
        if current_block:
            code_examples.append("\n".join(current_block).strip())

        # Filter out very short snippets
        code_examples = [ex for ex in code_examples if len(ex) > 20]

        return code_examples[:10]  # Limit to 10 examples

    def _get_docs_cache_key(
        self,
        library_id: str,
        topic: Optional[str],
        mode: DocumentationMode,
        page: int,
    ) -> str:
        """Generate cache key for documentation."""
        key_str = f"{library_id}:{topic or ''}:{mode.value}:{page}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def _get_cached_library(self, cache_key: str) -> Optional[LibraryInfo]:
        """Get cached library info."""
        with self._lock:
            if cache_key in self._library_cache:
                info, timestamp = self._library_cache[cache_key]
                if datetime.utcnow() - timestamp < self._cache_ttl:
                    return info
                else:
                    del self._library_cache[cache_key]
        return None

    def _cache_library(self, cache_key: str, info: LibraryInfo) -> None:
        """Cache library info."""
        with self._lock:
            if len(self._library_cache) >= self._max_cache_size:
                self._clean_library_cache()
            self._library_cache[cache_key] = (info, datetime.utcnow())

    def _get_cached_docs(self, cache_key: str) -> Optional[DocumentationResult]:
        """Get cached documentation."""
        with self._lock:
            if cache_key in self._docs_cache:
                docs, timestamp = self._docs_cache[cache_key]
                if datetime.utcnow() - timestamp < self._cache_ttl:
                    return docs
                else:
                    del self._docs_cache[cache_key]
        return None

    def _cache_docs(self, cache_key: str, docs: DocumentationResult) -> None:
        """Cache documentation."""
        with self._lock:
            if len(self._docs_cache) >= self._max_cache_size:
                self._clean_docs_cache()
            self._docs_cache[cache_key] = (docs, datetime.utcnow())

    def _clean_library_cache(self) -> None:
        """Clean expired library cache entries."""
        now = datetime.utcnow()
        expired = [
            key
            for key, (_, ts) in self._library_cache.items()
            if now - ts >= self._cache_ttl
        ]
        for key in expired:
            del self._library_cache[key]

    def _clean_docs_cache(self) -> None:
        """Clean expired docs cache entries."""
        now = datetime.utcnow()
        expired = [
            key
            for key, (_, ts) in self._docs_cache.items()
            if now - ts >= self._cache_ttl
        ]
        for key in expired:
            del self._docs_cache[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            **self._stats,
            "library_cache_size": len(self._library_cache),
            "docs_cache_size": len(self._docs_cache),
            "known_libraries": len(self.KNOWN_LIBRARIES),
        }

    def clear_cache(self) -> None:
        """Clear all caches."""
        with self._lock:
            self._library_cache.clear()
            self._docs_cache.clear()
        logger.info("Context7 caches cleared")


# Singleton instance
_service_instance: Optional[Context7Service] = None
_service_lock = threading.Lock()


def get_context7_service(**kwargs) -> Context7Service:
    """
    Get singleton Context7 service instance.

    Args:
        **kwargs: Optional configuration overrides

    Returns:
        Shared Context7Service instance
    """
    global _service_instance

    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = Context7Service(**kwargs)

    return _service_instance
