"""
Search system for the Knowledge Engine.

This module implements vector similarity search, hybrid search
(vector + keyword), result reranking, and relevance filtering for
semantic document retrieval.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import structlog

from morgan_server.knowledge.vectordb import VectorDBClient, SearchResult


logger = structlog.get_logger(__name__)


class SearchError(Exception):
    """Base exception for search errors."""


@dataclass
class SearchQuery:
    """Represents a search query with options."""

    query: str
    limit: int = 10
    score_threshold: float = 0.5
    filter_conditions: Optional[Dict[str, Any]] = None
    search_type: str = "vector"  # "vector", "keyword", "hybrid"
    rerank: bool = True
    keyword_weight: float = 0.3  # For hybrid search


@dataclass
class RankedResult:
    """Search result with ranking information."""

    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    rerank_score: Optional[float] = None
    rank: int = 0


class KeywordSearcher:
    """Simple keyword-based search using TF-IDF-like scoring."""

    def __init__(self):
        """Initialize the keyword searcher."""
        self.stop_words = self._load_stop_words()

    def _load_stop_words(self) -> Set[str]:
        """Load common English stop words."""
        return {
            "a", "an", "and", "are", "as", "at", "be", "by", "for",
            "from", "has", "he", "in", "is", "it", "its", "of", "on",
            "that", "the", "to", "was", "will", "with", "this",
        }

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Remove stop words
        return [t for t in tokens if t not in self.stop_words and len(t) > 2]

    def score_match(self, query: str, document: str) -> float:
        """
        Score how well a document matches a query using keyword overlap.

        Args:
            query: Search query
            document: Document text

        Returns:
            Score between 0 and 1
        """
        query_tokens = set(self.tokenize(query))
        doc_tokens = set(self.tokenize(document))

        if not query_tokens:
            return 0.0

        # Calculate Jaccard similarity
        intersection = query_tokens & doc_tokens
        union = query_tokens | doc_tokens

        if not union:
            return 0.0

        jaccard = len(intersection) / len(union)

        # Boost score if query terms appear in order
        query_lower = query.lower()
        doc_lower = document.lower()
        if query_lower in doc_lower:
            jaccard = min(1.0, jaccard * 1.5)

        return jaccard

    def search(
        self,
        query: str,
        documents: List[SearchResult],
        limit: int = 10,
    ) -> List[RankedResult]:
        """
        Search documents using keyword matching.

        Args:
            query: Search query
            documents: List of documents to search
            limit: Maximum number of results

        Returns:
            List of ranked results
        """
        results = []

        for doc in documents:
            content = doc.payload.get("content", "")
            score = self.score_match(query, content)

            if score > 0:
                results.append(
                    RankedResult(
                        id=doc.id,
                        content=content,
                        score=score,
                        metadata=doc.payload,
                        keyword_score=score,
                    )
                )

        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]


class ResultReranker:
    """
    Reranks search results using advanced reranking service.
    
    Supports:
    - Heuristic reranking (fast, no model required)
    - Local CrossEncoder reranking (high quality)
    - Remote reranking service (distributed setups)
    """

    def __init__(
        self,
        use_advanced_reranking: bool = False,
        remote_endpoint: Optional[str] = None,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        local_device: str = "cpu",
    ):
        """
        Initialize the reranker.

        Args:
            use_advanced_reranking: Whether to use advanced reranking service
            remote_endpoint: Remote reranking endpoint URL (optional)
            model: Model name for local CrossEncoder
            local_device: Device for local model ("cpu", "cuda", "mps")
        """
        self.use_advanced_reranking = use_advanced_reranking
        self._reranking_service = None
        
        if use_advanced_reranking:
            from morgan_server.knowledge.reranking import RerankingService
            self._reranking_service = RerankingService(
                remote_endpoint=remote_endpoint,
                model=model,
                local_device=local_device,
            )

    async def rerank(
        self,
        query: str,
        results: List[RankedResult],
        top_k: Optional[int] = None,
    ) -> List[RankedResult]:
        """
        Rerank search results for improved relevance.

        Args:
            query: Original search query
            results: List of results to rerank
            top_k: Number of top results to keep (None = keep all)

        Returns:
            Reranked list of results
        """
        if not results:
            return results

        logger.info("reranking_results", num_results=len(results))

        if self.use_advanced_reranking and self._reranking_service:
            reranked = await self._rerank_with_service(query, results)
        else:
            reranked = self._rerank_heuristic(query, results)

        # Update ranks
        for i, result in enumerate(reranked):
            result.rank = i + 1

        # Limit results if requested
        if top_k is not None:
            reranked = reranked[:top_k]

        logger.info("reranking_complete", num_results=len(reranked))
        return reranked

    async def _rerank_with_service(
        self,
        query: str,
        results: List[RankedResult],
    ) -> List[RankedResult]:
        """
        Rerank using advanced reranking service.

        Args:
            query: Search query
            results: Results to rerank

        Returns:
            Reranked results
        """
        # Extract documents and metadata
        documents = [result.content for result in results]
        metadata = [result.metadata for result in results]

        # Rerank using service
        reranked_results = await self._reranking_service.rerank(
            query=query,
            documents=documents,
            metadata=metadata,
        )

        # Map back to RankedResult objects
        reranked = []
        for rerank_result in reranked_results:
            original_result = results[rerank_result.original_index]
            original_result.rerank_score = rerank_result.score
            original_result.score = rerank_result.score
            reranked.append(original_result)

        return reranked

    def _rerank_heuristic(
        self,
        query: str,
        results: List[RankedResult],
    ) -> List[RankedResult]:
        """
        Rerank using heuristic methods (term overlap, position, etc.).

        Args:
            query: Search query
            results: Results to rerank

        Returns:
            Reranked results
        """
        query_terms = set(query.lower().split())

        for result in results:
            content_lower = result.content.lower()

            # Calculate term overlap
            content_terms = set(content_lower.split())
            overlap = len(query_terms & content_terms)
            total = len(query_terms | content_terms)
            term_score = overlap / total if total > 0 else 0.0

            # Check for exact phrase match
            phrase_bonus = 0.2 if query.lower() in content_lower else 0.0

            # Check for query terms in first 100 characters (position bonus)
            first_part = content_lower[:100]
            position_bonus = 0.0
            for term in query_terms:
                if term in first_part:
                    position_bonus += 0.05

            # Combine scores
            base_score = result.score
            rerank_score = (
                0.5 * base_score +
                0.3 * term_score +
                phrase_bonus +
                min(0.2, position_bonus)
            )

            result.rerank_score = rerank_score
            result.score = rerank_score

        # Sort by rerank score
        results.sort(key=lambda x: x.rerank_score or x.score, reverse=True)
        return results


class SearchSystem:
    """
    Comprehensive search system with vector, keyword, and hybrid search.

    Supports multiple search modes:
    - Vector similarity search
    - Keyword-based search
    - Hybrid search (combining vector and keyword)
    - Result reranking for improved relevance
    """

    def __init__(
        self,
        vectordb_client: VectorDBClient,
        collection_name: str = "knowledge_base",
        use_advanced_reranking: bool = False,
        reranking_remote_endpoint: Optional[str] = None,
        reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranking_device: str = "cpu",
    ):
        """
        Initialize the search system.

        Args:
            vectordb_client: Vector database client
            collection_name: Name of the collection to search
            use_advanced_reranking: Whether to use advanced reranking service
            reranking_remote_endpoint: Remote reranking endpoint URL (optional)
            reranking_model: Model name for local reranking
            reranking_device: Device for local reranking ("cpu", "cuda", "mps")
        """
        self.vectordb_client = vectordb_client
        self.collection_name = collection_name
        self.keyword_searcher = KeywordSearcher()
        self.reranker = ResultReranker(
            use_advanced_reranking=use_advanced_reranking,
            remote_endpoint=reranking_remote_endpoint,
            model=reranking_model,
            local_device=reranking_device,
        )

        logger.info(
            "search_system_initialized",
            collection_name=collection_name,
            use_advanced_reranking=use_advanced_reranking,
            reranking_remote_endpoint=reranking_remote_endpoint,
        )

    async def search(
        self,
        query: SearchQuery,
        query_vector: Optional[List[float]] = None,
    ) -> List[RankedResult]:
        """
        Execute a search query.

        Args:
            query: Search query with options
            query_vector: Pre-computed query vector (optional)

        Returns:
            List of ranked results

        Raises:
            SearchError: If search fails
        """
        try:
            logger.info(
                "search_started",
                query=query.query[:100],
                search_type=query.search_type,
            )

            if query.search_type == "vector":
                results = await self._vector_search(query, query_vector)
            elif query.search_type == "keyword":
                results = await self._keyword_search(query)
            elif query.search_type == "hybrid":
                results = await self._hybrid_search(query, query_vector)
            else:
                raise SearchError(f"Unknown search type: {query.search_type}")

            # Apply relevance filtering
            results = self._filter_by_relevance(results, query.score_threshold)

            # Apply reranking if requested
            if query.rerank and len(results) > 1:
                results = await self.reranker.rerank(
                    query.query,
                    results,
                    top_k=query.limit,
                )
            else:
                # Just limit results
                results = results[:query.limit]

            logger.info(
                "search_completed",
                query=query.query[:100],
                num_results=len(results),
            )

            return results

        except Exception as e:
            logger.error(
                "search_failed",
                query=query.query[:100],
                error=str(e),
            )
            raise SearchError(f"Search failed: {e}") from e

    async def _vector_search(
        self,
        query: SearchQuery,
        query_vector: Optional[List[float]] = None,
    ) -> List[RankedResult]:
        """
        Perform vector similarity search.

        Args:
            query: Search query
            query_vector: Pre-computed query vector

        Returns:
            List of ranked results
        """
        if query_vector is None:
            raise SearchError("Query vector required for vector search")

        # Search vector database
        search_results = await self.vectordb_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=query.limit * 2,  # Get more for reranking
            score_threshold=query.score_threshold,
            filter_conditions=query.filter_conditions,
        )

        # Convert to RankedResult
        results = [
            RankedResult(
                id=result.id,
                content=result.payload.get("content", ""),
                score=result.score,
                metadata=result.payload,
                vector_score=result.score,
            )
            for result in search_results
        ]

        return results

    async def _keyword_search(
        self,
        query: SearchQuery,
    ) -> List[RankedResult]:
        """
        Perform keyword-based search.

        Args:
            query: Search query

        Returns:
            List of ranked results
        """
        # For keyword search, we need to get all documents
        # In a real system, this would use a full-text search index
        # For now, we'll get top candidates using a dummy vector
        dummy_vector = [0.0] * 384  # Assuming 384-dim embeddings

        search_results = await self.vectordb_client.search(
            collection_name=self.collection_name,
            query_vector=dummy_vector,
            limit=100,  # Get more documents for keyword matching
            filter_conditions=query.filter_conditions,
        )

        # Apply keyword search
        results = self.keyword_searcher.search(
            query.query,
            search_results,
            limit=query.limit * 2,
        )

        return results

    async def _hybrid_search(
        self,
        query: SearchQuery,
        query_vector: Optional[List[float]] = None,
    ) -> List[RankedResult]:
        """
        Perform hybrid search (vector + keyword).

        Args:
            query: Search query
            query_vector: Pre-computed query vector

        Returns:
            List of ranked results
        """
        if query_vector is None:
            raise SearchError("Query vector required for hybrid search")

        # Get vector search results
        vector_results = await self._vector_search(query, query_vector)

        # Get all documents for keyword search
        all_docs = await self.vectordb_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=100,
            filter_conditions=query.filter_conditions,
        )

        # Calculate keyword scores
        keyword_scores = {}
        for doc in all_docs:
            content = doc.payload.get("content", "")
            score = self.keyword_searcher.score_match(query.query, content)
            keyword_scores[doc.id] = score

        # Combine scores
        combined_results = {}
        for result in vector_results:
            vector_score = result.vector_score or result.score
            keyword_score = keyword_scores.get(result.id, 0.0)

            # Weighted combination
            combined_score = (
                (1 - query.keyword_weight) * vector_score +
                query.keyword_weight * keyword_score
            )

            result.keyword_score = keyword_score
            result.score = combined_score
            combined_results[result.id] = result

        # Sort by combined score
        results = list(combined_results.values())
        results.sort(key=lambda x: x.score, reverse=True)

        return results

    def _filter_by_relevance(
        self,
        results: List[RankedResult],
        threshold: float,
    ) -> List[RankedResult]:
        """
        Filter results by relevance threshold.

        Args:
            results: List of results
            threshold: Minimum score threshold

        Returns:
            Filtered results
        """
        filtered = [r for r in results if r.score >= threshold]

        if len(filtered) < len(results):
            logger.info(
                "results_filtered",
                original_count=len(results),
                filtered_count=len(filtered),
                threshold=threshold,
            )

        return filtered

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get search system statistics.

        Returns:
            Dictionary with statistics
        """
        try:
            stats = await self.vectordb_client.get_collection_stats(
                self.collection_name
            )

            if stats:
                return {
                    "collection_name": stats.name,
                    "total_documents": stats.points_count,
                    "indexed_documents": stats.indexed_vectors_count,
                    "status": stats.status,
                }
            else:
                return {
                    "collection_name": self.collection_name,
                    "total_documents": 0,
                    "indexed_documents": 0,
                    "status": "not_found",
                }
        except Exception as e:
            logger.error("get_stats_failed", error=str(e))
            return {
                "collection_name": self.collection_name,
                "error": str(e),
            }
