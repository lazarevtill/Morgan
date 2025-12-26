"""
Multi-Stage Search Engine for Morgan RAG.

Implements six search strategies with hierarchical filtering and result fusion:
- Semantic search with hierarchical embeddings (coarse → medium → fine)
- Keyword search for exact term matching
- Category search for content type filtering
- Temporal search for recency-based ranking
- Memory search for conversation history
- Contextual search for personalized results

Features:
- Coarse-to-fine hierarchical filtering (90% candidate reduction)
- Reciprocal Rank Fusion for result merging
- Intelligent deduplication using cosine similarity
- Performance optimization with caching
"""

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from morgan.services.embeddings import get_embedding_service
from morgan.utils.logger import get_logger
from morgan.utils.request_context import get_request_id, set_request_id
from morgan.vector_db.client import VectorDBClient
from morgan.vectorization.hierarchical_embeddings import (
    get_hierarchical_embedding_service,
)

logger = get_logger(__name__)


class SearchStrategy(Enum):
    """Search strategies for different types of queries."""

    SEMANTIC = "semantic"  # Vector similarity across all scales
    KEYWORD = "keyword"  # Traditional text matching
    CATEGORY = "category"  # Filter by document categories
    TEMPORAL = "temporal"  # Weight by recency and relevance
    MEMORY = "memory"  # Search conversation history
    CONTEXTUAL = "contextual"  # Use conversation context for personalization


@dataclass
class SearchResult:
    """
    Single search result with relevance scoring and metadata.

    Enhanced version of the basic SearchResult with additional fields
    for multi-stage search and result fusion.
    """

    content: str
    source: str
    score: float
    result_type: str  # "knowledge", "memory", "web", etc.
    metadata: Dict[str, Any]

    # Multi-stage search specific fields
    strategy: Optional[str] = None  # Which strategy found this result
    rrf_score: Optional[float] = None  # Reciprocal Rank Fusion score
    hierarchical_scores: Dict[str, float] = field(
        default_factory=dict
    )  # Scores by scale
    dedup_signature: Optional[str] = None  # For deduplication

    def __post_init__(self):
        """Ensure metadata is always a dict."""
        if self.metadata is None:
            self.metadata = {}
        if not self.hierarchical_scores:
            self.hierarchical_scores = {}

    def summary(self, max_length: int = 100) -> str:
        """Get a short summary of the content."""
        if len(self.content) <= max_length:
            return self.content
        return self.content[: max_length - 3] + "..."

    def get_best_score(self) -> float:
        """Get the best score from all available scores."""
        scores = [self.score]
        if self.rrf_score is not None:
            scores.append(self.rrf_score)
        if self.hierarchical_scores:
            scores.extend(self.hierarchical_scores.values())
        return max(scores)


@dataclass
class SearchResults:
    """
    Collection of search results with metadata about the search process.

    Provides insights into search performance and strategy effectiveness.
    """

    results: List[SearchResult]
    total_candidates: int = 0
    filtered_candidates: int = 0
    strategies_used: List[str] = field(default_factory=list)
    search_time: float = 0.0
    fusion_applied: bool = False
    deduplication_applied: bool = False

    def __len__(self) -> int:
        """Number of results."""
        return len(self.results)

    def __iter__(self):
        """Iterate over results."""
        return iter(self.results)

    def __getitem__(self, index):
        """Get result by index."""
        return self.results[index]

    def get_by_strategy(self, strategy: str) -> List[SearchResult]:
        """Get results from a specific strategy."""
        return [r for r in self.results if r.strategy == strategy]

    def get_reduction_ratio(self) -> float:
        """Get candidate reduction ratio (0.0 to 1.0)."""
        if self.total_candidates == 0:
            return 0.0
        return 1.0 - (self.filtered_candidates / self.total_candidates)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring."""
        return {
            "total_results": len(self.results),
            "total_candidates": self.total_candidates,
            "filtered_candidates": self.filtered_candidates,
            "reduction_ratio": self.get_reduction_ratio(),
            "strategies_used": self.strategies_used,
            "search_time": self.search_time,
            "fusion_applied": self.fusion_applied,
            "deduplication_applied": self.deduplication_applied,
            "results_per_second": (
                len(self.results) / self.search_time if self.search_time > 0 else 0
            ),
        }


class MultiStageSearchEngine:
    """
    Advanced multi-stage search engine with hierarchical filtering and result fusion.

    Implements the complete search pipeline:
    1. Query analysis and strategy selection
    2. Hierarchical search with coarse-to-fine filtering
    3. Multiple search strategy execution
    4. Result fusion using Reciprocal Rank Fusion
    5. Intelligent deduplication
    6. Final ranking and optimization

    Performance targets:
    - 90% candidate reduction through hierarchical filtering
    - <500ms search latency for 10K documents
    - Effective result fusion from multiple strategies
    """

    def __init__(self):
        """Initialize multi-stage search engine."""
        self.embedding_service = get_embedding_service()
        self.hierarchical_service = get_hierarchical_embedding_service()
        self.vector_db = VectorDBClient()

        # Collection names
        self.knowledge_collection = "morgan_knowledge"
        self.memory_collection = "morgan_memories"  # Must match memory_processor.py

        # Search configuration
        self.coarse_filter_ratio = (
            0.1  # Keep 10% after coarse filtering (90% reduction)
        )
        self.medium_filter_ratio = 0.3  # Keep 30% after medium filtering
        self.similarity_threshold = 0.95  # For deduplication
        self.rrf_k = 60  # RRF parameter

        # Performance tracking
        self.search_stats = {
            "total_searches": 0,
            "average_search_time": 0.0,
            "average_candidates": 0,
            "average_reduction": 0.0,
        }

        logger.info("MultiStageSearchEngine initialized")

    def search(
        self,
        query: str,
        max_results: int = 10,
        strategies: Optional[List[SearchStrategy]] = None,
        min_score: float = 0.7,
        use_hierarchical: bool = True,
        conversation_context: Optional[str] = None,
        emotional_context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> SearchResults:
        """
        Execute multi-stage search with hierarchical filtering, result fusion, and companion awareness.

        Enhanced to support:
        - Emotional context integration in search result ranking
        - Companion-aware result filtering and personalization
        - Multi-stage search with hierarchical embeddings

        Args:
            query: Natural language search query
            max_results: Maximum number of results to return
            strategies: List of search strategies to use (all if None)
            min_score: Minimum relevance score threshold
            use_hierarchical: Whether to use hierarchical filtering
            conversation_context: Optional conversation context for personalization
            emotional_context: Optional emotional context for result enhancement
            user_id: Optional user ID for companion-aware personalization
            request_id: Optional request ID for tracing

        Returns:
            SearchResults with ranked and fused results enhanced with emotional and companion context

        Example:
            >>> engine = MultiStageSearchEngine()
            >>> emotional_ctx = {"primary_emotion": "frustration", "intensity": 0.8}
            >>> results = engine.search(
            ...     "How to deploy Docker containers?",
            ...     max_results=5,
            ...     emotional_context=emotional_ctx,
            ...     user_id="user123"
            ... )
            >>> print(f"Found {len(results)} enhanced results in {results.search_time:.3f}s")
        """
        # Get or generate request ID for tracing
        if request_id is None:
            request_id = get_request_id() or set_request_id()

        start_time = time.time()

        # Default to all strategies if none specified
        if strategies is None:
            strategies = [
                SearchStrategy.SEMANTIC,
                SearchStrategy.KEYWORD,
                SearchStrategy.CATEGORY,
                SearchStrategy.TEMPORAL,
                SearchStrategy.MEMORY,
                SearchStrategy.CONTEXTUAL,
            ]

        # Convert string strategies to SearchStrategy enums if needed
        converted_strategies = []
        for strategy in strategies:
            if isinstance(strategy, str):
                try:
                    converted_strategies.append(SearchStrategy(strategy))
                except ValueError:
                    logger.warning(f"Unknown strategy string: {strategy}")
                    continue
            else:
                converted_strategies.append(strategy)

        strategies = converted_strategies

        logger.debug(
            f"Starting multi-stage search: '{query}' "
            f"(strategies={[s.value for s in strategies]}, "
            f"hierarchical={use_hierarchical}, request_id={request_id})"
        )

        try:
            # Execute search strategies
            strategy_results = []
            total_candidates = 0

            for strategy in strategies:
                strategy_start = time.time()

                if strategy == SearchStrategy.SEMANTIC:
                    results, candidates = self._semantic_search(
                        query,
                        max_results * 2,
                        min_score,
                        use_hierarchical,
                        request_id,
                        emotional_context,
                        user_id,
                    )
                elif strategy == SearchStrategy.KEYWORD:
                    results, candidates = self._keyword_search(
                        query, max_results * 2, min_score, request_id
                    )
                elif strategy == SearchStrategy.CATEGORY:
                    results, candidates = self._category_search(
                        query, max_results * 2, min_score, request_id
                    )
                elif strategy == SearchStrategy.TEMPORAL:
                    results, candidates = self._temporal_search(
                        query, max_results * 2, min_score, request_id
                    )
                elif strategy == SearchStrategy.MEMORY:
                    results, candidates = self._memory_search(
                        query, max_results * 2, min_score, request_id
                    )
                elif strategy == SearchStrategy.CONTEXTUAL:
                    results, candidates = self._contextual_search(
                        query,
                        conversation_context,
                        max_results * 2,
                        min_score,
                        request_id,
                    )
                else:
                    logger.warning(f"Unknown search strategy: {strategy}")
                    continue

                # Tag results with strategy
                for result in results:
                    result.strategy = strategy.value

                strategy_results.append(results)
                total_candidates += candidates

                strategy_time = time.time() - strategy_start
                logger.debug(
                    f"Strategy {strategy.value}: {len(results)} results, "
                    f"{candidates} candidates in {strategy_time:.3f}s"
                )

            # Fusion results using Reciprocal Rank Fusion
            fused_results = self._fusion_results(strategy_results, max_results * 3)

            # Apply deduplication
            deduplicated_results = self._deduplicate_results(fused_results)

            # Final ranking and limiting
            final_results = self._final_ranking(
                deduplicated_results, max_results, min_score
            )

            # Calculate performance metrics
            search_time = time.time() - start_time
            filtered_candidates = len(final_results)

            # Update performance stats
            self._update_search_stats(
                search_time, total_candidates, filtered_candidates
            )

            # Create search results object
            search_results = SearchResults(
                results=final_results,
                total_candidates=total_candidates,
                filtered_candidates=filtered_candidates,
                strategies_used=[s.value for s in strategies],
                search_time=search_time,
                fusion_applied=len(strategy_results) > 1,
                deduplication_applied=len(fused_results) != len(deduplicated_results),
            )

            logger.info(
                f"Multi-stage search completed: {len(final_results)} results, "
                f"{total_candidates} candidates, {search_time:.3f}s "
                f"(reduction: {search_results.get_reduction_ratio():.1%}, request_id={request_id})"
            )

            return search_results

        except Exception as e:
            logger.error(f"Multi-stage search failed: {e}", exc_info=True)
            # Return empty results on failure
            return SearchResults(
                results=[],
                search_time=time.time() - start_time,
                strategies_used=[s.value for s in strategies],
            )

    def _semantic_search(
        self,
        query: str,
        max_results: int,
        min_score: float,
        use_hierarchical: bool,
        request_id: str,
        emotional_context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[List[SearchResult], int]:
        """
        Execute semantic search with optional hierarchical filtering.

        Returns:
            Tuple of (results, total_candidates_processed)
        """
        try:
            if use_hierarchical:
                return self._hierarchical_semantic_search(
                    query,
                    max_results,
                    min_score,
                    request_id,
                    emotional_context,
                    user_id,
                )
            else:
                return self._standard_semantic_search(
                    query,
                    max_results,
                    min_score,
                    request_id,
                    emotional_context,
                    user_id,
                )

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return [], 0

    def _hierarchical_semantic_search(
        self,
        query: str,
        max_results: int,
        min_score: float,
        request_id: str,
        emotional_context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[List[SearchResult], int]:
        """
        Execute hierarchical semantic search with coarse-to-fine filtering and emotional context integration.

        Enhanced to implement:
        1. Coarse: Category and topic filtering (90% reduction)
        2. Medium: Section and concept matching
        3. Fine: Precise content matching with emotional and companion awareness
        """
        try:
            # Generate query embeddings for all scales
            query_embeddings = self._generate_query_embeddings(query, request_id)

            # Use enhanced hierarchical search from vector database client
            hierarchical_results = self.vector_db.search_hierarchical(
                collection_name=self.knowledge_collection,
                coarse_vector=query_embeddings["coarse"],
                medium_vector=query_embeddings["medium"],
                fine_vector=query_embeddings["fine"],
                limit=max_results,
                coarse_limit=max_results * 10,  # 90% reduction target
                medium_limit=max_results * 3,  # Further filtering
                emotional_context=emotional_context,
                user_id=user_id,
            )

            # Convert to SearchResult objects with hierarchical metadata
            enhanced_results = []
            for result in hierarchical_results:
                search_result = self._convert_to_search_result(result, "semantic")

                # Add hierarchical search metadata
                search_result.metadata.update(
                    {
                        "hierarchical_search": True,
                        "emotional_enhanced": emotional_context is not None,
                        "companion_enhanced": user_id is not None,
                        "original_score": result.payload.get(
                            "original_score", result.score
                        ),
                        "enhancement_factors": result.payload.get(
                            "enhancement_factors", []
                        ),
                    }
                )

                # Store hierarchical scores if available
                if hasattr(result, "hierarchical_scores"):
                    search_result.hierarchical_scores = result.hierarchical_scores

                enhanced_results.append(search_result)

            # Apply additional companion-aware filtering and ranking
            if user_id or emotional_context:
                enhanced_results = self._apply_advanced_companion_filtering(
                    enhanced_results, query, emotional_context, user_id
                )

            total_candidates = (
                len(hierarchical_results) * 10
            )  # Estimate based on filtering ratios

            logger.debug(
                f"Enhanced hierarchical search completed: {len(enhanced_results)} results "
                f"with emotional_context={emotional_context is not None}, "
                f"user_id={user_id is not None}"
            )

            return enhanced_results, total_candidates

        except Exception as e:
            logger.error(f"Enhanced hierarchical semantic search failed: {e}")
            # Fallback to standard hierarchical search
            return self._fallback_hierarchical_search(
                query, max_results, min_score, request_id
            )

    def _apply_advanced_companion_filtering(
        self,
        results: List[SearchResult],
        query: str,
        emotional_context: Optional[Dict[str, Any]],
        user_id: Optional[str],
    ) -> List[SearchResult]:
        """
        Apply advanced companion-aware filtering and ranking to search results.

        Args:
            results: Search results to enhance
            query: Original search query
            emotional_context: Optional emotional context
            user_id: Optional user ID for personalization

        Returns:
            Enhanced and re-ranked search results
        """
        try:
            if not results:
                return results

            enhanced_results = []

            for result in results:
                enhancement_score = 0.0
                enhancement_details = []

                # Apply query-specific emotional enhancement
                if emotional_context:
                    emotional_boost = self._calculate_query_emotional_boost(
                        result, query, emotional_context
                    )
                    enhancement_score += emotional_boost
                    if emotional_boost > 0:
                        enhancement_details.append(
                            f"query_emotional:{emotional_boost:.3f}"
                        )

                # Apply user preference matching
                if user_id:
                    preference_boost = self._calculate_user_preference_boost(
                        result, query, user_id
                    )
                    enhancement_score += preference_boost
                    if preference_boost > 0:
                        enhancement_details.append(
                            f"user_preference:{preference_boost:.3f}"
                        )

                # Apply contextual relevance boost
                context_boost = self._calculate_contextual_relevance_boost(
                    result, query
                )
                enhancement_score += context_boost
                if context_boost > 0:
                    enhancement_details.append(f"contextual:{context_boost:.3f}")

                # Update result score and metadata
                result.score = min(1.0, result.score + enhancement_score)
                result.metadata.update(
                    {
                        "advanced_enhancement_score": enhancement_score,
                        "advanced_enhancement_details": enhancement_details,
                        "original_hierarchical_score": result.score - enhancement_score,
                    }
                )

                enhanced_results.append(result)

            # Re-sort by enhanced score
            enhanced_results.sort(key=lambda r: r.score, reverse=True)

            return enhanced_results

        except Exception as e:
            logger.warning(f"Advanced companion filtering failed: {e}")
            return results

    def _calculate_query_emotional_boost(
        self, result: SearchResult, query: str, emotional_context: Dict[str, Any]
    ) -> float:
        """Calculate emotional boost based on query and emotional context."""
        try:
            boost = 0.0
            content = result.content.lower()
            query_lower = query.lower()

            # Get emotional state
            primary_emotion = emotional_context.get("primary_emotion", "")
            intensity = emotional_context.get("intensity", 0.0)

            # Boost for emotional query-content alignment
            if primary_emotion == "sadness" and any(
                word in content for word in ["help", "support", "solution"]
            ):
                boost += 0.1 * intensity
            elif primary_emotion == "joy" and any(
                word in content for word in ["success", "achievement", "guide"]
            ):
                boost += 0.08 * intensity
            elif primary_emotion == "fear" and any(
                word in content for word in ["safe", "secure", "tutorial", "step"]
            ):
                boost += 0.12 * intensity
            elif primary_emotion == "anger" and any(
                word in content for word in ["fix", "solve", "troubleshoot"]
            ):
                boost += 0.1 * intensity

            # Boost for emotional indicators in query matching content
            emotional_indicators = emotional_context.get("emotional_indicators", [])
            for indicator in emotional_indicators:
                if indicator.lower() in content and indicator.lower() in query_lower:
                    boost += 0.05

            return min(0.2, boost)  # Cap at 20% boost

        except Exception as e:
            logger.warning(f"Failed to calculate query emotional boost: {e}")
            return 0.0

    def _calculate_user_preference_boost(
        self, result: SearchResult, query: str, user_id: str
    ) -> float:
        """Calculate boost based on user preferences and interaction history."""
        try:
            boost = 0.0
            content = result.content.lower()
            source = result.source.lower()

            # Simplified user preference matching
            # In practice, this would use actual user profile data

            # Boost for technical content (assuming technical user)
            technical_keywords = [
                "api",
                "code",
                "function",
                "class",
                "method",
                "implementation",
            ]
            tech_matches = sum(
                1 for keyword in technical_keywords if keyword in content
            )
            if tech_matches > 0:
                boost += min(0.1, tech_matches * 0.02)

            # Boost for documentation and tutorials
            if any(
                doc_type in source
                for doc_type in ["documentation", "tutorial", "guide", "reference"]
            ):
                boost += 0.05

            # Boost for comprehensive content
            if len(result.content) > 500:
                boost += 0.03

            return min(0.15, boost)  # Cap at 15% boost

        except Exception as e:
            logger.warning(f"Failed to calculate user preference boost: {e}")
            return 0.0

    def _calculate_contextual_relevance_boost(
        self, result: SearchResult, query: str
    ) -> float:
        """Calculate boost based on contextual relevance factors."""
        try:
            boost = 0.0
            content = result.content.lower()
            query_lower = query.lower()

            # Boost for query term density
            query_words = [word for word in query_lower.split() if len(word) > 3]
            if query_words:
                matches = sum(1 for word in query_words if word in content)
                match_ratio = matches / len(query_words)
                boost += match_ratio * 0.05

            # Boost for structured content
            if any(
                indicator in result.content
                for indicator in ["#", "##", "```", "- ", "1. "]
            ):
                boost += 0.02

            # Boost for complete answers (content that seems to fully address the query)
            if len(result.content) > 200 and any(
                word in content for word in ["how", "what", "why", "example"]
            ):
                boost += 0.03

            return min(0.08, boost)  # Cap at 8% boost

        except Exception as e:
            logger.warning(f"Failed to calculate contextual relevance boost: {e}")
            return 0.0

    def _fallback_hierarchical_search(
        self, query: str, max_results: int, min_score: float, request_id: str
    ) -> Tuple[List[SearchResult], int]:
        """Fallback hierarchical search without enhanced features."""
        try:
            # Generate query embeddings for all scales
            query_embeddings = self._generate_query_embeddings(query, request_id)

            total_candidates = 0

            # Stage 1: Coarse filtering (category and topic level)
            coarse_limit = max_results * 10  # Get more candidates for filtering
            coarse_results = self.vector_db.search(
                collection_name=self.knowledge_collection,
                query_vector=query_embeddings["coarse"],
                limit=coarse_limit,
                score_threshold=min_score * 0.8,  # Lower threshold for coarse
            )

            total_candidates += len(coarse_results)

            # Apply coarse filtering (keep top 10% for 90% reduction)
            coarse_filtered = coarse_results[
                : int(len(coarse_results) * self.coarse_filter_ratio)
            ]

            logger.debug(
                f"Fallback coarse filtering: {len(coarse_results)} → {len(coarse_filtered)} "
                f"({(1 - len(coarse_filtered)/len(coarse_results)):.1%} reduction)"
            )

            if not coarse_filtered:
                return [], total_candidates

            # Stage 2: Medium filtering (section and concept level)
            medium_results = []
            for result in coarse_filtered:
                # Re-score using medium embeddings
                medium_score = self._calculate_similarity(
                    query_embeddings["medium"],
                    result.payload.get("medium_embedding", []),
                )

                if medium_score >= min_score * 0.9:
                    search_result = self._convert_to_search_result(result, "semantic")
                    search_result.hierarchical_scores["medium"] = medium_score
                    medium_results.append(search_result)

            # Sort by medium score and apply medium filtering
            medium_results.sort(
                key=lambda r: r.hierarchical_scores.get("medium", 0), reverse=True
            )
            medium_filtered = medium_results[
                : int(len(medium_results) * self.medium_filter_ratio)
            ]

            logger.debug(
                f"Fallback medium filtering: {len(medium_results)} → {len(medium_filtered)} "
                f"({(1 - len(medium_filtered)/len(medium_results)):.1%} reduction)"
            )

            if not medium_filtered:
                return [], total_candidates

            # Stage 3: Fine filtering (precise content matching)
            fine_results = []
            for result in medium_filtered:
                # Re-score using fine embeddings
                fine_score = self._calculate_similarity(
                    query_embeddings["fine"], result.metadata.get("fine_embedding", [])
                )

                if fine_score >= min_score:
                    result.hierarchical_scores["fine"] = fine_score
                    result.score = fine_score  # Use fine score as primary score
                    fine_results.append(result)

            # Sort by fine score and limit results
            fine_results.sort(key=lambda r: r.score, reverse=True)
            final_results = fine_results[:max_results]

            logger.debug(
                f"Fallback fine filtering: {len(medium_filtered)} → {len(final_results)} "
                f"(final results)"
            )

            return final_results, total_candidates

        except Exception as e:
            logger.error(f"Fallback hierarchical search failed: {e}")
            return [], 0

    def _standard_semantic_search(
        self,
        query: str,
        max_results: int,
        min_score: float,
        request_id: str,
        emotional_context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[List[SearchResult], int]:
        """Execute standard semantic search with optional emotional context and companion awareness."""
        # Generate query embedding
        query_embedding = self.embedding_service.encode(
            text=query, instruction="query", request_id=request_id
        )

        # Use enhanced search if emotional context or user ID provided
        if emotional_context or user_id:
            search_results = self.vector_db.search_with_emotional_context(
                collection_name=self.knowledge_collection,
                query_vector=query_embedding,
                emotional_context=emotional_context or {},
                user_id=user_id,
                limit=max_results,
                score_threshold=min_score,
            )
        else:
            # Standard search
            search_results = self.vector_db.search(
                collection_name=self.knowledge_collection,
                query_vector=query_embedding,
                limit=max_results,
                score_threshold=min_score,
            )

        # Convert to SearchResult objects
        results = []
        for result in search_results:
            search_result = self._convert_to_search_result(result, "semantic")

            # Add enhancement metadata if available
            if emotional_context or user_id:
                search_result.metadata.update(
                    {
                        "emotional_enhanced": emotional_context is not None,
                        "companion_enhanced": user_id is not None,
                        "enhancement_factors": result.payload.get(
                            "enhancement_factors", []
                        ),
                    }
                )

            results.append(search_result)

        return results, len(search_results)

    def _keyword_search(
        self, query: str, max_results: int, min_score: float, request_id: str
    ) -> Tuple[List[SearchResult], int]:
        """
        Execute keyword-based search using text matching.

        Implements traditional text search with term frequency scoring.
        """
        try:
            # Extract keywords from query
            keywords = self._extract_keywords(query)

            if not keywords:
                return [], 0

            # Search using text filters (simplified approach)
            # In a full implementation, this would use a text search index
            results = []

            # For now, use vector search but with keyword-enhanced query
            keyword_query = " ".join(keywords)
            query_embedding = self.embedding_service.encode(
                text=keyword_query, instruction="query", request_id=request_id
            )

            search_results = self.vector_db.search(
                collection_name=self.knowledge_collection,
                query_vector=query_embedding,
                limit=max_results,
                score_threshold=min_score * 0.8,  # Lower threshold for keyword search
            )

            # Filter results that actually contain keywords
            for result in search_results:
                content = result.payload.get("content", "").lower()
                keyword_matches = sum(1 for kw in keywords if kw.lower() in content)

                if keyword_matches > 0:
                    search_result = self._convert_to_search_result(result, "keyword")
                    # Boost score based on keyword matches
                    search_result.score = min(
                        1.0, search_result.score + (keyword_matches * 0.1)
                    )
                    results.append(search_result)

            # Sort by score
            results.sort(key=lambda r: r.score, reverse=True)

            return results[:max_results], len(search_results)

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return [], 0

    def _category_search(
        self, query: str, max_results: int, min_score: float, request_id: str
    ) -> Tuple[List[SearchResult], int]:
        """
        Execute category-based search filtering by document types.
        """
        try:
            # Detect query category
            query_category = self._detect_query_category(query)

            if not query_category:
                return [], 0

            # Search with category filter
            filter_conditions = {"category": query_category}
            search_results = self.vector_db.search_with_filter(
                collection_name=self.knowledge_collection,
                filter_conditions=filter_conditions,
                limit=max_results * 2,
            )

            # Generate query embedding for scoring
            query_embedding = self.embedding_service.encode(
                text=query, instruction="query", request_id=request_id
            )

            # Score filtered results
            results = []
            for result in search_results:
                content_embedding = result.payload.get("fine_embedding", [])
                if content_embedding:
                    score = self._calculate_similarity(
                        query_embedding, content_embedding
                    )
                    if score >= min_score:
                        search_result = self._convert_to_search_result(
                            result, "category"
                        )
                        search_result.score = score
                        results.append(search_result)

            # Sort by score
            results.sort(key=lambda r: r.score, reverse=True)

            return results[:max_results], len(search_results)

        except Exception as e:
            logger.error(f"Category search failed: {e}")
            return [], 0

    def _temporal_search(
        self, query: str, max_results: int, min_score: float, request_id: str
    ) -> Tuple[List[SearchResult], int]:
        """
        Execute temporal search with recency-based ranking.
        """
        try:
            # Standard semantic search first
            query_embedding = self.embedding_service.encode(
                text=query, instruction="query", request_id=request_id
            )

            search_results = self.vector_db.search(
                collection_name=self.knowledge_collection,
                query_vector=query_embedding,
                limit=max_results * 2,
                score_threshold=min_score * 0.8,
            )

            # Apply temporal scoring
            results = []
            current_time = datetime.utcnow()

            for result in search_results:
                search_result = self._convert_to_search_result(result, "temporal")

                # Get document timestamp
                indexed_at = result.payload.get("indexed_at")
                if indexed_at:
                    try:
                        doc_time = datetime.fromisoformat(
                            indexed_at.replace("Z", "+00:00")
                        )
                        age_days = (current_time - doc_time).days

                        # Apply temporal decay (newer documents get higher scores)
                        temporal_boost = max(
                            0.1, 1.0 - (age_days / 365.0)
                        )  # Decay over 1 year
                        search_result.score = min(
                            1.0, search_result.score * temporal_boost
                        )

                    except (ValueError, TypeError):
                        # If timestamp parsing fails, use original score
                        pass

                if search_result.score >= min_score:
                    results.append(search_result)

            # Sort by temporal-adjusted score
            results.sort(key=lambda r: r.score, reverse=True)

            return results[:max_results], len(search_results)

        except Exception as e:
            logger.error(f"Temporal search failed: {e}")
            return [], 0

    def _memory_search(
        self, query: str, max_results: int, min_score: float, request_id: str
    ) -> Tuple[List[SearchResult], int]:
        """
        Execute memory search through conversation history with enhanced companion integration.

        Implements requirements 10.1-10.5:
        - Search both document knowledge and conversation memories
        - Surface previous answers and context for similar questions
        - Weight recent and relevant conversations higher in results
        - Provide conversation timestamps and context
        - Distinguish between document knowledge and conversation memories
        """
        try:
            from morgan.companion.relationship_manager import (
                CompanionRelationshipManager,
            )
            from morgan.intelligence.core.intelligence_engine import (
                get_emotional_intelligence_engine,
            )
            from morgan.memory.memory_processor import get_memory_processor
            from morgan.search.companion_memory_search import (
                get_companion_memory_search_engine,
            )

            # Use the dedicated companion memory search engine for enhanced integration
            companion_search = get_companion_memory_search_engine()
            get_memory_processor()
            CompanionRelationshipManager()
            emotional_engine = get_emotional_intelligence_engine()

            logger.debug(
                f"Executing enhanced memory search with companion integration: '{query}'"
            )

            # Analyze query for emotional context
            from morgan.intelligence.core.models import (
                ConversationContext,
                EmotionalState,
                EmotionType,
            )

            try:
                query_context = ConversationContext(
                    user_id="default_user",  # Will be enhanced with actual user ID
                    conversation_id=f"search_{request_id}",
                    message_text=query,
                    timestamp=datetime.utcnow(),
                )
                query_emotion = emotional_engine.analyze_emotion(query, query_context)
            except Exception as e:
                logger.warning(f"Failed to analyze query emotion: {e}")
                # Create a neutral emotional state as fallback
                query_emotion = EmotionalState(
                    primary_emotion=EmotionType.NEUTRAL,
                    intensity=0.0,
                    confidence=0.0,
                    secondary_emotions=[],
                    emotional_indicators=[],
                )

            # Execute companion-aware memory search with emotional context
            companion_results = companion_search.search_with_emotional_context(
                query=query,
                user_id="default_user",  # Will be enhanced with actual user ID
                max_results=max_results * 2,  # Get more for filtering
                include_emotional_moments=True,
                min_relationship_significance=0.1,  # Lower threshold for broader search
            )

            # Search for similar conversations to provide context continuity
            similar_conversations = companion_search.search_similar_conversations(
                current_query=query,
                user_id="default_user",
                max_results=max_results // 2,
                similarity_threshold=0.6,  # Lower threshold for more context
            )

            # Convert companion results to SearchResult format
            results = []
            total_candidates = len(companion_results) + len(similar_conversations)

            # Process main companion search results
            for comp_result in companion_results:
                # Enhanced metadata with companion context
                enhanced_metadata = {
                    "conversation_id": (
                        comp_result.source.split("(")[1].split(")")[0]
                        if "(" in comp_result.source
                        else ""
                    ),
                    "timestamp": comp_result.timestamp.isoformat(),
                    "emotional_context": comp_result.emotional_context,
                    "relationship_significance": comp_result.relationship_significance,
                    "personalization_factors": comp_result.personalization_factors,
                    "memory_type": comp_result.memory_type,
                    "user_engagement_score": comp_result.user_engagement_score,
                    "companion_enhanced": True,
                    "search_type": "companion_memory",
                }

                # Apply temporal weighting for recent conversations (requirement 10.3)
                temporal_boost = self._calculate_temporal_boost(comp_result.timestamp)
                enhanced_score = comp_result.score * (1.0 + temporal_boost)

                search_result = SearchResult(
                    content=comp_result.content,
                    source=f"Memory: {comp_result.source}",  # Clear distinction (requirement 10.5)
                    score=enhanced_score,
                    result_type="companion_memory",
                    strategy="memory",
                    metadata=enhanced_metadata,
                )
                results.append(search_result)

            # Process similar conversation results for context continuity
            for sim_result in similar_conversations:
                enhanced_metadata = {
                    "conversation_id": (
                        sim_result.source.split("(")[1].split(")")[0]
                        if "(" in sim_result.source
                        else ""
                    ),
                    "timestamp": sim_result.timestamp.isoformat(),
                    "emotional_context": sim_result.emotional_context,
                    "relationship_significance": sim_result.relationship_significance,
                    "memory_type": sim_result.memory_type,
                    "companion_enhanced": True,
                    "search_type": "similar_conversation",
                    "similarity_context": True,
                }

                # Apply similarity boost for context continuity
                similarity_boost = 0.1  # Boost similar conversations for context
                enhanced_score = sim_result.score * (1.0 + similarity_boost)

                search_result = SearchResult(
                    content=f"[Similar Context] {sim_result.content}",
                    source=f"Previous: {sim_result.source}",  # Clear distinction
                    score=enhanced_score,
                    result_type="similar_memory",
                    strategy="memory",
                    metadata=enhanced_metadata,
                )
                results.append(search_result)

            # Apply advanced companion-aware ranking and filtering
            final_results = self._apply_advanced_memory_ranking(
                results, query, query_emotion, max_results
            )

            logger.debug(
                f"Enhanced memory search completed: {len(final_results)} results "
                f"from {total_candidates} candidates with companion integration"
            )

            return final_results, total_candidates

        except Exception as e:
            logger.error(f"Enhanced companion memory search failed: {e}")
            # Fallback to basic memory search
            return self._basic_memory_search(query, max_results, min_score, request_id)

    def _basic_memory_search(
        self, query: str, max_results: int, min_score: float, request_id: str
    ) -> Tuple[List[SearchResult], int]:
        """Fallback basic memory search without companion features."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.encode(
                text=query, instruction="query", request_id=request_id
            )

            # Search memory collection
            search_results = self.vector_db.search(
                collection_name=self.memory_collection,
                query_vector=query_embedding,
                limit=max_results,
                score_threshold=min_score * 0.9,
            )

            # Convert to SearchResult objects
            results = []
            for result in search_results:
                payload = result.payload

                # Format conversation turn as content
                question = payload.get("question", "")
                answer = payload.get("answer", "")
                content = f"Q: {question}\nA: {answer}"

                search_result = SearchResult(
                    content=content,
                    source=f"Conversation ({payload.get('timestamp', 'Unknown')})",
                    score=result.score,
                    result_type="memory",
                    strategy="memory",
                    metadata={
                        "conversation_id": payload.get("conversation_id", ""),
                        "turn_id": payload.get("turn_id", ""),
                        "timestamp": payload.get("timestamp", ""),
                        "feedback_rating": payload.get("feedback_rating"),
                    },
                )
                results.append(search_result)

            return results, len(search_results)

        except Exception as e:
            logger.error(f"Basic memory search failed: {e}")
            return [], 0

    def _contextual_search(
        self,
        query: str,
        conversation_context: Optional[str],
        max_results: int,
        min_score: float,
        request_id: str,
    ) -> Tuple[List[SearchResult], int]:
        """
        Execute contextual search using conversation context and companion features for personalization.

        Enhanced to use companion relationship data and user preferences for better personalization.
        """
        try:
            from morgan.companion.relationship_manager import (
                CompanionRelationshipManager,
            )
            from morgan.intelligence.core.intelligence_engine import (
                get_emotional_intelligence_engine,
            )

            # Get companion manager for personalization
            companion_manager = CompanionRelationshipManager()
            emotional_engine = get_emotional_intelligence_engine()

            # Build enhanced contextual query
            enhanced_context = self._build_enhanced_context(
                query, conversation_context, companion_manager, emotional_engine
            )

            if not enhanced_context:
                # Fall back to standard semantic search
                return self._standard_semantic_search(
                    query, max_results, min_score, request_id
                )

            # Generate embedding for enhanced contextual query
            query_embedding = self.embedding_service.encode(
                text=enhanced_context, instruction="query", request_id=request_id
            )

            # Search knowledge collection with enhanced context
            search_results = self.vector_db.search(
                collection_name=self.knowledge_collection,
                query_vector=query_embedding,
                limit=max_results * 2,  # Get more results for personalization filtering
                score_threshold=min_score
                * 0.9,  # Lower threshold for contextual search
            )

            # Apply companion-aware result personalization
            personalized_results = self._apply_companion_personalization(
                search_results, query, companion_manager
            )

            # Convert to SearchResult objects with personalization metadata
            results = []
            for result, personalization_score in personalized_results:
                search_result = self._convert_to_search_result(result, "contextual")

                # Enhance score with personalization
                search_result.score = min(
                    1.0, search_result.score * (1.0 + personalization_score)
                )

                # Add personalization metadata
                search_result.metadata.update(
                    {
                        "personalization_score": personalization_score,
                        "original_score": result.score,
                        "enhanced_with_companion_context": True,
                    }
                )

                results.append(search_result)

            # Sort by enhanced score and limit results
            results.sort(key=lambda r: r.score, reverse=True)

            return results[:max_results], len(search_results)

        except Exception as e:
            logger.error(f"Enhanced contextual search failed: {e}")
            # Fall back to basic contextual search
            return self._basic_contextual_search(
                query, conversation_context, max_results, min_score, request_id
            )

    def _build_enhanced_context(
        self,
        query: str,
        conversation_context: Optional[str],
        companion_manager,
        emotional_engine,
    ) -> str:
        """Build enhanced context using companion and emotional intelligence."""
        try:
            context_parts = []

            # Add original conversation context
            if conversation_context:
                context_parts.append(f"Recent conversation: {conversation_context}")

            # Try to get user profile for personalization (simplified approach)
            # In a full implementation, this would use actual user ID
            user_id = "default_user"  # Placeholder - would come from request context

            # Add user interests and preferences if available
            try:
                # This is a simplified approach - in practice, you'd get the actual user profile
                context_parts.append(
                    "User interests: technology, programming, learning"
                )
                context_parts.append("Communication style: technical and detailed")
            except Exception:
                pass  # Continue without user profile if not available

            # Add emotional context if detectable from query
            try:
                from morgan.intelligence.core.models import ConversationContext

                conv_context = ConversationContext(
                    user_id=user_id,
                    conversation_id="search_context",
                    message_text=query,
                    timestamp=datetime.utcnow(),
                )

                emotional_state = emotional_engine.analyze_emotion(query, conv_context)

                if emotional_state.intensity > 0.5:
                    context_parts.append(
                        f"Emotional context: {emotional_state.primary_emotion.value} "
                        f"(intensity: {emotional_state.intensity:.2f})"
                    )
            except Exception:
                pass  # Continue without emotional context if not available

            # Combine all context parts
            if context_parts:
                enhanced_context = " | ".join(context_parts) + f" | Query: {query}"
                return enhanced_context
            else:
                return query

        except Exception as e:
            logger.warning(f"Failed to build enhanced context: {e}")
            return query

    def _apply_companion_personalization(
        self, search_results: List, query: str, companion_manager
    ) -> List[Tuple[Any, float]]:
        """Apply companion-based personalization to search results."""
        try:
            personalized_results = []

            for result in search_results:
                payload = result.payload
                content = payload.get("content", "")
                source = payload.get("source", "")
                payload.get("category", "")

                personalization_score = 0.0

                # Boost based on content category matching user interests
                # This is simplified - in practice, you'd use actual user profile
                technical_keywords = [
                    "python",
                    "javascript",
                    "docker",
                    "api",
                    "database",
                    "programming",
                    "code",
                    "development",
                    "software",
                ]

                content_lower = content.lower()
                matching_keywords = sum(
                    1 for keyword in technical_keywords if keyword in content_lower
                )

                if matching_keywords > 0:
                    personalization_score += min(0.3, matching_keywords * 0.1)

                # Boost based on source type preferences
                if "documentation" in source.lower() or ".md" in source:
                    personalization_score += 0.1  # Prefer documentation

                # Boost based on content depth (longer content often more comprehensive)
                if len(content) > 500:
                    personalization_score += 0.05
                elif len(content) > 1000:
                    personalization_score += 0.1

                # Boost based on query relevance (simple keyword matching)
                query_words = query.lower().split()
                query_matches = sum(
                    1 for word in query_words if len(word) > 3 and word in content_lower
                )

                if query_matches > 0:
                    personalization_score += min(0.2, query_matches * 0.05)

                personalized_results.append((result, personalization_score))

            # Sort by personalization score
            personalized_results.sort(key=lambda x: x[1], reverse=True)

            return personalized_results

        except Exception as e:
            logger.warning(f"Failed to apply companion personalization: {e}")
            return [(result, 0.0) for result in search_results]

    def _basic_contextual_search(
        self,
        query: str,
        conversation_context: Optional[str],
        max_results: int,
        min_score: float,
        request_id: str,
    ) -> Tuple[List[SearchResult], int]:
        """Fallback basic contextual search without companion features."""
        try:
            if not conversation_context:
                return self._standard_semantic_search(
                    query, max_results, min_score, request_id
                )

            # Enhance query with conversation context
            contextual_query = f"Context: {conversation_context}\nQuery: {query}"

            query_embedding = self.embedding_service.encode(
                text=contextual_query, instruction="query", request_id=request_id
            )

            # Search knowledge collection
            search_results = self.vector_db.search(
                collection_name=self.knowledge_collection,
                query_vector=query_embedding,
                limit=max_results,
                score_threshold=min_score,
            )

            # Convert to SearchResult objects
            results = []
            for result in search_results:
                search_result = self._convert_to_search_result(result, "contextual")
                results.append(search_result)

            return results, len(search_results)

        except Exception as e:
            logger.error(f"Basic contextual search failed: {e}")
            return [], 0

    def _fusion_results(
        self, strategy_results: List[List[SearchResult]], max_results: int
    ) -> List[SearchResult]:
        """
        Fuse results from multiple strategies using Reciprocal Rank Fusion.

        Implements the RRF algorithm with the following enhancements:
        - RRF formula: score = Σ(1 / (k + rank)) where k=60
        - Intelligent result deduplication using cosine similarity >95%
        - Result ranking and merging system with strategy weighting
        - Performance optimization for large result sets

        Args:
            strategy_results: List of result lists from different strategies
            max_results: Maximum number of results to return

        Returns:
            List of fused SearchResult objects with RRF scores
        """
        if not strategy_results or len(strategy_results) == 1:
            # No fusion needed for single strategy
            return strategy_results[0] if strategy_results else []

        logger.debug(
            f"Starting RRF fusion for {len(strategy_results)} strategies with "
            f"{sum(len(results) for results in strategy_results)} total results"
        )

        # Step 1: Collect all unique results by content signature with deduplication
        result_map = {}  # signature -> SearchResult (best version)
        result_ranks = defaultdict(
            list
        )  # signature -> [(strategy_idx, rank, original_score), ...]
        strategy_weights = self._calculate_strategy_weights(strategy_results)

        for strategy_idx, results in enumerate(strategy_results):
            for rank, result in enumerate(results):
                signature = self._get_result_signature(result)

                # Store the best version of each unique result
                if signature not in result_map:
                    result_map[signature] = result
                else:
                    # Keep the result with higher original score
                    if result.score > result_map[signature].score:
                        result_map[signature] = result

                # Track all occurrences for RRF calculation
                result_ranks[signature].append((strategy_idx, rank, result.score))

        # Step 2: Calculate RRF scores with strategy weighting
        fused_results = []
        for signature, result in result_map.items():
            rrf_score = 0.0
            strategy_count = 0
            total_original_score = 0.0

            # Calculate RRF score: Σ(weight * (1 / (k + rank)))
            for strategy_idx, rank, original_score in result_ranks[signature]:
                strategy_weight = strategy_weights.get(strategy_idx, 1.0)
                rrf_contribution = strategy_weight * (1.0 / (self.rrf_k + rank))
                rrf_score += rrf_contribution
                strategy_count += 1
                total_original_score += original_score

            # Normalize by number of strategies that found this result
            # Results found by multiple strategies get boosted
            strategy_boost = 1.0 + (
                0.1 * (strategy_count - 1)
            )  # 10% boost per additional strategy
            rrf_score *= strategy_boost

            # Combine RRF score with average original score for final ranking
            avg_original_score = total_original_score / strategy_count
            final_score = (0.7 * rrf_score) + (0.3 * avg_original_score)

            # Create enhanced result with RRF metadata
            fused_result = SearchResult(
                content=result.content,
                source=result.source,
                score=result.score,  # Keep original score
                result_type=result.result_type,
                metadata=result.metadata.copy(),
                strategy=self._get_combined_strategy_name(result_ranks[signature]),
                rrf_score=final_score,
                hierarchical_scores=result.hierarchical_scores.copy(),
                dedup_signature=signature,
            )

            # Add RRF metadata
            fused_result.metadata.update(
                {
                    "rrf_raw_score": rrf_score,
                    "strategy_count": strategy_count,
                    "strategy_boost": strategy_boost,
                    "avg_original_score": avg_original_score,
                    "strategies_found_in": [
                        strategy_idx for strategy_idx, _, _ in result_ranks[signature]
                    ],
                }
            )

            fused_results.append(fused_result)

        # Step 3: Sort by final RRF score
        fused_results.sort(key=lambda r: r.rrf_score, reverse=True)

        # Step 4: Apply intelligent deduplication using cosine similarity
        deduplicated_results = self._apply_rrf_deduplication(fused_results)

        logger.debug(
            f"RRF fusion completed: {sum(len(results) for results in strategy_results)} → "
            f"{len(fused_results)} unique → {len(deduplicated_results)} deduplicated results"
        )

        return deduplicated_results[:max_results]

    def _calculate_strategy_weights(
        self, strategy_results: List[List[SearchResult]]
    ) -> Dict[int, float]:
        """
        Calculate weights for different strategies based on result quality.

        Strategies with higher average scores get higher weights in RRF calculation.
        """
        weights = {}

        for strategy_idx, results in enumerate(strategy_results):
            if not results:
                weights[strategy_idx] = 0.5  # Low weight for empty results
                continue

            # Calculate average score for this strategy
            avg_score = sum(r.score for r in results) / len(results)

            # Weight based on average score and result count
            result_count_factor = min(
                1.0, len(results) / 10.0
            )  # Normalize by expected result count
            weights[strategy_idx] = (0.7 * avg_score) + (0.3 * result_count_factor)

        return weights

    def _get_combined_strategy_name(
        self, strategy_ranks: List[Tuple[int, int, float]]
    ) -> str:
        """
        Generate a combined strategy name for results found by multiple strategies.
        """
        strategy_names = {
            0: "semantic",
            1: "keyword",
            2: "category",
            3: "temporal",
            4: "memory",
            5: "contextual",
        }

        strategies = sorted({strategy_idx for strategy_idx, _, _ in strategy_ranks})
        strategy_labels = [
            strategy_names.get(idx, f"strategy_{idx}") for idx in strategies
        ]

        if len(strategy_labels) == 1:
            return strategy_labels[0]
        else:
            return f"fusion({'+'.join(strategy_labels)})"

    def _apply_rrf_deduplication(
        self, results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Apply intelligent deduplication using cosine similarity >95% threshold.

        This is more sophisticated than the general deduplication as it:
        1. Preserves the highest RRF scored result among duplicates
        2. Uses content embeddings for similarity comparison
        3. Considers both content and source similarity
        """
        if len(results) <= 1:
            return results

        deduplicated = []
        seen_embeddings = []

        for result in results:
            is_duplicate = False

            try:
                # Skip deduplication if embedding service is not available or fails
                if (
                    not hasattr(self, "embedding_service")
                    or self.embedding_service is None
                ):
                    deduplicated.append(result)
                    continue

                # Generate embedding for similarity comparison
                # Use more content for better deduplication accuracy
                content_for_embedding = result.content[
                    :1000
                ]  # Use more content than general dedup
                result_embedding = self.embedding_service.encode(
                    text=content_for_embedding, instruction="document"
                )

                # Ensure we got a valid embedding
                if not result_embedding or not isinstance(result_embedding, list):
                    deduplicated.append(result)
                    continue

                # Check similarity with previously seen results
                for seen_embedding in seen_embeddings:
                    similarity = self._calculate_similarity(
                        result_embedding, seen_embedding
                    )
                    if similarity > self.similarity_threshold:  # >95% similarity
                        is_duplicate = True
                        logger.debug(
                            f"RRF deduplication: Removing duplicate result "
                            f"(similarity: {similarity:.3f}) from {result.source}"
                        )
                        break

                if not is_duplicate:
                    deduplicated.append(result)
                    seen_embeddings.append(result_embedding)

            except Exception as e:
                logger.warning(
                    f"RRF deduplication failed for result from {result.source}: {e}"
                )
                # Include result if we can't check for duplicates to avoid losing content
                deduplicated.append(result)

        return deduplicated

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove duplicate results using cosine similarity threshold.

        Removes results with >95% similarity to higher-ranked results.
        """
        if len(results) <= 1:
            return results

        deduplicated = []
        seen_embeddings = []

        for result in results:
            is_duplicate = False

            # Get embedding for this result's content
            try:
                result_embedding = self.embedding_service.encode(
                    text=result.content[:500],  # Use first 500 chars for efficiency
                    instruction="document",
                )

                # Check similarity with previously seen results
                for seen_embedding in seen_embeddings:
                    similarity = self._calculate_similarity(
                        result_embedding, seen_embedding
                    )
                    if similarity > self.similarity_threshold:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    deduplicated.append(result)
                    seen_embeddings.append(result_embedding)

            except Exception as e:
                logger.warning(f"Failed to check duplicate for result: {e}")
                # Include result if we can't check for duplicates
                deduplicated.append(result)

        if len(deduplicated) != len(results):
            logger.debug(
                f"Deduplication: {len(results)} → {len(deduplicated)} "
                f"({len(results) - len(deduplicated)} duplicates removed)"
            )

        return deduplicated

    def _final_ranking(
        self, results: List[SearchResult], max_results: int, min_score: float
    ) -> List[SearchResult]:
        """
        Apply final ranking and filtering to results.
        """
        # Filter by minimum score
        filtered_results = []
        for result in results:
            best_score = result.get_best_score()
            if best_score >= min_score:
                filtered_results.append(result)

        # Sort by best available score
        filtered_results.sort(key=lambda r: r.get_best_score(), reverse=True)

        # Limit to max results
        return filtered_results[:max_results]

    def _generate_query_embeddings(
        self, query: str, request_id: str
    ) -> Dict[str, List[float]]:
        """Generate embeddings for all hierarchical scales."""
        # For query, we use the same text for all scales but could enhance this
        # by creating scale-specific query representations

        coarse_query = f"Query: {query}"  # Simple for now
        medium_query = query
        fine_query = query

        # Generate embeddings
        embeddings = self.embedding_service.encode_batch(
            [coarse_query, medium_query, fine_query],
            instruction="query",
            request_id=request_id,
        )

        return {"coarse": embeddings[0], "medium": embeddings[1], "fine": embeddings[2]}

    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        try:
            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(vec1, vec2))

            # Calculate norms
            norm_a = sum(a * a for a in vec1) ** 0.5
            norm_b = sum(b * b for b in vec2) ** 0.5

            if norm_a == 0 or norm_b == 0:
                return 0.0

            similarity = dot_product / (norm_a * norm_b)
            return float(similarity)

        except Exception as e:
            logger.warning("Similarity calculation failed: %s", e)
            return 0.0

    def _convert_to_search_result(self, db_result, strategy: str) -> SearchResult:
        """Convert database result to SearchResult object."""
        payload = db_result.payload

        return SearchResult(
            content=payload.get("content", ""),
            source=payload.get("source", "Unknown"),
            score=db_result.score,
            result_type="knowledge",
            strategy=strategy,
            metadata=payload.get("metadata", {}),
        )

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query text."""
        # Simple keyword extraction
        # Remove common stop words and extract meaningful terms
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "how",
            "what",
            "when",
            "where",
            "why",
            "which",
            "who",
        }

        # Extract words
        words = re.findall(r"\b\w+\b", query.lower())

        # Filter stop words and short words
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]

        return keywords

    def _detect_query_category(self, query: str) -> Optional[str]:
        """Detect the category of a query for category-based search."""
        query_lower = query.lower()

        # Category patterns
        category_patterns = {
            "code": ["function", "class", "method", "code", "programming", "script"],
            "documentation": ["guide", "tutorial", "how to", "documentation", "manual"],
            "api": ["api", "endpoint", "request", "response", "authentication"],
            "configuration": ["config", "setting", "environment", "setup", "install"],
            "troubleshooting": ["error", "issue", "problem", "debug", "fix", "solve"],
        }

        # Count matches for each category
        category_scores = {}
        for category, patterns in category_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                category_scores[category] = score

        # Return category with highest score
        if category_scores:
            return max(category_scores, key=category_scores.get)

        return None

    def _get_result_signature(self, result: SearchResult) -> str:
        """Generate a signature for result deduplication."""
        # Use content hash for deduplication
        content_key = result.content[:200].lower().strip()
        source_key = result.source.lower().strip()
        return f"{hash(content_key)}_{hash(source_key)}"

    def _extract_emotional_context_from_turn(
        self, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract emotional context from conversation turn payload."""
        try:
            # Look for emotional indicators in the question and answer
            question = payload.get("question", "")
            answer = payload.get("answer", "")

            # Simple emotional context extraction
            emotional_indicators = []

            # Check for emotional words in question
            emotional_words = [
                "happy",
                "sad",
                "angry",
                "frustrated",
                "excited",
                "worried",
                "confused",
                "grateful",
                "disappointed",
                "surprised",
                "afraid",
            ]

            for word in emotional_words:
                if word in question.lower() or word in answer.lower():
                    emotional_indicators.append(word)

            # Check for emotional punctuation
            if "!" in question:
                emotional_indicators.append("excitement_or_emphasis")
            if "?" in question and len(question.split("?")) > 2:
                emotional_indicators.append("confusion_or_curiosity")

            return {
                "emotional_indicators": emotional_indicators,
                "has_emotional_content": len(emotional_indicators) > 0,
                "question_length": len(question),
                "answer_length": len(answer),
            }

        except Exception as e:
            logger.warning(f"Failed to extract emotional context: {e}")
            return {}

    def _calculate_relationship_score_boost(
        self, payload: Dict[str, Any], companion_manager
    ) -> float:
        """Calculate relationship-based score boost for memory results."""
        try:
            # Get conversation metadata
            payload.get("conversation_id", "")
            feedback_rating = payload.get("feedback_rating")
            timestamp = payload.get("timestamp", "")

            boost = 0.0

            # Boost based on positive feedback
            if feedback_rating and feedback_rating >= 4:
                boost += 0.2  # 20% boost for highly rated interactions
            elif feedback_rating and feedback_rating >= 3:
                boost += 0.1  # 10% boost for positively rated interactions

            # Boost based on recency (more recent conversations are more relevant)
            if timestamp:
                try:
                    from datetime import datetime

                    turn_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    current_time = datetime.utcnow()
                    days_ago = (current_time - turn_time).days

                    # Boost recent conversations (within last 30 days)
                    if days_ago <= 7:
                        boost += 0.15  # 15% boost for last week
                    elif days_ago <= 30:
                        boost += 0.1  # 10% boost for last month

                except (ValueError, TypeError):
                    pass

            # Boost based on conversation length (longer conversations indicate engagement)
            question = payload.get("question", "")
            answer = payload.get("answer", "")
            total_length = len(question) + len(answer)

            if total_length > 500:
                boost += 0.1  # 10% boost for substantial conversations
            elif total_length > 200:
                boost += 0.05  # 5% boost for moderate conversations

            return min(boost, 0.5)  # Cap boost at 50%

        except Exception as e:
            logger.warning(f"Failed to calculate relationship boost: {e}")
            return 0.0

    def _calculate_emotional_score_boost(
        self, emotional_context: Dict[str, Any]
    ) -> float:
        """Calculate emotional-based score boost for enhanced memory results."""
        try:
            boost = 0.0

            # Boost based on emotional intensity
            intensity = emotional_context.get("intensity", 0.0)
            if intensity > 0.7:
                boost += 0.2  # 20% boost for high emotional intensity
            elif intensity > 0.5:
                boost += 0.1  # 10% boost for moderate emotional intensity

            # Boost based on emotional confidence
            confidence = emotional_context.get("confidence", 0.0)
            if confidence > 0.8:
                boost += 0.1  # 10% boost for high confidence emotions

            # Boost based on specific emotions that indicate important moments
            primary_emotion = emotional_context.get("primary_emotion", "")
            important_emotions = ["joy", "sadness", "surprise", "fear"]

            if primary_emotion in important_emotions:
                boost += 0.15  # 15% boost for emotionally significant moments

            return min(boost, 0.4)  # Cap boost at 40%

        except Exception as e:
            logger.warning(f"Failed to calculate emotional boost: {e}")
            return 0.0

    def _calculate_temporal_boost(self, timestamp: datetime) -> float:
        """
        Calculate temporal boost for recent conversations (requirement 10.3).

        Args:
            timestamp: Timestamp of the conversation/memory

        Returns:
            Boost factor (0.0 to 0.3)
        """
        try:
            current_time = datetime.utcnow()
            time_diff = current_time - timestamp
            days_ago = time_diff.days

            # Apply temporal weighting - more recent conversations get higher boost
            if days_ago <= 1:
                return 0.3  # Very recent (today/yesterday)
            elif days_ago <= 7:
                return 0.2  # Recent (this week)
            elif days_ago <= 30:
                return 0.1  # Somewhat recent (this month)
            elif days_ago <= 90:
                return 0.05  # Moderately old (this quarter)
            else:
                return 0.0  # Old conversations get no temporal boost

        except Exception as e:
            logger.warning(f"Failed to calculate temporal boost: {e}")
            return 0.0

    def _apply_advanced_memory_ranking(
        self,
        results: List[SearchResult],
        query: str,
        query_emotion: Any,
        max_results: int,
    ) -> List[SearchResult]:
        """
        Apply advanced companion-aware ranking with emotional and relationship context.

        Implements enhanced ranking that considers:
        - Emotional resonance between query and memory
        - Relationship significance
        - Temporal relevance
        - User engagement patterns
        """
        try:
            if not results:
                return results

            # Group results by type for balanced ranking
            companion_memories = [
                r for r in results if r.result_type == "companion_memory"
            ]
            similar_memories = [r for r in results if r.result_type == "similar_memory"]

            # Apply advanced scoring to companion memories
            for result in companion_memories:
                enhancement_score = 0.0

                # Emotional resonance boost
                emotional_context = result.metadata.get("emotional_context", {})
                if emotional_context and hasattr(query_emotion, "primary_emotion"):
                    emotional_boost = self._calculate_emotional_resonance(
                        query_emotion, emotional_context
                    )
                    enhancement_score += emotional_boost

                # Relationship significance boost
                relationship_sig = result.metadata.get("relationship_significance", 0.0)
                enhancement_score += relationship_sig * 0.2

                # User engagement boost
                engagement_score = result.metadata.get("user_engagement_score", 0.0)
                enhancement_score += engagement_score * 0.15

                # Apply enhancement
                result.score = min(1.0, result.score + enhancement_score)
                result.metadata["advanced_enhancement"] = enhancement_score

            # Sort all results by enhanced score
            all_results = companion_memories + similar_memories
            all_results.sort(key=lambda r: r.score, reverse=True)

            # Apply intelligent result mixing for diversity
            final_results = self._apply_intelligent_result_mixing(
                all_results, max_results
            )

            # Apply final deduplication
            deduplicated_results = self._deduplicate_memory_results(final_results)

            return deduplicated_results[:max_results]

        except Exception as e:
            logger.warning(f"Failed to apply advanced memory ranking: {e}")
            return results[:max_results]

    def _calculate_emotional_resonance(
        self, query_emotion: Any, memory_emotional_context: Dict[str, Any]
    ) -> float:
        """
        Calculate emotional resonance between query and memory.

        Args:
            query_emotion: Emotional state of the query
            memory_emotional_context: Emotional context of the memory

        Returns:
            Resonance boost (0.0 to 0.2)
        """
        try:
            boost = 0.0

            # Get query emotion details
            query_primary = getattr(query_emotion, "primary_emotion", None)
            query_intensity = getattr(query_emotion, "intensity", 0.0)

            if not query_primary:
                return 0.0

            # Get memory emotion details
            memory_emotions = memory_emotional_context.get("detected_emotions", [])
            memory_intensity = memory_emotional_context.get("intensity", 0.0)

            # Check for emotional alignment
            query_emotion_str = (
                query_primary.value
                if hasattr(query_primary, "value")
                else str(query_primary)
            )

            if query_emotion_str in memory_emotions:
                # Direct emotional match
                boost += 0.15 * min(query_intensity, memory_intensity)
            elif any(
                emotion in memory_emotions
                for emotion in ["joy", "sadness", "fear", "anger"]
            ):
                # Any strong emotion match
                boost += 0.08 * min(query_intensity, memory_intensity)

            # Boost for high emotional intensity in memory (emotionally significant moments)
            if memory_intensity > 0.7:
                boost += 0.05

            return min(0.2, boost)

        except Exception as e:
            logger.warning(f"Failed to calculate emotional resonance: {e}")
            return 0.0

    def _apply_intelligent_result_mixing(
        self, results: List[SearchResult], max_results: int
    ) -> List[SearchResult]:
        """
        Apply intelligent mixing of different result types for diversity.

        Ensures a good mix of:
        - High-scoring companion memories
        - Similar conversations for context
        - Different time periods
        - Various emotional contexts
        """
        try:
            if len(results) <= max_results:
                return results

            mixed_results = []
            companion_results = [
                r for r in results if r.result_type == "companion_memory"
            ]
            similar_results = [r for r in results if r.result_type == "similar_memory"]

            # Ensure we get the best companion memories first
            companion_count = min(
                len(companion_results), max_results * 3 // 4
            )  # 75% companion
            similar_count = min(len(similar_results), max_results - companion_count)

            # Add top companion memories
            mixed_results.extend(companion_results[:companion_count])

            # Add similar conversations for context
            mixed_results.extend(similar_results[:similar_count])

            # Sort final mix by score
            mixed_results.sort(key=lambda r: r.score, reverse=True)

            return mixed_results[:max_results]

        except Exception as e:
            logger.warning(f"Failed to apply intelligent result mixing: {e}")
            return results[:max_results]

    def _apply_companion_aware_ranking(
        self, results: List[SearchResult], query: str
    ) -> List[SearchResult]:
        """Apply companion-aware ranking to memory search results."""
        try:
            # Group results by type for balanced ranking
            enhanced_memories = [
                r for r in results if r.result_type == "enhanced_memory"
            ]
            conversation_memories = [r for r in results if r.result_type == "memory"]

            # Sort each group by score
            enhanced_memories.sort(key=lambda r: r.score, reverse=True)
            conversation_memories.sort(key=lambda r: r.score, reverse=True)

            # Interleave results to provide balanced mix
            final_results = []
            max_length = max(len(enhanced_memories), len(conversation_memories))

            for i in range(max_length):
                # Prefer enhanced memories slightly (they have richer context)
                if i < len(enhanced_memories):
                    final_results.append(enhanced_memories[i])
                if i < len(conversation_memories):
                    final_results.append(conversation_memories[i])

            # Apply final deduplication based on content similarity
            deduplicated_results = self._deduplicate_memory_results(final_results)

            return deduplicated_results

        except Exception as e:
            logger.warning(f"Failed to apply companion-aware ranking: {e}")
            return results

    def _deduplicate_memory_results(
        self, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Deduplicate memory results based on content and conversation context."""
        if len(results) <= 1:
            return results

        deduplicated = []
        seen_conversations = set()
        seen_content_hashes = set()

        for result in results:
            # Create deduplication keys
            conversation_id = result.metadata.get("conversation_id", "")
            content_hash = hash(result.content[:200].lower().strip())

            # Skip if we've seen very similar content from the same conversation
            conversation_content_key = f"{conversation_id}_{content_hash}"

            if conversation_content_key not in seen_content_hashes:
                seen_content_hashes.add(conversation_content_key)
                seen_conversations.add(conversation_id)
                deduplicated.append(result)
            elif result.result_type == "enhanced_memory":
                # Prefer enhanced memories over basic conversation turns
                # Replace if current result is enhanced and previous wasn't
                for i, existing in enumerate(deduplicated):
                    if (
                        existing.metadata.get("conversation_id") == conversation_id
                        and existing.result_type == "memory"
                    ):
                        deduplicated[i] = result
                        break

        return deduplicated

    def _update_search_stats(self, search_time: float, candidates: int, filtered: int):
        """Update performance statistics."""
        self.search_stats["total_searches"] += 1

        # Update running averages
        total = self.search_stats["total_searches"]
        self.search_stats["average_search_time"] = (
            self.search_stats["average_search_time"] * (total - 1) + search_time
        ) / total
        self.search_stats["average_candidates"] = (
            self.search_stats["average_candidates"] * (total - 1) + candidates
        ) / total

        if candidates > 0:
            reduction = 1.0 - (filtered / candidates)
            self.search_stats["average_reduction"] = (
                self.search_stats["average_reduction"] * (total - 1) + reduction
            ) / total

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.search_stats.copy()


# Singleton instance for global access
_multi_stage_search_engine = None


def get_multi_stage_search_engine() -> MultiStageSearchEngine:
    """
    Get singleton multi-stage search engine instance.

    Returns:
        Shared MultiStageSearchEngine instance
    """
    global _multi_stage_search_engine

    if _multi_stage_search_engine is None:
        _multi_stage_search_engine = MultiStageSearchEngine()

    return _multi_stage_search_engine
