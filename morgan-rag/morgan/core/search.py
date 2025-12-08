"""
Smart Search - Intelligent Information Retrieval

Simple, focused module for finding relevant information across all of Morgan's knowledge.

KISS Principle: One responsibility - find the most relevant information for any query.
Human-First: Return results that actually help humans, not just technically accurate matches.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from morgan.config import get_settings
from morgan.optimization.async_processor import get_async_processor
from morgan.optimization.emotional_optimizer import get_emotional_optimizer
from morgan.services.embedding_service import get_embedding_service
from morgan.utils.error_decorators import handle_search_errors, monitor_performance
from morgan.utils.error_handling import SearchError
from morgan.utils.logger import get_logger
from morgan.vector_db.client import VectorDBClient

logger = get_logger(__name__)


class SearchStrategy(Enum):
    """Different search strategies for different types of queries."""

    KNOWLEDGE = "knowledge"  # Search knowledge base
    MEMORY = "memory"  # Search conversation history
    HYBRID = "hybrid"  # Search both knowledge and memory
    SEMANTIC = "semantic"  # Pure semantic similarity
    KEYWORD = "keyword"  # Keyword-based search


@dataclass
class SearchResult:
    """
    A search result with human-friendly information.

    Simple structure that provides everything a human needs to understand
    where the information came from and how relevant it is.
    """

    content: str
    source: str
    score: float
    result_type: str  # "knowledge", "memory", "web", etc.
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Ensure metadata is always a dict."""
        if self.metadata is None:
            self.metadata = {}

    def summary(self, max_length: int = 100) -> str:
        """Get a short summary of the content."""
        if len(self.content) <= max_length:
            return self.content
        return self.content[: max_length - 3] + "..."


class SmartSearch:
    """
    Morgan's Smart Search Engine

    Finds relevant information by:
    - Understanding the intent behind queries
    - Searching across multiple knowledge sources
    - Ranking results by relevance and usefulness
    - Providing context about where information came from

    KISS: Single responsibility - be really good at finding relevant information.
    """

    def __init__(self):
        """Initialize smart search with optimization support."""
        self.settings = get_settings()
        self.embedding_service = get_embedding_service()
        self.vector_db = VectorDBClient()

        # Optimization components
        self.async_processor = get_async_processor()
        self.emotional_optimizer = get_emotional_optimizer()

        # Search settings
        self.default_max_results = getattr(
            self.settings, "morgan_default_search_results", 10
        )
        self.min_score_threshold = 0.7
        self.knowledge_collection = "morgan_knowledge"
        self.memory_collection = "morgan_turns"

        logger.info("Smart search initialized with optimization support")

    @handle_search_errors("find_relevant_info", "smart_search")
    @monitor_performance("find_relevant_info", "smart_search")
    def find_relevant_info(
        self,
        query: str,
        max_results: Optional[int] = None,
        strategy: SearchStrategy = SearchStrategy.HYBRID,
        min_score: Optional[float] = None,
        emotional_context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        use_enhanced_search: bool = True,
    ) -> List[SearchResult]:
        """
        Find information relevant to a query with enhanced emotional and companion awareness.

        This is the main search method enhanced with emotional intelligence and
        companion relationship features for more personalized results.

        Args:
            query: Natural language query
            max_results: Maximum number of results (uses default if None)
            strategy: Search strategy to use
            min_score: Minimum relevance score (uses default if None)
            emotional_context: Optional emotional context for result enhancement
            user_id: Optional user ID for companion-aware personalization
            use_enhanced_search: Whether to use enhanced multi-stage search

        Returns:
            List of relevant search results, ranked by relevance and enhanced with emotional/companion context

        Example:
            >>> search = SmartSearch()
            >>> emotional_ctx = {"primary_emotion": "frustration", "intensity": 0.8}
            >>> results = search.find_relevant_info(
            ...     "How do I deploy Docker containers?",
            ...     emotional_context=emotional_ctx,
            ...     user_id="user123"
            ... )
            >>> for result in results:
            ...     print(f"Score: {result.score:.2f}")
            ...     print(f"Enhanced: {result.metadata.get('emotional_enhanced', False)}")
        """
        if max_results is None:
            max_results = self.default_max_results

        if min_score is None:
            min_score = self.min_score_threshold

        logger.debug(
            f"Enhanced search for: '{query}' (strategy: {strategy.value}, "
            f"emotional_context: {emotional_context is not None}, "
            f"user_id: {user_id is not None})"
        )

        try:
            # Use enhanced multi-stage search if available and requested
            if use_enhanced_search and (emotional_context or user_id):
                try:
                    from morgan.search.multi_stage_search import (
                        get_multi_stage_search_engine,
                    )

                    multi_stage_engine = get_multi_stage_search_engine()

                    # Map strategy to multi-stage strategies
                    from morgan.search.multi_stage_search import (
                        SearchStrategy as MultiStageStrategy,
                    )

                    strategy_mapping = {
                        SearchStrategy.KNOWLEDGE: [MultiStageStrategy.SEMANTIC],
                        SearchStrategy.MEMORY: [MultiStageStrategy.MEMORY],
                        SearchStrategy.HYBRID: [
                            MultiStageStrategy.SEMANTIC,
                            MultiStageStrategy.MEMORY,
                            MultiStageStrategy.CONTEXTUAL,
                        ],
                        SearchStrategy.SEMANTIC: [MultiStageStrategy.SEMANTIC],
                    }

                    strategies = strategy_mapping.get(
                        strategy,
                        [MultiStageStrategy.SEMANTIC, MultiStageStrategy.MEMORY],
                    )

                    # Execute enhanced search
                    search_results = multi_stage_engine.search(
                        query=query,
                        max_results=max_results,
                        strategies=strategies,
                        min_score=min_score,
                        use_hierarchical=True,
                        emotional_context=emotional_context,
                        user_id=user_id,
                    )

                    # Convert to SearchResult objects
                    results = []
                    for result in search_results.results:
                        enhanced_result = SearchResult(
                            content=result.content,
                            source=result.source,
                            score=result.score,
                            result_type=result.result_type,
                            metadata={
                                **result.metadata,
                                "enhanced_multi_stage_search": True,
                                "search_time": search_results.search_time,
                                "strategies_used": search_results.strategies_used,
                                "fusion_applied": search_results.fusion_applied,
                            },
                        )
                        results.append(enhanced_result)

                    logger.debug(
                        f"Enhanced multi-stage search found {len(results)} results"
                    )
                    return results

                except Exception as e:
                    logger.warning(
                        f"Enhanced search failed, falling back to standard search: {e}"
                    )

            # Fallback to standard search strategies
            if strategy == SearchStrategy.KNOWLEDGE:
                results = self._search_knowledge_only(query, max_results, min_score)
            elif strategy == SearchStrategy.MEMORY:
                results = self._search_memory_only(query, max_results, min_score)
            elif strategy == SearchStrategy.HYBRID:
                results = self._search_hybrid(query, max_results, min_score)
            elif strategy == SearchStrategy.SEMANTIC:
                results = self._search_semantic(query, max_results, min_score)
            else:
                # Default to hybrid
                results = self._search_hybrid(query, max_results, min_score)

            # Post-process results for better human experience
            results = self._enhance_results(results, query)

            logger.debug(f"Found {len(results)} relevant results")
            return results

        except Exception as e:
            raise SearchError(
                f"Search failed for query '{query}': {e}",
                operation="find_relevant_info",
                component="smart_search",
                metadata={
                    "query": query[:100],  # Truncate long queries
                    "strategy": strategy.value,
                    "max_results": max_results,
                    "min_score": min_score,
                    "has_emotional_context": emotional_context is not None,
                    "has_user_id": user_id is not None,
                    "use_enhanced_search": use_enhanced_search,
                    "error_type": type(e).__name__,
                },
            ) from e

    def search_knowledge(
        self, query: str, max_results: int = 10, min_score: float = 0.7
    ) -> List[SearchResult]:
        """
        Search only the knowledge base (documents, not conversations).

        Args:
            query: Search query
            max_results: Maximum results to return
            min_score: Minimum relevance score

        Returns:
            List of knowledge-based search results
        """
        return self._search_knowledge_only(query, max_results, min_score)

    def search_conversations_optimized(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.7,
        user_id: str = "default_user",
        emotional_context: Optional[Dict[str, Any]] = None,
        use_async_processing: bool = True,
    ) -> List[SearchResult]:
        """
        Optimized conversation search with real-time emotional processing.

        Enhanced for companion interactions with:
        - Async processing for real-time responses
        - Optimized emotional context integration
        - Fast emotional state detection
        - Cached emotional patterns

        Args:
            query: Search query
            max_results: Maximum results to return
            min_score: Minimum relevance score
            user_id: User identifier for personalization
            emotional_context: Optional current emotional context
            use_async_processing: Use async processing for real-time performance

        Returns:
            List of optimized conversation search results
        """
        try:
            # Fast emotional state detection if context not provided
            if not emotional_context and query:
                emotional_context = self.emotional_optimizer.detect_emotion_fast(
                    text=query, user_id=user_id, use_cache=True
                ).__dict__

            # Use async processing for real-time companion interactions
            if use_async_processing:
                # Submit search as high-priority companion task
                task_id = self.async_processor.submit_companion_task(
                    self._search_conversations_with_emotion,
                    query=query,
                    max_results=max_results,
                    min_score=min_score,
                    user_id=user_id,
                    emotional_context=emotional_context,
                )

                # Wait for result with short timeout for real-time interaction
                result = self.async_processor.get_task_result(task_id, timeout=5.0)

                if result and result.success:
                    return result.result
                else:
                    logger.warning(
                        "Async conversation search failed or timed out, falling back to sync"
                    )

            # Fallback to synchronous processing
            return self._search_conversations_with_emotion(
                query=query,
                max_results=max_results,
                min_score=min_score,
                user_id=user_id,
                emotional_context=emotional_context,
            )

        except Exception as e:
            logger.error(f"Optimized conversation search failed: {e}")
            # Final fallback to basic search
            return self.search_conversations(
                query, max_results, min_score, user_id, True, emotional_context
            )

    def search_conversations(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.7,
        user_id: str = "default_user",
        include_emotional_context: bool = True,
        emotional_context: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search conversation history with enhanced companion features and emotional context integration.

        Args:
            query: Search query
            max_results: Maximum results to return
            min_score: Minimum relevance score
            user_id: User identifier for personalization
            include_emotional_context: Whether to include emotional context
            emotional_context: Optional current emotional context for enhanced ranking

        Returns:
            List of conversation-based search results enhanced with emotional and companion context
        """
        if include_emotional_context:
            try:
                from morgan.search.companion_memory_search import (
                    get_companion_memory_search_engine,
                )
                from morgan.search.multi_stage_search import (
                    get_multi_stage_search_engine,
                )

                # Use enhanced multi-stage search with emotional context
                multi_stage_engine = get_multi_stage_search_engine()

                # Search using memory strategy with emotional context
                from morgan.search.multi_stage_search import (
                    SearchStrategy as MultiStageStrategy,
                )

                search_results = multi_stage_engine.search(
                    query=query,
                    max_results=max_results,
                    strategies=[MultiStageStrategy.MEMORY],
                    min_score=min_score,
                    emotional_context=emotional_context,
                    user_id=user_id,
                )

                # Convert to SearchResult objects with enhanced metadata
                results = []
                for result in search_results.results:
                    enhanced_result = SearchResult(
                        content=result.content,
                        source=result.source,
                        score=result.score,
                        result_type=result.result_type,
                        metadata={
                            **result.metadata,
                            "enhanced_search": True,
                            "emotional_context_used": emotional_context is not None,
                            "companion_enhanced": True,
                            "search_strategy": result.strategy,
                        },
                    )
                    results.append(enhanced_result)

                # If enhanced search didn't return enough results, supplement with companion search
                if len(results) < max_results:
                    try:
                        companion_engine = get_companion_memory_search_engine()
                        companion_results = (
                            companion_engine.search_with_emotional_context(
                                query=query,
                                user_id=user_id,
                                max_results=max_results - len(results),
                                include_emotional_moments=True,
                                min_relationship_significance=0.0,
                            )
                        )

                        # Add companion results
                        for comp_result in companion_results:
                            search_result = SearchResult(
                                content=comp_result.content,
                                source=comp_result.source,
                                score=comp_result.score,
                                result_type=comp_result.result_type,
                                metadata={
                                    "emotional_context": comp_result.emotional_context,
                                    "relationship_significance": comp_result.relationship_significance,
                                    "personalization_factors": comp_result.personalization_factors,
                                    "memory_type": comp_result.memory_type,
                                    "user_engagement_score": comp_result.user_engagement_score,
                                    "timestamp": comp_result.timestamp.isoformat(),
                                    "companion_enhanced": True,
                                    "supplemental_result": True,
                                },
                            )
                            results.append(search_result)

                    except Exception as e:
                        logger.warning(f"Supplemental companion search failed: {e}")

                return results[:max_results]

            except Exception as e:
                logger.warning(
                    f"Enhanced conversation search failed, falling back to basic search: {e}"
                )

        # Fallback to basic memory search
        return self._search_memory_only(query, max_results, min_score)

    def find_similar_questions(
        self,
        question: str,
        max_results: int = 5,
        user_id: str = "default_user",
        use_companion_features: bool = True,
    ) -> List[SearchResult]:
        """
        Find similar questions from conversation history.

        Useful for suggesting related topics or finding previous answers
        to similar questions.

        Args:
            question: The question to find similar ones for
            max_results: Maximum number of similar questions
            user_id: User identifier for personalization
            use_companion_features: Whether to use companion features

        Returns:
            List of similar questions with their answers
        """
        try:
            if use_companion_features:
                try:
                    from morgan.search.companion_memory_search import (
                        get_companion_memory_search_engine,
                    )

                    # Use companion-aware similar conversation search
                    companion_engine = get_companion_memory_search_engine()
                    companion_results = companion_engine.search_similar_conversations(
                        current_query=question,
                        user_id=user_id,
                        max_results=max_results,
                        similarity_threshold=0.6,
                    )

                    # Convert to SearchResult objects
                    results = []
                    for comp_result in companion_results:
                        search_result = SearchResult(
                            content=comp_result.content,
                            source=comp_result.source,
                            score=comp_result.score,
                            result_type=comp_result.result_type,
                            metadata={
                                "emotional_context": comp_result.emotional_context,
                                "relationship_significance": comp_result.relationship_significance,
                                "memory_type": comp_result.memory_type,
                                "companion_enhanced": True,
                            },
                        )
                        results.append(search_result)

                    return results

                except Exception as e:
                    logger.warning(f"Companion similar questions search failed: {e}")

            # Fallback to basic search
            results = self._search_memory_only(question, max_results, min_score=0.6)

            # Filter to only include results that are actually questions
            question_results = []
            for result in results:
                # Simple heuristic: check if the content contains question words
                content_lower = result.content.lower()
                if any(
                    word in content_lower
                    for word in ["how", "what", "why", "when", "where", "which", "?"]
                ):
                    question_results.append(result)

            return question_results[:max_results]

        except Exception as e:
            logger.error(f"Failed to find similar questions: {e}")
            return []

    def search_relationship_memories(
        self,
        user_id: str = "default_user",
        max_results: int = 10,
        milestone_types: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Search for relationship-significant memories and milestones.

        Args:
            user_id: User identifier
            max_results: Maximum number of results
            milestone_types: Types of milestones to include

        Returns:
            List of relationship-focused search results
        """
        try:
            from morgan.search.companion_memory_search import (
                get_companion_memory_search_engine,
            )

            companion_engine = get_companion_memory_search_engine()
            companion_results = companion_engine.get_relationship_memories(
                user_id=user_id,
                milestone_types=milestone_types,
                max_results=max_results,
            )

            # Convert to SearchResult objects
            results = []
            for comp_result in companion_results:
                search_result = SearchResult(
                    content=comp_result.content,
                    source=comp_result.source,
                    score=comp_result.score,
                    result_type=comp_result.result_type,
                    metadata={
                        "emotional_context": comp_result.emotional_context,
                        "relationship_significance": comp_result.relationship_significance,
                        "personalization_factors": comp_result.personalization_factors,
                        "memory_type": comp_result.memory_type,
                        "user_engagement_score": comp_result.user_engagement_score,
                        "companion_enhanced": True,
                    },
                )
                results.append(search_result)

            return results

        except Exception as e:
            logger.error(f"Failed to search relationship memories: {e}")
            return []

    def get_personalized_memories(
        self,
        user_id: str = "default_user",
        max_results: int = 15,
        days_back: int = 30,
        memory_types: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Get personalized memories based on user relationship and preferences.

        Args:
            user_id: User identifier
            max_results: Maximum number of results
            days_back: Number of days to look back
            memory_types: Types of memories to include

        Returns:
            List of personalized memory results
        """
        try:
            from morgan.search.companion_memory_search import (
                get_companion_memory_search_engine,
            )

            companion_engine = get_companion_memory_search_engine()
            companion_results = companion_engine.get_personalized_memories(
                user_id=user_id,
                memory_types=memory_types,
                max_results=max_results,
                days_back=days_back,
            )

            # Convert to SearchResult objects
            results = []
            for comp_result in companion_results:
                search_result = SearchResult(
                    content=comp_result.content,
                    source=comp_result.source,
                    score=comp_result.score,
                    result_type=comp_result.result_type,
                    metadata={
                        "emotional_context": comp_result.emotional_context,
                        "relationship_significance": comp_result.relationship_significance,
                        "personalization_factors": comp_result.personalization_factors,
                        "memory_type": comp_result.memory_type,
                        "user_engagement_score": comp_result.user_engagement_score,
                        "timestamp": comp_result.timestamp.isoformat(),
                        "companion_enhanced": True,
                    },
                )
                results.append(search_result)

            return results

        except Exception as e:
            logger.error(f"Failed to get personalized memories: {e}")
            return []

    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """
        Get search suggestions based on a partial query.

        Args:
            partial_query: Partial search query

        Returns:
            List of suggested complete queries
        """
        try:
            # Simple approach: find similar content and extract key phrases
            if len(partial_query) < 3:
                return []

            # Search for partial matches
            results = self.find_relevant_info(
                query=partial_query, max_results=20, min_score=0.5
            )

            # Extract suggestions from results
            suggestions = set()
            for result in results:
                # Extract sentences that contain the partial query
                sentences = result.content.split(".")
                for sentence in sentences:
                    sentence = sentence.strip()
                    if partial_query.lower() in sentence.lower() and len(
                        sentence
                    ) > len(partial_query):
                        # Clean up the sentence to make it a good suggestion
                        if len(sentence) < 100:  # Keep suggestions short
                            suggestions.add(sentence)

            return list(suggestions)[:10]  # Return top 10 suggestions

        except Exception as e:
            logger.error(f"Failed to get search suggestions: {e}")
            return []

    def _search_knowledge_only(
        self, query: str, max_results: int, min_score: float
    ) -> List[SearchResult]:
        """Search only the knowledge base."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.encode(
                text=query, instruction="query"
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
                payload = result.payload
                search_result = SearchResult(
                    content=payload.get("content", ""),
                    source=payload.get("source", "Unknown"),
                    score=result.score,
                    result_type="knowledge",
                    metadata=payload.get("metadata", {}),
                )
                results.append(search_result)

            return results

        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return []

    def _search_memory_only(
        self, query: str, max_results: int, min_score: float
    ) -> List[SearchResult]:
        """Search only conversation memory."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.encode(
                text=query, instruction="query"
            )

            # Search memory collection
            search_results = self.vector_db.search(
                collection_name=self.memory_collection,
                query_vector=query_embedding,
                limit=max_results,
                score_threshold=min_score,
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
                    metadata={
                        "conversation_id": payload.get("conversation_id", ""),
                        "turn_id": payload.get("turn_id", ""),
                        "timestamp": payload.get("timestamp", ""),
                        "feedback_rating": payload.get("feedback_rating"),
                    },
                )
                results.append(search_result)

            return results

        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []

    def _search_hybrid(
        self, query: str, max_results: int, min_score: float
    ) -> List[SearchResult]:
        """
        Search both knowledge and memory, then merge results intelligently.

        This is the most powerful search strategy as it combines
        factual knowledge with conversational experience.
        """
        try:
            # Search both sources
            knowledge_results = self._search_knowledge_only(
                query,
                max_results // 2 + 2,  # Get a bit more from each source
                min_score,
            )

            memory_results = self._search_memory_only(
                query,
                max_results // 2 + 2,
                min_score * 0.9,  # Slightly lower threshold for memory
            )

            # Merge and rank results
            all_results = knowledge_results + memory_results

            # Sort by score (highest first)
            all_results.sort(key=lambda r: r.score, reverse=True)

            # Remove duplicates and limit results
            unique_results = []
            seen_content = set()

            for result in all_results:
                # Simple deduplication based on content similarity
                content_key = result.content[:100].lower().strip()
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    unique_results.append(result)

                if len(unique_results) >= max_results:
                    break

            return unique_results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    def _search_semantic(
        self, query: str, max_results: int, min_score: float
    ) -> List[SearchResult]:
        """
        Pure semantic search across all collections.

        This focuses purely on semantic similarity without
        considering source types or other factors.
        """
        # For now, just use hybrid search
        # Could be enhanced with more sophisticated semantic techniques
        return self._search_hybrid(query, max_results, min_score)

    def _search_conversations_with_emotion(
        self,
        query: str,
        max_results: int,
        min_score: float,
        user_id: str,
        emotional_context: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Internal method for conversation search with emotional context.

        Args:
            query: Search query
            max_results: Maximum results
            min_score: Minimum score
            user_id: User identifier
            emotional_context: Emotional context for enhancement

        Returns:
            List of emotionally enhanced search results
        """
        try:
            # Get user emotional pattern for personalization
            user_pattern = self.emotional_optimizer.get_user_emotional_pattern(user_id)

            # Generate query embedding
            query_embedding = self.embedding_service.encode(
                text=query, instruction="query"
            )

            # Search with emotional context if available
            if emotional_context and hasattr(
                self.vector_db, "search_with_emotional_context"
            ):
                search_results = self.vector_db.search_with_emotional_context(
                    collection_name=self.memory_collection,
                    query_vector=query_embedding,
                    emotional_context=emotional_context,
                    user_id=user_id,
                    limit=max_results,
                    score_threshold=min_score,
                )
            else:
                # Standard search
                search_results = self.vector_db.search(
                    collection_name=self.memory_collection,
                    query_vector=query_embedding,
                    limit=max_results,
                    score_threshold=min_score,
                )

            # Convert to SearchResult objects with emotional enhancement
            results = []
            for result in search_results:
                payload = (
                    result.payload if hasattr(result, "payload") else result.payload
                )

                # Format conversation turn as content
                question = payload.get("question", "")
                answer = payload.get("answer", "")
                content = f"Q: {question}\nA: {answer}"

                # Create enhanced search result
                search_result = SearchResult(
                    content=content,
                    source=f"Conversation ({payload.get('timestamp', 'Unknown')})",
                    score=(result.score if hasattr(result, "score") else result.score),
                    result_type="memory",
                    metadata={
                        "conversation_id": payload.get("conversation_id", ""),
                        "turn_id": payload.get("turn_id", ""),
                        "timestamp": payload.get("timestamp", ""),
                        "feedback_rating": payload.get("feedback_rating"),
                        "emotional_enhanced": emotional_context is not None,
                        "user_pattern_applied": user_pattern is not None,
                        "optimization_used": True,
                    },
                )
                results.append(search_result)

            # Apply additional emotional optimization if pattern available
            if user_pattern and emotional_context:
                # Generate optimized response parameters for ranking
                for result in results:
                    try:
                        # Create emotional state from context
                        from datetime import datetime

                        from morgan.optimization.emotional_optimizer import (
                            EmotionalState,
                        )

                        current_emotion = EmotionalState(
                            primary_emotion=emotional_context.get(
                                "primary_emotion", "neutral"
                            ),
                            intensity=emotional_context.get("intensity", 0.5),
                            confidence=emotional_context.get("confidence", 0.5),
                            secondary_emotions=emotional_context.get(
                                "secondary_emotions", []
                            ),
                            emotional_indicators=emotional_context.get(
                                "emotional_indicators", []
                            ),
                            timestamp=datetime.now(),
                            user_id=user_id,
                        )

                        # Get optimization parameters
                        optimization_params = self.emotional_optimizer.optimize_emotional_response_generation(
                            user_emotion=current_emotion,
                            response_context=result.content,
                            user_pattern=user_pattern,
                        )

                        # Boost score based on emotional relevance
                        empathy_boost = (
                            optimization_params.get("empathy_level", 0.5) * 0.1
                        )
                        result.score = min(1.0, result.score + empathy_boost)

                        # Add optimization metadata
                        result.metadata.update(
                            {
                                "empathy_level": optimization_params.get(
                                    "empathy_level"
                                ),
                                "emotional_tone": optimization_params.get(
                                    "emotional_tone"
                                ),
                                "personalization_factors": optimization_params.get(
                                    "personalization_factors", []
                                ),
                            }
                        )

                    except Exception as e:
                        logger.warning(
                            f"Failed to apply emotional optimization to result: {e}"
                        )

            # Sort by enhanced scores
            results.sort(key=lambda r: r.score, reverse=True)

            return results[:max_results]

        except Exception as e:
            logger.error(f"Emotional conversation search failed: {e}")
            # Fallback to basic memory search
            return self._search_memory_only(query, max_results, min_score)

    def _enhance_results(
        self, results: List[SearchResult], query: str
    ) -> List[SearchResult]:
        """
        Enhance search results for better human experience.

        This includes:
        - Highlighting relevant parts
        - Adding context
        - Improving source descriptions
        - Ranking by usefulness (not just similarity)
        """
        try:
            enhanced_results = []

            for result in results:
                # Enhance source description
                enhanced_source = self._enhance_source_description(
                    result.source, result.result_type
                )

                # Add query relevance context
                relevance_context = self._get_relevance_context(result.content, query)

                # Create enhanced result
                enhanced_result = SearchResult(
                    content=result.content,
                    source=enhanced_source,
                    score=result.score,
                    result_type=result.result_type,
                    metadata={
                        **result.metadata,
                        "relevance_context": relevance_context,
                        "enhanced": True,
                    },
                )

                enhanced_results.append(enhanced_result)

            return enhanced_results

        except Exception as e:
            logger.error(f"Failed to enhance results: {e}")
            return results  # Return original results if enhancement fails

    def _enhance_source_description(self, source: str, result_type: str) -> str:
        """Make source descriptions more human-friendly."""
        if result_type == "knowledge":
            if source.endswith(".md"):
                return f"Documentation: {source}"
            elif source.endswith(".pdf"):
                return f"PDF Document: {source}"
            elif "http" in source:
                return f"Web Page: {source}"
            else:
                return f"Document: {source}"
        elif result_type == "memory":
            return f"Previous Conversation: {source}"
        else:
            return source

    def _get_relevance_context(self, content: str, query: str) -> str:
        """
        Get context about why this content is relevant to the query.

        Simple approach: find sentences that contain query terms.
        """
        try:
            query_words = query.lower().split()
            sentences = content.split(".")

            relevant_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if any(word in sentence.lower() for word in query_words):
                    relevant_sentences.append(sentence)

            if relevant_sentences:
                return (
                    relevant_sentences[0][:200] + "..."
                    if len(relevant_sentences[0]) > 200
                    else relevant_sentences[0]
                )
            else:
                return content[:200] + "..." if len(content) > 200 else content

        except Exception:
            return content[:200] + "..." if len(content) > 200 else content


# Human-friendly helper functions
def quick_search(query: str, max_results: int = 5) -> List[str]:
    """
    Quick search that returns just the content.

    Args:
        query: Search query
        max_results: Maximum results to return

    Returns:
        List of relevant content snippets

    Example:
        >>> results = quick_search("Docker deployment")
        >>> for result in results:
        ...     print(result[:100] + "...")
    """
    search = SmartSearch()
    results = search.find_relevant_info(query, max_results=max_results)
    return [result.content for result in results]


def search_with_sources(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search that returns content with source information.

    Args:
        query: Search query
        max_results: Maximum results to return

    Returns:
        List of dictionaries with content and source

    Example:
        >>> results = search_with_sources("Docker networking")
        >>> for result in results:
        ...     print(f"Source: {result['source']}")
        ...     print(f"Content: {result['content'][:100]}...")
    """
    search = SmartSearch()
    results = search.find_relevant_info(query, max_results=max_results)

    return [
        {
            "content": result.content,
            "source": result.source,
            "score": result.score,
            "type": result.result_type,
        }
        for result in results
    ]


if __name__ == "__main__":
    # Demo smart search capabilities
    print("🔍 Morgan Smart Search Demo")
    print("=" * 40)

    search = SmartSearch()

    # Test different search strategies
    queries = [
        "Docker deployment",
        "How to install Python?",
        "What is machine learning?",
    ]

    for query in queries:
        print(f"\nSearching for: '{query}'")
        print("-" * 30)

        results = search.find_relevant_info(query, max_results=3)

        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. Score: {result.score:.2f} | Type: {result.result_type}")
                print(f"   Source: {result.source}")
                print(f"   Content: {result.summary(150)}")
                print()
        else:
            print("   No results found")

    # Test search suggestions
    print("\nSearch suggestions for 'Docker':")
    suggestions = search.get_search_suggestions("Docker")
    for i, suggestion in enumerate(suggestions[:5], 1):
        print(f"  {i}. {suggestion}")

    print("\n" + "=" * 40)
    print("Smart search demo completed!")
