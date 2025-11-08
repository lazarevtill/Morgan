"""
Morgan Assistant - Main orchestrator for the AI assistant.

Coordinates:
- Emotion detection
- Memory retrieval
- RAG search
- Context building
- Response generation
- Learning updates

Target: < 2s response latency (P95)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from morgan.core.context import ContextManager, ContextOverflowError
from morgan.core.memory import MemorySystem, MemoryRetrievalError
from morgan.core.response_generator import ResponseGenerator, GenerationError
from morgan.core.search import MultiStageSearch, SearchConfig, SearchResult
from morgan.core.types import (
    AssistantMetrics,
    AssistantResponse,
    ConversationContext,
    EmotionalState,
    Message,
    MessageRole,
    ProcessingContext,
    UserProfile,
)
from morgan.emotions.detector import EmotionDetector
from morgan.emotions.exceptions import EmotionDetectionError
from morgan.emotions.types import EmotionResult
from morgan.jina.reranking.service import RerankingService
from morgan.learning.engine import LearningEngine
from morgan.learning.exceptions import LearningError
from morgan.learning.types import FeedbackSignal, FeedbackType, LearningContext
from morgan.services.embedding_service import EmbeddingService
from morgan.vector_db.client import QdrantClient

logger = logging.getLogger(__name__)


class AssistantError(Exception):
    """Base exception for assistant errors."""

    def __init__(
        self,
        message: str,
        correlation_id: Optional[str] = None,
        recoverable: bool = True,
    ):
        super().__init__(message)
        self.correlation_id = correlation_id
        self.recoverable = recoverable


class MorganAssistant:
    """
    Main Morgan Assistant orchestrator.

    Integrates:
    - EmotionDetector: Emotion analysis
    - LearningEngine: Adaptive learning
    - MemorySystem: Multi-layer memory
    - ContextManager: Context handling
    - MultiStageSearch: RAG retrieval
    - ResponseGenerator: LLM generation

    Features:
    - Full async/await architecture
    - Parallel processing where possible
    - Circuit breakers for resilience
    - Comprehensive error handling
    - Performance tracking
    - Graceful degradation

    Performance targets:
    - Total latency: < 2s (P95)
    - Memory retrieval: < 100ms
    - Context building: < 50ms
    - Emotion detection: < 200ms
    """

    def __init__(
        self,
        # Storage
        storage_path: Optional[Path] = None,

        # LLM configuration
        llm_base_url: str = "http://localhost:11434",
        llm_model: str = "llama3.2:latest",

        # Vector DB
        vector_db: Optional[QdrantClient] = None,

        # Services
        embedding_service: Optional[EmbeddingService] = None,
        reranking_service: Optional[RerankingService] = None,

        # Feature flags
        enable_emotion_detection: bool = True,
        enable_learning: bool = True,
        enable_rag: bool = True,

        # Performance tuning
        max_concurrent_operations: int = 10,
    ):
        """
        Initialize Morgan Assistant.

        Args:
            storage_path: Path for persistent storage
            llm_base_url: LLM API base URL
            llm_model: LLM model name
            vector_db: Vector database client
            embedding_service: Embedding service
            reranking_service: Reranking service
            enable_emotion_detection: Enable emotion detection
            enable_learning: Enable learning system
            enable_rag: Enable RAG search
            max_concurrent_operations: Max concurrent operations
        """
        self.storage_path = storage_path or Path.home() / ".morgan"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Feature flags
        self.enable_emotion_detection = enable_emotion_detection
        self.enable_learning = enable_learning
        self.enable_rag = enable_rag

        # Core components
        self.memory_system = MemorySystem(
            storage_path=self.storage_path / "memory",
        )

        self.context_manager = ContextManager(
            max_context_tokens=8000,
            target_context_tokens=6000,
        )

        self.response_generator = ResponseGenerator(
            llm_base_url=llm_base_url,
            llm_model=llm_model,
        )

        # Optional components
        self.emotion_detector: Optional[EmotionDetector] = None
        if enable_emotion_detection:
            self.emotion_detector = EmotionDetector(
                enable_cache=True,
                enable_history=True,
                history_storage_path=self.storage_path / "emotions",
            )

        self.learning_engine: Optional[LearningEngine] = None
        if enable_learning:
            self.learning_engine = LearningEngine(
                enable_pattern_detection=True,
                enable_feedback_processing=True,
                enable_preference_learning=True,
                enable_adaptation=True,
                enable_consolidation=True,
            )

        self.search_engine: Optional[MultiStageSearch] = None
        if enable_rag and vector_db and embedding_service:
            self.search_engine = MultiStageSearch(
                vector_db=vector_db,
                embedding_service=embedding_service,
                reranking_service=reranking_service,
                config=SearchConfig(),
            )

        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent_operations)

        # Circuit breaker state
        self._failure_counts: Dict[str, int] = {
            "emotion": 0,
            "learning": 0,
            "rag": 0,
            "generation": 0,
        }
        self._max_failures = 5

        # Metrics
        self._metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "degraded_requests": 0,
        }

        logger.info(
            "Morgan Assistant initialized",
            extra={
                "storage_path": str(self.storage_path),
                "emotion_detection": enable_emotion_detection,
                "learning": enable_learning,
                "rag": enable_rag,
            },
        )

    async def initialize(self) -> None:
        """Initialize all components and start background tasks."""
        logger.info("Initializing Morgan Assistant")

        # Initialize memory system
        await self.memory_system.initialize()

        # Initialize emotion detector
        if self.emotion_detector:
            await self.emotion_detector.initialize()

        # Initialize learning engine
        if self.learning_engine:
            await self.learning_engine.initialize()

        logger.info("Morgan Assistant initialized successfully")

    async def cleanup(self) -> None:
        """Cleanup resources and stop background tasks."""
        logger.info("Cleaning up Morgan Assistant")

        # Cleanup components
        await self.memory_system.cleanup()

        if self.emotion_detector:
            await self.emotion_detector.cleanup()

        if self.learning_engine:
            await self.learning_engine.cleanup()

        await self.response_generator.cleanup()

        logger.info("Morgan Assistant cleaned up")

    async def process_message(
        self,
        user_id: str,
        message: str,
        session_id: str,
    ) -> AssistantResponse:
        """
        Process user message and generate response.

        Main orchestration flow:
        1. Detect emotion (parallel)
        2. Retrieve memories (parallel)
        3. RAG search (if needed)
        4. Build context
        5. Generate response
        6. Update memory and learning
        7. Return response

        Args:
            user_id: User identifier
            message: User message
            session_id: Session identifier

        Returns:
            Assistant response

        Raises:
            AssistantError: If processing fails critically
        """
        correlation_id = self._generate_correlation_id()
        start_time = time.time()

        # Initialize metrics
        metrics = AssistantMetrics(
            correlation_id=correlation_id,
            operation="process_message",
            started_at=datetime.now(),
        )

        # Initialize processing context
        ctx = ProcessingContext(
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            message=message,
            timestamp=datetime.now(),
            metrics=metrics,
        )

        self._metrics["total_requests"] += 1

        try:
            async with self._semaphore:
                # Phase 1: Parallel emotion detection and memory retrieval
                await self._phase1_gather_context(ctx)

                # Phase 2: RAG search (if needed and enabled)
                await self._phase2_rag_search(ctx)

                # Phase 3: Build conversation context
                await self._phase3_build_context(ctx)

                # Phase 4: Generate response
                response = await self._phase4_generate_response(ctx)

                # Phase 5: Update memory and learning (background)
                asyncio.create_task(self._phase5_update_systems(ctx, response))

                # Update metrics
                metrics.total_duration_ms = (time.time() - start_time) * 1000
                response.metadata["metrics"] = metrics.to_dict()

                self._metrics["successful_requests"] += 1

                logger.info(
                    "Message processed successfully",
                    extra={
                        "correlation_id": correlation_id,
                        "user_id": user_id,
                        "duration_ms": round(metrics.total_duration_ms, 2),
                        "degraded": metrics.degraded_mode,
                    },
                )

                return response

        except Exception as e:
            self._metrics["failed_requests"] += 1

            logger.error(
                "Failed to process message",
                extra={
                    "correlation_id": correlation_id,
                    "user_id": user_id,
                    "error": str(e),
                },
            )

            # Return degraded response
            return await self._create_fallback_response(
                message,
                correlation_id,
                error=str(e),
            )

    async def stream_response(
        self,
        user_id: str,
        message: str,
        session_id: str,
    ) -> AsyncIterator[str]:
        """
        Stream response to user.

        Args:
            user_id: User identifier
            message: User message
            session_id: Session identifier

        Yields:
            Response chunks
        """
        correlation_id = self._generate_correlation_id()

        try:
            # Build context (same as normal flow)
            ctx = ProcessingContext(
                correlation_id=correlation_id,
                user_id=user_id,
                session_id=session_id,
                message=message,
                timestamp=datetime.now(),
            )

            # Gather context
            await self._phase1_gather_context(ctx)
            await self._phase2_rag_search(ctx)
            await self._phase3_build_context(ctx)

            # Stream response
            if not ctx.conversation_context:
                raise AssistantError(
                    "Failed to build context",
                    correlation_id=correlation_id,
                )

            async for chunk in self.response_generator.generate_stream(
                context=ctx.conversation_context,
                user_message=message,
                rag_results=ctx.rag_sources,
                detected_emotion=ctx.detected_emotion,
            ):
                yield chunk

        except Exception as e:
            logger.error(
                "Streaming failed",
                extra={
                    "correlation_id": correlation_id,
                    "error": str(e),
                },
            )
            yield f"\n\n[Error: {str(e)}]"

    async def _phase1_gather_context(
        self,
        ctx: ProcessingContext,
    ) -> None:
        """
        Phase 1: Gather emotion and memory context in parallel.

        Args:
            ctx: Processing context
        """
        start_time = time.time()

        # Run in parallel
        tasks = []

        # Emotion detection
        if self.emotion_detector and self.enable_emotion_detection:
            tasks.append(self._detect_emotion_safe(ctx))

        # Memory retrieval
        tasks.append(self._retrieve_memories_safe(ctx))

        # User profile
        tasks.append(self._get_user_profile_safe(ctx))

        # Execute in parallel
        await asyncio.gather(*tasks, return_exceptions=True)

        ctx.metrics.emotion_detection_ms = (time.time() - start_time) * 1000

    async def _phase2_rag_search(
        self,
        ctx: ProcessingContext,
    ) -> None:
        """
        Phase 2: RAG search if needed.

        Args:
            ctx: Processing context
        """
        if not self.search_engine or not self.enable_rag:
            return

        start_time = time.time()

        try:
            # Perform search
            results, search_metrics = await self.search_engine.search(
                query=ctx.message,
                top_k=5,
            )

            ctx.rag_sources = results
            ctx.metrics.rag_sources_found = len(results)

            # Reset failure count on success
            self._failure_counts["rag"] = 0

        except Exception as e:
            logger.warning(
                "RAG search failed, continuing without",
                extra={
                    "correlation_id": ctx.correlation_id,
                    "error": str(e),
                },
            )
            self._failure_counts["rag"] += 1
            ctx.metrics.degraded_mode = True

        ctx.metrics.rag_search_ms = (time.time() - start_time) * 1000

    async def _phase3_build_context(
        self,
        ctx: ProcessingContext,
    ) -> None:
        """
        Phase 3: Build conversation context.

        Args:
            ctx: Processing context
        """
        start_time = time.time()

        try:
            context = await self.context_manager.build_context(
                messages=ctx.retrieved_memories,
                user_id=ctx.user_id,
                session_id=ctx.session_id,
                user_profile=ctx.user_profile,
                emotional_state=ctx.emotional_state,
            )

            ctx.conversation_context = context

        except ContextOverflowError as e:
            logger.error(
                "Context overflow",
                extra={
                    "correlation_id": ctx.correlation_id,
                    "error": str(e),
                },
            )
            raise

        ctx.metrics.context_building_ms = (time.time() - start_time) * 1000

    async def _phase4_generate_response(
        self,
        ctx: ProcessingContext,
    ) -> AssistantResponse:
        """
        Phase 4: Generate response.

        Args:
            ctx: Processing context

        Returns:
            Assistant response
        """
        start_time = time.time()

        if not ctx.conversation_context:
            raise AssistantError(
                "No conversation context available",
                correlation_id=ctx.correlation_id,
            )

        try:
            response = await self.response_generator.generate(
                context=ctx.conversation_context,
                user_message=ctx.message,
                rag_results=ctx.rag_sources,
                detected_emotion=ctx.detected_emotion,
            )

            # Reset failure count
            self._failure_counts["generation"] = 0

            return response

        except GenerationError as e:
            self._failure_counts["generation"] += 1
            logger.error(
                "Response generation failed",
                extra={
                    "correlation_id": ctx.correlation_id,
                    "error": str(e),
                },
            )
            raise

        finally:
            ctx.metrics.response_generation_ms = (time.time() - start_time) * 1000

    async def _phase5_update_systems(
        self,
        ctx: ProcessingContext,
        response: AssistantResponse,
    ) -> None:
        """
        Phase 5: Update memory and learning systems (background).

        Args:
            ctx: Processing context
            response: Generated response
        """
        start_time = time.time()

        try:
            # Store user message
            user_message = Message(
                role=MessageRole.USER,
                content=ctx.message,
                timestamp=ctx.timestamp,
                message_id=self._generate_id(),
                emotion=ctx.detected_emotion,
            )

            await self.memory_system.store_message(
                session_id=ctx.session_id,
                message=user_message,
                user_id=ctx.user_id,
            )

            # Store assistant response
            assistant_message = Message(
                role=MessageRole.ASSISTANT,
                content=response.content,
                timestamp=response.timestamp,
                message_id=response.response_id,
            )

            await self.memory_system.store_message(
                session_id=ctx.session_id,
                message=assistant_message,
                user_id=ctx.user_id,
            )

            # Update emotional state
            if ctx.detected_emotion:
                emotional_state = EmotionalState(
                    user_id=ctx.user_id,
                    timestamp=datetime.now(),
                    current_emotions=ctx.detected_emotion,
                )
                await self.memory_system.update_emotional_state(emotional_state)

            # Update learning system
            if self.learning_engine and self.enable_learning:
                await self._update_learning_safe(ctx, response)

        except Exception as e:
            logger.error(
                "Failed to update systems",
                extra={
                    "correlation_id": ctx.correlation_id,
                    "error": str(e),
                },
            )

        ctx.metrics.learning_update_ms = (time.time() - start_time) * 1000

    async def _detect_emotion_safe(
        self,
        ctx: ProcessingContext,
    ) -> None:
        """Detect emotion with error handling."""
        if not self.emotion_detector:
            return

        try:
            # Check circuit breaker
            if self._failure_counts["emotion"] >= self._max_failures:
                logger.warning("Emotion detection circuit breaker open")
                ctx.metrics.degraded_mode = True
                return

            result = await self.emotion_detector.detect(
                text=ctx.message,
                user_id=ctx.user_id,
            )

            ctx.detected_emotion = result

            # Reset failure count
            self._failure_counts["emotion"] = 0

        except Exception as e:
            logger.warning(
                "Emotion detection failed",
                extra={
                    "correlation_id": ctx.correlation_id,
                    "error": str(e),
                },
            )
            self._failure_counts["emotion"] += 1
            ctx.metrics.degraded_mode = True

    async def _retrieve_memories_safe(
        self,
        ctx: ProcessingContext,
    ) -> None:
        """Retrieve memories with error handling."""
        try:
            messages = await self.memory_system.retrieve_context(
                session_id=ctx.session_id,
                n_messages=20,
            )

            ctx.retrieved_memories = messages
            ctx.metrics.messages_retrieved = len(messages)

        except Exception as e:
            logger.warning(
                "Memory retrieval failed",
                extra={
                    "correlation_id": ctx.correlation_id,
                    "error": str(e),
                },
            )
            ctx.retrieved_memories = []
            ctx.metrics.degraded_mode = True

    async def _get_user_profile_safe(
        self,
        ctx: ProcessingContext,
    ) -> None:
        """Get user profile with error handling."""
        try:
            profile = await self.memory_system.get_user_profile(ctx.user_id)
            ctx.user_profile = profile

            emotional_state = await self.memory_system.get_emotional_state(ctx.user_id)
            ctx.emotional_state = emotional_state

        except Exception as e:
            logger.warning(
                "Failed to get user profile",
                extra={
                    "correlation_id": ctx.correlation_id,
                    "error": str(e),
                },
            )

    async def _update_learning_safe(
        self,
        ctx: ProcessingContext,
        response: AssistantResponse,
    ) -> None:
        """Update learning system with error handling."""
        if not self.learning_engine:
            return

        try:
            # Create learning context
            learning_ctx = LearningContext(
                user_id=ctx.user_id,
                session_id=ctx.session_id,
                timestamp=datetime.now(),
                interaction_type="message",
                metadata={
                    "message": ctx.message,
                    "response": response.content,
                    "emotion": ctx.detected_emotion.primary_emotion.emotion_type.value
                    if ctx.detected_emotion and ctx.detected_emotion.primary_emotion
                    else None,
                },
            )

            # Process interaction
            await self.learning_engine.learn_from_interaction(learning_ctx)

        except Exception as e:
            logger.warning(
                "Learning update failed",
                extra={
                    "correlation_id": ctx.correlation_id,
                    "error": str(e),
                },
            )

    async def _create_fallback_response(
        self,
        message: str,
        correlation_id: str,
        error: str,
    ) -> AssistantResponse:
        """
        Create fallback response when processing fails.

        Args:
            message: Original user message
            correlation_id: Correlation ID
            error: Error message

        Returns:
            Fallback response
        """
        return AssistantResponse(
            content="I apologize, but I'm having trouble processing your request right now. Please try again in a moment.",
            response_id=self._generate_id(),
            timestamp=datetime.now(),
            confidence=0.0,
            metadata={
                "fallback": True,
                "error": error,
                "correlation_id": correlation_id,
            },
        )

    def _generate_correlation_id(self) -> str:
        """Generate unique correlation ID."""
        return hashlib.sha256(
            f"{time.time()}{id(self)}".encode()
        ).hexdigest()[:16]

    def _generate_id(self) -> str:
        """Generate unique ID."""
        return hashlib.sha256(
            f"{time.time()}{id(self)}{asyncio.get_event_loop()}".encode()
        ).hexdigest()[:16]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get assistant statistics.

        Returns:
            Statistics dictionary
        """
        stats = {
            "metrics": self._metrics.copy(),
            "circuit_breakers": {
                name: {
                    "failure_count": count,
                    "is_open": count >= self._max_failures,
                }
                for name, count in self._failure_counts.items()
            },
        }

        # Add component stats
        stats["memory"] = self.memory_system.get_stats()
        stats["context"] = self.context_manager.get_stats()
        stats["response_generator"] = self.response_generator.get_stats()

        if self.search_engine:
            stats["search"] = self.search_engine.get_stats()

        return stats
