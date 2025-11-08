"""
Main Emotion Detector.

Orchestrates all 11 emotion detection modules to provide comprehensive
emotion analysis with <200ms response time.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import List, Optional

from morgan.emotions.base import CircuitBreaker, EmotionModule
from morgan.emotions.exceptions import (
    EmotionDetectionError,
    EmotionResourceError,
    EmotionValidationError,
)
from morgan.emotions.modules.aggregator import EmotionAggregator
from morgan.emotions.modules.cache import EmotionCache
from morgan.emotions.modules.classifier import EmotionClassifier
from morgan.emotions.modules.context_analyzer import ContextAnalyzer
from morgan.emotions.modules.history_tracker import EmotionHistoryTracker
from morgan.emotions.modules.intensity import IntensityAnalyzer
from morgan.emotions.modules.multi_emotion import MultiEmotionDetector
from morgan.emotions.modules.pattern_detector import PatternDetector
from morgan.emotions.modules.temporal_analyzer import TemporalAnalyzer
from morgan.emotions.modules.trigger_detector import TriggerDetector
from morgan.emotions.types import EmotionContext, EmotionResult


class EmotionDetector(EmotionModule):
    """
    Main emotion detection system.

    Orchestrates 11 specialized modules to provide comprehensive emotion analysis:
    1. EmotionClassifier - Classifies text into emotion types
    2. IntensityAnalyzer - Analyzes and adjusts emotion intensities
    3. PatternDetector - Detects emotional patterns over time
    4. TriggerDetector - Identifies emotional triggers
    5. EmotionHistoryTracker - Maintains emotional history
    6. ContextAnalyzer - Analyzes conversational context
    7. MultiEmotionDetector - Handles multiple simultaneous emotions
    8. TemporalAnalyzer - Analyzes temporal emotion changes
    9. EmotionCache - Caches results for performance
    10. EmotionAggregator - Aggregates all results
    11. Main orchestration and error handling

    Target: <200ms response time per message
    """

    def __init__(
        self,
        enable_cache: bool = True,
        enable_history: bool = True,
        history_storage_path: Optional[Path] = None,
        max_concurrent_operations: int = 5,
    ) -> None:
        """
        Initialize emotion detector.

        Args:
            enable_cache: Enable result caching
            enable_history: Enable history tracking
            history_storage_path: Optional path for history persistence
            max_concurrent_operations: Max concurrent async operations
        """
        super().__init__("EmotionDetector")

        # Initialize all modules
        self._classifier = EmotionClassifier()
        self._intensity_analyzer = IntensityAnalyzer()
        self._pattern_detector = PatternDetector()
        self._trigger_detector = TriggerDetector()
        self._context_analyzer = ContextAnalyzer()
        self._multi_emotion_detector = MultiEmotionDetector()
        self._temporal_analyzer = TemporalAnalyzer()
        self._aggregator = EmotionAggregator()

        # Optional modules
        self._cache = EmotionCache() if enable_cache else None
        self._history_tracker = (
            EmotionHistoryTracker(storage_path=history_storage_path)
            if enable_history
            else None
        )

        # Circuit breaker for resilience
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exception=EmotionDetectionError,
        )

        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent_operations)

        # Performance tracking
        self._total_detections = 0
        self._total_processing_time = 0.0
        self._cache_hits = 0

    async def initialize(self) -> None:
        """Initialize all modules in parallel."""
        modules = [
            self._classifier,
            self._intensity_analyzer,
            self._pattern_detector,
            self._trigger_detector,
            self._context_analyzer,
            self._multi_emotion_detector,
            self._temporal_analyzer,
            self._aggregator,
        ]

        if self._cache:
            modules.append(self._cache)
        if self._history_tracker:
            modules.append(self._history_tracker)

        # Initialize all modules concurrently
        try:
            await asyncio.gather(*[module.initialize() for module in modules])
        except Exception as e:
            raise EmotionResourceError(
                f"Failed to initialize emotion detector: {str(e)}",
                resource_type="modules",
            )

    async def cleanup(self) -> None:
        """Cleanup all modules."""
        modules = [
            self._classifier,
            self._intensity_analyzer,
            self._pattern_detector,
            self._trigger_detector,
            self._context_analyzer,
            self._multi_emotion_detector,
            self._temporal_analyzer,
            self._aggregator,
        ]

        if self._cache:
            modules.append(self._cache)
        if self._history_tracker:
            modules.append(self._history_tracker)

        await asyncio.gather(*[module.cleanup() for module in modules])

    async def detect(
        self,
        text: str,
        context: Optional[EmotionContext] = None,
        use_cache: bool = True,
    ) -> EmotionResult:
        """
        Detect emotions in text.

        Args:
            text: Text to analyze
            context: Optional conversation context
            use_cache: Whether to use cache (default: True)

        Returns:
            EmotionResult with comprehensive analysis

        Raises:
            EmotionValidationError: If input is invalid
            EmotionDetectionError: If detection fails
        """
        await self.ensure_initialized()

        # Validate input
        if not text or not text.strip():
            raise EmotionValidationError("Text cannot be empty", field="text")

        if len(text) > 10000:
            raise EmotionValidationError(
                "Text exceeds maximum length (10000 characters)", field="text"
            )

        # Use circuit breaker for resilience
        return await self._circuit_breaker.call(
            self._detect_internal, text, context, use_cache
        )

    async def _detect_internal(
        self,
        text: str,
        context: Optional[EmotionContext],
        use_cache: bool,
    ) -> EmotionResult:
        """Internal detection method with full pipeline."""
        start_time = time.perf_counter()

        # Acquire semaphore to limit concurrency
        async with self._semaphore:
            # Check cache first
            if use_cache and self._cache:
                context_key = self._get_context_key(context)
                cached = await self._cache.get(text, context_key)
                if cached:
                    self._cache_hits += 1
                    return cached

            # Run detection pipeline
            result = await self._run_detection_pipeline(text, context)

            # Calculate processing time
            processing_time = (time.perf_counter() - start_time) * 1000

            # Update result with processing time
            result = EmotionResult(
                primary_emotions=result.primary_emotions,
                dominant_emotion=result.dominant_emotion,
                valence=result.valence,
                arousal=result.arousal,
                triggers=result.triggers,
                patterns=result.patterns,
                context=result.context,
                timestamp=result.timestamp,
                processing_time_ms=processing_time,
                warnings=result.warnings,
            )

            # Cache result
            if use_cache and self._cache:
                context_key = self._get_context_key(context)
                await self._cache.set(text, result, context_key)

            # Update history
            if self._history_tracker and context and context.user_id:
                await self._history_tracker.add_result(context.user_id, result)

            # Update temporal analyzer
            if context and context.user_id:
                await self._temporal_analyzer.add_result(context.user_id, result)

            # Update stats
            self._total_detections += 1
            self._total_processing_time += processing_time

            return result

    async def _run_detection_pipeline(
        self,
        text: str,
        context: Optional[EmotionContext],
    ) -> EmotionResult:
        """Run the complete detection pipeline."""
        warnings: List[str] = []

        # STAGE 1: Classification (async)
        classified_emotions = await self._classifier.classify(text)

        if not classified_emotions:
            warnings.append("No emotions classified from text")

        # STAGE 2: Intensity Analysis (async)
        intensity_adjusted = await self._intensity_analyzer.analyze_intensity(
            classified_emotions, text, context
        )

        # STAGE 3: Context Analysis (async, only if we have context)
        if context:
            context_adjusted = await self._context_analyzer.analyze_context(
                intensity_adjusted, context
            )
        else:
            context_adjusted = intensity_adjusted

        # STAGE 4: Parallel analysis of triggers, patterns, multi-emotions
        trigger_task = self._trigger_detector.detect_triggers(
            text, context_adjusted
        )
        pattern_task = self._pattern_detector.detect_patterns(
            context_adjusted, context
        )
        multi_emotion_task = self._multi_emotion_detector.analyze_multi_emotions(
            context_adjusted
        )

        # Wait for all parallel tasks
        triggers, patterns, (dominant, valence, arousal) = await asyncio.gather(
            trigger_task,
            pattern_task,
            multi_emotion_task,
        )

        # STAGE 5: Aggregation
        result = await self._aggregator.aggregate(
            emotions=context_adjusted,
            dominant_emotion=dominant,
            valence=valence,
            arousal=arousal,
            triggers=triggers,
            patterns=patterns,
            context=context,
            warnings=warnings,
        )

        return result

    async def get_user_history(
        self,
        user_id: str,
        limit: int = 10,
    ) -> List[EmotionResult]:
        """
        Get emotion history for a user.

        Args:
            user_id: User identifier
            limit: Maximum results to return

        Returns:
            List of recent emotion results
        """
        if not self._history_tracker:
            return []

        return await self._history_tracker.get_recent(user_id, limit)

    async def get_user_trajectory(
        self,
        user_id: str,
    ) -> dict:
        """
        Get emotional trajectory for a user.

        Args:
            user_id: User identifier

        Returns:
            Trajectory analysis dictionary
        """
        return await self._temporal_analyzer.analyze_trajectory(user_id)

    async def clear_user_data(self, user_id: str) -> None:
        """
        Clear all data for a user.

        Args:
            user_id: User identifier
        """
        tasks = []

        if self._history_tracker:
            tasks.append(self._history_tracker.clear_history(user_id))

        if self._pattern_detector:
            tasks.append(self._pattern_detector.clear_history(user_id))

        if tasks:
            await asyncio.gather(*tasks)

    def _get_context_key(self, context: Optional[EmotionContext]) -> Optional[str]:
        """Generate cache key from context."""
        if not context:
            return None

        parts = []
        if context.user_id:
            parts.append(f"user:{context.user_id}")
        if context.conversation_id:
            parts.append(f"conv:{context.conversation_id}")

        return "|".join(parts) if parts else None

    @property
    def stats(self) -> dict:
        """Get detector statistics."""
        avg_time = (
            self._total_processing_time / self._total_detections
            if self._total_detections > 0
            else 0.0
        )

        cache_hit_rate = (
            self._cache_hits / self._total_detections
            if self._total_detections > 0
            else 0.0
        )

        return {
            "total_detections": self._total_detections,
            "average_processing_time_ms": avg_time,
            "cache_enabled": self._cache is not None,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "circuit_breaker_state": self._circuit_breaker.state,
        }
