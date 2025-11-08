"""
Feedback Processing Module.

Processes user feedback signals including explicit feedback (ratings, corrections)
and implicit feedback (interaction patterns) with sentiment analysis.
"""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from morgan.learning.base import (
    AsyncCache,
    BaseLearningModule,
    CircuitBreaker,
    HealthStatus,
)
from morgan.learning.exceptions import FeedbackProcessingError
from morgan.learning.types import (
    FeedbackSignal,
    FeedbackType,
    LearningContext,
)
from morgan.learning.utils import (
    aggregate_feedback,
    generate_id,
    validate_feedback_signal,
)


class FeedbackModule(BaseLearningModule):
    """
    Feedback processing and sentiment analysis module.

    Features:
    - Explicit feedback processing (ratings, corrections)
    - Implicit feedback inference (clicks, time, engagement)
    - Sentiment analysis
    - Feedback aggregation and trends
    - Actionable insight extraction
    """

    def __init__(
        self,
        enable_cache: bool = True,
        sentiment_threshold: float = 0.3,
        max_feedback_per_user: int = 500,
        feedback_window_days: int = 30,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Initialize feedback module.

        Args:
            enable_cache: Enable result caching
            sentiment_threshold: Threshold for sentiment classification
            max_feedback_per_user: Maximum feedback signals to store per user
            feedback_window_days: Days to keep feedback
            correlation_id: Optional correlation ID for request tracing
        """
        super().__init__("FeedbackModule", correlation_id)

        self._sentiment_threshold = sentiment_threshold
        self._max_feedback = max_feedback_per_user
        self._window_days = feedback_window_days

        # Feedback storage
        self._user_feedback: Dict[str, List[FeedbackSignal]] = defaultdict(list)
        self._feedback_by_message: Dict[str, List[FeedbackSignal]] = defaultdict(list)

        # Sentiment lexicon
        self._positive_words: Set[str] = set()
        self._negative_words: Set[str] = set()
        self._intensifiers: Set[str] = set()

        # Cache and circuit breaker
        self._cache = AsyncCache(max_size=500) if enable_cache else None
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exception=FeedbackProcessingError,
            name="feedback_processing",
        )

        # Concurrency control
        self._semaphore = asyncio.Semaphore(10)
        self._lock = asyncio.Lock()

        # Metrics
        self._feedback_processed = 0
        self._feedback_actionable = 0
        self._sentiment_analyzed = 0

    async def initialize(self) -> None:
        """Initialize the feedback module."""
        self._load_sentiment_lexicon()
        self._log_info("Feedback module initialized")

    async def cleanup(self) -> None:
        """Cleanup module resources."""
        async with self._lock:
            self._user_feedback.clear()
            self._feedback_by_message.clear()

        if self._cache:
            await self._cache.clear()

        self._log_info("Feedback module cleaned up")

    async def health_check(self) -> HealthStatus:
        """Check module health."""
        try:
            feedback_count = sum(len(f) for f in self._user_feedback.values())

            return HealthStatus(
                healthy=True,
                message="Feedback module healthy",
                details={
                    "feedback_tracked": feedback_count,
                    "users_tracked": len(self._user_feedback),
                    "feedback_processed": self._feedback_processed,
                    "actionable_ratio": (
                        self._feedback_actionable / self._feedback_processed
                        if self._feedback_processed > 0
                        else 0.0
                    ),
                    "circuit_breaker_state": self._circuit_breaker.state,
                },
                last_check=asyncio.get_event_loop().time(),
            )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                message=f"Feedback module unhealthy: {str(e)}",
                details={},
                last_check=asyncio.get_event_loop().time(),
            )

    async def process_feedback(
        self,
        user_id: str,
        feedback_type: FeedbackType,
        message_id: Optional[str] = None,
        rating: Optional[float] = None,
        text: Optional[str] = None,
        correction: Optional[str] = None,
        context: Optional[LearningContext] = None,
    ) -> FeedbackSignal:
        """
        Process a feedback signal.

        Args:
            user_id: User identifier
            feedback_type: Type of feedback
            message_id: Optional associated message ID
            rating: Optional numeric rating (0-1)
            text: Optional text feedback
            correction: Optional correction text
            context: Optional learning context

        Returns:
            Processed feedback signal

        Raises:
            FeedbackProcessingError: If processing fails
        """
        await self.ensure_initialized()

        try:
            # Create feedback signal
            feedback = await self._circuit_breaker.call(
                self._process_feedback_internal,
                user_id,
                feedback_type,
                message_id,
                rating,
                text,
                correction,
                context,
            )

            # Validate
            issues = validate_feedback_signal(feedback)
            if issues:
                self._log_warning(
                    f"Feedback validation issues: {issues}",
                    feedback_id=feedback.feedback_id,
                )

            # Store
            await self._store_feedback(feedback)

            self._feedback_processed += 1
            if feedback.is_actionable:
                self._feedback_actionable += 1

            return feedback

        except Exception as e:
            self._log_error("Feedback processing failed", e, user_id=user_id)
            raise FeedbackProcessingError(
                f"Failed to process feedback for user {user_id}",
                cause=e,
            )

    async def _process_feedback_internal(
        self,
        user_id: str,
        feedback_type: FeedbackType,
        message_id: Optional[str],
        rating: Optional[float],
        text: Optional[str],
        correction: Optional[str],
        context: Optional[LearningContext],
    ) -> FeedbackSignal:
        """Internal feedback processing implementation."""
        async with self._semaphore:
            # Analyze sentiment if text provided
            sentiment, sentiment_confidence = 0.0, 0.0
            if text:
                sentiment, sentiment_confidence = await self._analyze_sentiment(text)
                self._sentiment_analyzed += 1

            # Create feedback signal
            feedback = FeedbackSignal(
                feedback_id=generate_id("feedback"),
                user_id=user_id,
                feedback_type=feedback_type,
                timestamp=datetime.utcnow(),
                message_id=message_id,
                rating=rating,
                text=text,
                correction=correction,
                sentiment=sentiment,
                sentiment_confidence=sentiment_confidence,
                context=context.metadata if context else {},
                conversation_id=context.conversation_id if context else None,
            )

            return feedback

    async def _analyze_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (sentiment score -1 to +1, confidence 0 to 1)
        """
        if not text or not text.strip():
            return 0.0, 0.0

        # Normalize text
        normalized = text.lower().strip()
        words = re.findall(r"\b\w+\b", normalized)

        if not words:
            return 0.0, 0.0

        # Count positive and negative words
        positive_count = sum(1 for w in words if w in self._positive_words)
        negative_count = sum(1 for w in words if w in self._negative_words)

        # Check for intensifiers
        intensifier_count = sum(1 for w in words if w in self._intensifiers)
        intensity_boost = min(intensifier_count * 0.2, 0.5)

        # Calculate sentiment
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0.0, 0.0

        # Sentiment score
        sentiment = (positive_count - negative_count) / len(words)
        sentiment = max(-1.0, min(1.0, sentiment * (1.0 + intensity_boost)))

        # Confidence based on sentiment word density
        confidence = min(total_sentiment_words / len(words) * 2.0, 1.0)

        return sentiment, confidence

    async def _store_feedback(self, feedback: FeedbackSignal) -> None:
        """Store feedback signal."""
        async with self._lock:
            # Store by user
            self._user_feedback[feedback.user_id].append(feedback)

            # Trim old feedback
            cutoff = datetime.utcnow() - timedelta(days=self._window_days)
            self._user_feedback[feedback.user_id] = [
                f
                for f in self._user_feedback[feedback.user_id]
                if f.timestamp > cutoff
            ]

            # Limit per user
            if len(self._user_feedback[feedback.user_id]) > self._max_feedback:
                self._user_feedback[feedback.user_id] = self._user_feedback[
                    feedback.user_id
                ][-self._max_feedback :]

            # Store by message if available
            if feedback.message_id:
                self._feedback_by_message[feedback.message_id].append(feedback)

        # Invalidate cache
        if self._cache:
            await self._cache.invalidate(f"feedback_summary:{feedback.user_id}")

    async def get_user_feedback(
        self,
        user_id: str,
        limit: Optional[int] = None,
        feedback_type: Optional[FeedbackType] = None,
    ) -> List[FeedbackSignal]:
        """
        Get feedback signals for a user.

        Args:
            user_id: User identifier
            limit: Optional limit on results
            feedback_type: Optional filter by type

        Returns:
            List of feedback signals
        """
        async with self._lock:
            feedback = list(self._user_feedback.get(user_id, []))

        # Filter by type if specified
        if feedback_type:
            feedback = [f for f in feedback if f.feedback_type == feedback_type]

        # Sort by timestamp (newest first)
        feedback.sort(key=lambda f: f.timestamp, reverse=True)

        # Apply limit
        if limit:
            feedback = feedback[:limit]

        return feedback

    async def get_message_feedback(
        self,
        message_id: str,
    ) -> List[FeedbackSignal]:
        """
        Get all feedback for a specific message.

        Args:
            message_id: Message identifier

        Returns:
            List of feedback signals
        """
        async with self._lock:
            return list(self._feedback_by_message.get(message_id, []))

    async def get_feedback_summary(
        self,
        user_id: str,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Get aggregated feedback summary for a user.

        Args:
            user_id: User identifier
            use_cache: Whether to use cache

        Returns:
            Feedback summary dictionary
        """
        # Check cache
        if use_cache and self._cache:
            cache_key = f"feedback_summary:{user_id}"
            cached = await self._cache.get(cache_key)
            if cached:
                return cached

        # Get all feedback
        feedback = await self.get_user_feedback(user_id)

        # Aggregate
        summary = aggregate_feedback(feedback)

        # Add trends
        summary["trends"] = await self._analyze_trends(feedback)

        # Cache result
        if self._cache:
            cache_key = f"feedback_summary:{user_id}"
            await self._cache.set(cache_key, summary, ttl=300)

        return summary

    async def _analyze_trends(
        self,
        feedback_list: List[FeedbackSignal],
    ) -> Dict[str, Any]:
        """
        Analyze feedback trends over time.

        Args:
            feedback_list: List of feedback signals

        Returns:
            Trends dictionary
        """
        if not feedback_list:
            return {
                "sentiment_trend": "stable",
                "feedback_frequency_trend": "stable",
                "recent_shift": False,
            }

        # Sort by timestamp
        sorted_feedback = sorted(feedback_list, key=lambda f: f.timestamp)

        # Split into first half and second half
        mid_point = len(sorted_feedback) // 2
        first_half = sorted_feedback[:mid_point]
        second_half = sorted_feedback[mid_point:]

        if not first_half or not second_half:
            return {
                "sentiment_trend": "stable",
                "feedback_frequency_trend": "stable",
                "recent_shift": False,
            }

        # Compare sentiment
        first_avg_sentiment = sum(f.sentiment for f in first_half) / len(first_half)
        second_avg_sentiment = sum(f.sentiment for f in second_half) / len(
            second_half
        )

        sentiment_change = second_avg_sentiment - first_avg_sentiment

        if sentiment_change > 0.2:
            sentiment_trend = "improving"
        elif sentiment_change < -0.2:
            sentiment_trend = "declining"
        else:
            sentiment_trend = "stable"

        # Check for recent shift (last 5 feedback items)
        recent = sorted_feedback[-5:]
        if len(recent) >= 3:
            recent_avg = sum(f.sentiment for f in recent) / len(recent)
            overall_avg = sum(f.sentiment for f in sorted_feedback) / len(
                sorted_feedback
            )
            recent_shift = abs(recent_avg - overall_avg) > 0.3
        else:
            recent_shift = False

        return {
            "sentiment_trend": sentiment_trend,
            "sentiment_change": sentiment_change,
            "feedback_frequency_trend": "stable",
            "recent_shift": recent_shift,
        }

    async def infer_implicit_feedback(
        self,
        user_id: str,
        message_id: str,
        interaction_data: Dict[str, Any],
    ) -> FeedbackSignal:
        """
        Infer implicit feedback from interaction data.

        Args:
            user_id: User identifier
            message_id: Message identifier
            interaction_data: Interaction metrics (time, clicks, etc.)

        Returns:
            Inferred feedback signal
        """
        # Analyze interaction data
        time_spent = interaction_data.get("time_spent_seconds", 0)
        follow_up_question = interaction_data.get("follow_up_question", False)
        copied_text = interaction_data.get("copied_text", False)
        shared = interaction_data.get("shared", False)

        # Infer sentiment
        sentiment = 0.0

        # Time spent (positive if reasonable, negative if too short)
        if time_spent < 2:
            sentiment -= 0.3  # Too quick, likely not helpful
        elif time_spent > 5:
            sentiment += 0.3  # Engaged with content

        # Positive signals
        if follow_up_question:
            sentiment += 0.2
        if copied_text:
            sentiment += 0.3
        if shared:
            sentiment += 0.4

        # Clamp sentiment
        sentiment = max(-1.0, min(1.0, sentiment))

        # Determine feedback type
        if sentiment > 0.2:
            feedback_type = FeedbackType.IMPLICIT_POSITIVE
        elif sentiment < -0.2:
            feedback_type = FeedbackType.IMPLICIT_NEGATIVE
        else:
            feedback_type = FeedbackType.IMPLICIT_POSITIVE

        # Create feedback signal
        feedback = FeedbackSignal(
            feedback_id=generate_id("feedback_implicit"),
            user_id=user_id,
            feedback_type=feedback_type,
            timestamp=datetime.utcnow(),
            message_id=message_id,
            sentiment=sentiment,
            sentiment_confidence=0.5,  # Lower confidence for implicit
            context=interaction_data,
        )

        # Store
        await self._store_feedback(feedback)

        return feedback

    def _load_sentiment_lexicon(self) -> None:
        """Load sentiment lexicon."""
        self._positive_words = {
            "good",
            "great",
            "excellent",
            "awesome",
            "amazing",
            "wonderful",
            "fantastic",
            "perfect",
            "helpful",
            "useful",
            "clear",
            "love",
            "like",
            "appreciate",
            "thanks",
            "thank",
            "correct",
            "right",
            "accurate",
            "better",
            "best",
            "brilliant",
            "superb",
            "outstanding",
        }

        self._negative_words = {
            "bad",
            "poor",
            "terrible",
            "awful",
            "horrible",
            "wrong",
            "incorrect",
            "useless",
            "unhelpful",
            "unclear",
            "confusing",
            "hate",
            "dislike",
            "disappointing",
            "disappointed",
            "worse",
            "worst",
            "broken",
            "fail",
            "failed",
            "error",
            "mistake",
            "problem",
            "issue",
        }

        self._intensifiers = {
            "very",
            "extremely",
            "incredibly",
            "absolutely",
            "really",
            "so",
            "too",
            "quite",
            "fairly",
            "pretty",
        }

    async def clear_user_feedback(self, user_id: str) -> None:
        """Clear all feedback for a user."""
        async with self._lock:
            self._user_feedback.pop(user_id, None)

        if self._cache:
            await self._cache.invalidate(f"feedback_summary:{user_id}")

    @property
    def stats(self) -> Dict[str, Any]:
        """Get module statistics."""
        return {
            "feedback_processed": self._feedback_processed,
            "feedback_actionable": self._feedback_actionable,
            "sentiment_analyzed": self._sentiment_analyzed,
            "users_tracked": len(self._user_feedback),
            "circuit_breaker_state": self._circuit_breaker.state,
            "cache_stats": self._cache.stats if self._cache else None,
        }
