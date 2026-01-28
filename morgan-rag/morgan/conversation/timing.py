"""
Optimal timing detection module.

Detects optimal timing for conversations, responses, and interactions
based on user patterns, availability, and contextual factors.
"""

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from morgan.config import get_settings
from morgan.intelligence.core.models import ConversationContext, EmotionalState
from morgan.utils.cache import FileCache
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class TimingContext(Enum):
    """Context for timing analysis."""

    CONVERSATION_START = "conversation_start"
    RESPONSE_TIMING = "response_timing"
    TOPIC_INTRODUCTION = "topic_introduction"
    QUESTION_ASKING = "question_asking"
    EMOTIONAL_SUPPORT = "emotional_support"
    BREAK_SUGGESTION = "break_suggestion"


class AvailabilityLevel(Enum):
    """User availability levels."""

    HIGHLY_AVAILABLE = "highly_available"
    AVAILABLE = "available"
    SOMEWHAT_AVAILABLE = "somewhat_available"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"


@dataclass
class TimingPattern:
    """User's timing pattern for specific context."""

    context: TimingContext
    preferred_times: List[Tuple[time, time]]  # (start_time, end_time) pairs
    response_time_preferences: Dict[str, float]  # context -> preferred seconds
    availability_patterns: Dict[int, AvailabilityLevel]  # hour -> availability
    engagement_patterns: Dict[int, float]  # hour -> engagement level
    pattern_confidence: float = 0.0


@dataclass
class TimingRecommendation:
    """Timing recommendation for an action."""

    recommended_action: str
    optimal_timing: datetime
    confidence: float
    reasoning: str
    alternative_timings: List[datetime] = field(default_factory=list)
    urgency_level: str = "normal"  # low, normal, high, urgent


@dataclass
class ResponseTimingAnalysis:
    """Analysis of response timing patterns."""

    average_response_time: float  # seconds
    response_time_variance: float
    optimal_response_window: Tuple[float, float]  # (min_seconds, max_seconds)
    context_specific_timings: Dict[str, float]
    user_patience_level: float  # 0.0 to 1.0


class OptimalTimingDetector:
    """
    Optimal timing detection system.

    Features:
    - User availability pattern learning
    - Response timing optimization
    - Conversation flow timing
    - Emotional state timing considerations
    - Context-aware timing recommendations
    - Interruption avoidance
    """

    def __init__(self):
        """Initialize optimal timing detector."""
        self.settings = get_settings()

        # Setup cache for timing data
        cache_dir = self.settings.morgan_data_dir / "cache" / "timing"
        self.cache = FileCache(cache_dir)

        # User timing data
        self.user_timing_patterns: Dict[str, Dict[TimingContext, TimingPattern]] = (
            defaultdict(dict)
        )
        self.user_interaction_history: Dict[str, List[ConversationContext]] = (
            defaultdict(list)
        )
        self.response_timing_data: Dict[str, List[Tuple[datetime, float]]] = (
            defaultdict(list)
        )

        logger.info("Optimal Timing Detector initialized")

    def detect_optimal_timing(
        self,
        user_id: str,
        context: TimingContext,
        emotional_state: Optional[EmotionalState] = None,
        urgency_level: str = "normal",
        current_time: Optional[datetime] = None,
    ) -> TimingRecommendation:
        """
        Detect optimal timing for a specific context.

        Args:
            user_id: User identifier
            context: Timing context
            emotional_state: Current emotional state
            urgency_level: Urgency of the action
            current_time: Current time (defaults to now)

        Returns:
            Timing recommendation
        """
        if current_time is None:
            current_time = datetime.utcnow()

        # Get user's timing patterns
        timing_patterns = self.user_timing_patterns.get(user_id, {})
        pattern = timing_patterns.get(context)

        # Analyze current availability
        current_availability = self._analyze_current_availability(
            user_id, current_time
        )

        # Consider emotional state
        emotional_timing_factor = self._analyze_emotional_timing(
            emotional_state, context
        )

        # Generate timing recommendation
        recommendation = self._generate_timing_recommendation(
            user_id,
            context,
            pattern,
            current_availability,
            emotional_timing_factor,
            urgency_level,
            current_time,
        )

        logger.debug(
            f"Generated timing recommendation for user {user_id}, "
            f"context {context.value}: {recommendation.recommended_action} "
            f"at {recommendation.optimal_timing}"
        )

        return recommendation

    def learn_timing_patterns(
        self,
        user_id: str,
        conversation_context: ConversationContext,
        emotional_state: EmotionalState,
        response_time: Optional[float] = None,
    ) -> None:
        """
        Learn timing patterns from user interactions.

        Args:
            user_id: User identifier
            conversation_context: Conversation context
            emotional_state: User's emotional state
            response_time: Time taken to respond (seconds)
        """
        # Store interaction history
        self.user_interaction_history[user_id].append(conversation_context)

        # Keep only recent interactions (last 200)
        if len(self.user_interaction_history[user_id]) > 200:
            self.user_interaction_history[user_id] = self.user_interaction_history[
                user_id
            ][-200:]

        # Store response timing data
        if response_time is not None:
            self.response_timing_data[user_id].append(
                (conversation_context.timestamp, response_time)
            )

            # Keep only recent timing data (last 100)
            if len(self.response_timing_data[user_id]) > 100:
                self.response_timing_data[user_id] = self.response_timing_data[user_id][
                    -100:
                ]

        # Update timing patterns
        self._update_timing_patterns(user_id, conversation_context, emotional_state)

        # Cache updated patterns
        self._cache_timing_patterns(user_id)

        logger.debug(f"Updated timing patterns for user {user_id}")

    def analyze_response_timing(self, user_id: str) -> ResponseTimingAnalysis:
        """
        Analyze user's response timing patterns.

        Args:
            user_id: User identifier

        Returns:
            Response timing analysis
        """
        timing_data = self.response_timing_data.get(user_id, [])
        if not timing_data:
            # Return default analysis
            return ResponseTimingAnalysis(
                average_response_time=30.0,
                response_time_variance=15.0,
                optimal_response_window=(10.0, 60.0),
                context_specific_timings={},
                user_patience_level=0.5,
            )

        # Calculate statistics
        response_times = [rt for _, rt in timing_data]
        avg_response_time = sum(response_times) / len(response_times)
        
        # Calculate variance
        variance = sum((rt - avg_response_time) ** 2 for rt in response_times) / len(response_times)
        
        # Determine optimal response window
        min_optimal = max(5.0, avg_response_time - variance ** 0.5)
        max_optimal = avg_response_time + variance ** 0.5

        # Analyze context-specific timings
        context_timings = self._analyze_context_specific_timings(user_id)

        # Estimate user patience level
        patience_level = self._estimate_user_patience(response_times)

        return ResponseTimingAnalysis(
            average_response_time=avg_response_time,
            response_time_variance=variance,
            optimal_response_window=(min_optimal, max_optimal),
            context_specific_timings=context_timings,
            user_patience_level=patience_level,
        )

    def get_availability_forecast(
        self, user_id: str, forecast_hours: int = 24
    ) -> Dict[datetime, AvailabilityLevel]:
        """
        Get availability forecast for user.

        Args:
            user_id: User identifier
            forecast_hours: Hours to forecast

        Returns:
            Availability forecast by hour
        """
        forecast = {}
        current_time = datetime.utcnow()

        # Get user's availability patterns
        timing_patterns = self.user_timing_patterns.get(user_id, {})

        for hour_offset in range(forecast_hours):
            forecast_time = current_time + timedelta(hours=hour_offset)
            availability = self._predict_availability(user_id, forecast_time)
            forecast[forecast_time] = availability

        return forecast

    def suggest_conversation_break(
        self, user_id: str, conversation_duration: timedelta
    ) -> Optional[TimingRecommendation]:
        """
        Suggest when to take a conversation break.

        Args:
            user_id: User identifier
            conversation_duration: Current conversation duration

        Returns:
            Break timing recommendation or None
        """
        # Get user's conversation patterns
        interaction_history = self.user_interaction_history.get(user_id, [])
        
        if not interaction_history:
            return None

        # Analyze typical conversation lengths
        conversation_lengths = self._analyze_conversation_lengths(interaction_history)
        avg_length = sum(conversation_lengths) / len(conversation_lengths) if conversation_lengths else 30

        # Suggest break if current conversation is significantly longer
        if conversation_duration.total_seconds() / 60 > avg_length * 1.5:
            return TimingRecommendation(
                recommended_action="suggest_break",
                optimal_timing=datetime.utcnow() + timedelta(minutes=2),
                confidence=0.8,
                reasoning=f"Conversation duration ({conversation_duration.total_seconds()/60:.1f} min) "
                         f"exceeds typical length ({avg_length:.1f} min)",
                urgency_level="low",
            )

        return None

    def optimize_response_timing(
        self,
        user_id: str,
        message_complexity: str = "normal",
        emotional_urgency: float = 0.5,
    ) -> float:
        """
        Optimize response timing based on user patterns.

        Args:
            user_id: User identifier
            message_complexity: Complexity of response (simple, normal, complex)
            emotional_urgency: Emotional urgency level (0.0 to 1.0)

        Returns:
            Optimal response delay in seconds
        """
        # Get user's response timing analysis
        timing_analysis = self.analyze_response_timing(user_id)

        # Base timing from user patterns
        base_timing = timing_analysis.average_response_time

        # Adjust for message complexity
        complexity_multipliers = {
            "simple": 0.7,
            "normal": 1.0,
            "complex": 1.5,
        }
        complexity_factor = complexity_multipliers.get(message_complexity, 1.0)

        # Adjust for emotional urgency
        urgency_factor = 1.0 - (emotional_urgency * 0.5)  # Higher urgency = faster response

        # Calculate optimal timing
        optimal_timing = base_timing * complexity_factor * urgency_factor

        # Ensure within reasonable bounds
        min_timing, max_timing = timing_analysis.optimal_response_window
        optimal_timing = max(min_timing, min(max_timing, optimal_timing))

        return optimal_timing

    def _analyze_current_availability(
        self, user_id: str, current_time: datetime
    ) -> AvailabilityLevel:
        """Analyze user's current availability."""
        # Get user's historical patterns for this time
        interaction_history = self.user_interaction_history.get(user_id, [])
        
        if not interaction_history:
            return AvailabilityLevel.AVAILABLE

        # Analyze interactions at similar times
        current_hour = current_time.hour
        current_day_of_week = current_time.weekday()

        similar_time_interactions = [
            ctx for ctx in interaction_history
            if abs(ctx.timestamp.hour - current_hour) <= 1
            and ctx.timestamp.weekday() == current_day_of_week
        ]

        if not similar_time_interactions:
            return AvailabilityLevel.AVAILABLE

        # Calculate availability based on interaction frequency and response patterns
        interaction_frequency = len(similar_time_interactions) / max(1, len(interaction_history) / 7)
        
        if interaction_frequency > 0.8:
            return AvailabilityLevel.HIGHLY_AVAILABLE
        elif interaction_frequency > 0.5:
            return AvailabilityLevel.AVAILABLE
        elif interaction_frequency > 0.2:
            return AvailabilityLevel.SOMEWHAT_AVAILABLE
        else:
            return AvailabilityLevel.BUSY

    def _analyze_emotional_timing(
        self, emotional_state: Optional[EmotionalState], context: TimingContext
    ) -> float:
        """Analyze timing factor based on emotional state."""
        if not emotional_state:
            return 1.0

        # High emotional intensity may require immediate attention
        if emotional_state.intensity > 0.8:
            if context == TimingContext.EMOTIONAL_SUPPORT:
                return 0.1  # Respond immediately
            else:
                return 0.5  # Respond faster than normal

        # Low emotional intensity allows for normal timing
        if emotional_state.intensity < 0.3:
            return 1.2  # Can wait a bit longer

        # Specific emotional states
        if emotional_state.primary_emotion.value in ["anger", "fear", "sadness"]:
            return 0.7  # Respond sooner for negative emotions
        elif emotional_state.primary_emotion.value in ["joy", "excitement"]:
            return 1.0  # Normal timing for positive emotions

        return 1.0

    def _generate_timing_recommendation(
        self,
        user_id: str,
        context: TimingContext,
        pattern: Optional[TimingPattern],
        availability: AvailabilityLevel,
        emotional_factor: float,
        urgency_level: str,
        current_time: datetime,
    ) -> TimingRecommendation:
        """Generate timing recommendation based on analysis."""
        # Base timing calculation
        if pattern and pattern.response_time_preferences:
            base_delay = pattern.response_time_preferences.get(context.value, 30.0)
        else:
            base_delay = 30.0  # Default 30 seconds

        # Adjust for availability
        availability_multipliers = {
            AvailabilityLevel.HIGHLY_AVAILABLE: 0.5,
            AvailabilityLevel.AVAILABLE: 1.0,
            AvailabilityLevel.SOMEWHAT_AVAILABLE: 1.5,
            AvailabilityLevel.BUSY: 3.0,
            AvailabilityLevel.UNAVAILABLE: 10.0,
        }
        availability_factor = availability_multipliers[availability]

        # Adjust for urgency
        urgency_multipliers = {
            "low": 2.0,
            "normal": 1.0,
            "high": 0.5,
            "urgent": 0.1,
        }
        urgency_factor = urgency_multipliers.get(urgency_level, 1.0)

        # Calculate final timing
        final_delay = base_delay * availability_factor * emotional_factor * urgency_factor
        final_delay = max(1.0, min(300.0, final_delay))  # Between 1 second and 5 minutes

        optimal_timing = current_time + timedelta(seconds=final_delay)

        # Generate reasoning
        reasoning_parts = []
        reasoning_parts.append(f"User availability: {availability.value}")
        if emotional_factor != 1.0:
            reasoning_parts.append(f"Emotional timing factor: {emotional_factor:.1f}")
        reasoning_parts.append(f"Urgency level: {urgency_level}")

        reasoning = "; ".join(reasoning_parts)

        # Calculate confidence
        confidence = 0.7
        if pattern:
            confidence += 0.2
        if availability in [AvailabilityLevel.HIGHLY_AVAILABLE, AvailabilityLevel.AVAILABLE]:
            confidence += 0.1

        # Generate alternative timings
        alternatives = []
        for offset in [5, 15, 60]:  # 5 seconds, 15 seconds, 1 minute
            alt_time = optimal_timing + timedelta(seconds=offset)
            alternatives.append(alt_time)

        # Determine recommended action
        if urgency_level == "urgent":
            action = "respond_immediately"
        elif availability == AvailabilityLevel.UNAVAILABLE:
            action = "wait_for_availability"
        elif context == TimingContext.EMOTIONAL_SUPPORT:
            action = "provide_timely_support"
        else:
            action = "respond_optimally"

        return TimingRecommendation(
            recommended_action=action,
            optimal_timing=optimal_timing,
            confidence=confidence,
            reasoning=reasoning,
            alternative_timings=alternatives,
            urgency_level=urgency_level,
        )

    def _update_timing_patterns(
        self,
        user_id: str,
        context: ConversationContext,
        emotional_state: EmotionalState,
    ) -> None:
        """Update user's timing patterns."""
        interaction_time = context.timestamp
        hour = interaction_time.hour

        # Update availability patterns
        for timing_context in TimingContext:
            if timing_context not in self.user_timing_patterns[user_id]:
                self.user_timing_patterns[user_id][timing_context] = TimingPattern(
                    context=timing_context,
                    preferred_times=[],
                    response_time_preferences={},
                    availability_patterns={},
                    engagement_patterns={},
                )

            pattern = self.user_timing_patterns[user_id][timing_context]

            # Update availability for this hour
            current_availability = self._determine_availability_from_interaction(
                context, emotional_state
            )
            pattern.availability_patterns[hour] = current_availability

            # Update engagement patterns
            engagement_level = emotional_state.intensity
            if hour in pattern.engagement_patterns:
                # Weighted average with existing data
                pattern.engagement_patterns[hour] = (
                    pattern.engagement_patterns[hour] * 0.7 + engagement_level * 0.3
                )
            else:
                pattern.engagement_patterns[hour] = engagement_level

            # Update pattern confidence
            interaction_count = len(self.user_interaction_history[user_id])
            pattern.pattern_confidence = min(1.0, interaction_count / 50.0)

    def _determine_availability_from_interaction(
        self, context: ConversationContext, emotional_state: EmotionalState
    ) -> AvailabilityLevel:
        """Determine availability level from interaction characteristics."""
        # Quick responses suggest high availability
        if hasattr(context, "response_time") and context.response_time:
            if context.response_time < 10:
                return AvailabilityLevel.HIGHLY_AVAILABLE
            elif context.response_time < 30:
                return AvailabilityLevel.AVAILABLE
            elif context.response_time < 120:
                return AvailabilityLevel.SOMEWHAT_AVAILABLE
            else:
                return AvailabilityLevel.BUSY

        # Long messages suggest availability
        if len(context.message_text) > 200:
            return AvailabilityLevel.AVAILABLE
        elif len(context.message_text) < 50:
            return AvailabilityLevel.SOMEWHAT_AVAILABLE

        # High emotional engagement suggests availability
        if emotional_state.intensity > 0.7:
            return AvailabilityLevel.AVAILABLE

        return AvailabilityLevel.AVAILABLE  # Default

    def _predict_availability(
        self, user_id: str, forecast_time: datetime
    ) -> AvailabilityLevel:
        """Predict user availability at a specific time."""
        timing_patterns = self.user_timing_patterns.get(user_id, {})
        
        if not timing_patterns:
            return AvailabilityLevel.AVAILABLE

        hour = forecast_time.hour
        availability_votes = []

        # Collect availability predictions from all patterns
        for pattern in timing_patterns.values():
            if hour in pattern.availability_patterns:
                availability_votes.append(pattern.availability_patterns[hour])

        if not availability_votes:
            return AvailabilityLevel.AVAILABLE

        # Return most common availability level
        from collections import Counter
        availability_counts = Counter(availability_votes)
        most_common = availability_counts.most_common(1)[0][0]
        return most_common

    def _analyze_conversation_lengths(
        self, interaction_history: List[ConversationContext]
    ) -> List[float]:
        """Analyze typical conversation lengths in minutes."""
        if len(interaction_history) < 2:
            return [30.0]  # Default 30 minutes

        # Group interactions by conversation_id
        conversations = defaultdict(list)
        for ctx in interaction_history:
            conversations[ctx.conversation_id].append(ctx)

        lengths = []
        for conversation_turns in conversations.values():
            if len(conversation_turns) >= 2:
                # Calculate duration from first to last turn
                conversation_turns.sort(key=lambda x: x.timestamp)
                duration = (
                    conversation_turns[-1].timestamp - conversation_turns[0].timestamp
                ).total_seconds() / 60
                lengths.append(max(1.0, duration))  # At least 1 minute

        return lengths if lengths else [30.0]

    def _analyze_context_specific_timings(self, user_id: str) -> Dict[str, float]:
        """Analyze response timings for different contexts."""
        timing_data = self.response_timing_data.get(user_id, [])
        interaction_history = self.user_interaction_history.get(user_id, [])

        if not timing_data or not interaction_history:
            return {}

        # Create mapping of timestamps to contexts
        context_timings = defaultdict(list)
        
        for ctx in interaction_history:
            # Find corresponding timing data
            for timestamp, response_time in timing_data:
                if abs((timestamp - ctx.timestamp).total_seconds()) < 60:  # Within 1 minute
                    # Determine context based on message content
                    message_context = self._classify_message_context(ctx.message_text)
                    context_timings[message_context].append(response_time)
                    break

        # Calculate average timing for each context
        context_averages = {}
        for context, timings in context_timings.items():
            if timings:
                context_averages[context] = sum(timings) / len(timings)

        return context_averages

    def _classify_message_context(self, message_text: str) -> str:
        """Classify message context for timing analysis."""
        message_lower = message_text.lower()

        if any(word in message_lower for word in ["?", "what", "how", "why", "when", "where"]):
            return "question"
        elif any(word in message_lower for word in ["help", "problem", "issue", "trouble"]):
            return "support_request"
        elif any(word in message_lower for word in ["sad", "angry", "upset", "worried", "anxious"]):
            return "emotional_expression"
        elif any(word in message_lower for word in ["thanks", "thank you", "appreciate"]):
            return "gratitude"
        elif len(message_text) > 200:
            return "detailed_sharing"
        else:
            return "general_conversation"

    def _estimate_user_patience(self, response_times: List[float]) -> float:
        """Estimate user patience level based on response times."""
        if not response_times:
            return 0.5

        # Users with consistently quick responses may have lower patience
        avg_response_time = sum(response_times) / len(response_times)
        
        if avg_response_time < 10:
            return 0.3  # Low patience - expects quick responses
        elif avg_response_time < 30:
            return 0.5  # Moderate patience
        elif avg_response_time < 60:
            return 0.7  # Good patience
        else:
            return 0.9  # High patience

    def _cache_timing_patterns(self, user_id: str) -> None:
        """Cache user's timing patterns."""
        cache_key = f"timing_patterns_{user_id}"
        
        # Convert patterns to cacheable format
        patterns_data = {}
        for context, pattern in self.user_timing_patterns[user_id].items():
            patterns_data[context.value] = {
                "response_time_preferences": pattern.response_time_preferences,
                "availability_patterns": {
                    str(hour): availability.value
                    for hour, availability in pattern.availability_patterns.items()
                },
                "engagement_patterns": pattern.engagement_patterns,
                "pattern_confidence": pattern.pattern_confidence,
            }
        
        self.cache.set(cache_key, patterns_data)


# Singleton instance
_optimal_timing_detector_instance = None
_optimal_timing_detector_lock = threading.Lock()


def get_optimal_timing_detector() -> OptimalTimingDetector:
    """
    Get singleton optimal timing detector instance.

    Returns:
        Shared OptimalTimingDetector instance
    """
    global _optimal_timing_detector_instance

    if _optimal_timing_detector_instance is None:
        with _optimal_timing_detector_lock:
            if _optimal_timing_detector_instance is None:
                _optimal_timing_detector_instance = OptimalTimingDetector()

    return _optimal_timing_detector_instance