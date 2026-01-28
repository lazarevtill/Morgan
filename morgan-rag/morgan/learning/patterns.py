"""
Interaction Pattern Analysis for Morgan RAG.

Analyzes user interaction patterns to identify communication preferences,
topic interests, timing patterns, and behavioral tendencies for personalization.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional

from ..intelligence.core.models import CommunicationStyle, InteractionData, ResponseLength
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PatternType(Enum):
    """Types of interaction patterns."""

    COMMUNICATION = "communication"
    TOPIC = "topic"
    TIMING = "timing"
    BEHAVIORAL = "behavioral"
    EMOTIONAL = "emotional"


class PatternConfidence(Enum):
    """Confidence levels for pattern identification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class CommunicationPattern:
    """Communication style and preference patterns."""

    pattern_id: str
    user_id: str
    preferred_style: CommunicationStyle
    formality_level: float  # 0.0 (casual) to 1.0 (formal)
    technical_depth: float  # 0.0 (simple) to 1.0 (technical)
    response_length_preference: ResponseLength
    question_types: List[str]  # Types of questions user asks
    vocabulary_level: str  # "basic", "intermediate", "advanced"
    confidence: PatternConfidence
    sample_size: int
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TopicPattern:
    """Topic interest and domain patterns."""

    pattern_id: str
    user_id: str
    primary_topics: List[str]
    topic_frequencies: Dict[str, int]
    domain_expertise: Dict[str, float]  # domain -> expertise level (0.0-1.0)
    learning_areas: List[str]  # Areas user is actively learning
    avoided_topics: List[str]  # Topics user shows no interest in
    topic_transitions: Dict[str, List[str]]  # Common topic flow patterns
    confidence: PatternConfidence
    analysis_period: timedelta
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TimingPattern:
    """Temporal interaction patterns."""

    pattern_id: str
    user_id: str
    active_hours: List[int]  # Hours of day when user is most active (0-23)
    session_durations: Dict[str, float]  # Average session durations by time
    response_time_expectations: float  # Expected response time in seconds
    interaction_frequency: str  # "daily", "weekly", "sporadic"
    peak_activity_days: List[str]  # Days of week with highest activity
    timezone_preference: Optional[str]
    confidence: PatternConfidence
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BehavioralPattern:
    """Behavioral and interaction style patterns."""

    pattern_id: str
    user_id: str
    interaction_style: str  # "exploratory", "goal-oriented", "conversational"
    feedback_frequency: float  # How often user provides feedback
    help_seeking_behavior: str  # "direct", "indirect", "exploratory"
    error_tolerance: float  # 0.0 (low) to 1.0 (high)
    learning_style: str  # "visual", "textual", "interactive"
    social_cues: List[str]  # Social interaction preferences
    attention_span: str  # "short", "medium", "long"
    confidence: PatternConfidence
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class InteractionPatterns:
    """Complete set of interaction patterns for a user."""

    user_id: str
    analysis_period: timedelta
    communication_patterns: List[CommunicationPattern]
    topic_patterns: List[TopicPattern]
    timing_patterns: List[TimingPattern]
    behavioral_patterns: List[BehavioralPattern]
    overall_confidence: float  # 0.0 to 1.0
    pattern_stability: float  # How stable patterns are over time
    last_analysis: datetime = field(default_factory=datetime.utcnow)


class InteractionPatternAnalyzer:
    """
    Analyzes user interaction patterns for personalization.

    Identifies communication preferences, topic interests, timing patterns,
    and behavioral tendencies from user interactions.
    """

    # Pattern detection thresholds
    MIN_INTERACTIONS_FOR_PATTERN = 5
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    MEDIUM_CONFIDENCE_THRESHOLD = 0.6

    # Communication pattern indicators
    FORMALITY_INDICATORS = {
        "formal": [
            r"\b(please|thank you|would you|could you|may I)\b",
            r"\b(sir|madam|mr\.|ms\.|dr\.)\b",
            r"\b(appreciate|grateful|kindly|respectfully)\b",
        ],
        "casual": [
            r"\b(hey|hi|yeah|yep|nope|gonna|wanna)\b",
            r"\b(cool|awesome|great|nice|sweet)\b",
            r"[!]{2,}|[?]{2,}",  # Multiple punctuation
        ],
    }

    TECHNICAL_INDICATORS = [
        r"\b(algorithm|implementation|architecture|framework)\b",
        r"\b(API|SDK|database|server|client)\b",
        r"\b(function|method|class|variable|parameter)\b",
        r"\b(optimize|performance|scalability|efficiency)\b",
    ]

    def __init__(self):
        """Initialize the pattern analyzer."""
        self.user_patterns: Dict[str, InteractionPatterns] = {}
        logger.info("Interaction pattern analyzer initialized")

    def analyze_patterns(
        self,
        user_id: str,
        interactions: List[InteractionData],
        analysis_period: timedelta = timedelta(days=30),
    ) -> InteractionPatterns:
        """
        Analyze interaction patterns for a user.

        Args:
            user_id: User identifier
            interactions: List of interactions to analyze
            analysis_period: Time period for analysis

        Returns:
            InteractionPatterns: Identified patterns
        """
        logger.info(
            f"Analyzing patterns for user {user_id} with {len(interactions)} interactions"
        )

        if len(interactions) < self.MIN_INTERACTIONS_FOR_PATTERN:
            logger.warning(
                f"Insufficient interactions ({len(interactions)}) for pattern analysis"
            )
            return self._create_empty_patterns(user_id, analysis_period)

        # Analyze different pattern types
        communication_patterns = self._analyze_communication_patterns(
            user_id, interactions
        )
        topic_patterns = self._analyze_topic_patterns(
            user_id, interactions, analysis_period
        )
        timing_patterns = self._analyze_timing_patterns(user_id, interactions)
        behavioral_patterns = self._analyze_behavioral_patterns(user_id, interactions)

        # Calculate overall confidence and stability
        overall_confidence = self._calculate_overall_confidence(
            [
                communication_patterns,
                topic_patterns,
                timing_patterns,
                behavioral_patterns,
            ]
        )

        pattern_stability = self._calculate_pattern_stability(user_id, interactions)

        patterns = InteractionPatterns(
            user_id=user_id,
            analysis_period=analysis_period,
            communication_patterns=communication_patterns,
            topic_patterns=topic_patterns,
            timing_patterns=timing_patterns,
            behavioral_patterns=behavioral_patterns,
            overall_confidence=overall_confidence,
            pattern_stability=pattern_stability,
        )

        # Store patterns for future comparison
        self.user_patterns[user_id] = patterns

        logger.info(f"Pattern analysis complete for user {user_id}")
        return patterns

    def _analyze_communication_patterns(
        self, user_id: str, interactions: List[InteractionData]
    ) -> List[CommunicationPattern]:
        """Analyze communication style patterns."""
        logger.debug(f"Analyzing communication patterns for user {user_id}")

        # Collect text data
        messages = []
        for interaction in interactions:
            if hasattr(interaction.conversation_context, "message_text"):
                messages.append(interaction.conversation_context.message_text)

        if not messages:
            return []

        # Analyze formality level
        formality_level = self._calculate_formality_level(messages)

        # Analyze technical depth
        technical_depth = self._calculate_technical_depth(messages)

        # Determine preferred communication style
        preferred_style = self._determine_communication_style(
            formality_level, technical_depth
        )

        # Analyze response length preference
        response_length_pref = self._analyze_response_length_preference(interactions)

        # Analyze question types
        question_types = self._extract_question_types(messages)

        # Determine vocabulary level
        vocabulary_level = self._determine_vocabulary_level(messages)

        # Calculate confidence
        confidence = self._calculate_confidence(len(interactions))

        pattern = CommunicationPattern(
            pattern_id=f"comm_{user_id}_{datetime.now(timezone.utc).timestamp()}",
            user_id=user_id,
            preferred_style=preferred_style,
            formality_level=formality_level,
            technical_depth=technical_depth,
            response_length_preference=response_length_pref,
            question_types=question_types,
            vocabulary_level=vocabulary_level,
            confidence=confidence,
            sample_size=len(interactions),
        )

        return [pattern]

    def _analyze_topic_patterns(
        self,
        user_id: str,
        interactions: List[InteractionData],
        analysis_period: timedelta,
    ) -> List[TopicPattern]:
        """Analyze topic interest patterns."""
        logger.debug(f"Analyzing topic patterns for user {user_id}")

        # Extract topics from interactions
        all_topics = []
        topic_counter = Counter()

        for interaction in interactions:
            topics = getattr(interaction, "topics_discussed", [])
            all_topics.extend(topics)
            topic_counter.update(topics)

        if not all_topics:
            return []

        # Identify primary topics (top 5)
        primary_topics = [topic for topic, _ in topic_counter.most_common(5)]

        # Estimate domain expertise based on topic frequency and depth
        domain_expertise = self._estimate_domain_expertise(topic_counter, interactions)

        # Identify learning areas (topics with increasing frequency)
        learning_areas = self._identify_learning_areas(interactions)

        # Identify avoided topics (topics mentioned but not engaged with)
        avoided_topics = self._identify_avoided_topics(interactions)

        # Analyze topic transitions
        topic_transitions = self._analyze_topic_transitions(interactions)

        confidence = self._calculate_confidence(len(all_topics))

        pattern = TopicPattern(
            pattern_id=f"topic_{user_id}_{datetime.now(timezone.utc).timestamp()}",
            user_id=user_id,
            primary_topics=primary_topics,
            topic_frequencies=dict(topic_counter),
            domain_expertise=domain_expertise,
            learning_areas=learning_areas,
            avoided_topics=avoided_topics,
            topic_transitions=topic_transitions,
            confidence=confidence,
            analysis_period=analysis_period,
        )

        return [pattern]

    def _analyze_timing_patterns(
        self, user_id: str, interactions: List[InteractionData]
    ) -> List[TimingPattern]:
        """Analyze temporal interaction patterns."""
        logger.debug(f"Analyzing timing patterns for user {user_id}")

        # Extract timestamps
        timestamps = []
        for interaction in interactions:
            if hasattr(interaction.conversation_context, "timestamp"):
                timestamps.append(interaction.conversation_context.timestamp)

        if not timestamps:
            return []

        # Analyze active hours
        active_hours = self._analyze_active_hours(timestamps)

        # Analyze session durations
        session_durations = self._analyze_session_durations(interactions)

        # Estimate response time expectations
        response_time_expectations = self._estimate_response_time_expectations(
            interactions
        )

        # Determine interaction frequency
        interaction_frequency = self._determine_interaction_frequency(timestamps)

        # Identify peak activity days
        peak_activity_days = self._identify_peak_activity_days(timestamps)

        confidence = self._calculate_confidence(len(timestamps))

        pattern = TimingPattern(
            pattern_id=f"timing_{user_id}_{datetime.now(timezone.utc).timestamp()}",
            user_id=user_id,
            active_hours=active_hours,
            session_durations=session_durations,
            response_time_expectations=response_time_expectations,
            interaction_frequency=interaction_frequency,
            peak_activity_days=peak_activity_days,
            timezone_preference=None,  # Would need additional data
            confidence=confidence,
        )

        return [pattern]

    def _analyze_behavioral_patterns(
        self, user_id: str, interactions: List[InteractionData]
    ) -> List[BehavioralPattern]:
        """Analyze behavioral interaction patterns."""
        logger.debug(f"Analyzing behavioral patterns for user {user_id}")

        # Determine interaction style
        interaction_style = self._determine_interaction_style(interactions)

        # Calculate feedback frequency
        feedback_frequency = self._calculate_feedback_frequency(interactions)

        # Analyze help-seeking behavior
        help_seeking_behavior = self._analyze_help_seeking_behavior(interactions)

        # Estimate error tolerance
        error_tolerance = self._estimate_error_tolerance(interactions)

        # Determine learning style
        learning_style = self._determine_learning_style(interactions)

        # Extract social cues
        social_cues = self._extract_social_cues(interactions)

        # Estimate attention span
        attention_span = self._estimate_attention_span(interactions)

        confidence = self._calculate_confidence(len(interactions))

        pattern = BehavioralPattern(
            pattern_id=f"behavior_{user_id}_{datetime.now(timezone.utc).timestamp()}",
            user_id=user_id,
            interaction_style=interaction_style,
            feedback_frequency=feedback_frequency,
            help_seeking_behavior=help_seeking_behavior,
            error_tolerance=error_tolerance,
            learning_style=learning_style,
            social_cues=social_cues,
            attention_span=attention_span,
            confidence=confidence,
        )

        return [pattern]

    def _calculate_formality_level(self, messages: List[str]) -> float:
        """Calculate formality level from messages."""
        formal_count = 0
        casual_count = 0

        for message in messages:
            message_lower = message.lower()

            # Count formal indicators
            for pattern in self.FORMALITY_INDICATORS["formal"]:
                formal_count += len(re.findall(pattern, message_lower))

            # Count casual indicators
            for pattern in self.FORMALITY_INDICATORS["casual"]:
                casual_count += len(re.findall(pattern, message_lower))

        if formal_count + casual_count == 0:
            return 0.5  # Neutral

        return formal_count / (formal_count + casual_count)

    def _calculate_technical_depth(self, messages: List[str]) -> float:
        """Calculate technical depth from messages."""
        technical_count = 0
        total_words = 0

        for message in messages:
            words = message.lower().split()
            total_words += len(words)

            for pattern in self.TECHNICAL_INDICATORS:
                technical_count += len(re.findall(pattern, message.lower()))

        if total_words == 0:
            return 0.0

        return min(
            technical_count / (total_words / 100), 1.0
        )  # Normalize per 100 words

    def _determine_communication_style(
        self, formality_level: float, technical_depth: float
    ) -> CommunicationStyle:
        """Determine communication style from metrics."""
        if formality_level > 0.7:
            return CommunicationStyle.FORMAL
        elif technical_depth > 0.5:
            return CommunicationStyle.TECHNICAL
        elif formality_level > 0.4:
            return CommunicationStyle.PROFESSIONAL
        else:
            return CommunicationStyle.CASUAL

    def _analyze_response_length_preference(
        self, interactions: List[InteractionData]
    ) -> ResponseLength:
        """Analyze preferred response length."""
        # This would analyze user feedback on response lengths
        # For now, return default
        return ResponseLength.DETAILED

    def _extract_question_types(self, messages: List[str]) -> List[str]:
        """Extract types of questions user asks."""
        question_types = []

        for message in messages:
            if "?" in message:
                message_lower = message.lower()
                if any(
                    word in message_lower
                    for word in ["how", "what", "why", "when", "where"]
                ):
                    if "how" in message_lower:
                        question_types.append("how-to")
                    elif "what" in message_lower:
                        question_types.append("definition")
                    elif "why" in message_lower:
                        question_types.append("explanation")
                    elif "when" in message_lower:
                        question_types.append("temporal")
                    elif "where" in message_lower:
                        question_types.append("location")

        return list(set(question_types))

    def _determine_vocabulary_level(self, messages: List[str]) -> str:
        """Determine vocabulary complexity level."""
        # Simple heuristic based on average word length
        total_chars = sum(len(word) for message in messages for word in message.split())
        total_words = sum(len(message.split()) for message in messages)

        if total_words == 0:
            return "basic"

        avg_word_length = total_chars / total_words

        if avg_word_length > 6:
            return "advanced"
        elif avg_word_length > 4.5:
            return "intermediate"
        else:
            return "basic"

    def _calculate_confidence(self, sample_size: int) -> PatternConfidence:
        """Calculate confidence level based on sample size."""
        if sample_size >= 20:
            return PatternConfidence.HIGH
        elif sample_size >= 10:
            return PatternConfidence.MEDIUM
        else:
            return PatternConfidence.LOW

    def _estimate_domain_expertise(
        self, topic_counter: Counter, interactions: List[InteractionData]
    ) -> Dict[str, float]:
        """Estimate domain expertise levels."""
        # Simple heuristic based on topic frequency and interaction depth
        expertise = {}
        total_topics = sum(topic_counter.values())

        for topic, count in topic_counter.items():
            # Base expertise on frequency
            frequency_score = count / total_topics
            # Could add complexity analysis here
            expertise[topic] = min(frequency_score * 2, 1.0)

        return expertise

    def _identify_learning_areas(
        self, interactions: List[InteractionData]
    ) -> List[str]:
        """Identify areas user is actively learning."""
        # Would analyze question patterns and learning indicators
        return []

    def _identify_avoided_topics(
        self, interactions: List[InteractionData]
    ) -> List[str]:
        """Identify topics user avoids."""
        # Would analyze topic engagement patterns
        return []

    def _analyze_topic_transitions(
        self, interactions: List[InteractionData]
    ) -> Dict[str, List[str]]:
        """Analyze common topic transition patterns."""
        # Would analyze how topics flow in conversations
        return {}

    def _analyze_active_hours(self, timestamps: List[datetime]) -> List[int]:
        """Analyze most active hours of the day."""
        hour_counter = Counter(ts.hour for ts in timestamps)
        # Return top 3 most active hours
        return [hour for hour, _ in hour_counter.most_common(3)]

    def _analyze_session_durations(
        self, interactions: List[InteractionData]
    ) -> Dict[str, float]:
        """Analyze session duration patterns."""
        # Would group interactions into sessions and analyze durations
        return {"average": 300.0}  # Default 5 minutes

    def _estimate_response_time_expectations(
        self, interactions: List[InteractionData]
    ) -> float:
        """Estimate expected response time."""
        # Would analyze time between user messages
        return 30.0  # Default 30 seconds

    def _determine_interaction_frequency(self, timestamps: List[datetime]) -> str:
        """Determine interaction frequency pattern."""
        if not timestamps:
            return "sporadic"

        # Calculate average time between interactions
        timestamps.sort()
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i - 1]).total_seconds()
            intervals.append(interval)

        if not intervals:
            return "sporadic"

        avg_interval = sum(intervals) / len(intervals)

        if avg_interval < 86400:  # Less than 1 day
            return "daily"
        elif avg_interval < 604800:  # Less than 1 week
            return "weekly"
        else:
            return "sporadic"

    def _identify_peak_activity_days(self, timestamps: List[datetime]) -> List[str]:
        """Identify peak activity days of the week."""
        day_counter = Counter(ts.strftime("%A") for ts in timestamps)
        return [day for day, _ in day_counter.most_common(3)]

    def _determine_interaction_style(self, interactions: List[InteractionData]) -> str:
        """Determine overall interaction style."""
        # Would analyze question patterns and interaction depth
        return "conversational"  # Default

    def _calculate_feedback_frequency(
        self, interactions: List[InteractionData]
    ) -> float:
        """Calculate how often user provides feedback."""
        feedback_count = 0
        for interaction in interactions:
            if hasattr(interaction.conversation_context, "user_feedback"):
                if interaction.conversation_context.user_feedback is not None:
                    feedback_count += 1

        return feedback_count / len(interactions) if interactions else 0.0

    def _analyze_help_seeking_behavior(
        self, interactions: List[InteractionData]
    ) -> str:
        """Analyze how user seeks help."""
        # Would analyze question directness and patterns
        return "direct"  # Default

    def _estimate_error_tolerance(self, interactions: List[InteractionData]) -> float:
        """Estimate user's tolerance for errors."""
        # Would analyze reactions to incorrect responses
        return 0.5  # Default medium tolerance

    def _determine_learning_style(self, interactions: List[InteractionData]) -> str:
        """Determine preferred learning style."""
        # Would analyze content preferences
        return "textual"  # Default

    def _extract_social_cues(self, interactions: List[InteractionData]) -> List[str]:
        """Extract social interaction preferences."""
        # Would analyze social language patterns
        return ["polite", "collaborative"]  # Default

    def _estimate_attention_span(self, interactions: List[InteractionData]) -> str:
        """Estimate user's attention span."""
        # Would analyze session lengths and message complexity
        return "medium"  # Default

    def _calculate_overall_confidence(self, pattern_lists: List[List]) -> float:
        """Calculate overall confidence across all patterns."""
        confidences = []
        for pattern_list in pattern_lists:
            for pattern in pattern_list:
                if hasattr(pattern, "confidence"):
                    conf_value = {
                        PatternConfidence.LOW: 0.3,
                        PatternConfidence.MEDIUM: 0.6,
                        PatternConfidence.HIGH: 0.9,
                    }.get(pattern.confidence, 0.3)
                    confidences.append(conf_value)

        return sum(confidences) / len(confidences) if confidences else 0.0

    def _calculate_pattern_stability(
        self, user_id: str, interactions: List[InteractionData]
    ) -> float:
        """Calculate how stable patterns are over time."""
        # Would compare current patterns with historical patterns
        return 0.7  # Default moderate stability

    def _create_empty_patterns(
        self, user_id: str, analysis_period: timedelta
    ) -> InteractionPatterns:
        """Create empty patterns structure for insufficient data."""
        return InteractionPatterns(
            user_id=user_id,
            analysis_period=analysis_period,
            communication_patterns=[],
            topic_patterns=[],
            timing_patterns=[],
            behavioral_patterns=[],
            overall_confidence=0.0,
            pattern_stability=0.0,
        )
