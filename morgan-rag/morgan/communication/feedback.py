"""
Emotional feedback processing module.

Processes user feedback to improve emotional intelligence and communication
effectiveness through sentiment analysis, satisfaction tracking, and
adaptive response refinement.
"""

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List

from morgan.config import get_settings
from morgan.emotional.models import ConversationContext, EmotionalState
from morgan.services.llm_service import get_llm_service
from morgan.utils.cache import FileCache
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class FeedbackType(Enum):
    """Types of feedback that can be processed."""

    EXPLICIT_RATING = "explicit_rating"
    IMPLICIT_BEHAVIORAL = "implicit_behavioral"
    TEXTUAL_FEEDBACK = "textual_feedback"
    EMOTIONAL_RESPONSE = "emotional_response"
    ENGAGEMENT_METRICS = "engagement_metrics"


class FeedbackSentiment(Enum):
    """Sentiment of feedback."""

    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


@dataclass
class FeedbackAnalysis:
    """Analysis result of user feedback."""

    feedback_type: FeedbackType
    sentiment: FeedbackSentiment
    satisfaction_score: float  # 0.0 to 1.0
    specific_aspects: Dict[str, float]  # Aspect -> score mapping
    improvement_suggestions: List[str]
    confidence_score: float
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FeedbackPattern:
    """Pattern detected in user feedback over time."""

    pattern_type: str
    description: str
    frequency: int
    trend: str  # improving, declining, stable
    confidence: float
    examples: List[str]


class EmotionalFeedbackProcessor:
    """
    Emotional feedback processing system.

    Features:
    - Multi-modal feedback analysis (explicit, implicit, textual)
    - Sentiment analysis and satisfaction scoring
    - Feedback pattern recognition and trend analysis
    - Adaptive response improvement based on feedback
    - Emotional intelligence refinement through feedback loops
    """

    def __init__(self):
        """Initialize emotional feedback processor."""
        self.settings = get_settings()
        self.llm_service = get_llm_service()

        # Setup cache for feedback data
        cache_dir = self.settings.morgan_data_dir / "cache" / "feedback"
        self.cache = FileCache(cache_dir)

        # Feedback storage
        self.user_feedback_history: Dict[str, List[FeedbackAnalysis]] = defaultdict(
            list
        )
        self.feedback_patterns: Dict[str, List[FeedbackPattern]] = defaultdict(list)

        logger.info("Emotional Feedback Processor initialized")

    def process_feedback(
        self,
        user_id: str,
        feedback_data: Dict[str, Any],
        context: ConversationContext,
        emotional_state: EmotionalState,
    ) -> FeedbackAnalysis:
        """
        Process user feedback and extract insights.

        Args:
            user_id: User identifier
            feedback_data: Raw feedback data
            context: Conversation context
            emotional_state: User's emotional state

        Returns:
            Feedback analysis result
        """
        # Determine feedback type
        feedback_type = self._determine_feedback_type(feedback_data)

        # Analyze sentiment
        sentiment = self._analyze_sentiment(feedback_data, feedback_type)

        # Calculate satisfaction score
        satisfaction_score = self._calculate_satisfaction_score(
            feedback_data, sentiment, feedback_type
        )

        # Analyze specific aspects
        specific_aspects = self._analyze_specific_aspects(
            feedback_data, context, emotional_state
        )

        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(
            feedback_data, specific_aspects, context
        )

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            feedback_data, feedback_type, emotional_state
        )

        # Create analysis result
        analysis = FeedbackAnalysis(
            feedback_type=feedback_type,
            sentiment=sentiment,
            satisfaction_score=satisfaction_score,
            specific_aspects=specific_aspects,
            improvement_suggestions=improvement_suggestions,
            confidence_score=confidence_score,
        )

        # Store feedback
        self._store_feedback_analysis(user_id, analysis)

        # Update patterns
        self._update_feedback_patterns(user_id, analysis)

        logger.debug(
            f"Processed feedback for user {user_id}: "
            f"sentiment={sentiment.value}, satisfaction={satisfaction_score:.2f}"
        )

        return analysis

    def analyze_feedback_trends(
        self, user_id: str, timeframe_days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze feedback trends over time.

        Args:
            user_id: User identifier
            timeframe_days: Number of days to analyze

        Returns:
            Trend analysis results
        """
        if user_id not in self.user_feedback_history:
            return {"error": "No feedback history available"}

        cutoff_date = datetime.utcnow() - timedelta(days=timeframe_days)
        recent_feedback = [
            analysis
            for analysis in self.user_feedback_history[user_id]
            if analysis.analysis_timestamp >= cutoff_date
        ]

        if not recent_feedback:
            return {"error": "No recent feedback available"}

        # Analyze satisfaction trends
        satisfaction_trend = self._analyze_satisfaction_trend(recent_feedback)

        # Analyze sentiment distribution
        sentiment_distribution = self._analyze_sentiment_distribution(recent_feedback)

        # Analyze improvement areas
        improvement_areas = self._identify_improvement_areas(recent_feedback)

        # Analyze feedback patterns
        patterns = self._analyze_feedback_patterns(user_id, recent_feedback)

        return {
            "timeframe_days": timeframe_days,
            "feedback_count": len(recent_feedback),
            "satisfaction_trend": satisfaction_trend,
            "sentiment_distribution": sentiment_distribution,
            "improvement_areas": improvement_areas,
            "patterns": patterns,
        }

    def get_improvement_recommendations(
        self, user_id: str, context: ConversationContext
    ) -> List[Dict[str, Any]]:
        """
        Get improvement recommendations based on feedback history.

        Args:
            user_id: User identifier
            context: Current conversation context

        Returns:
            List of improvement recommendations
        """
        if user_id not in self.user_feedback_history:
            return []

        recent_feedback = self.user_feedback_history[user_id][
            -10:
        ]  # Last 10 feedback items

        recommendations = []

        # Analyze common improvement suggestions
        all_suggestions = []
        for analysis in recent_feedback:
            all_suggestions.extend(analysis.improvement_suggestions)

        # Count suggestion frequency
        suggestion_counts = {}
        for suggestion in all_suggestions:
            suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1

        # Create recommendations based on frequent suggestions
        for suggestion, count in suggestion_counts.items():
            if count >= 2:  # Appears in at least 2 feedback items
                recommendations.append(
                    {
                        "recommendation": suggestion,
                        "frequency": count,
                        "priority": "high" if count >= 3 else "medium",
                        "confidence": min(1.0, count / len(recent_feedback)),
                    }
                )

        # Sort by frequency
        recommendations.sort(key=lambda x: x["frequency"], reverse=True)

        return recommendations[:5]  # Top 5 recommendations

    def _determine_feedback_type(self, feedback_data: Dict[str, Any]) -> FeedbackType:
        """Determine the type of feedback."""
        if "rating" in feedback_data or "score" in feedback_data:
            return FeedbackType.EXPLICIT_RATING
        elif "text" in feedback_data or "message" in feedback_data:
            return FeedbackType.TEXTUAL_FEEDBACK
        elif "emotion" in feedback_data:
            return FeedbackType.EMOTIONAL_RESPONSE
        elif "engagement_time" in feedback_data or "interaction_count" in feedback_data:
            return FeedbackType.ENGAGEMENT_METRICS
        else:
            return FeedbackType.IMPLICIT_BEHAVIORAL

    def _analyze_sentiment(
        self, feedback_data: Dict[str, Any], feedback_type: FeedbackType
    ) -> FeedbackSentiment:
        """Analyze sentiment of feedback."""
        if feedback_type == FeedbackType.EXPLICIT_RATING:
            rating = feedback_data.get("rating", feedback_data.get("score", 0.5))
            if rating >= 0.9:
                return FeedbackSentiment.VERY_POSITIVE
            elif rating >= 0.7:
                return FeedbackSentiment.POSITIVE
            elif rating >= 0.4:
                return FeedbackSentiment.NEUTRAL
            elif rating >= 0.2:
                return FeedbackSentiment.NEGATIVE
            else:
                return FeedbackSentiment.VERY_NEGATIVE

        elif feedback_type == FeedbackType.TEXTUAL_FEEDBACK:
            text = feedback_data.get("text", feedback_data.get("message", ""))
            return self._analyze_text_sentiment(text)

        elif feedback_type == FeedbackType.EMOTIONAL_RESPONSE:
            emotion = feedback_data.get("emotion", "neutral")
            emotion_sentiment_map = {
                "joy": FeedbackSentiment.VERY_POSITIVE,
                "happiness": FeedbackSentiment.POSITIVE,
                "satisfaction": FeedbackSentiment.POSITIVE,
                "neutral": FeedbackSentiment.NEUTRAL,
                "disappointment": FeedbackSentiment.NEGATIVE,
                "frustration": FeedbackSentiment.NEGATIVE,
                "anger": FeedbackSentiment.VERY_NEGATIVE,
            }
            return emotion_sentiment_map.get(emotion, FeedbackSentiment.NEUTRAL)

        else:
            return FeedbackSentiment.NEUTRAL

    def _analyze_text_sentiment(self, text: str) -> FeedbackSentiment:
        """Analyze sentiment of text feedback."""
        text_lower = text.lower()

        # Simple keyword-based sentiment analysis
        positive_keywords = [
            "good",
            "great",
            "excellent",
            "amazing",
            "helpful",
            "useful",
            "love",
            "like",
            "appreciate",
            "thank",
            "perfect",
            "wonderful",
        ]

        negative_keywords = [
            "bad",
            "terrible",
            "awful",
            "useless",
            "hate",
            "dislike",
            "wrong",
            "error",
            "problem",
            "issue",
            "disappointed",
            "frustrated",
        ]

        positive_count = sum(
            1 for keyword in positive_keywords if keyword in text_lower
        )
        negative_count = sum(
            1 for keyword in negative_keywords if keyword in text_lower
        )

        if positive_count > negative_count + 1:
            return (
                FeedbackSentiment.POSITIVE
                if positive_count <= 2
                else FeedbackSentiment.VERY_POSITIVE
            )
        elif negative_count > positive_count + 1:
            return (
                FeedbackSentiment.NEGATIVE
                if negative_count <= 2
                else FeedbackSentiment.VERY_NEGATIVE
            )
        else:
            return FeedbackSentiment.NEUTRAL

    def _calculate_satisfaction_score(
        self,
        feedback_data: Dict[str, Any],
        sentiment: FeedbackSentiment,
        feedback_type: FeedbackType,
    ) -> float:
        """Calculate satisfaction score from feedback."""
        if feedback_type == FeedbackType.EXPLICIT_RATING:
            return float(feedback_data.get("rating", feedback_data.get("score", 0.5)))

        # Convert sentiment to satisfaction score
        sentiment_scores = {
            FeedbackSentiment.VERY_POSITIVE: 0.95,
            FeedbackSentiment.POSITIVE: 0.75,
            FeedbackSentiment.NEUTRAL: 0.5,
            FeedbackSentiment.NEGATIVE: 0.25,
            FeedbackSentiment.VERY_NEGATIVE: 0.05,
        }

        return sentiment_scores.get(sentiment, 0.5)

    def _analyze_specific_aspects(
        self,
        feedback_data: Dict[str, Any],
        context: ConversationContext,
        emotional_state: EmotionalState,
    ) -> Dict[str, float]:
        """Analyze specific aspects of the interaction."""
        aspects = {}

        # Analyze helpfulness
        if "helpful" in str(feedback_data).lower():
            aspects["helpfulness"] = 0.8
        elif "not helpful" in str(feedback_data).lower():
            aspects["helpfulness"] = 0.2
        else:
            aspects["helpfulness"] = 0.6

        # Analyze emotional appropriateness
        if emotional_state.intensity > 0.7:
            # High emotional intensity - check if response was appropriate
            if (
                "supportive" in str(feedback_data).lower()
                or "understanding" in str(feedback_data).lower()
            ):
                aspects["emotional_appropriateness"] = 0.9
            elif (
                "cold" in str(feedback_data).lower()
                or "insensitive" in str(feedback_data).lower()
            ):
                aspects["emotional_appropriateness"] = 0.2
            else:
                aspects["emotional_appropriateness"] = 0.6
        else:
            aspects["emotional_appropriateness"] = 0.7

        # Analyze response relevance
        if "relevant" in str(feedback_data).lower():
            aspects["relevance"] = 0.8
        elif (
            "off-topic" in str(feedback_data).lower()
            or "irrelevant" in str(feedback_data).lower()
        ):
            aspects["relevance"] = 0.2
        else:
            aspects["relevance"] = 0.6

        # Analyze communication clarity
        if (
            "clear" in str(feedback_data).lower()
            or "understand" in str(feedback_data).lower()
        ):
            aspects["clarity"] = 0.8
        elif (
            "confusing" in str(feedback_data).lower()
            or "unclear" in str(feedback_data).lower()
        ):
            aspects["clarity"] = 0.2
        else:
            aspects["clarity"] = 0.6

        return aspects

    def _generate_improvement_suggestions(
        self,
        feedback_data: Dict[str, Any],
        specific_aspects: Dict[str, float],
        context: ConversationContext,
    ) -> List[str]:
        """Generate improvement suggestions based on feedback."""
        suggestions = []

        # Check low-scoring aspects
        for aspect, score in specific_aspects.items():
            if score < 0.5:
                if aspect == "helpfulness":
                    suggestions.append(
                        "Provide more actionable and specific information"
                    )
                elif aspect == "emotional_appropriateness":
                    suggestions.append(
                        "Improve emotional sensitivity and empathy in responses"
                    )
                elif aspect == "relevance":
                    suggestions.append(
                        "Focus more closely on the user's specific question or concern"
                    )
                elif aspect == "clarity":
                    suggestions.append(
                        "Use clearer language and better structure in responses"
                    )

        # Analyze textual feedback for specific suggestions
        feedback_text = str(feedback_data).lower()

        if "too long" in feedback_text:
            suggestions.append("Provide more concise responses")
        elif "too short" in feedback_text:
            suggestions.append("Provide more detailed explanations")

        if "too formal" in feedback_text:
            suggestions.append("Use more casual and friendly language")
        elif "too casual" in feedback_text:
            suggestions.append("Use more professional language")

        if "more examples" in feedback_text:
            suggestions.append("Include more practical examples and illustrations")

        return suggestions

    def _calculate_confidence_score(
        self,
        feedback_data: Dict[str, Any],
        feedback_type: FeedbackType,
        emotional_state: EmotionalState,
    ) -> float:
        """Calculate confidence in feedback analysis."""
        confidence_factors = []

        # Feedback type confidence
        type_confidence = {
            FeedbackType.EXPLICIT_RATING: 0.9,
            FeedbackType.TEXTUAL_FEEDBACK: 0.8,
            FeedbackType.EMOTIONAL_RESPONSE: 0.7,
            FeedbackType.ENGAGEMENT_METRICS: 0.6,
            FeedbackType.IMPLICIT_BEHAVIORAL: 0.5,
        }
        confidence_factors.append(type_confidence.get(feedback_type, 0.5))

        # Emotional state confidence
        confidence_factors.append(emotional_state.confidence)

        # Feedback richness
        feedback_richness = len(str(feedback_data)) / 100.0  # Normalize by length
        confidence_factors.append(min(1.0, feedback_richness))

        return sum(confidence_factors) / len(confidence_factors)

    def _store_feedback_analysis(
        self, user_id: str, analysis: FeedbackAnalysis
    ) -> None:
        """Store feedback analysis."""
        self.user_feedback_history[user_id].append(analysis)

        # Keep only recent feedback (last 100 items)
        if len(self.user_feedback_history[user_id]) > 100:
            self.user_feedback_history[user_id] = self.user_feedback_history[user_id][
                -100:
            ]

        # Cache feedback
        self._cache_feedback_analysis(user_id, analysis)

    def _cache_feedback_analysis(
        self, user_id: str, analysis: FeedbackAnalysis
    ) -> None:
        """Cache feedback analysis."""
        cache_key = f"feedback_{user_id}_{analysis.analysis_timestamp.isoformat()}"
        analysis_dict = {
            "feedback_type": analysis.feedback_type.value,
            "sentiment": analysis.sentiment.value,
            "satisfaction_score": analysis.satisfaction_score,
            "specific_aspects": analysis.specific_aspects,
            "improvement_suggestions": analysis.improvement_suggestions,
            "confidence_score": analysis.confidence_score,
            "analysis_timestamp": analysis.analysis_timestamp.isoformat(),
        }
        self.cache.set(cache_key, analysis_dict)

    def _update_feedback_patterns(
        self, user_id: str, analysis: FeedbackAnalysis
    ) -> None:
        """Update feedback patterns for user."""
        # Simple pattern detection - in a real implementation,
        # this would use more sophisticated pattern recognition

        # Check for satisfaction pattern
        recent_analyses = self.user_feedback_history[user_id][
            -5:
        ]  # Last 5 feedback items
        if len(recent_analyses) >= 3:
            avg_satisfaction = sum(a.satisfaction_score for a in recent_analyses) / len(
                recent_analyses
            )

            if avg_satisfaction > 0.8:
                pattern = FeedbackPattern(
                    pattern_type="high_satisfaction",
                    description="User consistently provides positive feedback",
                    frequency=len(
                        [a for a in recent_analyses if a.satisfaction_score > 0.7]
                    ),
                    trend="stable",
                    confidence=0.8,
                    examples=[],
                )
                self.feedback_patterns[user_id] = [pattern]

    def _analyze_satisfaction_trend(
        self, feedback_list: List[FeedbackAnalysis]
    ) -> Dict[str, Any]:
        """Analyze satisfaction trend over time."""
        if len(feedback_list) < 2:
            return {"trend": "insufficient_data"}

        scores = [analysis.satisfaction_score for analysis in feedback_list]

        # Simple trend analysis
        first_half_avg = sum(scores[: len(scores) // 2]) / (len(scores) // 2)
        second_half_avg = sum(scores[len(scores) // 2 :]) / (
            len(scores) - len(scores) // 2
        )

        if second_half_avg > first_half_avg + 0.1:
            trend = "improving"
        elif second_half_avg < first_half_avg - 0.1:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "current_average": second_half_avg,
            "previous_average": first_half_avg,
            "overall_average": sum(scores) / len(scores),
        }

    def _analyze_sentiment_distribution(
        self, feedback_list: List[FeedbackAnalysis]
    ) -> Dict[str, Any]:
        """Analyze distribution of sentiments."""
        sentiment_counts = {}
        for analysis in feedback_list:
            sentiment = analysis.sentiment.value
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

        total_count = len(feedback_list)
        sentiment_percentages = {
            sentiment: (count / total_count) * 100
            for sentiment, count in sentiment_counts.items()
        }

        return {
            "counts": sentiment_counts,
            "percentages": sentiment_percentages,
            "most_common": (
                max(sentiment_counts.items(), key=lambda x: x[1])[0]
                if sentiment_counts
                else "neutral"
            ),
        }

    def _identify_improvement_areas(
        self, feedback_list: List[FeedbackAnalysis]
    ) -> List[Dict[str, Any]]:
        """Identify areas that need improvement."""
        # Aggregate specific aspects
        aspect_scores = defaultdict(list)
        for analysis in feedback_list:
            for aspect, score in analysis.specific_aspects.items():
                aspect_scores[aspect].append(score)

        # Calculate average scores and identify low-performing areas
        improvement_areas = []
        for aspect, scores in aspect_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score < 0.6:  # Below satisfactory threshold
                improvement_areas.append(
                    {
                        "aspect": aspect,
                        "average_score": avg_score,
                        "priority": "high" if avg_score < 0.4 else "medium",
                        "feedback_count": len(scores),
                    }
                )

        # Sort by priority and score
        improvement_areas.sort(key=lambda x: x["average_score"])

        return improvement_areas

    def _analyze_feedback_patterns(
        self, user_id: str, feedback_list: List[FeedbackAnalysis]
    ) -> List[Dict[str, Any]]:
        """Analyze patterns in feedback."""
        patterns = []

        # Check for consistent improvement suggestions
        all_suggestions = []
        for analysis in feedback_list:
            all_suggestions.extend(analysis.improvement_suggestions)

        suggestion_counts = {}
        for suggestion in all_suggestions:
            suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1

        for suggestion, count in suggestion_counts.items():
            if count >= 2:  # Appears multiple times
                patterns.append(
                    {
                        "pattern_type": "recurring_suggestion",
                        "description": suggestion,
                        "frequency": count,
                        "confidence": min(1.0, count / len(feedback_list)),
                    }
                )

        return patterns


# Singleton instance
_emotional_feedback_processor_instance = None
_emotional_feedback_processor_lock = threading.Lock()


def get_emotional_feedback_processor() -> EmotionalFeedbackProcessor:
    """
    Get singleton emotional feedback processor instance.

    Returns:
        Shared EmotionalFeedbackProcessor instance
    """
    global _emotional_feedback_processor_instance

    if _emotional_feedback_processor_instance is None:
        with _emotional_feedback_processor_lock:
            if _emotional_feedback_processor_instance is None:
                _emotional_feedback_processor_instance = EmotionalFeedbackProcessor()

    return _emotional_feedback_processor_instance
