"""
Feedback Processing for Morgan RAG.

Processes user feedback to improve personalization, including explicit ratings,
implicit behavioral signals, and conversation quality indicators.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from ..emotional.models import ConversationContext
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FeedbackType(Enum):
    """Types of user feedback."""

    EXPLICIT_RATING = "explicit_rating"  # Direct 1-5 star ratings
    THUMBS_UP_DOWN = "thumbs_up_down"  # Simple positive/negative
    BEHAVIORAL = "behavioral"  # Implicit behavior signals
    CONVERSATION_QUALITY = "conversation_quality"  # Quality indicators
    PREFERENCE_CORRECTION = "preference_correction"  # Explicit preference updates
    EMOTIONAL_RESPONSE = "emotional_response"  # Emotional reaction to responses


class FeedbackSentiment(Enum):
    """Sentiment of feedback."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


@dataclass
class UserFeedback:
    """Represents user feedback on assistant performance."""

    feedback_id: str
    user_id: str
    feedback_type: FeedbackType
    satisfaction_rating: Optional[float] = None  # 0.0 to 1.0
    sentiment: Optional[FeedbackSentiment] = None
    specific_aspects: Dict[str, float] = field(default_factory=dict)  # aspect -> rating
    feedback_text: Optional[str] = None
    context_id: Optional[str] = None  # Related conversation/response ID
    behavioral_signals: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Initialize feedback ID if not provided."""
        if not self.feedback_id:
            self.feedback_id = str(uuid.uuid4())


@dataclass
class FeedbackAnalysis:
    """Analysis results from feedback processing."""

    feedback_id: str
    user_id: str
    identified_issues: List[str]
    improvement_areas: List[str]
    positive_aspects: List[str]
    confidence_score: float  # 0.0 to 1.0
    actionable_insights: List[str]
    preference_implications: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LearningUpdate:
    """Learning updates derived from feedback."""

    update_id: str
    user_id: str
    preference_updates: List[Dict[str, Any]]
    adaptation_changes: List[Dict[str, Any]]
    confidence_adjustments: Dict[str, float]
    learning_insights: List[str]
    confidence_score: float  # Overall confidence in updates
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Initialize update ID if not provided."""
        if not self.update_id:
            self.update_id = str(uuid.uuid4())


class FeedbackProcessor:
    """
    Processes user feedback to improve personalization.

    Analyzes explicit and implicit feedback to identify improvement areas,
    update user preferences, and adjust behavioral adaptations.
    """

    # Behavioral signal patterns
    POSITIVE_SIGNALS = {
        "engagement": ["long_session", "multiple_questions", "follow_up_questions"],
        "satisfaction": [
            "positive_emotional_response",
            "continued_conversation",
            "return_user",
        ],
        "usefulness": ["copy_response", "share_response", "bookmark_conversation"],
    }

    NEGATIVE_SIGNALS = {
        "dissatisfaction": [
            "short_session",
            "abrupt_end",
            "negative_emotional_response",
        ],
        "confusion": [
            "repeated_questions",
            "clarification_requests",
            "frustration_indicators",
        ],
        "irrelevance": ["topic_change", "ignore_response", "new_query_immediately"],
    }

    # Feedback aspect categories
    FEEDBACK_ASPECTS = {
        "helpfulness": ["helpful", "useful", "valuable", "beneficial"],
        "accuracy": ["correct", "accurate", "right", "precise", "wrong", "incorrect"],
        "clarity": ["clear", "understandable", "confusing", "unclear", "complex"],
        "relevance": ["relevant", "on-topic", "related", "irrelevant", "off-topic"],
        "completeness": [
            "complete",
            "thorough",
            "comprehensive",
            "incomplete",
            "missing",
        ],
        "tone": ["friendly", "professional", "warm", "cold", "rude", "polite"],
    }

    def __init__(self):
        """Initialize feedback processor."""
        self.feedback_history: Dict[str, List[UserFeedback]] = {}
        self.analysis_cache: Dict[str, FeedbackAnalysis] = {}

        logger.info("Feedback processor initialized")

    def process_feedback(
        self, user_id: str, feedback: UserFeedback, context: ConversationContext
    ) -> LearningUpdate:
        """
        Process user feedback to generate learning updates.

        Args:
            user_id: User identifier
            feedback: User feedback to process
            context: Conversation context

        Returns:
            LearningUpdate: Learning updates derived from feedback
        """
        logger.info(f"Processing feedback from user {user_id}")

        # Store feedback in history
        if user_id not in self.feedback_history:
            self.feedback_history[user_id] = []
        self.feedback_history[user_id].append(feedback)

        # Analyze the feedback
        analysis = self._analyze_feedback(feedback, context)
        self.analysis_cache[feedback.feedback_id] = analysis

        # Generate learning updates
        learning_update = self._generate_learning_updates(
            user_id, feedback, analysis, context
        )

        logger.info(
            f"Generated {len(learning_update.preference_updates)} preference updates"
        )
        return learning_update

    def _analyze_feedback(
        self, feedback: UserFeedback, context: ConversationContext
    ) -> FeedbackAnalysis:
        """Analyze feedback to extract insights."""
        logger.debug(f"Analyzing feedback {feedback.feedback_id}")

        identified_issues = []
        improvement_areas = []
        positive_aspects = []
        preference_implications = {}

        # Analyze explicit ratings
        if feedback.satisfaction_rating is not None:
            if feedback.satisfaction_rating < 0.4:
                identified_issues.append("Low satisfaction rating")
                improvement_areas.append("Overall response quality")
            elif feedback.satisfaction_rating > 0.7:
                positive_aspects.append("High satisfaction rating")

        # Analyze specific aspect ratings
        for aspect, rating in feedback.specific_aspects.items():
            if rating < 0.4:
                identified_issues.append(f"Low {aspect} rating")
                improvement_areas.append(aspect)
            elif rating > 0.7:
                positive_aspects.append(f"High {aspect} rating")

        # Analyze feedback text
        if feedback.feedback_text:
            text_insights = self._analyze_feedback_text(feedback.feedback_text)
            identified_issues.extend(text_insights["issues"])
            improvement_areas.extend(text_insights["improvements"])
            positive_aspects.extend(text_insights["positives"])
            preference_implications.update(text_insights["preferences"])

        # Analyze behavioral signals
        if feedback.behavioral_signals:
            behavioral_insights = self._analyze_behavioral_signals(
                feedback.behavioral_signals
            )
            identified_issues.extend(behavioral_insights["issues"])
            improvement_areas.extend(behavioral_insights["improvements"])
            positive_aspects.extend(behavioral_insights["positives"])

        # Generate actionable insights
        actionable_insights = self._generate_actionable_insights(
            identified_issues, improvement_areas, positive_aspects
        )

        # Calculate confidence score
        confidence_score = self._calculate_analysis_confidence(feedback)

        return FeedbackAnalysis(
            feedback_id=feedback.feedback_id,
            user_id=feedback.user_id,
            identified_issues=identified_issues,
            improvement_areas=improvement_areas,
            positive_aspects=positive_aspects,
            confidence_score=confidence_score,
            actionable_insights=actionable_insights,
            preference_implications=preference_implications,
        )

    def _analyze_feedback_text(self, feedback_text: str) -> Dict[str, List[str]]:
        """Analyze textual feedback for insights."""

        text_lower = feedback_text.lower()
        insights = {
            "issues": [],
            "improvements": [],
            "positives": [],
            "preferences": {},
        }

        # Check for aspect-specific feedback
        for aspect, keywords in self.FEEDBACK_ASPECTS.items():
            positive_keywords = [
                k
                for k in keywords
                if k
                not in [
                    "wrong",
                    "incorrect",
                    "confusing",
                    "unclear",
                    "complex",
                    "irrelevant",
                    "off-topic",
                    "incomplete",
                    "missing",
                    "cold",
                    "rude",
                ]
            ]
            negative_keywords = [
                k
                for k in keywords
                if k
                in [
                    "wrong",
                    "incorrect",
                    "confusing",
                    "unclear",
                    "complex",
                    "irrelevant",
                    "off-topic",
                    "incomplete",
                    "missing",
                    "cold",
                    "rude",
                ]
            ]

            # Check for positive mentions
            for keyword in positive_keywords:
                if keyword in text_lower:
                    insights["positives"].append(f"User appreciated {aspect}")
                    break

            # Check for negative mentions
            for keyword in negative_keywords:
                if keyword in text_lower:
                    insights["issues"].append(f"User found issue with {aspect}")
                    insights["improvements"].append(aspect)
                    break

        # Check for preference indicators
        if any(word in text_lower for word in ["prefer", "like", "want", "need"]):
            if "brief" in text_lower or "short" in text_lower:
                insights["preferences"]["response_length"] = "brief"
            elif "detailed" in text_lower or "thorough" in text_lower:
                insights["preferences"]["response_length"] = "detailed"

            if "simple" in text_lower or "easy" in text_lower:
                insights["preferences"]["complexity"] = "simple"
            elif "technical" in text_lower or "detailed" in text_lower:
                insights["preferences"]["complexity"] = "technical"

        return insights

    def _analyze_behavioral_signals(
        self, behavioral_signals: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Analyze behavioral signals for insights."""
        insights = {"issues": [], "improvements": [], "positives": []}

        # Check for positive signals
        for category, signals in self.POSITIVE_SIGNALS.items():
            for signal in signals:
                if signal in behavioral_signals and behavioral_signals[signal]:
                    insights["positives"].append(
                        f"Positive {category} signal: {signal}"
                    )

        # Check for negative signals
        for category, signals in self.NEGATIVE_SIGNALS.items():
            for signal in signals:
                if signal in behavioral_signals and behavioral_signals[signal]:
                    insights["issues"].append(f"Negative {category} signal: {signal}")
                    insights["improvements"].append(category)

        return insights

    def _generate_actionable_insights(
        self, issues: List[str], improvements: List[str], positives: List[str]
    ) -> List[str]:
        """Generate actionable insights from analysis."""
        insights = []

        # Generate insights from issues
        if "accuracy" in improvements:
            insights.append("Improve fact-checking and information verification")
        if "clarity" in improvements:
            insights.append("Use simpler language and better explanations")
        if "relevance" in improvements:
            insights.append("Better understand user intent and context")
        if "completeness" in improvements:
            insights.append("Provide more comprehensive responses")
        if "tone" in improvements:
            insights.append("Adjust communication style to match user preferences")

        # Generate insights from positives
        if any("satisfaction" in p for p in positives):
            insights.append("Continue current approach - user is satisfied")
        if any("helpfulness" in p for p in positives):
            insights.append("Maintain helpful response style")

        return insights

    def _calculate_analysis_confidence(self, feedback: UserFeedback) -> float:
        """Calculate confidence in feedback analysis."""
        confidence_factors = []

        # Explicit ratings provide high confidence
        if feedback.satisfaction_rating is not None:
            confidence_factors.append(0.8)

        # Specific aspect ratings provide medium confidence
        if feedback.specific_aspects:
            confidence_factors.append(0.6)

        # Text feedback provides variable confidence based on length
        if feedback.feedback_text:
            text_length = len(feedback.feedback_text.split())
            if text_length > 10:
                confidence_factors.append(0.7)
            elif text_length > 3:
                confidence_factors.append(0.5)
            else:
                confidence_factors.append(0.3)

        # Behavioral signals provide lower confidence
        if feedback.behavioral_signals:
            confidence_factors.append(0.4)

        if not confidence_factors:
            return 0.1  # Very low confidence with no data

        return sum(confidence_factors) / len(confidence_factors)

    def _generate_learning_updates(
        self,
        user_id: str,
        feedback: UserFeedback,
        analysis: FeedbackAnalysis,
        context: ConversationContext,
    ) -> LearningUpdate:
        """Generate learning updates from feedback analysis."""
        preference_updates = []
        adaptation_changes = []
        confidence_adjustments = {}
        learning_insights = []

        # Generate preference updates from analysis
        for pref_key, pref_value in analysis.preference_implications.items():
            preference_updates.append(
                {
                    "category": (
                        "communication"
                        if pref_key in ["response_length", "complexity"]
                        else "content"
                    ),
                    "key": pref_key,
                    "value": pref_value,
                    "confidence": analysis.confidence_score,
                    "source": "explicit_feedback",
                }
            )

        # Generate adaptation changes based on issues
        if "clarity" in analysis.improvement_areas:
            adaptation_changes.append(
                {
                    "type": "response_style",
                    "parameter": "simplify_language",
                    "adjustment": 0.2,
                    "reason": "User feedback indicates clarity issues",
                }
            )

        if "relevance" in analysis.improvement_areas:
            adaptation_changes.append(
                {
                    "type": "content_selection",
                    "parameter": "context_weighting",
                    "adjustment": 0.3,
                    "reason": "User feedback indicates relevance issues",
                }
            )

        # Adjust confidence based on feedback sentiment
        if feedback.satisfaction_rating is not None:
            if feedback.satisfaction_rating < 0.4:
                # Lower confidence in current adaptations
                confidence_adjustments["overall"] = -0.1
            elif feedback.satisfaction_rating > 0.7:
                # Increase confidence in current adaptations
                confidence_adjustments["overall"] = 0.1

        # Generate learning insights
        learning_insights.extend(analysis.actionable_insights)

        if analysis.positive_aspects:
            learning_insights.append(
                f"User appreciated: {', '.join(analysis.positive_aspects[:3])}"
            )

        if analysis.identified_issues:
            learning_insights.append(
                f"Areas for improvement: {', '.join(analysis.improvement_areas[:3])}"
            )

        # Calculate overall confidence in updates
        update_confidence = min(analysis.confidence_score, 0.9)

        return LearningUpdate(
            update_id=str(uuid.uuid4()),
            user_id=user_id,
            preference_updates=preference_updates,
            adaptation_changes=adaptation_changes,
            confidence_adjustments=confidence_adjustments,
            learning_insights=learning_insights,
            confidence_score=update_confidence,
        )

    def get_feedback_history(self, user_id: str) -> List[UserFeedback]:
        """
        Get feedback history for a user.

        Args:
            user_id: User identifier

        Returns:
            List[UserFeedback]: User's feedback history
        """
        return self.feedback_history.get(user_id, [])

    def get_feedback_analysis(self, feedback_id: str) -> Optional[FeedbackAnalysis]:
        """
        Get analysis for specific feedback.

        Args:
            feedback_id: Feedback identifier

        Returns:
            Optional[FeedbackAnalysis]: Feedback analysis if available
        """
        return self.analysis_cache.get(feedback_id)

    def get_user_satisfaction_trend(
        self, user_id: str, days: int = 30
    ) -> Dict[str, float]:
        """
        Get user satisfaction trend over time.

        Args:
            user_id: User identifier
            days: Number of days to analyze

        Returns:
            Dict[str, float]: Satisfaction metrics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_feedback = [
            f
            for f in self.feedback_history.get(user_id, [])
            if f.timestamp >= cutoff_date and f.satisfaction_rating is not None
        ]

        if not recent_feedback:
            return {"average": 0.0, "trend": 0.0, "count": 0}

        ratings = [f.satisfaction_rating for f in recent_feedback]
        average_rating = sum(ratings) / len(ratings)

        # Calculate trend (simple linear regression slope)
        if len(ratings) > 1:
            n = len(ratings)
            x_values = list(range(n))
            x_mean = sum(x_values) / n
            y_mean = average_rating

            numerator = sum(
                (x - x_mean) * (y - y_mean) for x, y in zip(x_values, ratings)
            )
            denominator = sum((x - x_mean) ** 2 for x in x_values)

            trend = numerator / denominator if denominator != 0 else 0.0
        else:
            trend = 0.0

        return {
            "average": average_rating,
            "trend": trend,
            "count": len(recent_feedback),
        }
