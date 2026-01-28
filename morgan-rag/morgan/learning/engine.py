"""
Main Learning Engine for Morgan RAG.

Coordinates all learning activities including pattern analysis, preference extraction,
behavioral adaptation, and feedback processing to continuously improve personalization.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ..intelligence.core.models import CompanionProfile, ConversationContext, InteractionData
from ..utils.logger import get_logger
from .adaptation import AdaptationResult, BehavioralAdaptationEngine
from .feedback import FeedbackProcessor, LearningUpdate, UserFeedback
from .patterns import InteractionPatternAnalyzer, InteractionPatterns
from .preferences import PreferenceExtractor, PreferenceStorage, UserPreferenceProfile

logger = get_logger(__name__)


@dataclass
class LearningSession:
    """Represents a learning session for a user."""

    session_id: str
    user_id: str
    start_time: datetime
    interactions_processed: int = 0
    patterns_identified: int = 0
    preferences_updated: int = 0
    adaptations_applied: int = 0
    feedback_processed: int = 0
    learning_score: float = 0.0  # Overall learning effectiveness

    def __post_init__(self):
        """Initialize session ID if not provided."""
        if not self.session_id:
            self.session_id = str(uuid.uuid4())


@dataclass
class LearningMetrics:
    """Metrics for learning effectiveness."""

    total_interactions: int
    successful_adaptations: int
    user_satisfaction_trend: float  # -1.0 to 1.0
    preference_stability: float  # 0.0 to 1.0
    learning_velocity: float  # rate of improvement
    personalization_accuracy: float  # 0.0 to 1.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class LearningEngine:
    """
    Main Learning Engine for continuous personalization.

    Coordinates all learning activities:
    - Analyzes interaction patterns
    - Extracts and updates user preferences
    - Adapts behavior based on learning
    - Processes feedback for improvement

    Requirements addressed: 24.1, 24.2, 24.3, 24.4, 24.5
    """

    def __init__(self):
        """Initialize the learning engine with all components."""
        self.pattern_analyzer = InteractionPatternAnalyzer()
        self.preference_extractor = PreferenceExtractor()
        self.preference_storage = PreferenceStorage()
        self.adaptation_engine = BehavioralAdaptationEngine()
        self.feedback_processor = FeedbackProcessor()

        # Learning state
        self.active_sessions: Dict[str, LearningSession] = {}
        self.user_metrics: Dict[str, LearningMetrics] = {}

        logger.info("Learning engine initialized")

    def analyze_interaction_patterns(
        self, user_id: str, interactions: List[InteractionData]
    ) -> InteractionPatterns:
        """
        Analyze user interaction patterns for learning insights.

        Requirement 24.1: Analyze communication patterns and preferences

        Args:
            user_id: User identifier
            interactions: List of interaction data to analyze

        Returns:
            InteractionPatterns: Identified patterns and insights
        """
        logger.info(f"Analyzing interaction patterns for user {user_id}")

        try:
            # Use pattern analyzer to identify patterns
            patterns = self.pattern_analyzer.analyze_patterns(
                user_id=user_id, interactions=interactions
            )

            # Update learning metrics
            self._update_learning_metrics(
                user_id=user_id,
                patterns_identified=len(patterns.communication_patterns),
            )

            logger.info(
                f"Identified {len(patterns.communication_patterns)} communication patterns "
                f"and {len(patterns.topic_patterns)} topic patterns for user {user_id}"
            )

            return patterns

        except Exception as e:
            logger.error(f"Error analyzing patterns for user {user_id}: {e}")
            # Return empty patterns on error
            return InteractionPatterns(
                user_id=user_id,
                analysis_period=timedelta(days=30),
                communication_patterns=[],
                topic_patterns=[],
                timing_patterns=[],
                behavioral_patterns=[],
            )

    def learn_from_feedback(
        self, user_id: str, feedback: UserFeedback, context: ConversationContext
    ) -> LearningUpdate:
        """
        Process user feedback to improve personalization.

        Requirement 24.2: Adjust response styles based on feedback

        Args:
            user_id: User identifier
            feedback: User feedback data
            context: Conversation context

        Returns:
            LearningUpdate: Learning updates applied
        """
        logger.info(f"Processing feedback from user {user_id}")

        try:
            # Process feedback through feedback processor
            learning_update = self.feedback_processor.process_feedback(
                user_id=user_id, feedback=feedback, context=context
            )

            # Apply learning updates to user preferences
            if learning_update.preference_updates:
                self._apply_preference_updates(
                    user_id, learning_update.preference_updates
                )

            # Update adaptation strategies if needed
            if learning_update.adaptation_changes:
                self.adaptation_engine.update_strategies(
                    user_id=user_id, changes=learning_update.adaptation_changes
                )

            # Update learning metrics
            self._update_learning_metrics(
                user_id=user_id,
                feedback_processed=1,
                user_satisfaction=feedback.satisfaction_rating,
            )

            logger.info(
                f"Applied {len(learning_update.preference_updates)} preference updates"
            )

            return learning_update

        except Exception as e:
            logger.error(f"Error processing feedback for user {user_id}: {e}")
            return LearningUpdate(
                update_id=str(uuid.uuid4()),
                user_id=user_id,
                timestamp=datetime.now(timezone.utc),
                preference_updates=[],
                adaptation_changes=[],
                confidence_score=0.0,
            )

    def adapt_behavior(
        self,
        user_id: str,
        current_context: ConversationContext,
        user_profile: CompanionProfile,
    ) -> AdaptationResult:
        """
        Adapt behavior based on learned patterns and preferences.

        Requirement 24.3: Customize search weights and result ranking
        Requirement 24.4: Expand domain-specific vocabulary understanding

        Args:
            user_id: User identifier
            current_context: Current conversation context
            user_profile: User's companion profile

        Returns:
            AdaptationResult: Behavioral adaptations applied
        """
        logger.info(f"Adapting behavior for user {user_id}")

        try:
            # Get user preference profile
            preference_profile = self.preference_storage.get_user_preferences(user_id)

            # Apply behavioral adaptations
            adaptation_result = self.adaptation_engine.adapt_behavior(
                user_id=user_id,
                context=current_context,
                user_profile=user_profile,
                preference_profile=preference_profile,
            )

            # Update learning metrics
            self._update_learning_metrics(user_id=user_id, adaptations_applied=1)

            logger.info(
                f"Applied {len(adaptation_result.adaptations)} behavioral adaptations "
                f"with confidence {adaptation_result.confidence_score:.2f}"
            )

            return adaptation_result

        except Exception as e:
            logger.error(f"Error adapting behavior for user {user_id}: {e}")
            return AdaptationResult(
                user_id=user_id,
                adaptations=[],
                confidence_score=0.0,
                reasoning="Error occurred during adaptation",
            )

    def extract_preferences(
        self, user_id: str, interactions: List[InteractionData]
    ) -> UserPreferenceProfile:
        """
        Extract and update user preferences from interactions.

        Requirement 24.5: Reference past preferences and successful patterns

        Args:
            user_id: User identifier
            interactions: Recent interactions to analyze

        Returns:
            UserPreferenceProfile: Updated preference profile
        """
        logger.info(f"Extracting preferences for user {user_id}")

        try:
            # Extract preferences from interactions
            preference_updates = self.preference_extractor.extract_preferences(
                user_id=user_id, interactions=interactions
            )

            # Store updated preferences
            for update in preference_updates:
                self.preference_storage.update_preference(user_id, update)

            # Get complete preference profile
            preference_profile = self.preference_storage.get_user_preferences(user_id)

            # Update learning metrics
            self._update_learning_metrics(
                user_id=user_id, preferences_updated=len(preference_updates)
            )

            logger.info(
                f"Updated {len(preference_updates)} preferences for user {user_id}"
            )

            return preference_profile

        except Exception as e:
            logger.error(f"Error extracting preferences for user {user_id}: {e}")
            # Return empty profile on error
            return UserPreferenceProfile(
                user_id=user_id,
                preferences={},
                confidence_scores={},
                last_updated=datetime.now(timezone.utc),
            )

    def start_learning_session(self, user_id: str) -> str:
        """
        Start a new learning session for a user.

        Args:
            user_id: User identifier

        Returns:
            str: Session ID
        """
        session = LearningSession(
            session_id=str(uuid.uuid4()), user_id=user_id, start_time=datetime.now(timezone.utc)
        )

        self.active_sessions[session.session_id] = session
        logger.info(f"Started learning session {session.session_id} for user {user_id}")

        return session.session_id

    def end_learning_session(self, session_id: str) -> Optional[LearningSession]:
        """
        End a learning session and return results.

        Args:
            session_id: Session identifier

        Returns:
            Optional[LearningSession]: Completed session data
        """
        session = self.active_sessions.pop(session_id, None)
        if session:
            # Calculate learning score based on session metrics
            session.learning_score = self._calculate_learning_score(session)
            logger.info(
                f"Ended learning session {session_id} with score {session.learning_score:.2f}"
            )

        return session

    def get_learning_metrics(self, user_id: str) -> Optional[LearningMetrics]:
        """
        Get learning metrics for a user.

        Args:
            user_id: User identifier

        Returns:
            Optional[LearningMetrics]: User's learning metrics
        """
        return self.user_metrics.get(user_id)

    def _update_learning_metrics(
        self,
        user_id: str,
        interactions_processed: int = 0,
        patterns_identified: int = 0,
        preferences_updated: int = 0,
        adaptations_applied: int = 0,
        feedback_processed: int = 0,
        user_satisfaction: Optional[float] = None,
    ):
        """Update learning metrics for a user."""
        if user_id not in self.user_metrics:
            self.user_metrics[user_id] = LearningMetrics(
                total_interactions=0,
                successful_adaptations=0,
                user_satisfaction_trend=0.0,
                preference_stability=0.5,
                learning_velocity=0.0,
                personalization_accuracy=0.0,
            )

        metrics = self.user_metrics[user_id]
        metrics.total_interactions += interactions_processed
        metrics.successful_adaptations += adaptations_applied

        # Update satisfaction trend if provided
        if user_satisfaction is not None:
            # Simple moving average for satisfaction trend
            alpha = 0.1  # Learning rate
            metrics.user_satisfaction_trend = (
                alpha * user_satisfaction
                + (1 - alpha) * metrics.user_satisfaction_trend
            )

        metrics.last_updated = datetime.now(timezone.utc)

    def _apply_preference_updates(
        self, user_id: str, preference_updates: List[Dict[str, Any]]
    ):
        """Apply preference updates to user profile."""
        for update in preference_updates:
            # Convert dict to preference update object if needed
            # This would depend on the specific structure
            pass

    def _calculate_learning_score(self, session: LearningSession) -> float:
        """Calculate overall learning effectiveness score for a session."""
        if session.interactions_processed == 0:
            return 0.0

        # Weighted score based on different learning activities
        pattern_score = min(
            session.patterns_identified / session.interactions_processed, 1.0
        )
        preference_score = min(
            session.preferences_updated / session.interactions_processed, 1.0
        )
        adaptation_score = min(
            session.adaptations_applied / session.interactions_processed, 1.0
        )
        feedback_score = min(
            session.feedback_processed / session.interactions_processed, 1.0
        )

        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.2]  # pattern, preference, adaptation, feedback
        scores = [pattern_score, preference_score, adaptation_score, feedback_score]

        return sum(w * s for w, s in zip(weights, scores))


# Global learning engine instance
_learning_engine: Optional[LearningEngine] = None


def get_learning_engine() -> LearningEngine:
    """
    Get the global learning engine instance.

    Returns:
        LearningEngine: Global learning engine instance
    """
    global _learning_engine
    if _learning_engine is None:
        _learning_engine = LearningEngine()
    return _learning_engine
