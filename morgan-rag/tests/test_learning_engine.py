"""
Tests for main learning engine module.

Tests pattern analysis, feedback learning, behavior adaptation,
preference extraction, and learning metrics tracking.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

from morgan.learning.engine import LearningEngine, LearningSession, LearningMetrics
from morgan.learning.feedback import UserFeedback, FeedbackType, FeedbackSentiment
from morgan.intelligence.core.models import (
    InteractionData,
    ConversationContext,
    CompanionProfile,
    EmotionalState,
    EmotionType,
    UserPreferences,
)


class TestLearningEngine:
    """Test learning engine functionality."""

    @pytest.fixture
    def engine(self):
        """Create learning engine."""
        return LearningEngine()

    @pytest.fixture
    def sample_interactions(self):
        """Create sample interactions for testing."""
        interactions = []
        for i in range(10):
            interaction = InteractionData(
                conversation_context=ConversationContext(
                    user_id="test_user",
                    conversation_id="test_conv",
                    message_text=f"Test message {i}",
                    timestamp=datetime.now(timezone.utc) - timedelta(days=i),
                ),
                emotional_state=EmotionalState(
                    primary_emotion=EmotionType.NEUTRAL,
                    intensity=0.5,
                    confidence=0.7,
                    timestamp=datetime.now(timezone.utc),
                ),
            )
            interactions.append(interaction)
        return interactions

    @pytest.fixture
    def user_profile(self):
        """Create test user profile."""
        return CompanionProfile(
            user_id="test_user",
            preferred_name="friend",
            relationship_duration=timedelta(days=30),
            interaction_count=15,
            communication_preferences=UserPreferences(),
            trust_level=0.7,
            engagement_score=0.75,
        )

    @pytest.fixture
    def conversation_context(self):
        """Create conversation context."""
        return ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            message_text="This is a test message",
            timestamp=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def user_feedback(self):
        """Create user feedback."""
        return UserFeedback(
            feedback_id="test_feedback",
            user_id="test_user",
            feedback_type=FeedbackType.EXPLICIT_RATING,
            satisfaction_rating=0.8,
            sentiment=FeedbackSentiment.POSITIVE,
        )

    def test_analyze_interaction_patterns(self, engine, sample_interactions):
        """Test analyzing interaction patterns."""
        patterns = engine.analyze_interaction_patterns("test_user", sample_interactions)

        assert patterns is not None
        assert patterns.user_id == "test_user"
        assert isinstance(patterns.communication_patterns, list)
        assert isinstance(patterns.topic_patterns, list)

    def test_analyze_interaction_patterns_insufficient_data(self, engine):
        """Test analyzing patterns with insufficient interactions."""
        few_interactions = [
            InteractionData(
                conversation_context=ConversationContext(
                    user_id="test_user",
                    conversation_id="test_conv",
                    message_text="Test",
                    timestamp=datetime.now(timezone.utc),
                ),
                emotional_state=EmotionalState(
                    primary_emotion=EmotionType.NEUTRAL,
                    intensity=0.5,
                    confidence=0.7,
                    timestamp=datetime.now(timezone.utc),
                ),
            )
        ]

        patterns = engine.analyze_interaction_patterns("test_user", few_interactions)

        assert patterns is not None
        assert len(patterns.communication_patterns) == 0

    def test_learn_from_feedback(self, engine, user_feedback, conversation_context):
        """Test learning from user feedback."""
        learning_update = engine.learn_from_feedback(
            "test_user", user_feedback, conversation_context
        )

        assert learning_update is not None
        assert learning_update.user_id == "test_user"
        assert isinstance(learning_update.preference_updates, list)

    def test_adapt_behavior(self, engine, user_profile, conversation_context):
        """Test adapting behavior."""
        adaptation_result = engine.adapt_behavior(
            "test_user", conversation_context, user_profile
        )

        assert adaptation_result is not None
        assert adaptation_result.user_id == "test_user"
        assert isinstance(adaptation_result.adaptations, list)

    def test_extract_preferences(self, engine, sample_interactions):
        """Test extracting preferences from interactions."""
        preference_profile = engine.extract_preferences(
            "test_user", sample_interactions
        )

        assert preference_profile is not None
        assert preference_profile.user_id == "test_user"
        assert isinstance(preference_profile.preferences, dict)

    def test_start_learning_session(self, engine):
        """Test starting a learning session."""
        session_id = engine.start_learning_session("test_user")

        assert session_id is not None
        assert isinstance(session_id, str)
        assert session_id in engine.active_sessions

    def test_end_learning_session(self, engine):
        """Test ending a learning session."""
        session_id = engine.start_learning_session("test_user")
        session = engine.end_learning_session(session_id)

        assert session is not None
        assert session.session_id == session_id
        assert session.learning_score >= 0.0
        assert session.learning_score <= 1.0

    def test_end_learning_session_invalid_id(self, engine):
        """Test ending session with invalid ID."""
        result = engine.end_learning_session("invalid_session_id")

        assert result is None

    def test_get_learning_metrics(self, engine):
        """Test getting learning metrics."""
        metrics = engine.get_learning_metrics("test_user")

        # May be None if no metrics exist yet
        assert metrics is None or isinstance(metrics, LearningMetrics)

    def test_update_learning_metrics(self, engine):
        """Test updating learning metrics."""
        engine._update_learning_metrics(
            "test_user",
            interactions_processed=5,
            patterns_identified=3,
            user_satisfaction=0.8,
        )

        metrics = engine.get_learning_metrics("test_user")

        assert metrics is not None
        assert metrics.total_interactions == 5

    def test_calculate_learning_score(self, engine):
        """Test calculating learning score."""
        session = LearningSession(
            session_id="test_session",
            user_id="test_user",
            start_time=datetime.now(timezone.utc),
            interactions_processed=10,
            patterns_identified=8,
            preferences_updated=5,
            adaptations_applied=3,
            feedback_processed=2,
        )

        score = engine._calculate_learning_score(session)

        assert score >= 0.0
        assert score <= 1.0

    def test_calculate_learning_score_no_interactions(self, engine):
        """Test calculating learning score with no interactions."""
        session = LearningSession(
            session_id="test_session",
            user_id="test_user",
            start_time=datetime.now(timezone.utc),
            interactions_processed=0,
        )

        score = engine._calculate_learning_score(session)

        assert score == 0.0


class TestLearningMetrics:
    """Test learning metrics functionality."""

    def test_learning_metrics_creation(self):
        """Test creating learning metrics."""
        metrics = LearningMetrics(
            total_interactions=100,
            successful_adaptations=25,
            user_satisfaction_trend=0.75,
            preference_stability=0.8,
            learning_velocity=0.15,
            personalization_accuracy=0.85,
        )

        assert metrics.total_interactions == 100
        assert metrics.successful_adaptations == 25
        assert metrics.user_satisfaction_trend == 0.75


class TestLearningSession:
    """Test learning session functionality."""

    def test_learning_session_creation(self):
        """Test creating learning session."""
        session = LearningSession(
            session_id="test_session", user_id="test_user", start_time=datetime.now(timezone.utc)
        )

        assert session.session_id == "test_session"
        assert session.user_id == "test_user"
        assert session.interactions_processed == 0

    def test_learning_session_auto_id(self):
        """Test learning session with auto-generated ID."""
        session = LearningSession(
            session_id="", user_id="test_user", start_time=datetime.now(timezone.utc)
        )

        assert session.session_id is not None
        assert len(session.session_id) > 0
