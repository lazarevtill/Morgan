"""
Tests for feedback processing module.

Tests feedback analysis, learning update generation,
behavioral signal analysis, and satisfaction trend tracking.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from morgan.learning.feedback import (
    FeedbackProcessor, UserFeedback, FeedbackType, FeedbackSentiment,
    FeedbackAnalysis, LearningUpdate
)
from morgan.emotional.models import ConversationContext


class TestFeedbackProcessor:
    """Test feedback processor functionality."""

    @pytest.fixture
    def processor(self):
        """Create feedback processor."""
        return FeedbackProcessor()

    @pytest.fixture
    def conversation_context(self):
        """Create conversation context."""
        return ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            message_text="This response was helpful",
            timestamp=datetime.utcnow()
        )

    @pytest.fixture
    def positive_feedback(self):
        """Create positive feedback."""
        return UserFeedback(
            feedback_id="test_positive",
            user_id="test_user",
            feedback_type=FeedbackType.EXPLICIT_RATING,
            satisfaction_rating=0.9,
            sentiment=FeedbackSentiment.POSITIVE,
            feedback_text="This was very helpful and clear",
            specific_aspects={"helpfulness": 0.95, "clarity": 0.9}
        )

    @pytest.fixture
    def negative_feedback(self):
        """Create negative feedback."""
        return UserFeedback(
            feedback_id="test_negative",
            user_id="test_user",
            feedback_type=FeedbackType.EXPLICIT_RATING,
            satisfaction_rating=0.3,
            sentiment=FeedbackSentiment.NEGATIVE,
            feedback_text="This was confusing and not helpful",
            specific_aspects={"helpfulness": 0.2, "clarity": 0.3}
        )

    def test_process_feedback_positive(self, processor, positive_feedback, conversation_context):
        """Test processing positive feedback."""
        learning_update = processor.process_feedback("test_user", positive_feedback, conversation_context)

        assert learning_update is not None
        assert learning_update.user_id == "test_user"
        assert isinstance(learning_update.preference_updates, list)
        assert isinstance(learning_update.learning_insights, list)

    def test_process_feedback_negative(self, processor, negative_feedback, conversation_context):
        """Test processing negative feedback."""
        learning_update = processor.process_feedback("test_user", negative_feedback, conversation_context)

        assert learning_update is not None
        assert len(learning_update.adaptation_changes) > 0 or len(learning_update.learning_insights) > 0

    def test_analyze_feedback_high_satisfaction(self, processor, positive_feedback, conversation_context):
        """Test analyzing high satisfaction feedback."""
        analysis = processor._analyze_feedback(positive_feedback, conversation_context)

        assert isinstance(analysis, FeedbackAnalysis)
        assert len(analysis.positive_aspects) > 0
        assert analysis.confidence_score > 0.5

    def test_analyze_feedback_low_satisfaction(self, processor, negative_feedback, conversation_context):
        """Test analyzing low satisfaction feedback."""
        analysis = processor._analyze_feedback(negative_feedback, conversation_context)

        assert isinstance(analysis, FeedbackAnalysis)
        assert len(analysis.identified_issues) > 0
        assert len(analysis.improvement_areas) > 0

    def test_analyze_feedback_text_positive(self, processor):
        """Test analyzing positive feedback text."""
        insights = processor._analyze_feedback_text("This was helpful and accurate")

        assert isinstance(insights, dict)
        assert "positives" in insights
        assert len(insights["positives"]) > 0

    def test_analyze_feedback_text_negative(self, processor):
        """Test analyzing negative feedback text."""
        insights = processor._analyze_feedback_text("This was confusing and incorrect")

        assert isinstance(insights, dict)
        assert "issues" in insights
        assert len(insights["issues"]) > 0

    def test_analyze_feedback_text_preference_indicators(self, processor):
        """Test analyzing feedback text with preference indicators."""
        insights = processor._analyze_feedback_text("I prefer brief and simple explanations")

        assert "preferences" in insights
        assert len(insights["preferences"]) > 0

    def test_analyze_behavioral_signals_positive(self, processor):
        """Test analyzing positive behavioral signals."""
        signals = {
            "long_session": True,
            "follow_up_questions": True,
            "copy_response": True
        }

        insights = processor._analyze_behavioral_signals(signals)

        assert isinstance(insights, dict)
        assert len(insights["positives"]) > 0

    def test_analyze_behavioral_signals_negative(self, processor):
        """Test analyzing negative behavioral signals."""
        signals = {
            "short_session": True,
            "abrupt_end": True,
            "topic_change": True
        }

        insights = processor._analyze_behavioral_signals(signals)

        assert len(insights["issues"]) > 0

    def test_generate_actionable_insights(self, processor):
        """Test generating actionable insights."""
        issues = ["Low accuracy rating"]
        improvements = ["accuracy", "clarity"]
        positives = ["High helpfulness rating"]

        insights = processor._generate_actionable_insights(issues, improvements, positives)

        assert isinstance(insights, list)
        assert len(insights) > 0

    def test_calculate_analysis_confidence_high(self, processor):
        """Test calculating high analysis confidence."""
        feedback = UserFeedback(
            feedback_id="test",
            user_id="test_user",
            feedback_type=FeedbackType.EXPLICIT_RATING,
            satisfaction_rating=0.8,
            feedback_text="This was very helpful and clear with good examples",
            specific_aspects={"helpfulness": 0.9}
        )

        confidence = processor._calculate_analysis_confidence(feedback)

        assert confidence > 0.5

    def test_calculate_analysis_confidence_low(self, processor):
        """Test calculating low analysis confidence."""
        feedback = UserFeedback(
            feedback_id="test",
            user_id="test_user",
            feedback_type=FeedbackType.BEHAVIORAL
        )

        confidence = processor._calculate_analysis_confidence(feedback)

        assert confidence >= 0.1
        assert confidence <= 0.5

    def test_generate_learning_updates(self, processor, positive_feedback, conversation_context):
        """Test generating learning updates from analysis."""
        analysis = processor._analyze_feedback(positive_feedback, conversation_context)
        learning_update = processor._generate_learning_updates(
            "test_user", positive_feedback, analysis, conversation_context
        )

        assert isinstance(learning_update, LearningUpdate)
        assert learning_update.confidence_score > 0.0

    def test_get_feedback_history(self, processor, positive_feedback, conversation_context):
        """Test getting feedback history."""
        # First add some feedback
        processor.process_feedback("test_user", positive_feedback, conversation_context)

        history = processor.get_feedback_history("test_user")

        assert isinstance(history, list)
        assert len(history) > 0

    def test_get_feedback_history_empty(self, processor):
        """Test getting feedback history for user with no feedback."""
        history = processor.get_feedback_history("unknown_user")

        assert isinstance(history, list)
        assert len(history) == 0

    def test_get_feedback_analysis(self, processor, positive_feedback, conversation_context):
        """Test getting feedback analysis."""
        processor.process_feedback("test_user", positive_feedback, conversation_context)

        analysis = processor.get_feedback_analysis(positive_feedback.feedback_id)

        assert analysis is not None
        assert isinstance(analysis, FeedbackAnalysis)

    def test_get_user_satisfaction_trend(self, processor):
        """Test getting user satisfaction trend."""
        # Add some feedback
        for i in range(5):
            feedback = UserFeedback(
                feedback_id=f"test_{i}",
                user_id="test_user",
                feedback_type=FeedbackType.EXPLICIT_RATING,
                satisfaction_rating=0.7 + (i * 0.05),
                timestamp=datetime.utcnow() - timedelta(days=i)
            )
            context = ConversationContext(
                user_id="test_user",
                conversation_id=f"conv_{i}",
                message_text="Test",
                timestamp=datetime.utcnow()
            )
            processor.process_feedback("test_user", feedback, context)

        trend = processor.get_user_satisfaction_trend("test_user", days=30)

        assert isinstance(trend, dict)
        assert "average" in trend
        assert "trend" in trend
        assert "count" in trend

    def test_get_user_satisfaction_trend_no_data(self, processor):
        """Test getting satisfaction trend with no data."""
        trend = processor.get_user_satisfaction_trend("unknown_user", days=30)

        assert trend["count"] == 0
        assert trend["average"] == 0.0


class TestUserFeedback:
    """Test user feedback model."""

    def test_user_feedback_creation(self):
        """Test creating user feedback."""
        feedback = UserFeedback(
            feedback_id="test",
            user_id="test_user",
            feedback_type=FeedbackType.EXPLICIT_RATING,
            satisfaction_rating=0.8,
            sentiment=FeedbackSentiment.POSITIVE
        )

        assert feedback.feedback_id == "test"
        assert feedback.satisfaction_rating == 0.8

    def test_user_feedback_auto_id(self):
        """Test user feedback with auto-generated ID."""
        feedback = UserFeedback(
            feedback_id="",
            user_id="test_user",
            feedback_type=FeedbackType.THUMBS_UP_DOWN
        )

        assert feedback.feedback_id is not None
        assert len(feedback.feedback_id) > 0


class TestLearningUpdate:
    """Test learning update model."""

    def test_learning_update_creation(self):
        """Test creating learning update."""
        update = LearningUpdate(
            update_id="test",
            user_id="test_user",
            preference_updates=[],
            adaptation_changes=[],
            confidence_adjustments={},
            learning_insights=[],
            confidence_score=0.8
        )

        assert update.update_id == "test"
        assert update.confidence_score == 0.8

    def test_learning_update_auto_id(self):
        """Test learning update with auto-generated ID."""
        update = LearningUpdate(
            update_id="",
            user_id="test_user",
            preference_updates=[],
            adaptation_changes=[],
            confidence_adjustments={},
            learning_insights=[],
            confidence_score=0.7
        )

        assert update.update_id is not None
        assert len(update.update_id) > 0
