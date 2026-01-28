"""
Integration tests for learning module integration.

Tests that the LearningEngine and its sub-components are properly
initialized and can be accessed via singleton factory functions.
"""
import pytest
from unittest.mock import MagicMock, patch


class TestLearningEngineIntegration:
    """Test LearningEngine integration in orchestrator."""

    def test_learning_engine_instantiation(self):
        """LearningEngine should instantiate with all sub-components."""
        from morgan.learning.engine import LearningEngine

        engine = LearningEngine()
        assert engine.pattern_analyzer is not None
        assert engine.preference_extractor is not None
        assert engine.preference_storage is not None
        assert engine.adaptation_engine is not None
        assert engine.feedback_processor is not None

    def test_learning_engine_singleton(self):
        """get_learning_engine should return the same singleton instance."""
        from morgan.learning.engine import get_learning_engine

        e1 = get_learning_engine()
        e2 = get_learning_engine()
        assert e1 is e2, (
            "get_learning_engine must return the same singleton instance"
        )

    def test_pattern_analyzer_instantiation(self):
        """InteractionPatternAnalyzer should instantiate cleanly."""
        from morgan.learning.patterns import InteractionPatternAnalyzer

        analyzer = InteractionPatternAnalyzer()
        assert analyzer is not None

    def test_preference_extractor_instantiation(self):
        """PreferenceExtractor should instantiate cleanly."""
        from morgan.learning.preferences import PreferenceExtractor

        extractor = PreferenceExtractor()
        assert extractor is not None

    def test_adaptation_engine_instantiation(self):
        """BehavioralAdaptationEngine should instantiate cleanly."""
        from morgan.learning.adaptation import BehavioralAdaptationEngine

        engine = BehavioralAdaptationEngine()
        assert engine is not None

    def test_feedback_processor_instantiation(self):
        """FeedbackProcessor should instantiate cleanly."""
        from morgan.learning.feedback import FeedbackProcessor

        processor = FeedbackProcessor()
        assert processor is not None

    def test_learning_engine_start_session(self):
        """LearningEngine should start a learning session."""
        from morgan.learning.engine import LearningEngine

        engine = LearningEngine()
        session_id = engine.start_learning_session("test_user_123")
        assert session_id is not None
        assert isinstance(session_id, str)
        assert session_id in engine.active_sessions
        assert engine.active_sessions[session_id].user_id == "test_user_123"
