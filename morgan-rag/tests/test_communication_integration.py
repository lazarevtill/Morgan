"""
Integration tests for communication module integration.

Tests that the communication module singletons (CommunicationStyleAdapter,
NonVerbalCueDetector, CulturalEmotionalAwareness, UserPreferenceLearner)
are properly implemented as singletons -- calling the getter function
multiple times returns the same instance.
"""
import pytest
from unittest.mock import MagicMock, patch


class TestCommunicationIntegration:
    """Test communication module integration in orchestrator."""

    def test_style_adapter_singleton(self):
        """get_communication_style_adapter should return the same singleton instance."""
        from morgan.communication.style import get_communication_style_adapter
        a1 = get_communication_style_adapter()
        a2 = get_communication_style_adapter()
        assert a1 is a2, (
            "get_communication_style_adapter must return the same singleton instance"
        )

    def test_nonverbal_detector_singleton(self):
        """get_nonverbal_cue_detector should return the same singleton instance."""
        from morgan.communication.nonverbal import get_nonverbal_cue_detector
        d1 = get_nonverbal_cue_detector()
        d2 = get_nonverbal_cue_detector()
        assert d1 is d2, (
            "get_nonverbal_cue_detector must return the same singleton instance"
        )

    def test_cultural_awareness_singleton(self):
        """get_cultural_emotional_awareness should return the same singleton instance."""
        from morgan.communication.cultural import get_cultural_emotional_awareness
        c1 = get_cultural_emotional_awareness()
        c2 = get_cultural_emotional_awareness()
        assert c1 is c2, (
            "get_cultural_emotional_awareness must return the same singleton instance"
        )

    def test_preference_learner_singleton(self):
        """get_user_preference_learner should return the same singleton instance."""
        from morgan.communication.preferences import get_user_preference_learner
        p1 = get_user_preference_learner()
        p2 = get_user_preference_learner()
        assert p1 is p2, (
            "get_user_preference_learner must return the same singleton instance"
        )
