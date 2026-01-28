"""
Integration tests for conversation intelligence module integration.

Tests that the conversation module singletons (ConversationFlowManager,
ConversationQualityAssessor, InterruptionHandler, TopicPreferenceLearner)
are properly implemented as singletons -- calling the getter function
multiple times returns the same instance.
"""
import pytest
from unittest.mock import MagicMock, patch


class TestConversationFlowIntegration:
    """Test ConversationFlowManager integration in orchestrator."""

    def test_flow_manager_singleton(self):
        """get_conversation_flow_manager should return the same singleton instance."""
        from morgan.conversation.flow import get_conversation_flow_manager
        mgr1 = get_conversation_flow_manager()
        mgr2 = get_conversation_flow_manager()
        assert mgr1 is mgr2, (
            "get_conversation_flow_manager must return the same singleton instance"
        )

    def test_quality_assessor_singleton(self):
        """get_conversation_quality_assessor should return the same singleton instance."""
        from morgan.conversation.quality import get_conversation_quality_assessor
        a1 = get_conversation_quality_assessor()
        a2 = get_conversation_quality_assessor()
        assert a1 is a2, (
            "get_conversation_quality_assessor must return the same singleton instance"
        )

    def test_interruption_handler_singleton(self):
        """get_interruption_handler should return the same singleton instance."""
        from morgan.conversation.interruption import get_interruption_handler
        h1 = get_interruption_handler()
        h2 = get_interruption_handler()
        assert h1 is h2, (
            "get_interruption_handler must return the same singleton instance"
        )

    def test_topic_learner_singleton(self):
        """get_topic_preference_learner should return the same singleton instance."""
        from morgan.conversation.topics import get_topic_preference_learner
        l1 = get_topic_preference_learner()
        l2 = get_topic_preference_learner()
        assert l1 is l2, (
            "get_topic_preference_learner must return the same singleton instance"
        )
