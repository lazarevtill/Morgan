"""
Tests for crisis detection and support module.

Tests crisis detection patterns, support response generation,
safety planning, and ongoing risk assessment capabilities.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from morgan.empathy.support import CrisisSupport, CrisisLevel, SupportType
from morgan.emotional.models import (
    EmotionalState, EmotionType, ConversationContext
)


class TestCrisisSupport:
    """Test crisis support functionality."""

    @pytest.fixture
    def crisis_support(self):
        """Create crisis support system for testing."""
        with patch('morgan.empathy.support.get_llm_service'):
            with patch('morgan.empathy.support.get_settings'):
                return CrisisSupport()

    @pytest.fixture
    def emotional_state_sadness(self):
        """Create emotional state - high intensity sadness."""
        return EmotionalState(
            primary_emotion=EmotionType.SADNESS,
            intensity=0.9,
            confidence=0.85,
            timestamp=datetime.utcnow()
        )

    @pytest.fixture
    def conversation_context(self):
        """Create conversation context."""
        return ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            message_text="I'm having a really hard time dealing with everything",
            timestamp=datetime.utcnow()
        )

    def test_detect_crisis_none(self, crisis_support, emotional_state_sadness, conversation_context):
        """Test crisis detection with no crisis."""
        normal_text = "I'm feeling a bit down but I'll be okay"

        crisis_level, crisis_types = crisis_support.detect_crisis(
            normal_text, emotional_state_sadness, conversation_context
        )

        assert crisis_level in [CrisisLevel.NONE, CrisisLevel.LOW]

    def test_detect_crisis_high_risk(self, crisis_support, conversation_context):
        """Test detection of high-risk crisis situation."""
        crisis_text = "I don't want to live anymore, everything feels hopeless"

        high_intensity_state = EmotionalState(
            primary_emotion=EmotionType.SADNESS,
            intensity=0.95,
            confidence=0.9,
            timestamp=datetime.utcnow()
        )

        crisis_level, crisis_types = crisis_support.detect_crisis(
            crisis_text, high_intensity_state, conversation_context
        )

        assert crisis_level in [CrisisLevel.HIGH, CrisisLevel.CRITICAL]
        assert len(crisis_types) > 0

    def test_detect_crisis_moderate_risk(self, crisis_support, emotional_state_sadness, conversation_context):
        """Test detection of moderate-risk crisis."""
        moderate_text = "I feel so worthless and hopeless about my situation"

        crisis_level, crisis_types = crisis_support.detect_crisis(
            moderate_text, emotional_state_sadness, conversation_context
        )

        assert crisis_level in [CrisisLevel.LOW, CrisisLevel.MEDIUM, CrisisLevel.HIGH]

    def test_generate_crisis_response_none(self, crisis_support, emotional_state_sadness, conversation_context):
        """Test generating response for non-crisis situation."""
        response = crisis_support.generate_crisis_response(
            CrisisLevel.NONE, [], emotional_state_sadness, conversation_context
        )

        assert isinstance(response, dict)
        assert "support_message" in response
        assert response["crisis_level"] == "none"

    def test_generate_crisis_response_critical(self, crisis_support, emotional_state_sadness, conversation_context):
        """Test generating response for critical crisis."""
        response = crisis_support.generate_crisis_response(
            CrisisLevel.CRITICAL, ["suicide_risk"], emotional_state_sadness, conversation_context
        )

        assert response["crisis_level"] == "critical"
        assert "immediate_resources" in response
        assert "emergency_contacts" in response
        assert response["follow_up_needed"] is True
        assert response["professional_help_recommended"] is True

    def test_generate_crisis_response_medium(self, crisis_support, emotional_state_sadness, conversation_context):
        """Test generating response for medium crisis."""
        response = crisis_support.generate_crisis_response(
            CrisisLevel.MEDIUM, ["severe_depression"], emotional_state_sadness, conversation_context
        )

        assert response["crisis_level"] == "medium"
        assert "support_message" in response
        assert "immediate_resources" in response
        assert response["professional_help_recommended"] is True

    def test_create_safety_plan(self, crisis_support):
        """Test creating safety plan."""
        safety_plan = crisis_support.create_safety_plan(
            ["suicide_risk"], {"location": "home", "support_available": True}
        )

        assert isinstance(safety_plan, dict)
        assert "warning_signs" in safety_plan
        assert "coping_strategies" in safety_plan
        assert "support_contacts" in safety_plan
        assert "professional_contacts" in safety_plan
        assert "emergency_plan" in safety_plan

    def test_assess_ongoing_risk_no_history(self, crisis_support):
        """Test ongoing risk assessment with no history."""
        assessment = crisis_support.assess_ongoing_risk("new_user", timeframe_days=7)

        assert assessment["risk_level"] == "low"

    def test_get_immediate_resources(self, crisis_support):
        """Test getting immediate crisis resources."""
        resources = crisis_support._get_immediate_resources(["suicide_risk"])

        assert isinstance(resources, list)
        assert len(resources) > 0

    def test_get_ongoing_resources(self, crisis_support):
        """Test getting ongoing support resources."""
        resources = crisis_support._get_ongoing_resources(["severe_depression"])

        assert isinstance(resources, list)
        assert len(resources) > 0

    def test_generate_safety_suggestions_high_risk(self, crisis_support):
        """Test generating safety suggestions for high-risk crisis."""
        suggestions = crisis_support._generate_safety_suggestions(
            CrisisLevel.HIGH, ["suicide_risk"]
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_generate_safety_suggestions_medium_risk(self, crisis_support):
        """Test generating safety suggestions for medium-risk crisis."""
        suggestions = crisis_support._generate_safety_suggestions(
            CrisisLevel.MEDIUM, ["panic_crisis"]
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_get_emergency_contacts(self, crisis_support):
        """Test getting emergency contact information."""
        contacts = crisis_support._get_emergency_contacts()

        assert isinstance(contacts, dict)
        assert len(contacts) > 0

    def test_assess_crisis_level_no_detection(self, crisis_support, emotional_state_sadness):
        """Test assessing crisis level with no detected crisis."""
        level = crisis_support._assess_crisis_level([], {}, emotional_state_sadness, "normal text")

        assert level in [CrisisLevel.NONE, CrisisLevel.LOW]

    def test_identify_warning_signs(self, crisis_support):
        """Test identifying warning signs for safety planning."""
        signs = crisis_support._identify_warning_signs(["suicide_risk", "self_harm"])

        assert isinstance(signs, list)
        assert len(signs) > 0

    def test_suggest_coping_strategies(self, crisis_support):
        """Test suggesting coping strategies."""
        strategies = crisis_support._suggest_coping_strategies(["panic_crisis", "severe_depression"])

        assert isinstance(strategies, list)
        assert len(strategies) > 0

    def test_suggest_environment_safety(self, crisis_support):
        """Test suggesting environment safety measures."""
        measures = crisis_support._suggest_environment_safety(["substance_crisis"])

        assert isinstance(measures, list)
        assert len(measures) > 0

    def test_create_emergency_plan(self, crisis_support):
        """Test creating emergency plan."""
        plan = crisis_support._create_emergency_plan(["suicide_risk"])

        assert isinstance(plan, list)
        assert len(plan) > 0

    def test_general_support_message_generation(self, crisis_support, emotional_state_sadness):
        """Test generating general support message."""
        message = crisis_support._generate_general_support_message(emotional_state_sadness)

        assert message is not None
        assert len(message) > 0

    def test_get_general_coping_suggestions(self, crisis_support, emotional_state_sadness):
        """Test getting general coping suggestions."""
        suggestions = crisis_support._get_general_coping_suggestions(emotional_state_sadness)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_get_general_mental_health_resources(self, crisis_support):
        """Test getting general mental health resources."""
        resources = crisis_support._get_general_mental_health_resources()

        assert isinstance(resources, list)
        assert len(resources) > 0
