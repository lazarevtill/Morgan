"""
Tests for empathetic response generator module.

Tests empathetic response generation, emotional acknowledgment,
supportive responses, and relationship-aware responses.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from morgan.empathy.generator import EmpatheticResponseGenerator
from morgan.emotional.models import (
    EmotionalState, EmotionType, ConversationContext,
    CompanionProfile, RelationshipMilestone, EmpatheticResponse
)


class TestEmpatheticResponseGenerator:
    """Test empathetic response generator functionality."""

    @pytest.fixture
    def generator(self):
        """Create empathetic response generator for testing."""
        with patch('morgan.empathy.generator.get_llm_service'):
            with patch('morgan.empathy.generator.get_settings'):
                return EmpatheticResponseGenerator()

    @pytest.fixture
    def emotional_state_joy(self):
        """Create sample emotional state - joy."""
        return EmotionalState(
            primary_emotion=EmotionType.JOY,
            intensity=0.8,
            confidence=0.9,
            timestamp=datetime.utcnow()
        )

    @pytest.fixture
    def emotional_state_sadness(self):
        """Create sample emotional state - sadness."""
        return EmotionalState(
            primary_emotion=EmotionType.SADNESS,
            intensity=0.7,
            confidence=0.85,
            timestamp=datetime.utcnow()
        )

    @pytest.fixture
    def conversation_context(self):
        """Create sample conversation context."""
        return ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            message_text="I'm feeling so happy about my new job!",
            timestamp=datetime.utcnow()
        )

    @pytest.fixture
    def companion_profile(self):
        """Create sample companion profile."""
        return CompanionProfile(
            user_id="test_user",
            preferred_name="friend",
            relationship_duration=timedelta(days=60),
            interaction_count=25,
            trust_level=0.75,
            engagement_score=0.8
        )

    def test_generate_empathetic_response_joy(self, generator, emotional_state_joy, conversation_context):
        """Test generating empathetic response for joy."""
        response = generator.generate_empathetic_response(
            emotional_state_joy, conversation_context
        )

        assert isinstance(response, EmpatheticResponse)
        assert response.response_text is not None
        assert len(response.response_text) > 0
        assert response.emotional_tone is not None
        assert response.empathy_level > 0.0
        assert response.empathy_level <= 1.0

    def test_generate_empathetic_response_sadness(self, generator, emotional_state_sadness, conversation_context):
        """Test generating empathetic response for sadness."""
        context = ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            message_text="I'm feeling really down today",
            timestamp=datetime.utcnow()
        )

        response = generator.generate_empathetic_response(
            emotional_state_sadness, context
        )

        assert response is not None
        assert response.empathy_level > 0.5  # Higher empathy for sadness
        assert "gentle" in response.emotional_tone or "supportive" in response.emotional_tone

    def test_create_emotional_acknowledgment(self, generator, emotional_state_joy):
        """Test creating emotional acknowledgment."""
        acknowledgment = generator.create_emotional_acknowledgment(emotional_state_joy)

        assert acknowledgment is not None
        assert len(acknowledgment) > 0
        assert isinstance(acknowledgment, str)

    def test_generate_supportive_response_emotional(self, generator, emotional_state_sadness, conversation_context):
        """Test generating emotional supportive response."""
        response = generator.generate_supportive_response(
            emotional_state_sadness, conversation_context, support_type="emotional"
        )

        assert response is not None
        assert len(response) > 0

    def test_generate_supportive_response_practical(self, generator, emotional_state_sadness, conversation_context):
        """Test generating practical supportive response."""
        response = generator.generate_supportive_response(
            emotional_state_sadness, conversation_context, support_type="practical"
        )

        assert response is not None
        assert isinstance(response, str)

    def test_create_relationship_aware_response(self, generator, emotional_state_joy, conversation_context, companion_profile):
        """Test creating relationship-aware response."""
        response = generator.create_relationship_aware_response(
            emotional_state_joy, conversation_context, companion_profile
        )

        assert response is not None
        assert len(response) > 0

    def test_empathy_level_calculation_high_intensity(self, generator, conversation_context):
        """Test empathy level calculation for high intensity emotions."""
        high_intensity_state = EmotionalState(
            primary_emotion=EmotionType.SADNESS,
            intensity=0.9,
            confidence=0.85,
            timestamp=datetime.utcnow()
        )

        empathy_level = generator._calculate_empathy_level(high_intensity_state, conversation_context)

        assert empathy_level > 0.7  # High empathy for high intensity
        assert empathy_level <= 1.0

    def test_empathy_level_calculation_low_intensity(self, generator, conversation_context):
        """Test empathy level calculation for low intensity emotions."""
        low_intensity_state = EmotionalState(
            primary_emotion=EmotionType.NEUTRAL,
            intensity=0.2,
            confidence=0.6,
            timestamp=datetime.utcnow()
        )

        empathy_level = generator._calculate_empathy_level(low_intensity_state, conversation_context)

        assert empathy_level >= 0.3  # Minimum empathy level
        assert empathy_level < 0.7  # Lower for low intensity

    def test_determine_emotional_tone_joy(self, generator, emotional_state_joy):
        """Test determining emotional tone for joy."""
        tone = generator._determine_emotional_tone(emotional_state_joy)

        assert tone is not None
        assert "celebratory" in tone or "warm" in tone or "enthusiastic" in tone

    def test_determine_emotional_tone_sadness(self, generator, emotional_state_sadness):
        """Test determining emotional tone for sadness."""
        tone = generator._determine_emotional_tone(emotional_state_sadness)

        assert tone is not None
        assert "gentle" in tone or "supportive" in tone

    def test_personalization_elements_high_intensity(self, generator, conversation_context, companion_profile):
        """Test identifying personalization elements for high intensity."""
        high_intensity_state = EmotionalState(
            primary_emotion=EmotionType.FEAR,
            intensity=0.85,
            confidence=0.9,
            timestamp=datetime.utcnow()
        )

        elements = generator._identify_personalization_elements(
            high_intensity_state, conversation_context, companion_profile
        )

        assert isinstance(elements, list)
        assert "high_intensity_support" in elements

    def test_build_relationship_context_established(self, generator, conversation_context, companion_profile):
        """Test building relationship context for established relationship."""
        context_str = generator._build_relationship_context(conversation_context, companion_profile)

        assert context_str is not None
        assert "established_relationship" in context_str or "developing_relationship" in context_str

    def test_build_relationship_context_new(self, generator, conversation_context):
        """Test building relationship context for new relationship."""
        new_profile = CompanionProfile(
            user_id="new_user",
            preferred_name="friend",
            relationship_duration=timedelta(days=2),
            interaction_count=3,
            trust_level=0.3,
            engagement_score=0.5
        )

        context_str = generator._build_relationship_context(conversation_context, new_profile)

        assert "new_relationship" in context_str

    def test_response_confidence_calculation(self, generator, emotional_state_joy, conversation_context, companion_profile):
        """Test response confidence calculation."""
        confidence = generator._calculate_response_confidence(
            emotional_state_joy, conversation_context, companion_profile
        )

        assert confidence > 0.0
        assert confidence <= 1.0

    def test_empathy_template_selection(self, generator, emotional_state_joy):
        """Test empathy template selection."""
        template = generator._get_empathy_template(emotional_state_joy)

        assert template is not None
        assert len(template) > 0
        assert isinstance(template, str)
