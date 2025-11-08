"""
Tests for emotional validation response module.

Tests validation response generation, affirmation creation,
comprehensive emotional validation, and validation confidence.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from morgan.empathy.validator import EmotionalValidator
from morgan.emotional.models import (
    EmotionalState, EmotionType, ConversationContext
)


class TestEmotionalValidator:
    """Test emotional validator functionality."""

    @pytest.fixture
    def validator(self):
        """Create emotional validator for testing."""
        with patch('morgan.empathy.validator.get_llm_service'):
            with patch('morgan.empathy.validator.get_settings'):
                return EmotionalValidator()

    @pytest.fixture
    def emotional_state_joy(self):
        """Create emotional state - joy."""
        return EmotionalState(
            primary_emotion=EmotionType.JOY,
            intensity=0.8,
            confidence=0.9,
            timestamp=datetime.utcnow()
        )

    @pytest.fixture
    def emotional_state_sadness(self):
        """Create emotional state - sadness."""
        return EmotionalState(
            primary_emotion=EmotionType.SADNESS,
            intensity=0.7,
            confidence=0.85,
            timestamp=datetime.utcnow()
        )

    @pytest.fixture
    def emotional_state_anger(self):
        """Create emotional state - anger."""
        return EmotionalState(
            primary_emotion=EmotionType.ANGER,
            intensity=0.75,
            confidence=0.8,
            timestamp=datetime.utcnow()
        )

    @pytest.fixture
    def conversation_context(self):
        """Create conversation context."""
        return ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            message_text="I'm feeling really happy about this achievement",
            timestamp=datetime.utcnow()
        )

    def test_generate_validation_response_joy(self, validator, emotional_state_joy, conversation_context):
        """Test generating validation response for joy."""
        response = validator.generate_validation_response(emotional_state_joy, conversation_context)

        assert response is not None
        assert len(response) > 0
        assert isinstance(response, str)

    def test_generate_validation_response_sadness(self, validator, emotional_state_sadness, conversation_context):
        """Test generating validation response for sadness."""
        sad_context = ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            message_text="I'm feeling really down today",
            timestamp=datetime.utcnow()
        )

        response = validator.generate_validation_response(emotional_state_sadness, sad_context)

        assert response is not None
        assert len(response) > 0

    def test_generate_validation_response_high_personalization(self, validator, emotional_state_anger, conversation_context):
        """Test generating validation with high personalization."""
        response = validator.generate_validation_response(
            emotional_state_anger, conversation_context, personalization_level=0.9
        )

        assert response is not None

    def test_generate_validation_response_low_personalization(self, validator, emotional_state_joy, conversation_context):
        """Test generating validation with low personalization."""
        response = validator.generate_validation_response(
            emotional_state_joy, conversation_context, personalization_level=0.3
        )

        assert response is not None

    def test_validate_emotional_experience(self, validator, emotional_state_sadness):
        """Test comprehensive emotional validation."""
        user_description = "I'm feeling sad and overwhelmed by everything happening"

        validation = validator.validate_emotional_experience(emotional_state_sadness, user_description)

        assert isinstance(validation, dict)
        assert "primary_validation" in validation
        assert "intensity_validation" in validation
        assert "context_validation" in validation
        assert "normalization" in validation
        assert "support_message" in validation
        assert "validation_confidence" in validation

    def test_create_affirmation_response_joy(self, validator, emotional_state_joy):
        """Test creating affirmation response for joy."""
        affirmation = validator.create_affirmation_response(emotional_state_joy)

        assert affirmation is not None
        assert len(affirmation) > 0

    def test_create_affirmation_response_sadness(self, validator, emotional_state_sadness):
        """Test creating affirmation response for sadness."""
        affirmation = validator.create_affirmation_response(emotional_state_sadness)

        assert affirmation is not None
        assert len(affirmation) > 0

    def test_create_affirmation_with_specific_concern(self, validator, emotional_state_anger):
        """Test creating affirmation with specific concern."""
        affirmation = validator.create_affirmation_response(
            emotional_state_anger, specific_concern="I feel like I'm overreacting"
        )

        assert affirmation is not None

    def test_get_base_validation_high_confidence(self, validator):
        """Test getting base validation for high confidence emotion."""
        high_conf_state = EmotionalState(
            primary_emotion=EmotionType.JOY,
            intensity=0.8,
            confidence=0.95,
            timestamp=datetime.utcnow()
        )

        validation = validator._get_base_validation(high_conf_state)

        assert validation is not None
        assert len(validation) > 0

    def test_get_base_validation_low_confidence(self, validator):
        """Test getting base validation for low confidence emotion."""
        low_conf_state = EmotionalState(
            primary_emotion=EmotionType.NEUTRAL,
            intensity=0.4,
            confidence=0.45,
            timestamp=datetime.utcnow()
        )

        validation = validator._get_base_validation(low_conf_state)

        assert validation is not None

    def test_get_intensity_modifier_high(self, validator):
        """Test getting intensity modifier for high intensity."""
        modifier = validator._get_intensity_modifier(0.85)

        assert modifier is not None

    def test_get_intensity_modifier_medium(self, validator):
        """Test getting intensity modifier for medium intensity."""
        modifier = validator._get_intensity_modifier(0.55)

        assert modifier is not None

    def test_get_intensity_modifier_low(self, validator):
        """Test getting intensity modifier for low intensity."""
        modifier = validator._get_intensity_modifier(0.25)

        assert modifier is not None

    def test_validate_primary_emotion_joy(self, validator, emotional_state_joy):
        """Test validating primary emotion - joy."""
        validation = validator._validate_primary_emotion(emotional_state_joy)

        assert validation is not None
        assert len(validation) > 0

    def test_validate_primary_emotion_sadness(self, validator, emotional_state_sadness):
        """Test validating primary emotion - sadness."""
        validation = validator._validate_primary_emotion(emotional_state_sadness)

        assert validation is not None

    def test_validate_emotion_intensity_high(self, validator):
        """Test validating high emotion intensity."""
        high_intensity_state = EmotionalState(
            primary_emotion=EmotionType.ANGER,
            intensity=0.9,
            confidence=0.85,
            timestamp=datetime.utcnow()
        )

        validation = validator._validate_emotion_intensity(high_intensity_state)

        assert validation is not None

    def test_validate_emotion_intensity_low(self, validator):
        """Test validating low emotion intensity."""
        low_intensity_state = EmotionalState(
            primary_emotion=EmotionType.SADNESS,
            intensity=0.3,
            confidence=0.7,
            timestamp=datetime.utcnow()
        )

        validation = validator._validate_emotion_intensity(low_intensity_state)

        assert validation is not None

    def test_validate_emotional_context(self, validator):
        """Test validating emotional context."""
        description = "This situation is really difficult and challenging for me"

        validation = validator._validate_emotional_context(description)

        assert validation is not None

    def test_normalize_emotional_experience(self, validator, emotional_state_sadness):
        """Test normalizing emotional experience."""
        normalization = validator._normalize_emotional_experience(emotional_state_sadness)

        assert normalization is not None
        assert len(normalization) > 0

    def test_generate_support_message(self, validator, emotional_state_sadness):
        """Test generating support message."""
        message = validator._generate_support_message(emotional_state_sadness)

        assert message is not None
        assert len(message) > 0

    def test_calculate_validation_confidence_joy(self, validator, emotional_state_joy):
        """Test calculating validation confidence for joy."""
        confidence = validator._calculate_validation_confidence(emotional_state_joy)

        assert confidence >= 0.0
        assert confidence <= 1.0

    def test_calculate_validation_confidence_neutral(self, validator):
        """Test calculating validation confidence for neutral."""
        neutral_state = EmotionalState(
            primary_emotion=EmotionType.NEUTRAL,
            intensity=0.3,
            confidence=0.6,
            timestamp=datetime.utcnow()
        )

        confidence = validator._calculate_validation_confidence(neutral_state)

        assert confidence >= 0.0
        assert confidence <= 1.0

    def test_address_specific_concern_wrong(self, validator, emotional_state_anger):
        """Test addressing 'wrong' concern."""
        response = validator._address_specific_concern("I think there's something wrong with me", emotional_state_anger)

        assert response is not None

    def test_address_specific_concern_weak(self, validator, emotional_state_sadness):
        """Test addressing 'weak' concern."""
        response = validator._address_specific_concern("I feel weak for crying", emotional_state_sadness)

        assert response is not None

    def test_address_specific_concern_alone(self, validator, emotional_state_sadness):
        """Test addressing 'alone' concern."""
        response = validator._address_specific_concern("I feel so alone in this", emotional_state_sadness)

        assert response is not None

    def test_get_fallback_validation(self, validator, emotional_state_joy):
        """Test getting fallback validation response."""
        fallback = validator._get_fallback_validation(emotional_state_joy)

        assert fallback is not None
        assert len(fallback) > 0
