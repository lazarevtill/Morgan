"""
Unit tests for the Roleplay System module.

Tests cover:
- Roleplay configuration loading
- Context-aware responses
- Emotional integration
"""

import pytest
from morgan_server.empathic import (
    RoleplaySystem,
    RoleplayConfig,
    RoleplayContext,
    RoleplayResponse,
    RoleplayTone,
    ResponseStyle,
    EmotionalTone,
    EmotionalDetection,
    EmotionalIntelligence,
    PersonalitySystem,
    PersonalityConfig,
    PersonalityTrait,
)


class TestRoleplayConfiguration:
    """Test roleplay configuration loading and management."""

    def test_default_config(self):
        """Test that default configuration is loaded correctly."""
        system = RoleplaySystem()

        assert system.config.character_name == "Morgan"
        assert system.config.tone == RoleplayTone.FRIENDLY
        assert system.config.response_style == ResponseStyle.CONVERSATIONAL
        assert system.config.emotional_intelligence_enabled is True
        assert system.config.relationship_awareness_enabled is True

    def test_custom_config(self):
        """Test creating system with custom configuration."""
        config = RoleplayConfig(
            character_name="TestBot",
            character_description="A helpful test assistant",
            tone=RoleplayTone.PROFESSIONAL,
            response_style=ResponseStyle.TECHNICAL,
            background_story="Expert in testing",
            expertise_areas=["testing", "QA"]
        )
        system = RoleplaySystem(config)

        assert system.config.character_name == "TestBot"
        assert system.config.character_description == "A helpful test assistant"
        assert system.config.tone == RoleplayTone.PROFESSIONAL
        assert system.config.response_style == ResponseStyle.TECHNICAL
        assert system.config.background_story == "Expert in testing"
        assert "testing" in system.config.expertise_areas
        assert "QA" in system.config.expertise_areas

    def test_config_with_personality_traits(self):
        """Test configuration with custom personality traits."""
        config = RoleplayConfig(
            personality_traits={
                PersonalityTrait.WARMTH: 0.9,
                PersonalityTrait.FORMALITY: 0.2,
                PersonalityTrait.HUMOR: 0.7
            }
        )
        system = RoleplaySystem(config)

        # Personality system should have these traits
        assert system.personality_system.get_trait(PersonalityTrait.WARMTH) == 0.9
        assert system.personality_system.get_trait(PersonalityTrait.FORMALITY) == 0.2
        assert system.personality_system.get_trait(PersonalityTrait.HUMOR) == 0.7

    def test_update_config(self):
        """Test updating configuration after initialization."""
        system = RoleplaySystem()

        # Update configuration
        system.update_config(
            tone=RoleplayTone.PLAYFUL,
            response_style=ResponseStyle.CONCISE
        )

        assert system.config.tone == RoleplayTone.PLAYFUL
        assert system.config.response_style == ResponseStyle.CONCISE

    def test_update_config_personality_traits(self):
        """Test updating personality traits through config update."""
        system = RoleplaySystem()

        # Update personality traits
        system.update_config(
            personality_traits={PersonalityTrait.WARMTH: 0.5}
        )

        assert system.personality_system.get_trait(PersonalityTrait.WARMTH) == 0.5

    def test_config_with_communication_preferences(self):
        """Test configuration with communication preferences."""
        config = RoleplayConfig(
            communication_preferences={
                "response_length": "concise",
                "formality": "casual"
            }
        )
        system = RoleplaySystem(config)

        assert system.config.communication_preferences["response_length"] == "concise"
        assert system.config.communication_preferences["formality"] == "casual"


class TestContextAwareResponses:
    """Test context-aware response generation."""

    def test_generate_response_basic(self):
        """Test basic response generation."""
        system = RoleplaySystem()
        base_response = "Hello! How can I help you today?"

        result = system.generate_response(base_response)

        assert isinstance(result, RoleplayResponse)
        assert result.response_text == base_response
        assert result.tone_applied in RoleplayTone
        assert result.style_applied in ResponseStyle

    def test_generate_response_with_context(self):
        """Test response generation with context."""
        system = RoleplaySystem()
        base_response = "I understand your concern."

        context = RoleplayContext(
            user_id="user123",
            relationship_depth=0.6
        )

        result = system.generate_response(base_response, context)

        assert isinstance(result, RoleplayResponse)
        assert "familiar_relationship" in result.relationship_notes
        assert context.relationship_depth in result.context_used.values()

    def test_generate_response_close_relationship(self):
        """Test response generation with close relationship."""
        system = RoleplaySystem()
        base_response = "Great to hear from you!"

        context = RoleplayContext(
            user_id="user123",
            relationship_depth=0.9
        )

        result = system.generate_response(base_response, context)

        assert "familiar_relationship" in result.relationship_notes
        assert "close_relationship" in result.relationship_notes

    def test_generate_response_new_relationship(self):
        """Test response generation with new relationship."""
        system = RoleplaySystem()
        base_response = "Hello!"

        context = RoleplayContext(
            user_id="user123",
            relationship_depth=0.1
        )

        result = system.generate_response(base_response, context)

        assert "familiar_relationship" not in result.relationship_notes
        assert "close_relationship" not in result.relationship_notes

    def test_generate_response_with_user_preferences(self):
        """Test response generation with user preferences."""
        system = RoleplaySystem()
        base_response = "Here's the information you requested."

        context = RoleplayContext(
            user_preferences={
                "response_length": "concise",
                "technical_level": "high"
            }
        )

        result = system.generate_response(base_response, context)

        # Style should be adjusted based on preferences
        assert result.style_applied in [ResponseStyle.CONCISE, ResponseStyle.TECHNICAL]

    def test_generate_response_with_conversation_history(self):
        """Test response generation with conversation history."""
        system = RoleplaySystem()
        base_response = "Based on what we discussed earlier..."

        context = RoleplayContext(
            conversation_history=[
                {"role": "user", "content": "Tell me about Python"},
                {"role": "assistant", "content": "Python is a programming language..."}
            ]
        )

        result = system.generate_response(base_response, context)

        assert isinstance(result, RoleplayResponse)
        assert len(context.conversation_history) == 2


class TestEmotionalIntegration:
    """Test emotional intelligence integration."""

    def test_emotional_integration_enabled(self):
        """Test that emotional intelligence is integrated when enabled."""
        config = RoleplayConfig(emotional_intelligence_enabled=True)
        system = RoleplaySystem(config)

        assert system.emotional_intelligence is not None
        assert system.config.emotional_intelligence_enabled is True

    def test_emotional_integration_disabled(self):
        """Test behavior when emotional intelligence is disabled."""
        config = RoleplayConfig(emotional_intelligence_enabled=False)
        system = RoleplaySystem(config)

        # Should still have EI system but not use it
        assert system.emotional_intelligence is not None
        assert system.config.emotional_intelligence_enabled is False

    def test_generate_response_with_emotional_state(self):
        """Test response generation with emotional state."""
        system = RoleplaySystem()
        base_response = "I'm here to help."

        emotional_state = EmotionalDetection(
            primary_tone=EmotionalTone.SAD,
            confidence=0.8,
            indicators=["sad", "down"]
        )

        context = RoleplayContext(
            user_id="user123",
            emotional_state=emotional_state
        )

        result = system.generate_response(base_response, context)

        assert result.emotional_adjustment is not None
        assert "emotional_tone" in result.context_used
        assert "emotional_intensity" in result.context_used

    def test_generate_response_with_joyful_emotion(self):
        """Test response generation with joyful emotional state."""
        system = RoleplaySystem()
        base_response = "That's wonderful!"

        emotional_state = EmotionalDetection(
            primary_tone=EmotionalTone.JOYFUL,
            confidence=0.9,
            indicators=["happy", "wonderful"]
        )

        context = RoleplayContext(
            user_id="user123",
            emotional_state=emotional_state
        )

        result = system.generate_response(base_response, context)

        assert result.emotional_adjustment == EmotionalTone.JOYFUL.value

    def test_generate_response_with_support_message(self):
        """Test that support messages are included for negative emotions."""
        system = RoleplaySystem()
        base_response = "I understand."

        # Create some negative emotional history
        system.emotional_intelligence.track_pattern(
            "user123",
            EmotionalTone.SAD,
            0.8
        )
        system.emotional_intelligence.track_pattern(
            "user123",
            EmotionalTone.ANXIOUS,
            0.7
        )

        emotional_state = EmotionalDetection(
            primary_tone=EmotionalTone.SAD,
            confidence=0.8,
            indicators=["sad"]
        )

        context = RoleplayContext(
            user_id="user123",
            emotional_state=emotional_state
        )

        result = system.generate_response(base_response, context)

        # Should have support in relationship notes
        support_notes = [n for n in result.relationship_notes if "support:" in n]
        assert len(support_notes) > 0

    def test_detect_and_integrate_emotion(self):
        """Test detecting and integrating emotion from message."""
        system = RoleplaySystem()
        message = "I'm feeling really happy today!"

        detection = system.detect_and_integrate_emotion(message, "user123")

        assert detection.primary_tone == EmotionalTone.JOYFUL
        assert detection.confidence > 0.5

    def test_detect_emotion_when_disabled(self):
        """Test emotion detection when EI is disabled."""
        config = RoleplayConfig(emotional_intelligence_enabled=False)
        system = RoleplaySystem(config)
        message = "I'm feeling really happy today!"

        detection = system.detect_and_integrate_emotion(message, "user123")

        # Should return neutral when disabled
        assert detection.primary_tone == EmotionalTone.NEUTRAL

    def test_get_emotional_trend(self):
        """Test getting emotional trend analysis."""
        system = RoleplaySystem()
        user_id = "user123"

        # Track some patterns
        system.emotional_intelligence.track_pattern(
            user_id,
            EmotionalTone.JOYFUL,
            0.8
        )
        system.emotional_intelligence.track_pattern(
            user_id,
            EmotionalTone.CONTENT,
            0.7
        )

        trend = system.get_emotional_trend(user_id)

        assert "dominant_tone" in trend
        assert "tone_distribution" in trend
        assert "trend" in trend

    def test_get_emotional_trend_when_disabled(self):
        """Test emotional trend when EI is disabled."""
        config = RoleplayConfig(emotional_intelligence_enabled=False)
        system = RoleplaySystem(config)

        trend = system.get_emotional_trend("user123")

        assert trend["dominant_tone"] is None
        assert trend["trend"] == "unknown"


class TestSystemPromptGeneration:
    """Test system prompt generation."""

    def test_get_system_prompt_basic(self):
        """Test basic system prompt generation."""
        system = RoleplaySystem()

        prompt = system.get_system_prompt()

        assert "Morgan" in prompt
        assert "friendly" in prompt.lower()
        assert "conversational" in prompt.lower()

    def test_get_system_prompt_with_expertise(self):
        """Test prompt includes expertise areas."""
        config = RoleplayConfig(
            expertise_areas=["programming", "AI", "testing"]
        )
        system = RoleplaySystem(config)

        prompt = system.get_system_prompt()

        assert "programming" in prompt
        assert "AI" in prompt
        assert "testing" in prompt

    def test_get_system_prompt_with_emotional_intelligence(self):
        """Test prompt includes emotional intelligence instructions."""
        config = RoleplayConfig(emotional_intelligence_enabled=True)
        system = RoleplaySystem(config)

        prompt = system.get_system_prompt()

        assert "emotionally intelligent" in prompt.lower()

    def test_get_system_prompt_with_relationship_context(self):
        """Test prompt adapts to relationship depth."""
        system = RoleplaySystem()

        # New relationship
        context_new = RoleplayContext(relationship_depth=0.1)
        prompt_new = system.get_system_prompt(context_new)
        assert "established relationship" not in prompt_new

        # Established relationship
        context_established = RoleplayContext(relationship_depth=0.6)
        prompt_established = system.get_system_prompt(context_established)
        assert "established relationship" in prompt_established

        # Close relationship
        context_close = RoleplayContext(relationship_depth=0.9)
        prompt_close = system.get_system_prompt(context_close)
        assert "close" in prompt_close.lower()

    def test_get_system_prompt_with_user_preferences(self):
        """Test prompt includes user preferences."""
        system = RoleplaySystem()

        context = RoleplayContext(
            user_preferences={
                "response_length": "concise",
                "formality": "casual"
            }
        )

        prompt = system.get_system_prompt(context)

        assert "concise" in prompt.lower()
        assert "casual" in prompt.lower()

    def test_get_system_prompt_with_tone(self):
        """Test prompt includes roleplay tone."""
        config = RoleplayConfig(tone=RoleplayTone.PLAYFUL)
        system = RoleplaySystem(config)

        prompt = system.get_system_prompt()

        assert "playful" in prompt.lower()

    def test_get_system_prompt_with_style(self):
        """Test prompt includes response style."""
        config = RoleplayConfig(response_style=ResponseStyle.TECHNICAL)
        system = RoleplaySystem(config)

        prompt = system.get_system_prompt()

        assert "technical" in prompt.lower()


class TestToneAndStyleApplication:
    """Test tone and style application logic."""

    def test_tone_adapts_to_relationship(self):
        """Test that tone adapts based on relationship depth."""
        config = RoleplayConfig(tone=RoleplayTone.PROFESSIONAL)
        system = RoleplaySystem(config)

        # New relationship - should stay professional
        context_new = RoleplayContext(relationship_depth=0.2)
        result_new = system.generate_response("Hello", context_new)
        assert result_new.tone_applied == RoleplayTone.PROFESSIONAL

        # Close relationship - should become friendlier
        context_close = RoleplayContext(relationship_depth=0.8)
        result_close = system.generate_response("Hello", context_close)
        assert result_close.tone_applied in [
            RoleplayTone.FRIENDLY,
            RoleplayTone.COMPANION
        ]

    def test_style_adapts_to_preferences(self):
        """Test that style adapts based on user preferences."""
        system = RoleplaySystem()

        # Concise preference
        context_concise = RoleplayContext(
            user_preferences={"response_length": "concise"}
        )
        result_concise = system.generate_response("Info", context_concise)
        assert result_concise.style_applied == ResponseStyle.CONCISE

        # Detailed preference
        context_detailed = RoleplayContext(
            user_preferences={"response_length": "detailed"}
        )
        result_detailed = system.generate_response("Info", context_detailed)
        assert result_detailed.style_applied == ResponseStyle.DETAILED

        # Technical preference
        context_technical = RoleplayContext(
            user_preferences={"technical_level": "high"}
        )
        result_technical = system.generate_response("Info", context_technical)
        assert result_technical.style_applied == ResponseStyle.TECHNICAL


class TestContextSummary:
    """Test context summary generation."""

    def test_get_context_summary(self):
        """Test getting context summary."""
        system = RoleplaySystem()

        context = RoleplayContext(
            user_id="user123",
            relationship_depth=0.7,
            conversation_history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        )

        summary = system.get_context_summary(context)

        assert summary["user_id"] == "user123"
        assert summary["relationship_depth"] == 0.7
        assert summary["conversation_length"] == 2
        assert summary["roleplay_tone"] == RoleplayTone.FRIENDLY.value
        assert summary["response_style"] == ResponseStyle.CONVERSATIONAL.value

    def test_get_context_summary_with_emotional_state(self):
        """Test context summary includes emotional state."""
        system = RoleplaySystem()

        emotional_state = EmotionalDetection(
            primary_tone=EmotionalTone.JOYFUL,
            confidence=0.8,
            indicators=["happy"]
        )

        context = RoleplayContext(
            user_id="user123",
            emotional_state=emotional_state
        )

        summary = system.get_context_summary(context)

        assert summary["emotional_state"] == EmotionalTone.JOYFUL.value

    def test_get_context_summary_no_emotional_state(self):
        """Test context summary with no emotional state."""
        system = RoleplaySystem()

        context = RoleplayContext(user_id="user123")

        summary = system.get_context_summary(context)

        assert summary["emotional_state"] is None


class TestIntegrationWithOtherSystems:
    """Test integration with emotional intelligence and personality systems."""

    def test_integration_with_custom_emotional_intelligence(self):
        """Test using custom emotional intelligence system."""
        ei = EmotionalIntelligence(pattern_window_days=7)
        system = RoleplaySystem(emotional_intelligence=ei)

        assert system.emotional_intelligence is ei
        assert system.emotional_intelligence.pattern_window_days == 7

    def test_integration_with_custom_personality_system(self):
        """Test using custom personality system."""
        personality_config = PersonalityConfig(
            name="CustomBot",
            traits={PersonalityTrait.WARMTH: 0.5}
        )
        personality = PersonalitySystem(personality_config)
        system = RoleplaySystem(personality_system=personality)

        assert system.personality_system is personality
        assert system.personality_system.config.name == "CustomBot"

    def test_personality_system_created_from_roleplay_config(self):
        """Test that personality system is created from roleplay config."""
        config = RoleplayConfig(
            character_name="TestBot",
            character_description="A test assistant",
            personality_traits={PersonalityTrait.WARMTH: 0.9},
            background_story="Expert tester",
            expertise_areas=["testing"]
        )
        system = RoleplaySystem(config)

        # Personality system should reflect roleplay config
        assert system.personality_system.config.name == "TestBot"
        assert system.personality_system.config.roleplay_description == "A test assistant"
        assert system.personality_system.config.background == "Expert tester"
        assert "testing" in system.personality_system.config.interests
        assert system.personality_system.get_trait(PersonalityTrait.WARMTH) == 0.9

    def test_full_integration_workflow(self):
        """Test complete workflow with all systems integrated."""
        system = RoleplaySystem()

        # Detect emotion
        message = "I'm so happy about this progress!"
        emotional_state = system.detect_and_integrate_emotion(message, "user123")

        # Generate response with full context
        context = RoleplayContext(
            user_id="user123",
            relationship_depth=0.7,
            emotional_state=emotional_state,
            user_preferences={"response_length": "conversational"}
        )

        result = system.generate_response(
            "That's wonderful to hear!",
            context
        )

        # Verify all systems contributed
        assert result.emotional_adjustment is not None
        assert len(result.personality_notes) > 0
        assert len(result.relationship_notes) > 0
        assert "emotional_tone" in result.context_used
        assert "personality_traits" in result.context_used
        assert "relationship_depth" in result.context_used
