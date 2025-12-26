"""
Unit tests for the Personality System module.

Tests cover:
- Personality trait application
- Consistency across conversations
- Adaptive behavior based on relationship depth
"""

import pytest
from morgan_server.empathic.personality import (
    PersonalitySystem,
    PersonalityConfig,
    PersonalityTrait,
    ConversationalStyle,
    PersonalityApplication,
)


class TestPersonalityTraits:
    """Test personality trait management."""

    def test_default_traits(self):
        """Test that default traits are set correctly."""
        system = PersonalitySystem()

        # Check all default traits are present
        assert PersonalityTrait.WARMTH in system.config.traits
        assert PersonalityTrait.FORMALITY in system.config.traits
        assert PersonalityTrait.HUMOR in system.config.traits
        assert PersonalityTrait.EMPATHY in system.config.traits
        assert PersonalityTrait.ENTHUSIASM in system.config.traits
        assert PersonalityTrait.DIRECTNESS in system.config.traits
        assert PersonalityTrait.CURIOSITY in system.config.traits

        # Check default values are in valid range
        for trait in PersonalityTrait:
            value = system.get_trait(trait)
            assert 0.0 <= value <= 1.0

    def test_get_trait(self):
        """Test getting a specific trait value."""
        config = PersonalityConfig(traits={PersonalityTrait.WARMTH: 0.9})
        system = PersonalitySystem(config)

        assert system.get_trait(PersonalityTrait.WARMTH) == 0.9

    def test_set_trait(self):
        """Test setting a trait value."""
        system = PersonalitySystem()

        system.set_trait(PersonalityTrait.WARMTH, 0.5)
        assert system.get_trait(PersonalityTrait.WARMTH) == 0.5

        system.set_trait(PersonalityTrait.HUMOR, 0.0)
        assert system.get_trait(PersonalityTrait.HUMOR) == 0.0

        system.set_trait(PersonalityTrait.EMPATHY, 1.0)
        assert system.get_trait(PersonalityTrait.EMPATHY) == 1.0

    def test_set_trait_invalid_value(self):
        """Test that setting invalid trait values raises error."""
        system = PersonalitySystem()

        with pytest.raises(ValueError):
            system.set_trait(PersonalityTrait.WARMTH, 1.5)

        with pytest.raises(ValueError):
            system.set_trait(PersonalityTrait.WARMTH, -0.1)

    def test_custom_traits(self):
        """Test creating system with custom trait values."""
        custom_traits = {
            PersonalityTrait.WARMTH: 0.3,
            PersonalityTrait.FORMALITY: 0.9,
            PersonalityTrait.HUMOR: 0.1,
        }
        config = PersonalityConfig(traits=custom_traits)
        system = PersonalitySystem(config)

        assert system.get_trait(PersonalityTrait.WARMTH) == 0.3
        assert system.get_trait(PersonalityTrait.FORMALITY) == 0.9
        assert system.get_trait(PersonalityTrait.HUMOR) == 0.1


class TestPersonalityApplication:
    """Test applying personality to responses."""

    def test_apply_personality_basic(self):
        """Test basic personality application."""
        system = PersonalitySystem()
        response = "Hello, how can I help you?"

        result = system.apply_personality(response)

        assert isinstance(result, PersonalityApplication)
        assert result.adjusted_response == response
        assert len(result.traits_applied) > 0
        assert isinstance(result.style_notes, list)

    def test_apply_personality_high_warmth(self):
        """Test that high warmth trait is reflected in style notes."""
        config = PersonalityConfig(traits={PersonalityTrait.WARMTH: 0.9})
        system = PersonalitySystem(config)

        result = system.apply_personality("Hello!")

        assert PersonalityTrait.WARMTH in result.traits_applied
        assert result.traits_applied[PersonalityTrait.WARMTH] == 0.9
        assert "warm_tone" in result.style_notes

    def test_apply_personality_low_formality(self):
        """Test that low formality trait is reflected in style notes."""
        config = PersonalityConfig(traits={PersonalityTrait.FORMALITY: 0.2})
        system = PersonalitySystem(config)

        result = system.apply_personality("Hey there!")

        assert PersonalityTrait.FORMALITY in result.traits_applied
        assert result.traits_applied[PersonalityTrait.FORMALITY] == 0.2
        assert "casual_language" in result.style_notes

    def test_apply_personality_high_formality(self):
        """Test that high formality trait is reflected in style notes."""
        config = PersonalityConfig(traits={PersonalityTrait.FORMALITY: 0.8})
        system = PersonalitySystem(config)

        result = system.apply_personality("Good day.")

        assert "formal_language" in result.style_notes

    def test_apply_personality_high_empathy(self):
        """Test that high empathy trait is reflected in style notes."""
        config = PersonalityConfig(traits={PersonalityTrait.EMPATHY: 0.9})
        system = PersonalitySystem(config)

        result = system.apply_personality("I understand.")

        assert PersonalityTrait.EMPATHY in result.traits_applied
        assert result.traits_applied[PersonalityTrait.EMPATHY] == 0.9
        assert "empathetic_approach" in result.style_notes

    def test_apply_personality_high_enthusiasm(self):
        """Test that high enthusiasm trait is reflected in style notes."""
        config = PersonalityConfig(traits={PersonalityTrait.ENTHUSIASM: 0.9})
        system = PersonalitySystem(config)

        result = system.apply_personality("That's great!")

        assert PersonalityTrait.ENTHUSIASM in result.traits_applied
        assert result.traits_applied[PersonalityTrait.ENTHUSIASM] == 0.9
        assert "enthusiastic_tone" in result.style_notes

    def test_apply_personality_with_relationship_depth(self):
        """Test adaptive behavior based on relationship depth."""
        system = PersonalitySystem()

        # Low relationship depth
        result_low = system.apply_personality("Hello", relationship_depth=0.2)
        assert "familiar_tone" not in result_low.style_notes
        assert "close_relationship" not in result_low.style_notes

        # Medium relationship depth
        result_med = system.apply_personality("Hello", relationship_depth=0.6)
        assert "familiar_tone" in result_med.style_notes
        assert "close_relationship" not in result_med.style_notes

        # High relationship depth
        result_high = system.apply_personality("Hello", relationship_depth=0.9)
        assert "familiar_tone" in result_high.style_notes
        assert "close_relationship" in result_high.style_notes

    def test_apply_personality_tracks_interaction(self):
        """Test that interactions are tracked when user_id is provided."""
        system = PersonalitySystem()
        user_id = "user123"

        # Initially no interactions
        assert system.get_conversation_count(user_id) == 0

        # Apply personality with user_id
        system.apply_personality("Hello", user_id=user_id)

        # Should have tracked the interaction
        assert system.get_conversation_count(user_id) == 1

        # Apply again
        system.apply_personality("How are you?", user_id=user_id)
        assert system.get_conversation_count(user_id) == 2


class TestPersonalityPrompt:
    """Test personality prompt generation."""

    def test_get_personality_prompt_default(self):
        """Test generating prompt with default personality."""
        system = PersonalitySystem()

        prompt = system.get_personality_prompt()

        assert "Morgan" in prompt
        assert "AI assistant" in prompt
        assert len(prompt) > 0

    def test_get_personality_prompt_high_warmth(self):
        """Test prompt reflects high warmth."""
        config = PersonalityConfig(traits={PersonalityTrait.WARMTH: 0.9})
        system = PersonalitySystem(config)

        prompt = system.get_personality_prompt()

        assert "warm" in prompt.lower()

    def test_get_personality_prompt_low_formality(self):
        """Test prompt reflects low formality."""
        config = PersonalityConfig(traits={PersonalityTrait.FORMALITY: 0.2})
        system = PersonalitySystem(config)

        prompt = system.get_personality_prompt()

        assert "casual" in prompt.lower()

    def test_get_personality_prompt_high_formality(self):
        """Test prompt reflects high formality."""
        config = PersonalityConfig(traits={PersonalityTrait.FORMALITY: 0.9})
        system = PersonalitySystem(config)

        prompt = system.get_personality_prompt()

        assert "professional" in prompt.lower() or "formal" in prompt.lower()

    def test_get_personality_prompt_with_humor(self):
        """Test prompt reflects humor trait."""
        config = PersonalityConfig(traits={PersonalityTrait.HUMOR: 0.8})
        system = PersonalitySystem(config)

        prompt = system.get_personality_prompt()

        assert "humor" in prompt.lower()

    def test_get_personality_prompt_with_empathy(self):
        """Test prompt reflects empathy trait."""
        config = PersonalityConfig(traits={PersonalityTrait.EMPATHY: 0.9})
        system = PersonalitySystem(config)

        prompt = system.get_personality_prompt()

        assert "empathetic" in prompt.lower()

    def test_get_personality_prompt_with_enthusiasm(self):
        """Test prompt reflects enthusiasm trait."""
        config = PersonalityConfig(traits={PersonalityTrait.ENTHUSIASM: 0.9})
        system = PersonalitySystem(config)

        prompt = system.get_personality_prompt()

        assert "enthusiastic" in prompt.lower()

    def test_get_personality_prompt_with_directness(self):
        """Test prompt reflects directness trait."""
        config = PersonalityConfig(traits={PersonalityTrait.DIRECTNESS: 0.9})
        system = PersonalitySystem(config)

        prompt = system.get_personality_prompt()

        assert "direct" in prompt.lower()

    def test_get_personality_prompt_with_curiosity(self):
        """Test prompt reflects curiosity trait."""
        config = PersonalityConfig(traits={PersonalityTrait.CURIOSITY: 0.9})
        system = PersonalitySystem(config)

        prompt = system.get_personality_prompt()

        assert "curious" in prompt.lower()

    def test_get_personality_prompt_with_roleplay(self):
        """Test prompt includes roleplay description."""
        config = PersonalityConfig(
            roleplay_description="You are a helpful coding assistant."
        )
        system = PersonalitySystem(config)

        prompt = system.get_personality_prompt()

        assert "helpful coding assistant" in prompt

    def test_get_personality_prompt_with_background(self):
        """Test prompt includes background."""
        config = PersonalityConfig(
            background="You have 10 years of experience in software development."
        )
        system = PersonalitySystem(config)

        prompt = system.get_personality_prompt()

        assert "10 years of experience" in prompt

    def test_get_personality_prompt_with_interests(self):
        """Test prompt includes interests."""
        config = PersonalityConfig(interests=["programming", "AI", "music"])
        system = PersonalitySystem(config)

        prompt = system.get_personality_prompt()

        assert "programming" in prompt
        assert "AI" in prompt
        assert "music" in prompt

    def test_get_personality_prompt_with_relationship_depth(self):
        """Test prompt adapts to relationship depth."""
        system = PersonalitySystem()

        # Low relationship
        prompt_low = system.get_personality_prompt(relationship_depth=0.2)
        assert "established relationship" not in prompt_low
        assert "close relationship" not in prompt_low

        # Medium relationship
        prompt_med = system.get_personality_prompt(relationship_depth=0.6)
        assert "established relationship" in prompt_med
        assert "close relationship" not in prompt_med

        # High relationship
        prompt_high = system.get_personality_prompt(relationship_depth=0.9)
        assert "close relationship" in prompt_high

    def test_get_personality_prompt_with_conversational_style(self):
        """Test prompt includes conversational style."""
        config = PersonalityConfig(conversational_style=ConversationalStyle.PLAYFUL)
        system = PersonalitySystem(config)

        prompt = system.get_personality_prompt()

        assert "playful" in prompt.lower()


class TestConsistencyAcrossConversations:
    """Test personality consistency across conversations."""

    def test_consistency_check_no_history(self):
        """Test consistency check with no history returns True."""
        system = PersonalitySystem()
        user_id = "user123"

        current_traits = {PersonalityTrait.WARMTH: 0.8}

        assert system.is_consistent_with_previous(user_id, current_traits) is True

    def test_consistency_check_consistent_traits(self):
        """Test consistency check with consistent traits."""
        system = PersonalitySystem()
        user_id = "user123"

        # First interaction
        system.apply_personality("Hello", user_id=user_id)

        # Check with same traits
        current_traits = system.config.traits.copy()
        assert system.is_consistent_with_previous(user_id, current_traits) is True

    def test_consistency_check_small_difference(self):
        """Test consistency check with small trait difference."""
        system = PersonalitySystem()
        user_id = "user123"

        # First interaction
        system.apply_personality("Hello", user_id=user_id)

        # Check with slightly different traits (within tolerance)
        current_traits = system.config.traits.copy()
        current_traits[PersonalityTrait.WARMTH] += 0.1

        assert (
            system.is_consistent_with_previous(user_id, current_traits, tolerance=0.2)
            is True
        )

    def test_consistency_check_large_difference(self):
        """Test consistency check with large trait difference."""
        system = PersonalitySystem()
        user_id = "user123"

        # First interaction with high warmth
        config = PersonalityConfig(traits={PersonalityTrait.WARMTH: 0.9})
        system = PersonalitySystem(config)
        system.apply_personality("Hello", user_id=user_id)

        # Check with very different traits (outside tolerance)
        current_traits = {PersonalityTrait.WARMTH: 0.2}

        assert (
            system.is_consistent_with_previous(user_id, current_traits, tolerance=0.2)
            is False
        )

    def test_consistency_check_custom_tolerance(self):
        """Test consistency check with custom tolerance."""
        system = PersonalitySystem()
        user_id = "user123"

        # First interaction
        system.apply_personality("Hello", user_id=user_id)

        # Check with different traits
        current_traits = system.config.traits.copy()
        current_traits[PersonalityTrait.WARMTH] += 0.15

        # Should be consistent with high tolerance
        assert (
            system.is_consistent_with_previous(user_id, current_traits, tolerance=0.3)
            is True
        )

        # Should be inconsistent with low tolerance
        assert (
            system.is_consistent_with_previous(user_id, current_traits, tolerance=0.1)
            is False
        )

    def test_conversation_count_tracking(self):
        """Test that conversation count is tracked correctly."""
        system = PersonalitySystem()
        user_id = "user123"

        assert system.get_conversation_count(user_id) == 0

        # Add multiple interactions
        for i in range(5):
            system.apply_personality(f"Message {i}", user_id=user_id)

        assert system.get_conversation_count(user_id) == 5

    def test_conversation_history_limit(self):
        """Test that conversation history is limited to 100 interactions."""
        system = PersonalitySystem()
        user_id = "user123"

        # Add more than 100 interactions
        for i in range(150):
            system.apply_personality(f"Message {i}", user_id=user_id)

        # Should be limited to 100
        assert system.get_conversation_count(user_id) == 100

    def test_multiple_users_tracked_separately(self):
        """Test that multiple users are tracked separately."""
        system = PersonalitySystem()

        user1 = "user1"
        user2 = "user2"

        # Add interactions for user1
        for i in range(3):
            system.apply_personality(f"User1 message {i}", user_id=user1)

        # Add interactions for user2
        for i in range(5):
            system.apply_personality(f"User2 message {i}", user_id=user2)

        assert system.get_conversation_count(user1) == 3
        assert system.get_conversation_count(user2) == 5


class TestAdaptiveBehavior:
    """Test adaptive behavior based on relationship depth."""

    def test_adaptive_behavior_new_relationship(self):
        """Test behavior with new relationship (low depth)."""
        system = PersonalitySystem()

        result = system.apply_personality("Hello", relationship_depth=0.1)

        # Should not have familiar or close relationship notes
        assert "familiar_tone" not in result.style_notes
        assert "close_relationship" not in result.style_notes

    def test_adaptive_behavior_established_relationship(self):
        """Test behavior with established relationship (medium depth)."""
        system = PersonalitySystem()

        result = system.apply_personality("Hello", relationship_depth=0.6)

        # Should have familiar tone but not close relationship
        assert "familiar_tone" in result.style_notes
        assert "close_relationship" not in result.style_notes

    def test_adaptive_behavior_close_relationship(self):
        """Test behavior with close relationship (high depth)."""
        system = PersonalitySystem()

        result = system.apply_personality("Hello", relationship_depth=0.9)

        # Should have both familiar tone and close relationship
        assert "familiar_tone" in result.style_notes
        assert "close_relationship" in result.style_notes

    def test_adaptive_behavior_in_prompt(self):
        """Test that relationship depth affects prompt generation."""
        system = PersonalitySystem()

        # New relationship
        prompt_new = system.get_personality_prompt(relationship_depth=0.1)
        assert "established relationship" not in prompt_new

        # Established relationship
        prompt_established = system.get_personality_prompt(relationship_depth=0.6)
        assert "established relationship" in prompt_established

        # Close relationship
        prompt_close = system.get_personality_prompt(relationship_depth=0.9)
        assert "close relationship" in prompt_close

    def test_adaptive_behavior_boundary_values(self):
        """Test adaptive behavior at boundary values."""
        system = PersonalitySystem()

        # Exactly at 0.5 threshold
        result_boundary = system.apply_personality("Hello", relationship_depth=0.5)
        assert "familiar_tone" not in result_boundary.style_notes

        # Just above 0.5 threshold
        result_above = system.apply_personality("Hello", relationship_depth=0.51)
        assert "familiar_tone" in result_above.style_notes

        # Exactly at 0.8 threshold
        result_boundary2 = system.apply_personality("Hello", relationship_depth=0.8)
        assert "close_relationship" not in result_boundary2.style_notes

        # Just above 0.8 threshold
        result_above2 = system.apply_personality("Hello", relationship_depth=0.81)
        assert "close_relationship" in result_above2.style_notes


class TestPersonalityConfig:
    """Test personality configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PersonalityConfig()

        assert config.name == "Morgan"
        assert config.conversational_style == ConversationalStyle.FRIENDLY
        assert config.roleplay_description is None
        assert config.background is None
        assert len(config.interests) == 0
        assert len(config.traits) > 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = PersonalityConfig(
            name="CustomBot",
            conversational_style=ConversationalStyle.PROFESSIONAL,
            roleplay_description="A professional assistant",
            background="Expert in technology",
            interests=["coding", "AI"],
        )

        assert config.name == "CustomBot"
        assert config.conversational_style == ConversationalStyle.PROFESSIONAL
        assert config.roleplay_description == "A professional assistant"
        assert config.background == "Expert in technology"
        assert "coding" in config.interests
        assert "AI" in config.interests

    def test_config_with_partial_traits(self):
        """Test that partial traits are filled with defaults."""
        config = PersonalityConfig(traits={PersonalityTrait.WARMTH: 0.5})

        # Should have the specified trait
        assert config.traits[PersonalityTrait.WARMTH] == 0.5

        # Should have default values for other traits
        assert PersonalityTrait.FORMALITY in config.traits
        assert PersonalityTrait.HUMOR in config.traits
        assert PersonalityTrait.EMPATHY in config.traits
