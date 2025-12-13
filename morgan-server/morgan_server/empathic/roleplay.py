"""
Roleplay System module for the Empathic Engine.

This module provides roleplay configuration and context-aware response logic
to make Morgan feel like a natural, consistent character with emotional
intelligence and relationship awareness.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .emotional import EmotionalTone, EmotionalDetection, EmotionalIntelligence
from .personality import PersonalitySystem, PersonalityConfig, PersonalityTrait


class RoleplayTone(str, Enum):
    """Roleplay tone options."""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    CASUAL = "casual"
    PLAYFUL = "playful"
    SUPPORTIVE = "supportive"
    MENTOR = "mentor"
    COMPANION = "companion"


class ResponseStyle(str, Enum):
    """Response style options."""
    CONCISE = "concise"
    DETAILED = "detailed"
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"
    EMPATHETIC = "empathetic"


@dataclass
class RoleplayConfig:
    """Configuration for roleplay behavior."""
    character_name: str = "Morgan"
    character_description: Optional[str] = None
    tone: RoleplayTone = RoleplayTone.FRIENDLY
    response_style: ResponseStyle = ResponseStyle.CONVERSATIONAL
    personality_traits: Dict[PersonalityTrait, float] = field(default_factory=dict)
    background_story: Optional[str] = None
    expertise_areas: List[str] = field(default_factory=list)
    communication_preferences: Dict[str, Any] = field(default_factory=dict)
    emotional_intelligence_enabled: bool = True
    relationship_awareness_enabled: bool = True


@dataclass
class RoleplayContext:
    """Context for roleplay response generation."""
    user_id: Optional[str] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    relationship_depth: float = 0.0
    emotional_state: Optional[EmotionalDetection] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    session_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoleplayResponse:
    """Result of roleplay response generation."""
    response_text: str
    tone_applied: RoleplayTone
    style_applied: ResponseStyle
    emotional_adjustment: Optional[str] = None
    relationship_notes: List[str] = field(default_factory=list)
    personality_notes: List[str] = field(default_factory=list)
    context_used: Dict[str, Any] = field(default_factory=dict)


class RoleplaySystem:
    """
    Roleplay system for context-aware, emotionally intelligent responses.

    This class provides:
    - Base roleplay configuration (personality, tone, style)
    - Context-aware response logic
    - Emotional intelligence integration
    - Relationship-aware behavior
    """

    def __init__(
        self,
        config: Optional[RoleplayConfig] = None,
        emotional_intelligence: Optional[EmotionalIntelligence] = None,
        personality_system: Optional[PersonalitySystem] = None
    ):
        """
        Initialize the roleplay system.

        Args:
            config: Roleplay configuration. If None, uses default.
            emotional_intelligence: Optional EI system for integration
            personality_system: Optional personality system for integration
        """
        self.config = config or RoleplayConfig()

        # Initialize or use provided emotional intelligence
        self.emotional_intelligence = (
            emotional_intelligence or EmotionalIntelligence()
        )

        # Initialize or use provided personality system
        if personality_system:
            self.personality_system = personality_system
        else:
            # Create personality config from roleplay config
            personality_config = PersonalityConfig(
                name=self.config.character_name,
                traits=self.config.personality_traits.copy(),
                roleplay_description=self.config.character_description,
                background=self.config.background_story,
                interests=self.config.expertise_areas.copy()
            )
            self.personality_system = PersonalitySystem(personality_config)

    def generate_response(
        self,
        base_response: str,
        context: Optional[RoleplayContext] = None
    ) -> RoleplayResponse:
        """
        Generate a roleplay-enhanced response.

        Args:
            base_response: The base response text to enhance
            context: Optional context for personalization

        Returns:
            RoleplayResponse with enhanced response and metadata
        """
        context = context or RoleplayContext()
        response_text = base_response
        relationship_notes = []
        personality_notes = []
        context_used = {}

        # Apply emotional intelligence if enabled and emotional state provided
        emotional_adjustment = None
        if (self.config.emotional_intelligence_enabled and
                context.emotional_state):
            adjustment = self.emotional_intelligence.adjust_response_tone(
                context.emotional_state,
                user_id=context.user_id
            )
            emotional_adjustment = adjustment.target_tone.value
            context_used["emotional_tone"] = adjustment.target_tone.value
            context_used["emotional_intensity"] = adjustment.intensity

            # Add emotional notes
            if adjustment.celebration:
                relationship_notes.append(f"celebration: {adjustment.celebration}")
            if adjustment.support:
                relationship_notes.append(f"support: {adjustment.support}")

        # Apply personality if relationship-aware behavior is enabled
        if self.config.relationship_awareness_enabled:
            personality_app = self.personality_system.apply_personality(
                response_text,
                user_id=context.user_id,
                relationship_depth=context.relationship_depth
            )
            response_text = personality_app.adjusted_response
            personality_notes = personality_app.style_notes
            context_used["personality_traits"] = personality_app.traits_applied
            context_used["relationship_depth"] = context.relationship_depth

        # Apply tone adjustments based on roleplay configuration
        tone_applied = self._apply_tone(
            response_text,
            context.relationship_depth
        )

        # Apply style adjustments
        style_applied = self._apply_style(
            response_text,
            context.user_preferences
        )

        # Add relationship-aware notes
        if context.relationship_depth > 0.5:
            relationship_notes.append("familiar_relationship")
        if context.relationship_depth > 0.8:
            relationship_notes.append("close_relationship")

        return RoleplayResponse(
            response_text=response_text,
            tone_applied=tone_applied,
            style_applied=style_applied,
            emotional_adjustment=emotional_adjustment,
            relationship_notes=relationship_notes,
            personality_notes=personality_notes,
            context_used=context_used
        )

    def get_system_prompt(
        self,
        context: Optional[RoleplayContext] = None
    ) -> str:
        """
        Generate a system prompt for the LLM based on roleplay configuration.

        Args:
            context: Optional context for personalization

        Returns:
            System prompt string
        """
        context = context or RoleplayContext()

        # Start with personality prompt
        prompt = self.personality_system.get_personality_prompt(
            relationship_depth=context.relationship_depth
        )

        # Add roleplay-specific instructions
        prompt += f"\n\nYour roleplay tone is {self.config.tone.value}."
        prompt += f" Your response style is {self.config.response_style.value}."

        # Add expertise areas
        if self.config.expertise_areas:
            areas = ", ".join(self.config.expertise_areas)
            prompt += f" You have expertise in: {areas}."

        # Add emotional intelligence instructions
        if self.config.emotional_intelligence_enabled:
            prompt += (
                " You are emotionally intelligent and adapt your responses "
                "to the user's emotional state."
            )

        # Add relationship awareness instructions
        if self.config.relationship_awareness_enabled:
            if context.relationship_depth > 0.5:
                prompt += (
                    " You have an established relationship with this user "
                    "and can be more familiar and personal."
                )
            if context.relationship_depth > 0.8:
                prompt += (
                    " You have a close, trusted relationship with this user "
                    "and can be very warm and supportive."
                )

        # Add communication preferences
        if context.user_preferences:
            if "response_length" in context.user_preferences:
                length = context.user_preferences["response_length"]
                prompt += f" The user prefers {length} responses."

            if "formality" in context.user_preferences:
                formality = context.user_preferences["formality"]
                prompt += f" The user prefers a {formality} communication style."

        return prompt

    def update_config(self, **kwargs) -> None:
        """
        Update roleplay configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Update personality system if personality traits changed
        if "personality_traits" in kwargs:
            for trait, value in kwargs["personality_traits"].items():
                self.personality_system.set_trait(trait, value)

    def get_context_summary(
        self,
        context: RoleplayContext
    ) -> Dict[str, Any]:
        """
        Get a summary of the current context for debugging/logging.

        Args:
            context: The roleplay context

        Returns:
            Dictionary with context summary
        """
        summary = {
            "user_id": context.user_id,
            "relationship_depth": context.relationship_depth,
            "conversation_length": len(context.conversation_history),
            "emotional_state": (
                context.emotional_state.primary_tone.value
                if context.emotional_state else None
            ),
            "roleplay_tone": self.config.tone.value,
            "response_style": self.config.response_style.value,
            "emotional_intelligence_enabled": (
                self.config.emotional_intelligence_enabled
            ),
            "relationship_awareness_enabled": (
                self.config.relationship_awareness_enabled
            )
        }

        return summary

    def _apply_tone(
        self,
        response: str,
        relationship_depth: float
    ) -> RoleplayTone:
        """
        Apply tone adjustments based on configuration and relationship.

        Args:
            response: The response text
            relationship_depth: Relationship depth (0.0 to 1.0)

        Returns:
            The tone that was applied
        """
        # Base tone from config
        tone = self.config.tone

        # Adjust tone based on relationship depth
        if relationship_depth > 0.7:
            # Closer relationships can be more casual/friendly
            if tone == RoleplayTone.PROFESSIONAL:
                tone = RoleplayTone.FRIENDLY
            elif tone == RoleplayTone.FRIENDLY:
                tone = RoleplayTone.COMPANION

        return tone

    def _apply_style(
        self,
        response: str,
        user_preferences: Dict[str, Any]
    ) -> ResponseStyle:
        """
        Apply style adjustments based on configuration and preferences.

        Args:
            response: The response text
            user_preferences: User preferences

        Returns:
            The style that was applied
        """
        # Base style from config
        style = self.config.response_style

        # Adjust based on user preferences
        if user_preferences.get("response_length") == "concise":
            style = ResponseStyle.CONCISE
        elif user_preferences.get("response_length") == "detailed":
            style = ResponseStyle.DETAILED

        if user_preferences.get("technical_level") == "high":
            style = ResponseStyle.TECHNICAL

        return style

    def detect_and_integrate_emotion(
        self,
        message: str,
        user_id: Optional[str] = None
    ) -> EmotionalDetection:
        """
        Detect emotional tone from a message and integrate with EI system.

        Args:
            message: The user's message
            user_id: Optional user ID for pattern tracking

        Returns:
            EmotionalDetection result
        """
        if not self.config.emotional_intelligence_enabled:
            # Return neutral if EI is disabled
            return EmotionalDetection(
                primary_tone=EmotionalTone.NEUTRAL,
                confidence=0.5,
                indicators=[]
            )

        return self.emotional_intelligence.detect_tone(message, user_id)

    def get_emotional_trend(
        self,
        user_id: str,
        days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get emotional trend analysis for a user.

        Args:
            user_id: User identifier
            days: Number of days to analyze

        Returns:
            Emotional trend analysis
        """
        if not self.config.emotional_intelligence_enabled:
            return {
                "dominant_tone": None,
                "tone_distribution": {},
                "trend": "unknown",
                "recent_shift": False
            }

        return self.emotional_intelligence.analyze_emotional_trend(
            user_id,
            days
        )
