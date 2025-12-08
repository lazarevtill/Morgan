"""
Personality System module for the Empathic Engine.

This module provides consistent personality traits and roleplay configuration
to make Morgan feel like a consistent, natural person across conversations.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class PersonalityTrait(str, Enum):
    """Core personality traits."""
    WARMTH = "warmth"  # How warm and friendly
    FORMALITY = "formality"  # How formal vs casual
    HUMOR = "humor"  # How much humor to use
    EMPATHY = "empathy"  # How empathetic
    ENTHUSIASM = "enthusiasm"  # How enthusiastic
    DIRECTNESS = "directness"  # How direct vs indirect
    CURIOSITY = "curiosity"  # How curious and inquisitive


class ConversationalStyle(str, Enum):
    """Conversational style options."""
    CASUAL = "casual"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    WARM = "warm"
    PLAYFUL = "playful"


@dataclass
class PersonalityConfig:
    """Configuration for Morgan's personality."""
    name: str = "Morgan"
    traits: Dict[PersonalityTrait, float] = field(default_factory=dict)  # 0.0 to 1.0
    conversational_style: ConversationalStyle = ConversationalStyle.FRIENDLY
    roleplay_description: Optional[str] = None
    background: Optional[str] = None
    interests: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Set default trait values if not provided."""
        default_traits = {
            PersonalityTrait.WARMTH: 0.8,
            PersonalityTrait.FORMALITY: 0.3,
            PersonalityTrait.HUMOR: 0.6,
            PersonalityTrait.EMPATHY: 0.9,
            PersonalityTrait.ENTHUSIASM: 0.7,
            PersonalityTrait.DIRECTNESS: 0.6,
            PersonalityTrait.CURIOSITY: 0.7,
        }
        for trait, value in default_traits.items():
            if trait not in self.traits:
                self.traits[trait] = value


@dataclass
class PersonalityApplication:
    """Result of applying personality to a response."""
    adjusted_response: str
    traits_applied: Dict[PersonalityTrait, float]
    style_notes: List[str] = field(default_factory=list)


class PersonalitySystem:
    """
    Personality system for maintaining consistent personality traits.
    
    This class provides:
    - Base roleplay configuration defining Morgan's personality traits
    - Consistent personality across conversations
    - Adaptive behavior based on relationship depth
    - Natural conversational style (not robotic)
    """
    
    def __init__(self, config: Optional[PersonalityConfig] = None):
        """
        Initialize the personality system.
        
        Args:
            config: Personality configuration. If None, uses default.
        """
        self.config = config or PersonalityConfig()
        self._conversation_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def get_trait(self, trait: PersonalityTrait) -> float:
        """
        Get the value of a specific personality trait.
        
        Args:
            trait: The trait to retrieve
            
        Returns:
            Trait value between 0.0 and 1.0
        """
        return self.config.traits.get(trait, 0.5)
    
    def set_trait(self, trait: PersonalityTrait, value: float) -> None:
        """
        Set the value of a specific personality trait.
        
        Args:
            trait: The trait to set
            value: Value between 0.0 and 1.0
            
        Raises:
            ValueError: If value is not between 0.0 and 1.0
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Trait value must be between 0.0 and 1.0, got {value}")
        self.config.traits[trait] = value
    
    def apply_personality(
        self,
        response: str,
        user_id: Optional[str] = None,
        relationship_depth: float = 0.0
    ) -> PersonalityApplication:
        """
        Apply personality traits to a response.
        
        Args:
            response: The base response to adjust
            user_id: Optional user ID for conversation tracking
            relationship_depth: Relationship depth (0.0 to 1.0) for adaptive behavior
            
        Returns:
            PersonalityApplication with adjusted response and metadata
        """
        adjusted = response
        traits_applied = {}
        style_notes = []
        
        # Apply warmth
        warmth = self.get_trait(PersonalityTrait.WARMTH)
        if warmth > 0.7:
            style_notes.append("warm_tone")
        traits_applied[PersonalityTrait.WARMTH] = warmth
        
        # Apply formality
        formality = self.get_trait(PersonalityTrait.FORMALITY)
        if formality < 0.4:
            style_notes.append("casual_language")
        elif formality > 0.7:
            style_notes.append("formal_language")
        traits_applied[PersonalityTrait.FORMALITY] = formality
        
        # Apply empathy
        empathy = self.get_trait(PersonalityTrait.EMPATHY)
        if empathy > 0.7:
            style_notes.append("empathetic_approach")
        traits_applied[PersonalityTrait.EMPATHY] = empathy
        
        # Apply enthusiasm
        enthusiasm = self.get_trait(PersonalityTrait.ENTHUSIASM)
        if enthusiasm > 0.7:
            style_notes.append("enthusiastic_tone")
        traits_applied[PersonalityTrait.ENTHUSIASM] = enthusiasm
        
        # Adapt based on relationship depth
        if relationship_depth > 0.5:
            style_notes.append("familiar_tone")
        if relationship_depth > 0.8:
            style_notes.append("close_relationship")
        
        # Track conversation for consistency
        if user_id:
            self._track_interaction(user_id, response, traits_applied)
        
        return PersonalityApplication(
            adjusted_response=adjusted,
            traits_applied=traits_applied,
            style_notes=style_notes
        )
    
    def get_personality_prompt(self, relationship_depth: float = 0.0) -> str:
        """
        Generate a personality prompt for the LLM.
        
        Args:
            relationship_depth: Relationship depth (0.0 to 1.0) for adaptive behavior
            
        Returns:
            Prompt describing Morgan's personality
        """
        traits = []
        
        # Warmth
        warmth = self.get_trait(PersonalityTrait.WARMTH)
        if warmth > 0.7:
            traits.append("warm and caring")
        elif warmth > 0.4:
            traits.append("friendly")
        
        # Formality
        formality = self.get_trait(PersonalityTrait.FORMALITY)
        if formality < 0.4:
            traits.append("casual and relaxed")
        elif formality > 0.7:
            traits.append("professional and formal")
        
        # Humor
        humor = self.get_trait(PersonalityTrait.HUMOR)
        if humor > 0.6:
            traits.append("with a good sense of humor")
        
        # Empathy
        empathy = self.get_trait(PersonalityTrait.EMPATHY)
        if empathy > 0.7:
            traits.append("deeply empathetic")
        
        # Enthusiasm
        enthusiasm = self.get_trait(PersonalityTrait.ENTHUSIASM)
        if enthusiasm > 0.7:
            traits.append("enthusiastic and energetic")
        
        # Directness
        directness = self.get_trait(PersonalityTrait.DIRECTNESS)
        if directness > 0.7:
            traits.append("direct and straightforward")
        elif directness < 0.4:
            traits.append("gentle and indirect")
        
        # Curiosity
        curiosity = self.get_trait(PersonalityTrait.CURIOSITY)
        if curiosity > 0.7:
            traits.append("curious and inquisitive")
        
        # Build prompt
        trait_desc = ", ".join(traits) if traits else "balanced and adaptable"
        
        prompt = f"You are {self.config.name}, a {trait_desc} AI assistant."
        
        if self.config.roleplay_description:
            prompt += f" {self.config.roleplay_description}"
        
        if self.config.background:
            prompt += f" Background: {self.config.background}"
        
        if self.config.interests:
            interests_str = ", ".join(self.config.interests)
            prompt += f" You're interested in: {interests_str}."
        
        # Adapt based on relationship
        if relationship_depth > 0.5:
            prompt += " You have an established relationship with this user and can be more familiar."
        if relationship_depth > 0.8:
            prompt += " You have a close relationship with this user and can be very personal and warm."
        
        prompt += f" Your conversational style is {self.config.conversational_style.value}."
        
        return prompt
    
    def is_consistent_with_previous(
        self,
        user_id: str,
        current_traits: Dict[PersonalityTrait, float],
        tolerance: float = 0.2
    ) -> bool:
        """
        Check if current traits are consistent with previous interactions.
        
        Args:
            user_id: User ID to check consistency for
            current_traits: Current trait values to check
            tolerance: Maximum allowed difference (0.0 to 1.0)
            
        Returns:
            True if consistent, False otherwise
        """
        if user_id not in self._conversation_history:
            return True  # No history, so consistent by default
        
        history = self._conversation_history[user_id]
        if not history:
            return True
        
        # Get the most recent interaction's traits
        last_traits = history[-1].get("traits", {})
        
        # Check each trait
        for trait, value in current_traits.items():
            if trait in last_traits:
                diff = abs(value - last_traits[trait])
                if diff > tolerance:
                    return False
        
        return True
    
    def get_conversation_count(self, user_id: str) -> int:
        """
        Get the number of tracked interactions for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Number of interactions
        """
        return len(self._conversation_history.get(user_id, []))
    
    def _track_interaction(
        self,
        user_id: str,
        response: str,
        traits: Dict[PersonalityTrait, float]
    ) -> None:
        """
        Track an interaction for consistency checking.
        
        Args:
            user_id: User ID
            response: The response given
            traits: Traits applied in this interaction
        """
        if user_id not in self._conversation_history:
            self._conversation_history[user_id] = []
        
        self._conversation_history[user_id].append({
            "timestamp": datetime.now(),
            "response": response,
            "traits": traits.copy()
        })
        
        # Keep only last 100 interactions per user
        if len(self._conversation_history[user_id]) > 100:
            self._conversation_history[user_id] = self._conversation_history[user_id][-100:]
