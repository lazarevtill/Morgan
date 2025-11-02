"""
Emotional validation response module.

Provides emotional validation responses that acknowledge and validate user emotions,
helping users feel heard and understood in their emotional experiences.
"""

import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

from morgan.config import get_settings
from morgan.utils.logger import get_logger
from morgan.services.llm_service import get_llm_service
from morgan.emotional.models import (
    EmotionalState, EmotionType, ConversationContext
)

logger = get_logger(__name__)


class EmotionalValidator:
    """
    Emotional validation response generator.
    
    Provides validation responses that:
    - Acknowledge user emotions without judgment
    - Validate the user's right to feel their emotions
    - Normalize emotional experiences
    - Provide emotional support and understanding
    """
    
    # Validation response templates by emotion type
    VALIDATION_TEMPLATES = {
        EmotionType.JOY: [
            "It's wonderful to see you feeling so happy! Your joy is completely valid and beautiful.",
            "I can feel your excitement, and it's absolutely delightful. You deserve to feel this good!",
            "Your happiness is infectious! It's perfectly natural to feel this way about something meaningful to you."
        ],
        EmotionType.SADNESS: [
            "I can sense the sadness you're experiencing, and it's completely okay to feel this way.",
            "Your feelings are valid and understandable. It's natural to feel sad when things are difficult.",
            "I want you to know that feeling sad is a normal human response, and you're not alone in this."
        ],
        EmotionType.ANGER: [
            "I can understand why you're feeling angry. Your frustration is completely valid.",
            "It's natural to feel angry when things don't go as expected. Your feelings make perfect sense.",
            "Your anger is a valid response to this situation. It's okay to feel frustrated."
        ],
        EmotionType.FEAR: [
            "Feeling scared or worried is a completely normal human response. Your concerns are valid.",
            "It's understandable to feel anxious about this. Fear is our mind's way of trying to protect us.",
            "Your worries are real and valid. It's natural to feel uncertain in situations like this."
        ],
        EmotionType.SURPRISE: [
            "Being surprised is such a natural reaction! It's wonderful when life brings unexpected moments.",
            "Your surprise is completely understandable. Unexpected things can really catch us off guard.",
            "It's perfectly normal to feel surprised by this. Life has a way of bringing unexpected turns."
        ],
        EmotionType.DISGUST: [
            "I can understand why this would bother you. Your reaction is completely valid.",
            "It's natural to feel uncomfortable about things that don't align with your values.",
            "Your feelings of discomfort are valid and understandable in this situation."
        ],
        EmotionType.NEUTRAL: [
            "I appreciate you sharing your thoughts with me. All feelings are valid, including feeling neutral.",
            "It's perfectly okay to feel calm or neutral about things. Not everything needs to evoke strong emotions.",
            "Your balanced perspective is valuable. Sometimes a neutral stance is exactly what's needed."
        ]
    }
    
    # Intensity-based validation modifiers
    INTENSITY_MODIFIERS = {
        "high": [
            "I can feel the intensity of your emotions, and that's completely okay.",
            "Strong emotions like this are a sign of how much this matters to you.",
            "The depth of your feelings shows how significant this is for you."
        ],
        "medium": [
            "Your feelings are important and deserve to be acknowledged.",
            "It's natural to have these kinds of emotional responses.",
            "Your emotional reaction is completely understandable."
        ],
        "low": [
            "Even subtle emotions are important and valid.",
            "Sometimes gentle feelings can be just as meaningful.",
            "Your quiet emotional response is perfectly valid."
        ]
    }
    
    def __init__(self):
        """Initialize emotional validator."""
        self.settings = get_settings()
        self.llm_service = get_llm_service()
        
        logger.info("Emotional Validator initialized")
    
    def generate_validation_response(
        self,
        emotional_state: EmotionalState,
        context: ConversationContext,
        personalization_level: float = 0.7
    ) -> str:
        """
        Generate an emotional validation response.
        
        Args:
            emotional_state: User's current emotional state
            context: Conversation context
            personalization_level: How personalized to make the response (0.0-1.0)
            
        Returns:
            Validation response text
        """
        try:
            # Get base validation template
            base_validation = self._get_base_validation(emotional_state)
            
            # Add intensity-appropriate modifier
            intensity_modifier = self._get_intensity_modifier(emotional_state.intensity)
            
            # Generate personalized validation if requested
            if personalization_level > 0.5:
                personalized_validation = self._generate_personalized_validation(
                    emotional_state, context, base_validation
                )
                if personalized_validation:
                    return personalized_validation
            
            # Combine base validation with intensity modifier
            if intensity_modifier and emotional_state.intensity > 0.3:
                return f"{base_validation} {intensity_modifier}"
            
            return base_validation
            
        except Exception as e:
            logger.warning(f"Failed to generate validation response: {e}")
            return self._get_fallback_validation(emotional_state)
    
    def validate_emotional_experience(
        self,
        emotional_state: EmotionalState,
        user_description: str
    ) -> Dict[str, Any]:
        """
        Provide comprehensive emotional validation.
        
        Args:
            emotional_state: Detected emotional state
            user_description: User's description of their experience
            
        Returns:
            Validation analysis with multiple validation approaches
        """
        validation_analysis = {
            "primary_validation": self._validate_primary_emotion(emotional_state),
            "intensity_validation": self._validate_emotion_intensity(emotional_state),
            "context_validation": self._validate_emotional_context(user_description),
            "normalization": self._normalize_emotional_experience(emotional_state),
            "support_message": self._generate_support_message(emotional_state),
            "validation_confidence": self._calculate_validation_confidence(emotional_state)
        }
        
        return validation_analysis
    
    def create_affirmation_response(
        self,
        emotional_state: EmotionalState,
        specific_concern: Optional[str] = None
    ) -> str:
        """
        Create an affirmation response for the user's emotional state.
        
        Args:
            emotional_state: User's emotional state
            specific_concern: Specific concern to address
            
        Returns:
            Affirmation response text
        """
        affirmations = {
            EmotionType.JOY: [
                "You deserve all the happiness you're feeling right now.",
                "Your joy is a gift to yourself and those around you.",
                "It's beautiful to see you embracing this positive moment."
            ],
            EmotionType.SADNESS: [
                "You are strong, even when you don't feel like it.",
                "It's okay to not be okay sometimes. You're still worthy of love and support.",
                "Your sadness doesn't define you - it's just what you're experiencing right now."
            ],
            EmotionType.ANGER: [
                "Your anger shows that you care deeply about what's right.",
                "It's okay to feel angry when your boundaries are crossed.",
                "You have the right to feel frustrated when things aren't fair."
            ],
            EmotionType.FEAR: [
                "You are braver than you believe and stronger than you feel.",
                "It's okay to feel scared - courage isn't the absence of fear.",
                "You've overcome challenges before, and you can do it again."
            ],
            EmotionType.SURPRISE: [
                "Life's surprises show how full of possibilities the world is.",
                "Your openness to unexpected experiences is a wonderful quality.",
                "Surprise moments often lead to the most meaningful experiences."
            ],
            EmotionType.DISGUST: [
                "Your moral compass is working - it's good to feel uncomfortable with wrong things.",
                "Having strong values that guide your reactions is admirable.",
                "Your sense of what's right and wrong is an important part of who you are."
            ],
            EmotionType.NEUTRAL: [
                "Your balanced perspective is a strength in itself.",
                "Sometimes the wisest response is a calm, measured one.",
                "Your ability to stay centered is valuable in chaotic times."
            ]
        }
        
        base_affirmations = affirmations.get(emotional_state.primary_emotion, [
            "You are valued and your feelings matter.",
            "You have the strength to handle whatever comes your way.",
            "Your emotional experience is valid and important."
        ])
        
        # Select affirmation based on intensity
        if emotional_state.intensity > 0.7:
            # High intensity - use more supportive affirmation
            selected_affirmation = base_affirmations[0] if base_affirmations else "You are strong and capable."
        else:
            # Lower intensity - use gentler affirmation
            selected_affirmation = base_affirmations[-1] if base_affirmations else "Your feelings are valid."
        
        # Add specific concern addressing if provided
        if specific_concern:
            concern_response = self._address_specific_concern(specific_concern, emotional_state)
            if concern_response:
                selected_affirmation = f"{selected_affirmation} {concern_response}"
        
        return selected_affirmation
    
    def _get_base_validation(self, emotional_state: EmotionalState) -> str:
        """Get base validation response for emotion type."""
        templates = self.VALIDATION_TEMPLATES.get(emotional_state.primary_emotion, [])
        if not templates:
            return "Your feelings are completely valid and understandable."
        
        # Select template based on confidence
        if emotional_state.confidence > 0.8:
            return templates[0]  # Most confident template
        elif emotional_state.confidence > 0.5:
            return templates[1] if len(templates) > 1 else templates[0]
        else:
            return templates[-1] if len(templates) > 2 else templates[0]
    
    def _get_intensity_modifier(self, intensity: float) -> Optional[str]:
        """Get intensity-appropriate modifier."""
        if intensity > 0.7:
            modifiers = self.INTENSITY_MODIFIERS["high"]
        elif intensity > 0.4:
            modifiers = self.INTENSITY_MODIFIERS["medium"]
        else:
            modifiers = self.INTENSITY_MODIFIERS["low"]
        
        # Return first modifier for consistency
        return modifiers[0] if modifiers else None
    
    def _generate_personalized_validation(
        self,
        emotional_state: EmotionalState,
        context: ConversationContext,
        base_validation: str
    ) -> Optional[str]:
        """Generate personalized validation using LLM."""
        try:
            prompt = f"""
            Create a personalized emotional validation response for a user experiencing {emotional_state.primary_emotion.value} 
            with intensity {emotional_state.intensity:.1f}.
            
            User's message: "{context.message_text}"
            Base validation: "{base_validation}"
            
            Guidelines:
            - Acknowledge their specific emotional experience
            - Validate their right to feel this way
            - Be warm, understanding, and non-judgmental
            - Reference their specific situation if appropriate
            - Keep it natural and conversational
            - Make it feel personal and genuine
            
            Personalized validation response:
            """
            
            response = self.llm_service.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=120
            )
            
            return response.content.strip()
            
        except Exception as e:
            logger.warning(f"Failed to generate personalized validation: {e}")
            return None
    
    def _validate_primary_emotion(self, emotional_state: EmotionalState) -> str:
        """Validate the primary emotion."""
        emotion_validations = {
            EmotionType.JOY: "Feeling joy and happiness is one of life's greatest gifts. Your positive emotions are wonderful.",
            EmotionType.SADNESS: "Sadness is a natural and healthy emotional response. It shows your capacity for deep feeling.",
            EmotionType.ANGER: "Anger can be a healthy emotion that signals when something needs attention or change.",
            EmotionType.FEAR: "Fear is a protective emotion that helps us navigate uncertain situations. It's completely normal.",
            EmotionType.SURPRISE: "Surprise keeps life interesting and shows your openness to new experiences.",
            EmotionType.DISGUST: "Feeling disgusted can be your moral compass working to protect your values.",
            EmotionType.NEUTRAL: "Feeling neutral or calm is perfectly valid. Not every moment needs intense emotion."
        }
        
        return emotion_validations.get(
            emotional_state.primary_emotion,
            "Your emotional response is completely natural and valid."
        )
    
    def _validate_emotion_intensity(self, emotional_state: EmotionalState) -> str:
        """Validate the intensity of the emotion."""
        if emotional_state.intensity > 0.8:
            return "The intensity of your feelings shows how much this matters to you, and that's completely okay."
        elif emotional_state.intensity > 0.5:
            return "The level of emotion you're experiencing is perfectly normal for this situation."
        else:
            return "Even gentle emotions are meaningful and deserve acknowledgment."
    
    def _validate_emotional_context(self, user_description: str) -> str:
        """Validate the emotional context from user description."""
        # Simple keyword-based context validation
        if any(word in user_description.lower() for word in ["difficult", "hard", "challenging"]):
            return "Difficult situations naturally bring up strong emotions. Your response is understandable."
        elif any(word in user_description.lower() for word in ["exciting", "amazing", "wonderful"]):
            return "Positive experiences deserve to be celebrated and felt fully."
        elif any(word in user_description.lower() for word in ["confused", "uncertain", "unsure"]):
            return "Uncertainty can bring up many emotions. It's natural to feel this way when things are unclear."
        else:
            return "Your emotional response to this situation makes complete sense."
    
    def _normalize_emotional_experience(self, emotional_state: EmotionalState) -> str:
        """Normalize the emotional experience."""
        normalizations = {
            EmotionType.JOY: "Many people experience joy in similar ways. You're not alone in feeling this happiness.",
            EmotionType.SADNESS: "Sadness is one of the most universal human experiences. You're not alone in feeling this way.",
            EmotionType.ANGER: "Anger is a common response to frustration or injustice. Many people would feel similarly.",
            EmotionType.FEAR: "Fear and anxiety are among the most common human emotions. You're definitely not alone.",
            EmotionType.SURPRISE: "Being surprised is a natural human reaction to unexpected events.",
            EmotionType.DISGUST: "Having strong reactions to things that conflict with your values is completely normal.",
            EmotionType.NEUTRAL: "Feeling calm or neutral is a healthy emotional state that many people experience."
        }
        
        return normalizations.get(
            emotional_state.primary_emotion,
            "Your emotional experience is shared by many people in similar situations."
        )
    
    def _generate_support_message(self, emotional_state: EmotionalState) -> str:
        """Generate supportive message based on emotion."""
        support_messages = {
            EmotionType.JOY: "I'm here to celebrate these positive moments with you.",
            EmotionType.SADNESS: "I'm here to support you through this difficult time.",
            EmotionType.ANGER: "I'm here to help you work through these frustrating feelings.",
            EmotionType.FEAR: "I'm here to provide reassurance and support as you navigate this uncertainty.",
            EmotionType.SURPRISE: "I'm here to help you process this unexpected experience.",
            EmotionType.DISGUST: "I'm here to support you in dealing with this uncomfortable situation.",
            EmotionType.NEUTRAL: "I'm here whenever you need someone to talk to."
        }
        
        return support_messages.get(
            emotional_state.primary_emotion,
            "I'm here to support you through whatever you're experiencing."
        )
    
    def _calculate_validation_confidence(self, emotional_state: EmotionalState) -> float:
        """Calculate confidence in validation approach."""
        # Base confidence on emotion detection confidence
        base_confidence = emotional_state.confidence
        
        # Adjust based on emotion type (some are easier to validate)
        if emotional_state.primary_emotion in [EmotionType.JOY, EmotionType.SADNESS]:
            base_confidence += 0.1  # These are easier to validate
        elif emotional_state.primary_emotion == EmotionType.NEUTRAL:
            base_confidence -= 0.1  # Neutral is harder to validate meaningfully
        
        return min(1.0, max(0.0, base_confidence))
    
    def _address_specific_concern(self, concern: str, emotional_state: EmotionalState) -> Optional[str]:
        """Address a specific concern in the validation."""
        concern_lower = concern.lower()
        
        if "wrong" in concern_lower or "bad" in concern_lower:
            return "There's nothing wrong with feeling this way."
        elif "weak" in concern_lower:
            return "Having emotions doesn't make you weak - it makes you human."
        elif "overreacting" in concern_lower:
            return "Your reaction is proportionate to what you're experiencing."
        elif "alone" in concern_lower:
            return "You're not alone in feeling this way."
        else:
            return None
    
    def _get_fallback_validation(self, emotional_state: EmotionalState) -> str:
        """Get fallback validation response."""
        return f"Your feelings of {emotional_state.primary_emotion.value} are completely valid and understandable. I'm here to support you."


# Singleton instance
_emotional_validator_instance = None
_emotional_validator_lock = threading.Lock()


def get_emotional_validator() -> EmotionalValidator:
    """
    Get singleton emotional validator instance.
    
    Returns:
        Shared EmotionalValidator instance
    """
    global _emotional_validator_instance
    
    if _emotional_validator_instance is None:
        with _emotional_validator_lock:
            if _emotional_validator_instance is None:
                _emotional_validator_instance = EmotionalValidator()
    
    return _emotional_validator_instance