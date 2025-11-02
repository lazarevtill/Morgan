"""
Communication style adaptation module.

Adapts communication style based on user preferences, emotional state,
and relationship context to provide personalized interaction experiences.
"""

import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from morgan.config import get_settings
from morgan.utils.logger import get_logger
from morgan.services.llm_service import get_llm_service
from morgan.emotional.models import (
    EmotionalState, ConversationContext, CommunicationStyle,
    ResponseLength, CompanionProfile
)

logger = get_logger(__name__)


class CommunicationTone(Enum):
    """Communication tone options."""
    FORMAL = "formal"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    WARM = "warm"
    SUPPORTIVE = "supportive"
    ENCOURAGING = "encouraging"
    EMPATHETIC = "empathetic"


class CommunicationComplexity(Enum):
    """Communication complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    DETAILED = "detailed"
    TECHNICAL = "technical"


@dataclass
class StyleAdaptationResult:
    """Result of communication style adaptation."""
    adapted_style: CommunicationStyle
    tone: CommunicationTone
    complexity: CommunicationComplexity
    personalization_elements: List[str]
    confidence_score: float
    adaptation_reasoning: str


class CommunicationStyleAdapter:
    """
    Communication style adaptation system.
    
    Features:
    - Dynamic style adaptation based on user preferences
    - Emotional state-aware communication adjustments
    - Relationship context integration
    - Cultural sensitivity in communication
    - Learning from user feedback and interactions
    """
    
    def __init__(self):
        """Initialize communication style adapter."""
        self.settings = get_settings()
        self.llm_service = get_llm_service()
        
        # Style adaptation history for learning
        self.adaptation_history: Dict[str, List[StyleAdaptationResult]] = {}
        
        logger.info("Communication Style Adapter initialized")
    
    def adapt_communication_style(
        self,
        user_id: str,
        emotional_state: EmotionalState,
        context: ConversationContext,
        companion_profile: Optional[CompanionProfile] = None,
        message_content: str = ""
    ) -> StyleAdaptationResult:
        """
        Adapt communication style based on user context and emotional state.
        
        Args:
            user_id: User identifier
            emotional_state: Current emotional state of user
            context: Conversation context
            companion_profile: User's companion profile
            message_content: Content to be communicated
            
        Returns:
            Style adaptation result with recommendations
        """
        # Analyze current communication needs
        base_style = self._determine_base_style(
            emotional_state, context, companion_profile
        )
        
        # Adapt tone based on emotional state
        adapted_tone = self._adapt_tone_for_emotion(
            emotional_state, base_style
        )
        
        # Determine complexity level
        complexity = self._determine_complexity_level(
            context, companion_profile, message_content
        )
        
        # Apply personalization elements
        personalization_elements = self._identify_personalization_elements(
            user_id, emotional_state, context, companion_profile
        )
        
        # Create adapted style
        adapted_style = self._create_adapted_style(
            base_style, adapted_tone, complexity, personalization_elements
        )
        
        # Generate adaptation reasoning
        reasoning = self._generate_adaptation_reasoning(
            emotional_state, context, adapted_tone, complexity
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_adaptation_confidence(
            emotional_state, context, companion_profile
        )
        
        result = StyleAdaptationResult(
            adapted_style=adapted_style,
            tone=adapted_tone,
            complexity=complexity,
            personalization_elements=personalization_elements,
            confidence_score=confidence_score,
            adaptation_reasoning=reasoning
        )
        
        # Store adaptation for learning
        self._store_adaptation_result(user_id, result)
        
        logger.debug(
            f"Adapted communication style for user {user_id}: "
            f"tone={adapted_tone.value}, complexity={complexity.value}, "
            f"confidence={confidence_score:.2f}"
        )
        
        return result
    
    def learn_from_feedback(
        self,
        user_id: str,
        adaptation_result: StyleAdaptationResult,
        feedback_score: float,
        feedback_text: Optional[str] = None
    ) -> None:
        """
        Learn from user feedback on communication style adaptation.
        
        Args:
            user_id: User identifier
            adaptation_result: Previous adaptation result
            feedback_score: Feedback score (0.0-1.0)
            feedback_text: Optional feedback text
        """
        # Update adaptation history with feedback
        if user_id in self.adaptation_history:
            for result in self.adaptation_history[user_id]:
                if result == adaptation_result:
                    # Add feedback information
                    result.confidence_score = (
                        result.confidence_score * 0.7 + feedback_score * 0.3
                    )
                    break
        
        # Analyze feedback for style preferences
        if feedback_text:
            self._analyze_feedback_for_preferences(
                user_id, feedback_text, adaptation_result
            )
        
        logger.debug(
            f"Learned from feedback for user {user_id}: "
            f"score={feedback_score:.2f}"
        )
    
    def get_style_recommendations(
        self,
        user_id: str,
        emotional_state: EmotionalState,
        context: ConversationContext
    ) -> Dict[str, Any]:
        """
        Get style recommendations for current context.
        
        Args:
            user_id: User identifier
            emotional_state: Current emotional state
            context: Conversation context
            
        Returns:
            Style recommendations dictionary
        """
        # Get historical successful adaptations
        successful_adaptations = self._get_successful_adaptations(user_id)
        
        # Analyze current context requirements
        context_requirements = self._analyze_context_requirements(
            emotional_state, context
        )
        
        # Generate recommendations
        recommendations = {
            "recommended_tone": self._recommend_tone(
                emotional_state, successful_adaptations
            ),
            "recommended_complexity": self._recommend_complexity(
                context, successful_adaptations
            ),
            "personalization_suggestions": self._suggest_personalization(
                user_id, context
            ),
            "emotional_considerations": self._get_emotional_considerations(
                emotional_state
            ),
            "context_factors": context_requirements
        }
        
        return recommendations
    
    def _determine_base_style(
        self,
        emotional_state: EmotionalState,
        context: ConversationContext,
        companion_profile: Optional[CompanionProfile]
    ) -> CommunicationStyle:
        """Determine base communication style."""
        # Start with user's preferred style if available
        if companion_profile and companion_profile.communication_preferences:
            base_style = companion_profile.communication_preferences.communication_style
        else:
            base_style = CommunicationStyle.FRIENDLY  # Default
        
        # Adjust based on emotional state
        if emotional_state.primary_emotion.value in ["sadness", "fear"]:
            base_style = CommunicationStyle.SUPPORTIVE
        elif emotional_state.primary_emotion.value == "anger":
            base_style = CommunicationStyle.CALM
        elif emotional_state.primary_emotion.value == "joy":
            base_style = CommunicationStyle.WARM
        
        return base_style
    
    def _adapt_tone_for_emotion(
        self,
        emotional_state: EmotionalState,
        base_style: CommunicationStyle
    ) -> CommunicationTone:
        """Adapt communication tone based on emotional state."""
        emotion_tone_map = {
            "joy": CommunicationTone.WARM,
            "sadness": CommunicationTone.SUPPORTIVE,
            "anger": CommunicationTone.EMPATHETIC,
            "fear": CommunicationTone.SUPPORTIVE,
            "surprise": CommunicationTone.ENCOURAGING,
            "disgust": CommunicationTone.EMPATHETIC,
            "neutral": CommunicationTone.FRIENDLY
        }
        
        # Get emotion-based tone
        emotion_tone = emotion_tone_map.get(
            emotional_state.primary_emotion.value,
            CommunicationTone.FRIENDLY
        )
        
        # Adjust intensity based on emotional intensity
        if emotional_state.intensity > 0.7:
            # High intensity emotions need more supportive tone
            if emotion_tone in [CommunicationTone.SUPPORTIVE, CommunicationTone.EMPATHETIC]:
                return emotion_tone
            else:
                return CommunicationTone.SUPPORTIVE
        
        return emotion_tone
    
    def _determine_complexity_level(
        self,
        context: ConversationContext,
        companion_profile: Optional[CompanionProfile],
        message_content: str
    ) -> CommunicationComplexity:
        """Determine appropriate complexity level."""
        # Start with user preference if available
        if companion_profile and companion_profile.communication_preferences:
            pref_length = companion_profile.communication_preferences.preferred_response_length
            if pref_length == ResponseLength.BRIEF:
                return CommunicationComplexity.SIMPLE
            elif pref_length == ResponseLength.DETAILED:
                return CommunicationComplexity.DETAILED
        
        # Analyze message complexity
        message_length = len(context.message_text)
        if message_length > 200:
            return CommunicationComplexity.DETAILED
        elif message_length < 50:
            return CommunicationComplexity.SIMPLE
        
        # Check for technical content
        technical_indicators = [
            "algorithm", "function", "code", "technical", "implementation",
            "architecture", "system", "database", "API", "framework"
        ]
        
        if any(indicator in context.message_text.lower() for indicator in technical_indicators):
            return CommunicationComplexity.TECHNICAL
        
        return CommunicationComplexity.MODERATE
    
    def _identify_personalization_elements(
        self,
        user_id: str,
        emotional_state: EmotionalState,
        context: ConversationContext,
        companion_profile: Optional[CompanionProfile]
    ) -> List[str]:
        """Identify personalization elements for communication."""
        elements = []
        
        # Emotional personalization
        elements.append(f"emotion_aware:{emotional_state.primary_emotion.value}")
        
        if emotional_state.intensity > 0.7:
            elements.append("high_emotional_intensity")
        
        # Relationship personalization
        if companion_profile:
            if companion_profile.preferred_name != "friend":
                elements.append(f"personal_name:{companion_profile.preferred_name}")
            
            if companion_profile.trust_level > 0.7:
                elements.append("high_trust_relationship")
            
            if companion_profile.interaction_count > 10:
                elements.append("established_relationship")
        
        # Context personalization
        if len(context.message_text) > 150:
            elements.append("detailed_context_provided")
        
        if context.previous_messages:
            elements.append("conversation_history_available")
        
        return elements
    
    def _create_adapted_style(
        self,
        base_style: CommunicationStyle,
        tone: CommunicationTone,
        complexity: CommunicationComplexity,
        personalization_elements: List[str]
    ) -> CommunicationStyle:
        """Create adapted communication style."""
        # For now, return the base style
        # In a more complex implementation, this would create a new style
        # that combines all the adaptation factors
        return base_style
    
    def _generate_adaptation_reasoning(
        self,
        emotional_state: EmotionalState,
        context: ConversationContext,
        tone: CommunicationTone,
        complexity: CommunicationComplexity
    ) -> str:
        """Generate reasoning for style adaptation."""
        reasoning_parts = []
        
        # Emotional reasoning
        reasoning_parts.append(
            f"Adapted for {emotional_state.primary_emotion.value} emotion "
            f"with {emotional_state.intensity:.1f} intensity"
        )
        
        # Tone reasoning
        reasoning_parts.append(f"Selected {tone.value} tone")
        
        # Complexity reasoning
        reasoning_parts.append(f"Using {complexity.value} complexity level")
        
        # Context reasoning
        if len(context.message_text) > 100:
            reasoning_parts.append("Detailed context provided by user")
        
        return "; ".join(reasoning_parts)
    
    def _calculate_adaptation_confidence(
        self,
        emotional_state: EmotionalState,
        context: ConversationContext,
        companion_profile: Optional[CompanionProfile]
    ) -> float:
        """Calculate confidence in style adaptation."""
        confidence_factors = []
        
        # Emotional state confidence
        confidence_factors.append(emotional_state.confidence)
        
        # Context richness
        if len(context.message_text) > 50:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        # Profile availability
        if companion_profile:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
        
        # Relationship maturity
        if companion_profile and companion_profile.interaction_count > 5:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _store_adaptation_result(
        self,
        user_id: str,
        result: StyleAdaptationResult
    ) -> None:
        """Store adaptation result for learning."""
        if user_id not in self.adaptation_history:
            self.adaptation_history[user_id] = []
        
        self.adaptation_history[user_id].append(result)
        
        # Keep only recent adaptations (last 50)
        if len(self.adaptation_history[user_id]) > 50:
            self.adaptation_history[user_id] = self.adaptation_history[user_id][-50:]
    
    def _analyze_feedback_for_preferences(
        self,
        user_id: str,
        feedback_text: str,
        adaptation_result: StyleAdaptationResult
    ) -> None:
        """Analyze feedback text for style preferences."""
        feedback_lower = feedback_text.lower()
        
        # Analyze tone preferences
        if "too formal" in feedback_lower:
            logger.debug(f"User {user_id} prefers less formal tone")
        elif "too casual" in feedback_lower:
            logger.debug(f"User {user_id} prefers more formal tone")
        
        # Analyze complexity preferences
        if "too complex" in feedback_lower or "too detailed" in feedback_lower:
            logger.debug(f"User {user_id} prefers simpler communication")
        elif "too simple" in feedback_lower or "more detail" in feedback_lower:
            logger.debug(f"User {user_id} prefers more detailed communication")
    
    def _get_successful_adaptations(
        self,
        user_id: str
    ) -> List[StyleAdaptationResult]:
        """Get historically successful adaptations for user."""
        if user_id not in self.adaptation_history:
            return []
        
        # Return adaptations with high confidence scores
        return [
            result for result in self.adaptation_history[user_id]
            if result.confidence_score > 0.7
        ]
    
    def _analyze_context_requirements(
        self,
        emotional_state: EmotionalState,
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Analyze context requirements for communication."""
        requirements = {
            "emotional_support_needed": emotional_state.intensity > 0.6,
            "detailed_response_expected": len(context.message_text) > 150,
            "technical_content_present": self._has_technical_content(context.message_text),
            "urgent_tone_needed": "urgent" in context.message_text.lower(),
            "personal_sharing_detected": len(context.message_text) > 200
        }
        
        return requirements
    
    def _recommend_tone(
        self,
        emotional_state: EmotionalState,
        successful_adaptations: List[StyleAdaptationResult]
    ) -> CommunicationTone:
        """Recommend communication tone."""
        # Check successful adaptations first
        if successful_adaptations:
            tone_counts = {}
            for adaptation in successful_adaptations:
                tone = adaptation.tone
                tone_counts[tone] = tone_counts.get(tone, 0) + 1
            
            # Return most successful tone
            if tone_counts:
                return max(tone_counts.items(), key=lambda x: x[1])[0]
        
        # Fallback to emotion-based recommendation
        return self._adapt_tone_for_emotion(emotional_state, CommunicationStyle.FRIENDLY)
    
    def _recommend_complexity(
        self,
        context: ConversationContext,
        successful_adaptations: List[StyleAdaptationResult]
    ) -> CommunicationComplexity:
        """Recommend communication complexity."""
        # Check successful adaptations first
        if successful_adaptations:
            complexity_counts = {}
            for adaptation in successful_adaptations:
                complexity = adaptation.complexity
                complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
            
            # Return most successful complexity
            if complexity_counts:
                return max(complexity_counts.items(), key=lambda x: x[1])[0]
        
        # Fallback to context-based recommendation
        return self._determine_complexity_level(context, None, "")
    
    def _suggest_personalization(
        self,
        user_id: str,
        context: ConversationContext
    ) -> List[str]:
        """Suggest personalization elements."""
        suggestions = []
        
        # Context-based suggestions
        if len(context.message_text) > 100:
            suggestions.append("acknowledge_detailed_sharing")
        
        if context.previous_messages:
            suggestions.append("reference_conversation_history")
        
        # User-specific suggestions
        if user_id in self.adaptation_history:
            suggestions.append("apply_learned_preferences")
        
        return suggestions
    
    def _get_emotional_considerations(
        self,
        emotional_state: EmotionalState
    ) -> List[str]:
        """Get emotional considerations for communication."""
        considerations = []
        
        considerations.append(f"primary_emotion:{emotional_state.primary_emotion.value}")
        
        if emotional_state.intensity > 0.7:
            considerations.append("high_emotional_intensity")
        
        if emotional_state.confidence < 0.5:
            considerations.append("uncertain_emotional_state")
        
        if emotional_state.secondary_emotions:
            considerations.append("mixed_emotions_present")
        
        return considerations
    
    def _has_technical_content(self, text: str) -> bool:
        """Check if text contains technical content."""
        technical_indicators = [
            "algorithm", "function", "code", "technical", "implementation",
            "architecture", "system", "database", "API", "framework",
            "programming", "software", "development", "debugging"
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in technical_indicators)


# Singleton instance
_communication_style_adapter_instance = None
_communication_style_adapter_lock = threading.Lock()


def get_communication_style_adapter() -> CommunicationStyleAdapter:
    """
    Get singleton communication style adapter instance.
    
    Returns:
        Shared CommunicationStyleAdapter instance
    """
    global _communication_style_adapter_instance
    
    if _communication_style_adapter_instance is None:
        with _communication_style_adapter_lock:
            if _communication_style_adapter_instance is None:
                _communication_style_adapter_instance = CommunicationStyleAdapter()
    
    return _communication_style_adapter_instance