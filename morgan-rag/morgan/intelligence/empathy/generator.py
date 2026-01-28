"""
Empathetic response creation module.

Generates emotionally aware and empathetic responses that acknowledge user emotions,
provide appropriate support, and maintain relationship context.
"""

import threading
from datetime import datetime
from typing import List, Optional

from morgan.config import get_settings
from morgan.intelligence.core.models import (
    CompanionProfile,
    ConversationContext,
    EmotionalState,
    EmotionType,
    EmpatheticResponse,
)
from morgan.services.llm import get_llm_service
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class EmpatheticResponseGenerator:
    """
    Empathetic response generation system.

    Provides capabilities to:
    - Generate emotionally aware responses
    - Create empathetic acknowledgments
    - Provide contextual emotional support
    - Maintain relationship awareness in responses
    """

    # Empathy templates by emotion type
    EMPATHY_TEMPLATES = {
        EmotionType.JOY: [
            "I can feel your happiness and excitement! It's wonderful to see you feeling so positive.",
            "Your joy is really coming through, and it's beautiful to witness.",
            "I'm so glad to hear about this positive experience you're having.",
        ],
        EmotionType.SADNESS: [
            "I can sense the sadness you're carrying, and I want you to know I'm here with you.",
            "I feel the weight of what you're going through, and your pain is valid.",
            "I'm holding space for your sadness and want to support you through this.",
        ],
        EmotionType.ANGER: [
            "I can feel your frustration and anger, and I understand why you're feeling this way.",
            "Your anger makes complete sense given what you've experienced.",
            "I can sense how upset this situation has made you, and that's completely valid.",
        ],
        EmotionType.FEAR: [
            "I can sense your worry and concern, and it's natural to feel anxious about this.",
            "I feel the uncertainty you're experiencing, and I'm here to support you.",
            "Your fears are understandable, and you don't have to face them alone.",
        ],
        EmotionType.SURPRISE: [
            "I can feel your surprise and amazement at this unexpected turn of events.",
            "Your astonishment is coming through clearly, and I can understand why this caught you off guard.",
            "I sense how unexpected this was for you, and that's quite a lot to process.",
        ],
        EmotionType.DISGUST: [
            "I can sense how much this situation bothers you and conflicts with your values.",
            "I feel your discomfort with this, and your reaction is completely understandable.",
            "I can see how this goes against what feels right to you.",
        ],
        EmotionType.NEUTRAL: [
            "I appreciate you sharing your thoughts with me in such a balanced way.",
            "I can sense your thoughtful and measured approach to this situation.",
            "Your calm perspective is valuable, and I'm here to explore this with you.",
        ],
    }

    def __init__(self):
        """Initialize empathetic response generator."""
        self.settings = get_settings()
        self.llm_service = get_llm_service()

        logger.info("Empathetic Response Generator initialized")

    def generate_empathetic_response(
        self,
        emotional_state: EmotionalState,
        context: ConversationContext,
        companion_profile: Optional[CompanionProfile] = None,
        response_focus: str = "support",
    ) -> EmpatheticResponse:
        """
        Generate comprehensive empathetic response.

        Args:
            emotional_state: User's current emotional state
            context: Conversation context
            companion_profile: User's companion profile for personalization
            response_focus: Focus of response (support, validation, encouragement)

        Returns:
            Complete empathetic response
        """
        # Generate core empathetic response
        response_text = self._generate_core_empathetic_response(
            emotional_state, context, companion_profile, response_focus
        )

        # Determine emotional tone
        emotional_tone = self._determine_emotional_tone(emotional_state)

        # Calculate empathy level
        empathy_level = self._calculate_empathy_level(emotional_state, context)

        # Identify personalization elements
        personalization_elements = self._identify_personalization_elements(
            emotional_state, context, companion_profile
        )

        # Build relationship context
        relationship_context = self._build_relationship_context(
            context, companion_profile
        )

        # Calculate confidence score
        confidence_score = self._calculate_response_confidence(
            emotional_state, context, companion_profile
        )

        return EmpatheticResponse(
            response_text=response_text,
            emotional_tone=emotional_tone,
            empathy_level=empathy_level,
            personalization_elements=personalization_elements,
            relationship_context=relationship_context,
            confidence_score=confidence_score,
        )

    def create_emotional_acknowledgment(
        self, emotional_state: EmotionalState, specific_context: Optional[str] = None
    ) -> str:
        """
        Create emotional acknowledgment response.

        Args:
            emotional_state: User's emotional state to acknowledge
            specific_context: Specific context to reference

        Returns:
            Emotional acknowledgment text
        """
        # Get base empathy template
        base_template = self._get_empathy_template(emotional_state)

        # Add specific context if provided
        if specific_context:
            contextual_acknowledgment = self._create_contextual_acknowledgment(
                emotional_state, specific_context, base_template
            )
            if contextual_acknowledgment:
                return contextual_acknowledgment

        return base_template

    def generate_supportive_response(
        self,
        emotional_state: EmotionalState,
        context: ConversationContext,
        support_type: str = "emotional",
    ) -> str:
        """
        Generate supportive response based on emotional state.

        Args:
            emotional_state: User's emotional state
            context: Conversation context
            support_type: Type of support (emotional, practical, informational)

        Returns:
            Supportive response text
        """
        try:
            # Generate support using LLM
            llm_support = self._generate_llm_support(
                emotional_state, context, support_type
            )

            if llm_support:
                return llm_support

            # Fallback to template-based support
            return self._generate_template_support(emotional_state, support_type)

        except Exception as e:
            logger.warning(f"Failed to generate supportive response: {e}")
            return self._get_fallback_support(emotional_state)

    def create_relationship_aware_response(
        self,
        emotional_state: EmotionalState,
        context: ConversationContext,
        companion_profile: CompanionProfile,
        include_milestones: bool = True,
    ) -> str:
        """
        Create response that incorporates relationship history and milestones.

        Args:
            emotional_state: User's emotional state
            context: Conversation context
            companion_profile: User's companion profile
            include_milestones: Whether to reference relationship milestones

        Returns:
            Relationship-aware empathetic response
        """
        # Build relationship context
        relationship_elements = []

        # Add relationship duration awareness
        if companion_profile.relationship_duration.days > 30:
            relationship_elements.append(
                f"relationship_duration:{companion_profile.relationship_duration.days}days"
            )

        # Add interaction history awareness
        if companion_profile.interaction_count > 10:
            relationship_elements.append(
                f"interaction_history:{companion_profile.interaction_count}conversations"
            )

        # Add milestone awareness
        if include_milestones and companion_profile.relationship_milestones:
            recent_milestones = [
                m
                for m in companion_profile.relationship_milestones
                if (datetime.utcnow() - m.timestamp).days <= 7
            ]
            if recent_milestones:
                relationship_elements.append("recent_milestones:true")

        # Generate relationship-aware response
        return self._generate_relationship_response(
            emotional_state, context, relationship_elements, companion_profile
        )

    def _generate_core_empathetic_response(
        self,
        emotional_state: EmotionalState,
        context: ConversationContext,
        companion_profile: Optional[CompanionProfile],
        response_focus: str,
    ) -> str:
        """Generate core empathetic response."""
        try:
            # Build context for LLM
            llm_context = self._build_llm_context(
                emotional_state, context, companion_profile, response_focus
            )

            prompt = f"""
            Generate an empathetic response for a user experiencing {emotional_state.primary_emotion.value}
            with intensity {emotional_state.intensity:.1f}.

            Context: {llm_context}
            Response focus: {response_focus}

            Guidelines:
            - Be genuinely empathetic and emotionally aware
            - Acknowledge their emotional state appropriately
            - Provide {response_focus} that feels authentic
            - Keep response natural and conversational
            - Show emotional intelligence and understanding
            - Reference their specific situation when appropriate

            Empathetic response:
            """

            response = self.llm_service.generate(
                prompt=prompt, temperature=0.7, max_tokens=150
            )

            return response.content.strip()

        except Exception as e:
            logger.warning(f"Failed to generate core empathetic response: {e}")
            return self._get_empathy_template(emotional_state)

    def _get_empathy_template(self, emotional_state: EmotionalState) -> str:
        """Get empathy template for emotion type."""
        templates = self.EMPATHY_TEMPLATES.get(emotional_state.primary_emotion, [])
        if not templates:
            return "I can sense what you're experiencing, and I want you to know I'm here to support you."

        # Select template based on confidence
        if emotional_state.confidence > 0.8:
            return templates[0]  # Most confident template
        elif emotional_state.confidence > 0.5:
            return templates[1] if len(templates) > 1 else templates[0]
        else:
            return templates[-1] if len(templates) > 2 else templates[0]

    def _determine_emotional_tone(self, emotional_state: EmotionalState) -> str:
        """Determine appropriate emotional tone for response."""
        tone_map = {
            EmotionType.JOY: "warm and celebratory",
            EmotionType.SADNESS: "gentle and supportive",
            EmotionType.ANGER: "calm and understanding",
            EmotionType.FEAR: "reassuring and stable",
            EmotionType.SURPRISE: "curious and supportive",
            EmotionType.DISGUST: "respectful and validating",
            EmotionType.NEUTRAL: "warm and engaging",
        }

        return tone_map.get(
            emotional_state.primary_emotion, "empathetic and supportive"
        )

    def _calculate_empathy_level(
        self, emotional_state: EmotionalState, context: ConversationContext
    ) -> float:
        """Calculate appropriate empathy level."""
        # Base empathy on emotional intensity
        base_empathy = emotional_state.intensity * 0.8

        # Adjust based on emotion type
        empathy_adjustments = {
            EmotionType.SADNESS: 0.2,
            EmotionType.FEAR: 0.15,
            EmotionType.ANGER: 0.1,
            EmotionType.JOY: -0.1,  # Less empathy needed for positive emotions
            EmotionType.SURPRISE: 0.0,
            EmotionType.DISGUST: 0.05,
            EmotionType.NEUTRAL: -0.2,
        }

        adjustment = empathy_adjustments.get(emotional_state.primary_emotion, 0.0)
        empathy_level = base_empathy + adjustment

        # Adjust based on context length (more context = more empathy possible)
        if len(context.message_text) > 100:
            empathy_level += 0.1

        return min(1.0, max(0.3, empathy_level))

    def _identify_personalization_elements(
        self,
        emotional_state: EmotionalState,
        context: ConversationContext,
        companion_profile: Optional[CompanionProfile],
    ) -> List[str]:
        """Identify personalization elements for response."""
        elements = []

        # Emotional personalization
        elements.append(
            f"emotion_acknowledgment:{emotional_state.primary_emotion.value}"
        )

        if emotional_state.intensity > 0.7:
            elements.append("high_intensity_support")

        # Context personalization
        if len(context.message_text) > 150:
            elements.append("detailed_context_awareness")

        # Relationship personalization
        if companion_profile:
            if companion_profile.interaction_count > 5:
                elements.append("relationship_history_awareness")

            if companion_profile.trust_level > 0.7:
                elements.append("high_trust_relationship")

            if companion_profile.preferred_name != "friend":
                elements.append(f"personal_name:{companion_profile.preferred_name}")

        return elements

    def _build_relationship_context(
        self,
        context: ConversationContext,
        companion_profile: Optional[CompanionProfile],
    ) -> str:
        """Build relationship context string."""
        if not companion_profile:
            return "new_relationship"

        context_elements = []

        # Relationship age
        if companion_profile.relationship_duration.days > 30:
            context_elements.append("established_relationship")
        elif companion_profile.relationship_duration.days > 7:
            context_elements.append("developing_relationship")
        else:
            context_elements.append("new_relationship")

        # Trust level
        if companion_profile.trust_level > 0.8:
            context_elements.append("high_trust")
        elif companion_profile.trust_level > 0.5:
            context_elements.append("moderate_trust")
        else:
            context_elements.append("building_trust")

        # Engagement level
        if companion_profile.engagement_score > 0.7:
            context_elements.append("high_engagement")

        return "_".join(context_elements)

    def _calculate_response_confidence(
        self,
        emotional_state: EmotionalState,
        context: ConversationContext,
        companion_profile: Optional[CompanionProfile],
    ) -> float:
        """Calculate confidence in empathetic response."""
        base_confidence = emotional_state.confidence

        # Adjust based on context richness
        if len(context.message_text) > 50:
            base_confidence += 0.1

        # Adjust based on relationship knowledge
        if companion_profile and companion_profile.interaction_count > 5:
            base_confidence += 0.1

        # Adjust based on emotion clarity
        if emotional_state.primary_emotion != EmotionType.NEUTRAL:
            base_confidence += 0.05

        return min(1.0, max(0.3, base_confidence))

    def _create_contextual_acknowledgment(
        self, emotional_state: EmotionalState, specific_context: str, base_template: str
    ) -> Optional[str]:
        """Create contextual acknowledgment using LLM."""
        try:
            prompt = f"""
            Create an empathetic acknowledgment that references the user's specific context.

            User's emotion: {emotional_state.primary_emotion.value} (intensity: {emotional_state.intensity:.1f})
            Specific context: "{specific_context}"
            Base acknowledgment: "{base_template}"

            Guidelines:
            - Acknowledge their specific situation
            - Show understanding of their emotional experience
            - Be warm and empathetic
            - Keep it natural and conversational
            - Reference their context appropriately

            Contextual acknowledgment:
            """

            response = self.llm_service.generate(
                prompt=prompt, temperature=0.7, max_tokens=100
            )

            return response.content.strip()

        except Exception as e:
            logger.warning(f"Failed to create contextual acknowledgment: {e}")
            return None

    def _generate_llm_support(
        self,
        emotional_state: EmotionalState,
        context: ConversationContext,
        support_type: str,
    ) -> Optional[str]:
        """Generate supportive response using LLM."""
        try:
            prompt = f"""
            Generate a {support_type} supportive response for a user experiencing {emotional_state.primary_emotion.value}.

            User's message: "{context.message_text}"
            Emotional intensity: {emotional_state.intensity:.1f}
            Support type: {support_type}

            Guidelines:
            - Provide genuine {support_type} support
            - Be empathetic and understanding
            - Offer appropriate help or comfort
            - Keep response natural and caring
            - Match the emotional tone appropriately

            Supportive response:
            """

            response = self.llm_service.generate(
                prompt=prompt, temperature=0.7, max_tokens=120
            )

            return response.content.strip()

        except Exception as e:
            logger.warning(f"Failed to generate LLM support: {e}")
            return None

    def _generate_template_support(
        self, emotional_state: EmotionalState, support_type: str
    ) -> str:
        """Generate template-based support."""
        support_templates = {
            "emotional": {
                EmotionType.SADNESS: "I'm here to support you through this difficult time. You don't have to face this alone.",
                EmotionType.ANGER: "I understand your frustration. Let's work through this together.",
                EmotionType.FEAR: "Your concerns are valid. I'm here to help you feel more secure.",
                EmotionType.JOY: "I'm so happy to share in your joy! This is wonderful news.",
            },
            "practical": {
                EmotionType.SADNESS: "Let's think about what practical steps might help you feel better.",
                EmotionType.ANGER: "What would be most helpful in addressing this frustrating situation?",
                EmotionType.FEAR: "Let's break this down into manageable steps you can take.",
                EmotionType.JOY: "How can we build on this positive momentum?",
            },
            "informational": {
                EmotionType.SADNESS: "Would it help to explore some resources that might support you?",
                EmotionType.ANGER: "Let me share some information that might help with this situation.",
                EmotionType.FEAR: "Here's some information that might help ease your concerns.",
                EmotionType.JOY: "I'd love to share more about this positive development with you.",
            },
        }

        templates = support_templates.get(support_type, support_templates["emotional"])
        return templates.get(
            emotional_state.primary_emotion,
            f"I'm here to provide {support_type} support for whatever you're experiencing.",
        )

    def _get_fallback_support(self, emotional_state: EmotionalState) -> str:
        """Get fallback support response."""
        return f"I can sense you're experiencing {emotional_state.primary_emotion.value}, and I want you to know I'm here to support you through this."

    def _build_llm_context(
        self,
        emotional_state: EmotionalState,
        context: ConversationContext,
        companion_profile: Optional[CompanionProfile],
        response_focus: str,
    ) -> str:
        """Build context string for LLM."""
        context_parts = [
            f"User message: '{context.message_text}'",
            f"Emotional state: {emotional_state.primary_emotion.value} (intensity: {emotional_state.intensity:.1f})",
        ]

        if companion_profile:
            context_parts.append(
                f"Relationship: {companion_profile.interaction_count} interactions over {companion_profile.relationship_duration.days} days"
            )

            if companion_profile.preferred_name != "friend":
                context_parts.append(
                    f"User prefers to be called: {companion_profile.preferred_name}"
                )

        if context.previous_messages:
            context_parts.append("Recent conversation context available")

        return " | ".join(context_parts)

    def _generate_relationship_response(
        self,
        emotional_state: EmotionalState,
        context: ConversationContext,
        relationship_elements: List[str],
        companion_profile: CompanionProfile,
    ) -> str:
        """Generate relationship-aware response."""
        try:
            relationship_context = ", ".join(relationship_elements)

            prompt = f"""
            Generate an empathetic response that incorporates relationship history and context.

            User's emotion: {emotional_state.primary_emotion.value} (intensity: {emotional_state.intensity:.1f})
            User's message: "{context.message_text}"
            Relationship context: {relationship_context}
            User's preferred name: {companion_profile.preferred_name}

            Guidelines:
            - Reference your relationship history appropriately
            - Use their preferred name if it's not "friend"
            - Show awareness of your shared experiences
            - Be empathetic and emotionally intelligent
            - Keep response natural and caring
            - Don't over-reference the relationship - be subtle

            Relationship-aware response:
            """

            response = self.llm_service.generate(
                prompt=prompt, temperature=0.7, max_tokens=150
            )

            return response.content.strip()

        except Exception as e:
            logger.warning(f"Failed to generate relationship response: {e}")
            return self._get_empathy_template(emotional_state)


# Singleton instance
_empathetic_response_generator_instance = None
_empathetic_response_generator_lock = threading.Lock()


def get_empathetic_response_generator() -> EmpatheticResponseGenerator:
    """
    Get singleton empathetic response generator instance.

    Returns:
        Shared EmpatheticResponseGenerator instance
    """
    global _empathetic_response_generator_instance

    if _empathetic_response_generator_instance is None:
        with _empathetic_response_generator_lock:
            if _empathetic_response_generator_instance is None:
                _empathetic_response_generator_instance = EmpatheticResponseGenerator()

    return _empathetic_response_generator_instance
