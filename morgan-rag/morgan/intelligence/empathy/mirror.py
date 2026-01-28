"""
Emotional mirroring and reflection module.

Provides emotional mirroring capabilities that reflect user emotions back to them
in a supportive way, helping users feel understood and creating emotional connection.
"""

import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from morgan.config import get_settings
from morgan.intelligence.core.models import (
    ConversationContext,
    EmotionalState,
    EmotionType,
)
from morgan.services.llm import get_llm_service
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class EmotionalMirror:
    """
    Emotional mirroring and reflection system.

    Provides capabilities to:
    - Mirror user emotions in responses
    - Reflect emotional states back for user awareness
    - Create emotional resonance and connection
    - Help users process emotions through reflection
    """

    # Mirroring phrases by emotion type
    MIRRORING_PHRASES = {
        EmotionType.JOY: [
            "I can feel your excitement and happiness",
            "Your joy is really coming through",
            "I sense the delight in your words",
            "Your enthusiasm is wonderful to experience",
        ],
        EmotionType.SADNESS: [
            "I can sense the sadness you're carrying",
            "I feel the weight of what you're going through",
            "Your pain is coming through clearly",
            "I can feel the heaviness in your words",
        ],
        EmotionType.ANGER: [
            "I can feel your frustration and anger",
            "Your irritation is really coming through",
            "I sense how upset this has made you",
            "I can feel the intensity of your frustration",
        ],
        EmotionType.FEAR: [
            "I can sense your worry and concern",
            "I feel the anxiety you're experiencing",
            "Your nervousness is coming through",
            "I can sense how unsettled you're feeling",
        ],
        EmotionType.SURPRISE: [
            "I can feel your surprise and amazement",
            "Your astonishment is really coming through",
            "I sense how unexpected this was for you",
            "I can feel how this caught you off guard",
        ],
        EmotionType.DISGUST: [
            "I can sense how this bothers you",
            "I feel your discomfort with this situation",
            "Your distaste is coming through clearly",
            "I can sense how this conflicts with your values",
        ],
        EmotionType.NEUTRAL: [
            "I sense your calm and balanced perspective",
            "I can feel your thoughtful approach",
            "Your measured response comes through",
            "I sense your centered state of mind",
        ],
    }

    # Reflection prompts to help users explore emotions
    REFLECTION_PROMPTS = {
        EmotionType.JOY: [
            "What aspects of this situation bring you the most joy?",
            "How does this happiness feel in your body?",
            "What would you like to do with this positive energy?",
        ],
        EmotionType.SADNESS: [
            "What feels most difficult about this situation?",
            "Where do you feel this sadness most strongly?",
            "What would help you feel supported right now?",
        ],
        EmotionType.ANGER: [
            "What specifically triggered this anger for you?",
            "What boundary or value feels like it was crossed?",
            "What would help you feel heard in this situation?",
        ],
        EmotionType.FEAR: [
            "What are you most concerned might happen?",
            "What would help you feel more secure?",
            "What resources do you have to handle this?",
        ],
        EmotionType.SURPRISE: [
            "What was most unexpected about this?",
            "How are you processing this surprise?",
            "What new possibilities does this open up?",
        ],
        EmotionType.DISGUST: [
            "What values does this situation conflict with?",
            "What would need to change for this to feel right?",
            "How can you protect your boundaries here?",
        ],
        EmotionType.NEUTRAL: [
            "What's helping you maintain this balanced perspective?",
            "How does this calm state serve you?",
            "What insights are emerging from this centered place?",
        ],
    }

    def __init__(self):
        """Initialize emotional mirror."""
        self.settings = get_settings()
        self.llm_service = get_llm_service()

        logger.info("Emotional Mirror initialized")

    def mirror_emotion(
        self,
        emotional_state: EmotionalState,
        context: ConversationContext,
        mirroring_intensity: float = 0.8,
    ) -> str:
        """
        Mirror the user's emotion back to them.

        Args:
            emotional_state: User's emotional state to mirror
            context: Conversation context
            mirroring_intensity: How strongly to mirror (0.0-1.0)

        Returns:
            Mirroring response text
        """
        try:
            # Get base mirroring phrase
            base_mirror = self._get_mirroring_phrase(emotional_state)

            # Adjust intensity based on user's emotional intensity
            adjusted_intensity = min(
                1.0, emotional_state.intensity * mirroring_intensity
            )

            # Generate contextual mirroring if intensity is high enough
            if adjusted_intensity > 0.6:
                contextual_mirror = self._generate_contextual_mirror(
                    emotional_state, context, base_mirror
                )
                if contextual_mirror:
                    return contextual_mirror

            # Add intensity modifier to base mirror
            intensity_modifier = self._get_intensity_modifier(adjusted_intensity)
            if intensity_modifier:
                return f"{base_mirror} {intensity_modifier}"

            return base_mirror

        except Exception as e:
            logger.warning(f"Failed to generate emotional mirror: {e}")
            return self._get_fallback_mirror(emotional_state)

    def create_reflection_prompt(
        self, emotional_state: EmotionalState, context: ConversationContext
    ) -> str:
        """
        Create a reflection prompt to help user explore their emotions.

        Args:
            emotional_state: User's emotional state
            context: Conversation context

        Returns:
            Reflection prompt text
        """
        # Get base reflection prompt
        base_prompts = self.REFLECTION_PROMPTS.get(emotional_state.primary_emotion, [])
        if not base_prompts:
            return "What are you noticing about your emotional experience right now?"

        # Select prompt based on emotional intensity
        if emotional_state.intensity > 0.7:
            # High intensity - use more direct prompt
            selected_prompt = base_prompts[0]
        elif emotional_state.intensity > 0.4:
            # Medium intensity - use exploratory prompt
            selected_prompt = (
                base_prompts[1] if len(base_prompts) > 1 else base_prompts[0]
            )
        else:
            # Low intensity - use gentle prompt
            selected_prompt = (
                base_prompts[-1] if len(base_prompts) > 2 else base_prompts[0]
            )

        return selected_prompt

    def generate_emotional_reflection(
        self,
        emotional_state: EmotionalState,
        context: ConversationContext,
        reflection_depth: str = "medium",
    ) -> Dict[str, Any]:
        """
        Generate comprehensive emotional reflection.

        Args:
            emotional_state: User's emotional state
            context: Conversation context
            reflection_depth: "light", "medium", or "deep"

        Returns:
            Reflection analysis with multiple components
        """
        reflection = {
            "emotion_mirror": self.mirror_emotion(emotional_state, context),
            "reflection_prompt": self.create_reflection_prompt(
                emotional_state, context
            ),
            "emotional_patterns": self._identify_emotional_patterns(
                emotional_state, context
            ),
            "growth_opportunities": self._identify_growth_opportunities(
                emotional_state
            ),
            "coping_suggestions": self._suggest_coping_strategies(emotional_state),
            "reflection_confidence": self._calculate_reflection_confidence(
                emotional_state
            ),
        }

        # Add deeper analysis for medium/deep reflection
        if reflection_depth in ["medium", "deep"]:
            reflection.update(
                {
                    "emotional_triggers": self._identify_emotional_triggers(context),
                    "emotional_needs": self._identify_emotional_needs(emotional_state),
                    "relationship_impact": self._analyze_relationship_impact(
                        emotional_state, context
                    ),
                }
            )

        # Add deepest analysis for deep reflection
        if reflection_depth == "deep":
            reflection.update(
                {
                    "underlying_values": self._identify_underlying_values(
                        emotional_state, context
                    ),
                    "personal_growth_insights": self._generate_growth_insights(
                        emotional_state, context
                    ),
                    "future_emotional_preparation": self._suggest_future_preparation(
                        emotional_state
                    ),
                }
            )

        return reflection

    def create_empathetic_echo(
        self, emotional_state: EmotionalState, user_words: str
    ) -> str:
        """
        Create an empathetic echo of the user's emotional expression.

        Args:
            emotional_state: User's emotional state
            user_words: User's actual words to echo

        Returns:
            Empathetic echo response
        """
        try:
            # Extract key emotional phrases from user's words
            emotional_phrases = self._extract_emotional_phrases(
                user_words, emotional_state
            )

            # Create echo that incorporates their language
            if emotional_phrases:
                echo_base = f"I hear that you're feeling {emotional_phrases[0]}"

                # Add validation
                validation = self._get_echo_validation(emotional_state)

                return f"{echo_base}, and {validation}"
            else:
                # Fallback to general mirroring
                return self.mirror_emotion(
                    emotional_state,
                    ConversationContext(
                        user_id="",
                        conversation_id="",
                        message_text=user_words,
                        timestamp=datetime.utcnow(),
                    ),
                )

        except Exception as e:
            logger.warning(f"Failed to create empathetic echo: {e}")
            return f"I can sense the {emotional_state.primary_emotion.value} in your words, and that's completely understandable."

    def _get_mirroring_phrase(self, emotional_state: EmotionalState) -> str:
        """Get appropriate mirroring phrase for emotion."""
        phrases = self.MIRRORING_PHRASES.get(emotional_state.primary_emotion, [])
        if not phrases:
            return f"I can sense the {emotional_state.primary_emotion.value} you're experiencing"

        # Select phrase based on confidence level
        if emotional_state.confidence > 0.8:
            return phrases[0]  # Most direct phrase
        elif emotional_state.confidence > 0.5:
            return phrases[1] if len(phrases) > 1 else phrases[0]
        else:
            return phrases[-1] if len(phrases) > 2 else phrases[0]

    def _get_intensity_modifier(self, intensity: float) -> Optional[str]:
        """Get intensity modifier for mirroring."""
        if intensity > 0.8:
            return "- it's really strong and powerful."
        elif intensity > 0.6:
            return "- it's quite noticeable and significant."
        elif intensity > 0.4:
            return "- it's present and meaningful."
        else:
            return "- it's gentle but still important."

    def _generate_contextual_mirror(
        self,
        emotional_state: EmotionalState,
        context: ConversationContext,
        base_mirror: str,
    ) -> Optional[str]:
        """Generate contextual mirroring using LLM."""
        try:
            prompt = f"""
            Create an empathetic mirroring response that reflects the user's emotional state back to them.

            User's emotion: {emotional_state.primary_emotion.value} (intensity: {emotional_state.intensity:.1f})
            User's message: "{context.message_text}"
            Base mirror: "{base_mirror}"

            Guidelines:
            - Mirror their emotion back to show you understand
            - Use their own language and context when possible
            - Be warm and empathetic
            - Help them feel seen and understood
            - Keep it natural and conversational
            - Don't be clinical or analytical

            Mirroring response:
            """

            response = self.llm_service.generate(
                prompt=prompt, temperature=0.7, max_tokens=100
            )

            return response.content.strip()

        except Exception as e:
            logger.warning(f"Failed to generate contextual mirror: {e}")
            return None

    def _identify_emotional_patterns(
        self, emotional_state: EmotionalState, context: ConversationContext
    ) -> List[str]:
        """Identify patterns in the emotional expression."""
        patterns = []

        # Intensity patterns
        if emotional_state.intensity > 0.8:
            patterns.append("high_emotional_intensity")
        elif emotional_state.intensity < 0.3:
            patterns.append("subtle_emotional_expression")

        # Confidence patterns
        if emotional_state.confidence < 0.5:
            patterns.append("mixed_emotional_signals")

        # Secondary emotion patterns
        if len(emotional_state.secondary_emotions) > 1:
            patterns.append("complex_emotional_state")

        # Context patterns
        message_length = len(context.message_text)
        if message_length > 200:
            patterns.append("detailed_emotional_sharing")
        elif message_length < 50:
            patterns.append("brief_emotional_expression")

        return patterns

    def _identify_growth_opportunities(
        self, emotional_state: EmotionalState
    ) -> List[str]:
        """Identify growth opportunities from emotional state."""

        growth_maps = {
            EmotionType.JOY: [
                "Savoring positive moments",
                "Sharing joy with others",
                "Building on positive momentum",
            ],
            EmotionType.SADNESS: [
                "Developing emotional resilience",
                "Learning self-compassion",
                "Finding meaning in difficult experiences",
            ],
            EmotionType.ANGER: [
                "Setting healthy boundaries",
                "Developing assertiveness skills",
                "Channeling energy into positive change",
            ],
            EmotionType.FEAR: [
                "Building courage and confidence",
                "Developing coping strategies",
                "Learning to trust your abilities",
            ],
            EmotionType.SURPRISE: [
                "Embracing flexibility and adaptability",
                "Finding opportunity in unexpected situations",
                "Developing openness to new experiences",
            ],
            EmotionType.DISGUST: [
                "Clarifying personal values",
                "Developing healthy boundaries",
                "Learning to navigate value conflicts",
            ],
            EmotionType.NEUTRAL: [
                "Appreciating moments of peace",
                "Developing mindful awareness",
                "Building emotional stability",
            ],
        }

        return growth_maps.get(
            emotional_state.primary_emotion,
            [
                "Developing emotional awareness",
                "Building emotional intelligence",
                "Learning from emotional experiences",
            ],
        )

    def _suggest_coping_strategies(self, emotional_state: EmotionalState) -> List[str]:
        """Suggest coping strategies based on emotional state."""
        coping_strategies = {
            EmotionType.JOY: [
                "Share your joy with someone you care about",
                "Take a moment to fully appreciate this feeling",
                "Consider what led to this positive experience",
            ],
            EmotionType.SADNESS: [
                "Allow yourself to feel the sadness without judgment",
                "Reach out to someone who cares about you",
                "Practice gentle self-care activities",
            ],
            EmotionType.ANGER: [
                "Take some deep breaths to center yourself",
                "Consider what boundary or value was crossed",
                "Find a healthy way to express your frustration",
            ],
            EmotionType.FEAR: [
                "Focus on what you can control in this situation",
                "Break down the challenge into smaller steps",
                "Remind yourself of times you've overcome difficulties",
            ],
            EmotionType.SURPRISE: [
                "Take time to process what happened",
                "Consider what this change might bring",
                "Stay open to new possibilities",
            ],
            EmotionType.DISGUST: [
                "Honor your values and what feels right to you",
                "Consider how to protect your boundaries",
                "Find ways to align your actions with your values",
            ],
            EmotionType.NEUTRAL: [
                "Appreciate this moment of calm",
                "Use this clarity to reflect on what matters",
                "Consider what you'd like to focus on next",
            ],
        }

        return coping_strategies.get(
            emotional_state.primary_emotion,
            [
                "Take time to understand what you're feeling",
                "Be patient and gentle with yourself",
                "Consider what support you might need",
            ],
        )

    def _calculate_reflection_confidence(
        self, emotional_state: EmotionalState
    ) -> float:
        """Calculate confidence in reflection accuracy."""
        base_confidence = emotional_state.confidence

        # Adjust based on emotion clarity
        if emotional_state.primary_emotion != EmotionType.NEUTRAL:
            base_confidence += 0.1

        # Adjust based on intensity (clearer emotions are easier to reflect)
        if emotional_state.intensity > 0.6:
            base_confidence += 0.1
        elif emotional_state.intensity < 0.3:
            base_confidence -= 0.1

        return min(1.0, max(0.0, base_confidence))

    def _identify_emotional_triggers(self, context: ConversationContext) -> List[str]:
        """Identify potential emotional triggers from context."""
        triggers = []
        message_lower = context.message_text.lower()

        # Common trigger patterns
        trigger_patterns = {
            "rejection": ["rejected", "dismissed", "ignored", "excluded"],
            "criticism": ["criticized", "judged", "blamed", "attacked"],
            "loss": ["lost", "gone", "ended", "over", "died"],
            "failure": ["failed", "mistake", "wrong", "messed up"],
            "uncertainty": ["don't know", "unsure", "confused", "unclear"],
            "overwhelm": ["too much", "overwhelmed", "stressed", "pressure"],
        }

        for trigger_type, patterns in trigger_patterns.items():
            if any(pattern in message_lower for pattern in patterns):
                triggers.append(trigger_type)

        return triggers

    def _identify_emotional_needs(self, emotional_state: EmotionalState) -> List[str]:
        """Identify emotional needs based on current state."""
        needs_map = {
            EmotionType.JOY: ["celebration", "sharing", "appreciation"],
            EmotionType.SADNESS: ["comfort", "understanding", "support"],
            EmotionType.ANGER: ["validation", "justice", "boundaries"],
            EmotionType.FEAR: ["safety", "reassurance", "control"],
            EmotionType.SURPRISE: ["processing time", "understanding", "adaptation"],
            EmotionType.DISGUST: ["values alignment", "boundaries", "respect"],
            EmotionType.NEUTRAL: ["clarity", "direction", "purpose"],
        }

        return needs_map.get(
            emotional_state.primary_emotion, ["understanding", "support", "acceptance"]
        )

    def _analyze_relationship_impact(
        self, emotional_state: EmotionalState, context: ConversationContext
    ) -> Dict[str, Any]:
        """Analyze how emotions might impact relationships."""
        impact_analysis = {
            "communication_style": self._assess_communication_impact(emotional_state),
            "relationship_needs": self._assess_relationship_needs(emotional_state),
            "connection_opportunities": self._identify_connection_opportunities(
                emotional_state
            ),
        }

        return impact_analysis

    def _assess_communication_impact(self, emotional_state: EmotionalState) -> str:
        """Assess how emotion might impact communication."""
        communication_impacts = {
            EmotionType.JOY: "likely to be more open and expressive",
            EmotionType.SADNESS: "may need more gentle and patient communication",
            EmotionType.ANGER: "might benefit from direct but respectful communication",
            EmotionType.FEAR: "may need reassuring and supportive communication",
            EmotionType.SURPRISE: "might need time to process before deep communication",
            EmotionType.DISGUST: "may need space to process conflicting values",
            EmotionType.NEUTRAL: "open to various communication styles",
        }

        return communication_impacts.get(
            emotional_state.primary_emotion,
            "communication style may be influenced by current emotional state",
        )

    def _assess_relationship_needs(self, emotional_state: EmotionalState) -> List[str]:
        """Assess relationship needs based on emotional state."""
        relationship_needs = {
            EmotionType.JOY: ["sharing positive experiences", "celebration together"],
            EmotionType.SADNESS: ["emotional support", "presence and understanding"],
            EmotionType.ANGER: ["validation of feelings", "respectful boundaries"],
            EmotionType.FEAR: ["reassurance and stability", "patient support"],
            EmotionType.SURPRISE: ["processing support", "adaptability from others"],
            EmotionType.DISGUST: ["respect for values", "understanding of boundaries"],
            EmotionType.NEUTRAL: ["authentic connection", "meaningful interaction"],
        }

        return relationship_needs.get(
            emotional_state.primary_emotion, ["understanding", "acceptance"]
        )

    def _identify_connection_opportunities(
        self, emotional_state: EmotionalState
    ) -> List[str]:
        """Identify opportunities for deeper connection."""
        opportunities = {
            EmotionType.JOY: ["sharing what brings happiness", "celebrating together"],
            EmotionType.SADNESS: ["offering comfort", "showing vulnerability"],
            EmotionType.ANGER: [
                "discussing values and boundaries",
                "problem-solving together",
            ],
            EmotionType.FEAR: ["providing reassurance", "building trust"],
            EmotionType.SURPRISE: [
                "exploring new experiences together",
                "adapting together",
            ],
            EmotionType.DISGUST: [
                "discussing values and ethics",
                "supporting boundaries",
            ],
            EmotionType.NEUTRAL: [
                "exploring interests together",
                "building understanding",
            ],
        }

        return opportunities.get(
            emotional_state.primary_emotion,
            ["building understanding", "showing empathy"],
        )

    def _identify_underlying_values(
        self, emotional_state: EmotionalState, context: ConversationContext
    ) -> List[str]:
        """Identify underlying values from emotional response."""
        # This is a simplified approach - in practice, this would be more sophisticated
        values = []
        message_lower = context.message_text.lower()

        value_indicators = {
            "fairness": ["unfair", "unjust", "not right", "wrong"],
            "respect": ["disrespected", "dismissed", "ignored"],
            "autonomy": ["controlled", "forced", "no choice"],
            "connection": ["alone", "isolated", "disconnected"],
            "growth": ["learning", "improving", "developing"],
            "security": ["unsafe", "uncertain", "worried"],
        }

        for value, indicators in value_indicators.items():
            if any(indicator in message_lower for indicator in indicators):
                values.append(value)

        return values if values else ["authenticity", "understanding", "respect"]

    def _generate_growth_insights(
        self, emotional_state: EmotionalState, context: ConversationContext
    ) -> List[str]:
        """Generate personal growth insights."""
        insights = []

        # Intensity-based insights
        if emotional_state.intensity > 0.7:
            insights.append("Strong emotions often signal what matters most to you")

        # Emotion-specific insights
        emotion_insights = {
            EmotionType.JOY: ["Joy shows you what aligns with your authentic self"],
            EmotionType.SADNESS: ["Sadness can deepen your capacity for compassion"],
            EmotionType.ANGER: ["Anger often points to important boundaries or values"],
            EmotionType.FEAR: ["Fear can highlight areas for growth and courage"],
            EmotionType.SURPRISE: ["Surprise keeps you open to new possibilities"],
            EmotionType.DISGUST: ["Strong reactions reveal your core values"],
            EmotionType.NEUTRAL: ["Calm moments offer clarity and perspective"],
        }

        insights.extend(
            emotion_insights.get(
                emotional_state.primary_emotion,
                ["Every emotion offers an opportunity for self-understanding"],
            )
        )

        return insights

    def _suggest_future_preparation(self, emotional_state: EmotionalState) -> List[str]:
        """Suggest ways to prepare for similar emotional situations."""
        preparation_strategies = {
            EmotionType.JOY: [
                "Notice what conditions create joy for you",
                "Consider how to cultivate more of these experiences",
            ],
            EmotionType.SADNESS: [
                "Develop a self-care toolkit for difficult times",
                "Build a support network you can reach out to",
            ],
            EmotionType.ANGER: [
                "Practice identifying your triggers early",
                "Develop healthy ways to express frustration",
            ],
            EmotionType.FEAR: [
                "Build confidence through small challenges",
                "Develop coping strategies for uncertainty",
            ],
            EmotionType.SURPRISE: [
                "Practice flexibility and adaptability",
                "Develop comfort with uncertainty",
            ],
            EmotionType.DISGUST: [
                "Clarify your values and boundaries",
                "Practice communicating your limits",
            ],
            EmotionType.NEUTRAL: [
                "Appreciate and cultivate moments of peace",
                "Use calm times for reflection and planning",
            ],
        }

        return preparation_strategies.get(
            emotional_state.primary_emotion,
            [
                "Develop emotional awareness and coping skills",
                "Build a toolkit for managing various emotional states",
            ],
        )

    def _extract_emotional_phrases(
        self, user_words: str, emotional_state: EmotionalState
    ) -> List[str]:
        """Extract key emotional phrases from user's words."""
        phrases = []
        words_lower = user_words.lower()

        # Look for direct emotional expressions
        emotion_words = {
            EmotionType.JOY: [
                "happy",
                "excited",
                "thrilled",
                "delighted",
                "joyful",
                "elated",
            ],
            EmotionType.SADNESS: [
                "sad",
                "depressed",
                "down",
                "upset",
                "heartbroken",
                "miserable",
            ],
            EmotionType.ANGER: [
                "angry",
                "mad",
                "furious",
                "frustrated",
                "irritated",
                "annoyed",
            ],
            EmotionType.FEAR: [
                "scared",
                "afraid",
                "worried",
                "anxious",
                "nervous",
                "terrified",
            ],
            EmotionType.SURPRISE: [
                "surprised",
                "shocked",
                "amazed",
                "astonished",
                "unexpected",
            ],
            EmotionType.DISGUST: [
                "disgusted",
                "revolted",
                "sick",
                "appalled",
                "horrified",
            ],
            EmotionType.NEUTRAL: ["calm", "okay", "fine", "neutral", "balanced"],
        }

        emotion_words_for_state = emotion_words.get(emotional_state.primary_emotion, [])
        for word in emotion_words_for_state:
            if word in words_lower:
                phrases.append(word)

        return phrases[:3]  # Return up to 3 phrases

    def _get_echo_validation(self, emotional_state: EmotionalState) -> str:
        """Get validation phrase for empathetic echo."""
        validations = {
            EmotionType.JOY: "that's such a wonderful feeling to experience",
            EmotionType.SADNESS: "it makes complete sense that you'd feel this way",
            EmotionType.ANGER: "your frustration is completely understandable",
            EmotionType.FEAR: "it's natural to feel concerned about this",
            EmotionType.SURPRISE: "that must have been quite unexpected",
            EmotionType.DISGUST: "I can understand why this would bother you",
            EmotionType.NEUTRAL: "your balanced perspective is valuable",
        }

        return validations.get(
            emotional_state.primary_emotion, "your feelings are completely valid"
        )

    def _get_fallback_mirror(self, emotional_state: EmotionalState) -> str:
        """Get fallback mirroring response."""
        return f"I can sense the {emotional_state.primary_emotion.value} you're experiencing, and I want you to know that I'm here with you in this moment."


# Singleton instance
_emotional_mirror_instance = None
_emotional_mirror_lock = threading.Lock()


def get_emotional_mirror() -> EmotionalMirror:
    """
    Get singleton emotional mirror instance.

    Returns:
        Shared EmotionalMirror instance
    """
    global _emotional_mirror_instance

    if _emotional_mirror_instance is None:
        with _emotional_mirror_lock:
            if _emotional_mirror_instance is None:
                _emotional_mirror_instance = EmotionalMirror()

    return _emotional_mirror_instance
