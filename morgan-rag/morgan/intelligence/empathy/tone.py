"""
Emotional tone matching module.

Provides emotional tone matching capabilities that adapt response tone and style
to match and complement the user's emotional state and communication preferences.
"""

import threading
from enum import Enum
from typing import Any, Dict, List, Optional

from morgan.config import get_settings
from morgan.intelligence.core.models import (
    CommunicationStyle,
    ConversationContext,
    EmotionalState,
    EmotionType,
)
from morgan.services.llm_service import get_llm_service
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class ToneType(Enum):
    """Types of emotional tones."""

    WARM_SUPPORTIVE = "warm_supportive"
    GENTLE_CALMING = "gentle_calming"
    ENERGETIC_ENTHUSIASTIC = "energetic_enthusiastic"
    CALM_REASSURING = "calm_reassuring"
    UNDERSTANDING_VALIDATING = "understanding_validating"
    ENCOURAGING_UPLIFTING = "encouraging_uplifting"
    RESPECTFUL_NEUTRAL = "respectful_neutral"
    COMPASSIONATE_CARING = "compassionate_caring"


class ToneIntensity(Enum):
    """Intensity levels for tone matching."""

    SUBTLE = "subtle"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class EmotionalToneManager:
    """
    Emotional tone matching and management system.

    Provides capabilities to:
    - Match response tone to user's emotional state
    - Adapt communication style based on user preferences
    - Modulate tone intensity appropriately
    - Create emotionally resonant responses
    """

    # Tone mapping for different emotional states
    EMOTION_TONE_MAP = {
        EmotionType.JOY: {
            "primary": ToneType.ENERGETIC_ENTHUSIASTIC,
            "secondary": [ToneType.WARM_SUPPORTIVE, ToneType.ENCOURAGING_UPLIFTING],
            "intensity_modifiers": {
                "high": ToneIntensity.STRONG,
                "medium": ToneIntensity.MODERATE,
                "low": ToneIntensity.SUBTLE,
            },
        },
        EmotionType.SADNESS: {
            "primary": ToneType.GENTLE_CALMING,
            "secondary": [
                ToneType.COMPASSIONATE_CARING,
                ToneType.UNDERSTANDING_VALIDATING,
            ],
            "intensity_modifiers": {
                "high": ToneIntensity.STRONG,
                "medium": ToneIntensity.MODERATE,
                "low": ToneIntensity.SUBTLE,
            },
        },
        EmotionType.ANGER: {
            "primary": ToneType.CALM_REASSURING,
            "secondary": [
                ToneType.UNDERSTANDING_VALIDATING,
                ToneType.RESPECTFUL_NEUTRAL,
            ],
            "intensity_modifiers": {
                "high": ToneIntensity.VERY_STRONG,
                "medium": ToneIntensity.STRONG,
                "low": ToneIntensity.MODERATE,
            },
        },
        EmotionType.FEAR: {
            "primary": ToneType.CALM_REASSURING,
            "secondary": [ToneType.WARM_SUPPORTIVE, ToneType.ENCOURAGING_UPLIFTING],
            "intensity_modifiers": {
                "high": ToneIntensity.STRONG,
                "medium": ToneIntensity.MODERATE,
                "low": ToneIntensity.SUBTLE,
            },
        },
        EmotionType.SURPRISE: {
            "primary": ToneType.ENERGETIC_ENTHUSIASTIC,
            "secondary": [ToneType.WARM_SUPPORTIVE, ToneType.UNDERSTANDING_VALIDATING],
            "intensity_modifiers": {
                "high": ToneIntensity.MODERATE,
                "medium": ToneIntensity.MODERATE,
                "low": ToneIntensity.SUBTLE,
            },
        },
        EmotionType.DISGUST: {
            "primary": ToneType.RESPECTFUL_NEUTRAL,
            "secondary": [
                ToneType.UNDERSTANDING_VALIDATING,
                ToneType.COMPASSIONATE_CARING,
            ],
            "intensity_modifiers": {
                "high": ToneIntensity.MODERATE,
                "medium": ToneIntensity.MODERATE,
                "low": ToneIntensity.SUBTLE,
            },
        },
        EmotionType.NEUTRAL: {
            "primary": ToneType.WARM_SUPPORTIVE,
            "secondary": [ToneType.RESPECTFUL_NEUTRAL, ToneType.ENCOURAGING_UPLIFTING],
            "intensity_modifiers": {
                "high": ToneIntensity.MODERATE,
                "medium": ToneIntensity.MODERATE,
                "low": ToneIntensity.SUBTLE,
            },
        },
    }

    # Tone characteristics and language patterns
    TONE_CHARACTERISTICS = {
        ToneType.WARM_SUPPORTIVE: {
            "adjectives": ["warm", "caring", "supportive", "kind", "gentle"],
            "phrases": [
                "I'm here for you",
                "You're not alone in this",
                "I care about what you're going through",
                "Your feelings matter to me",
            ],
            "sentence_starters": [
                "I can see that...",
                "It sounds like...",
                "I understand that...",
                "I'm glad you shared...",
            ],
            "punctuation_style": "gentle",  # More periods, fewer exclamations
            "word_choice": "inclusive",  # "we", "us", "together"
        },
        ToneType.GENTLE_CALMING: {
            "adjectives": ["gentle", "peaceful", "soothing", "calm", "soft"],
            "phrases": [
                "Take your time",
                "It's okay to feel this way",
                "You're safe here",
                "Let's take this slowly",
            ],
            "sentence_starters": [
                "Gently...",
                "Softly...",
                "Take a moment to...",
                "Allow yourself to...",
            ],
            "punctuation_style": "minimal",  # Fewer punctuation marks
            "word_choice": "soothing",  # "breathe", "rest", "peace"
        },
        ToneType.ENERGETIC_ENTHUSIASTIC: {
            "adjectives": [
                "exciting",
                "wonderful",
                "amazing",
                "fantastic",
                "brilliant",
            ],
            "phrases": [
                "That's fantastic!",
                "How exciting!",
                "I love your enthusiasm!",
                "This is wonderful news!",
            ],
            "sentence_starters": [
                "Wow!",
                "That's incredible!",
                "I'm so excited that...",
                "What a wonderful...",
            ],
            "punctuation_style": "expressive",  # More exclamations
            "word_choice": "energetic",  # "amazing", "incredible", "fantastic"
        },
        ToneType.CALM_REASSURING: {
            "adjectives": ["steady", "reliable", "stable", "grounded", "secure"],
            "phrases": [
                "Everything will be okay",
                "You can handle this",
                "You're stronger than you know",
                "This too shall pass",
            ],
            "sentence_starters": [
                "You can trust that...",
                "Rest assured that...",
                "You have the strength to...",
                "Remember that...",
            ],
            "punctuation_style": "steady",  # Consistent, measured
            "word_choice": "grounding",  # "stable", "secure", "steady"
        },
        ToneType.UNDERSTANDING_VALIDATING: {
            "adjectives": [
                "valid",
                "understandable",
                "reasonable",
                "natural",
                "normal",
            ],
            "phrases": [
                "That makes complete sense",
                "Your feelings are valid",
                "Anyone would feel that way",
                "You have every right to feel this",
            ],
            "sentence_starters": [
                "Of course you feel...",
                "It's completely natural to...",
                "Anyone in your situation would...",
                "Your reaction is...",
            ],
            "punctuation_style": "affirming",  # Clear, definitive
            "word_choice": "validating",  # "valid", "understandable", "natural"
        },
        ToneType.ENCOURAGING_UPLIFTING: {
            "adjectives": [
                "hopeful",
                "positive",
                "inspiring",
                "uplifting",
                "motivating",
            ],
            "phrases": [
                "You've got this",
                "I believe in you",
                "You're capable of amazing things",
                "Better days are ahead",
            ],
            "sentence_starters": [
                "You have the power to...",
                "I believe you can...",
                "You're capable of...",
                "There's hope in...",
            ],
            "punctuation_style": "uplifting",  # Mix of periods and exclamations
            "word_choice": "empowering",  # "capable", "strong", "possible"
        },
        ToneType.RESPECTFUL_NEUTRAL: {
            "adjectives": [
                "respectful",
                "balanced",
                "thoughtful",
                "considerate",
                "measured",
            ],
            "phrases": [
                "I respect your perspective",
                "That's a thoughtful point",
                "I appreciate you sharing",
                "Thank you for being open",
            ],
            "sentence_starters": [
                "I appreciate that...",
                "Thank you for...",
                "I respect that...",
                "That's an important...",
            ],
            "punctuation_style": "balanced",  # Professional, measured
            "word_choice": "respectful",  # "appreciate", "respect", "understand"
        },
        ToneType.COMPASSIONATE_CARING: {
            "adjectives": ["compassionate", "caring", "loving", "tender", "heartfelt"],
            "phrases": [
                "My heart goes out to you",
                "I feel for what you're going through",
                "You deserve compassion",
                "I'm holding space for you",
            ],
            "sentence_starters": [
                "With compassion...",
                "My heart...",
                "I feel deeply...",
                "With love and care...",
            ],
            "punctuation_style": "heartfelt",  # Emotional but gentle
            "word_choice": "caring",  # "heart", "care", "love", "compassion"
        },
    }

    def __init__(self):
        """Initialize emotional tone manager."""
        self.settings = get_settings()
        self.llm_service = get_llm_service()

        # Cache for tone adaptations
        self.tone_cache: Dict[str, Dict[str, Any]] = {}

        logger.info("Emotional Tone Manager initialized")

    def match_emotional_tone(
        self,
        emotional_state: EmotionalState,
        context: ConversationContext,
        user_communication_style: Optional[CommunicationStyle] = None,
    ) -> Dict[str, Any]:
        """
        Match emotional tone to user's state and preferences.

        Args:
            emotional_state: User's current emotional state
            context: Conversation context
            user_communication_style: User's preferred communication style

        Returns:
            Tone matching configuration
        """
        # Get base tone for emotion
        emotion_tone_config = self.EMOTION_TONE_MAP.get(
            emotional_state.primary_emotion, self.EMOTION_TONE_MAP[EmotionType.NEUTRAL]
        )

        # Determine intensity level
        intensity_level = self._determine_intensity_level(emotional_state.intensity)

        # Select primary tone
        primary_tone = emotion_tone_config["primary"]

        # Select secondary tone based on context or user style
        secondary_tones = emotion_tone_config["secondary"]
        secondary_tone = self._select_secondary_tone(
            secondary_tones, context, user_communication_style
        )

        # Get tone intensity
        tone_intensity = emotion_tone_config["intensity_modifiers"][intensity_level]

        # Create tone configuration
        tone_config = {
            "primary_tone": primary_tone,
            "secondary_tone": secondary_tone,
            "tone_intensity": tone_intensity,
            "emotional_resonance": self._calculate_emotional_resonance(emotional_state),
            "adaptation_confidence": self._calculate_adaptation_confidence(
                emotional_state, context
            ),
            "tone_characteristics": self._get_combined_tone_characteristics(
                primary_tone, secondary_tone, tone_intensity
            ),
        }

        return tone_config

    def adapt_response_tone(
        self,
        response_text: str,
        tone_config: Dict[str, Any],
        adaptation_strength: float = 0.8,
    ) -> str:
        """
        Adapt response text to match desired emotional tone.

        Args:
            response_text: Original response text
            tone_config: Tone configuration from match_emotional_tone
            adaptation_strength: How strongly to adapt (0.0-1.0)

        Returns:
            Tone-adapted response text
        """
        if adaptation_strength < 0.3:
            return response_text  # Minimal adaptation

        try:
            # Use LLM to adapt tone
            adapted_response = self._adapt_tone_with_llm(
                response_text, tone_config, adaptation_strength
            )

            if adapted_response:
                return adapted_response

            # Fallback to rule-based adaptation
            return self._adapt_tone_rule_based(response_text, tone_config)

        except Exception as e:
            logger.warning(f"Failed to adapt response tone: {e}")
            return response_text

    def create_tone_matched_response(
        self,
        content: str,
        emotional_state: EmotionalState,
        context: ConversationContext,
        response_type: str = "supportive",
    ) -> str:
        """
        Create a response with tone matched to emotional state.

        Args:
            content: Core content to communicate
            emotional_state: User's emotional state
            context: Conversation context
            response_type: Type of response (supportive, informative, etc.)

        Returns:
            Tone-matched response
        """
        # Get tone configuration
        tone_config = self.match_emotional_tone(emotional_state, context)

        # Generate base response with tone awareness
        base_response = self._generate_tone_aware_response(
            content, tone_config, response_type
        )

        # Adapt the response tone
        final_response = self.adapt_response_tone(base_response, tone_config)

        return final_response

    def analyze_user_tone_preferences(
        self,
        conversation_history: List[ConversationContext],
        user_feedback_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze user's tone preferences from conversation history.

        Args:
            conversation_history: User's conversation history
            user_feedback_history: User's feedback on responses

        Returns:
            Analysis of user's tone preferences
        """
        if not conversation_history:
            return {"preferences": "insufficient_data"}

        # Analyze message characteristics
        message_analysis = self._analyze_message_characteristics(conversation_history)

        # Analyze feedback patterns
        feedback_analysis = self._analyze_feedback_patterns(user_feedback_history)

        # Determine preferred tones
        preferred_tones = self._determine_preferred_tones(
            message_analysis, feedback_analysis
        )

        return {
            "preferred_tones": preferred_tones,
            "communication_style": message_analysis.get("style", "balanced"),
            "formality_preference": message_analysis.get("formality", "moderate"),
            "emotional_expressiveness": message_analysis.get(
                "expressiveness", "moderate"
            ),
            "response_length_preference": message_analysis.get(
                "length_preference", "medium"
            ),
            "confidence_score": self._calculate_preference_confidence(
                len(conversation_history), len(user_feedback_history)
            ),
        }

    def _determine_intensity_level(self, intensity: float) -> str:
        """Determine intensity level from emotional intensity."""
        if intensity > 0.8:
            return "high"
        elif intensity > 0.5:
            return "medium"
        else:
            return "low"

    def _select_secondary_tone(
        self,
        secondary_tones: List[ToneType],
        context: ConversationContext,
        user_style: Optional[CommunicationStyle],
    ) -> ToneType:
        """Select appropriate secondary tone."""
        if not secondary_tones:
            return ToneType.WARM_SUPPORTIVE

        # Consider user communication style
        if user_style:
            style_tone_preferences = {
                CommunicationStyle.FORMAL: ToneType.RESPECTFUL_NEUTRAL,
                CommunicationStyle.CASUAL: ToneType.WARM_SUPPORTIVE,
                CommunicationStyle.TECHNICAL: ToneType.RESPECTFUL_NEUTRAL,
                CommunicationStyle.FRIENDLY: ToneType.WARM_SUPPORTIVE,
                CommunicationStyle.PROFESSIONAL: ToneType.RESPECTFUL_NEUTRAL,
            }

            preferred_tone = style_tone_preferences.get(user_style)
            if preferred_tone in secondary_tones:
                return preferred_tone

        # Default to first secondary tone
        return secondary_tones[0]

    def _calculate_emotional_resonance(self, emotional_state: EmotionalState) -> float:
        """Calculate how well we can resonate with the emotional state."""
        base_resonance = emotional_state.confidence

        # Adjust based on emotion type (some are easier to resonate with)
        resonance_adjustments = {
            EmotionType.JOY: 0.1,
            EmotionType.SADNESS: 0.05,
            EmotionType.ANGER: -0.05,  # Harder to resonate with anger
            EmotionType.FEAR: 0.0,
            EmotionType.SURPRISE: 0.0,
            EmotionType.DISGUST: -0.1,  # Harder to resonate with disgust
            EmotionType.NEUTRAL: -0.05,
        }

        adjustment = resonance_adjustments.get(emotional_state.primary_emotion, 0.0)
        return min(1.0, max(0.0, base_resonance + adjustment))

    def _calculate_adaptation_confidence(
        self, emotional_state: EmotionalState, context: ConversationContext
    ) -> float:
        """Calculate confidence in tone adaptation."""
        base_confidence = emotional_state.confidence

        # Adjust based on context richness
        if len(context.message_text) > 100:
            base_confidence += 0.1  # More context helps
        elif len(context.message_text) < 20:
            base_confidence -= 0.1  # Less context hurts

        # Adjust based on previous messages
        if context.previous_messages and len(context.previous_messages) > 2:
            base_confidence += 0.05  # Conversation history helps

        return min(1.0, max(0.0, base_confidence))

    def _get_combined_tone_characteristics(
        self, primary_tone: ToneType, secondary_tone: ToneType, intensity: ToneIntensity
    ) -> Dict[str, Any]:
        """Combine characteristics from primary and secondary tones."""
        primary_chars = self.TONE_CHARACTERISTICS.get(primary_tone, {})
        secondary_chars = self.TONE_CHARACTERISTICS.get(secondary_tone, {})

        # Combine characteristics with primary taking precedence
        combined = {
            "adjectives": primary_chars.get("adjectives", [])
            + secondary_chars.get("adjectives", [])[:2],
            "phrases": primary_chars.get("phrases", [])
            + secondary_chars.get("phrases", [])[:2],
            "sentence_starters": primary_chars.get("sentence_starters", [])
            + secondary_chars.get("sentence_starters", [])[:2],
            "punctuation_style": primary_chars.get("punctuation_style", "balanced"),
            "word_choice": primary_chars.get("word_choice", "neutral"),
            "intensity": intensity.value,
        }

        return combined

    def _adapt_tone_with_llm(
        self,
        response_text: str,
        tone_config: Dict[str, Any],
        adaptation_strength: float,
    ) -> Optional[str]:
        """Adapt tone using LLM."""
        try:
            primary_tone = tone_config["primary_tone"].value
            tone_intensity = tone_config["tone_intensity"].value

            prompt = f"""
            Adapt the following response to match the specified emotional tone and intensity.

            Original response: "{response_text}"

            Target tone: {primary_tone}
            Tone intensity: {tone_intensity}
            Adaptation strength: {adaptation_strength:.1f} (0.0 = minimal, 1.0 = maximum)

            Guidelines:
            - Maintain the core message and meaning
            - Adjust language, word choice, and style to match the tone
            - Keep the response natural and conversational
            - Match the specified intensity level
            - Don't change factual content, only the emotional presentation

            Adapted response:
            """

            response = self.llm_service.generate(
                prompt=prompt, temperature=0.6, max_tokens=200
            )

            return response.content.strip()

        except Exception as e:
            logger.warning(f"LLM tone adaptation failed: {e}")
            return None

    def _adapt_tone_rule_based(
        self, response_text: str, tone_config: Dict[str, Any]
    ) -> str:
        """Adapt tone using rule-based approach."""
        adapted_text = response_text
        tone_chars = tone_config.get("tone_characteristics", {})

        # Simple rule-based adaptations
        punctuation_style = tone_chars.get("punctuation_style", "balanced")

        if punctuation_style == "expressive":
            # Add more exclamation marks for energetic tone
            adapted_text = adapted_text.replace(".", "!")
        elif punctuation_style == "gentle":
            # Replace exclamation marks with periods for gentle tone
            adapted_text = adapted_text.replace("!", ".")
        elif punctuation_style == "minimal":
            # Remove excessive punctuation for calming tone
            adapted_text = adapted_text.replace("!!", "!").replace("??", "?")

        # Add tone-appropriate sentence starters
        sentence_starters = tone_chars.get("sentence_starters", [])
        if sentence_starters and not any(
            adapted_text.startswith(starter.split()[0]) for starter in sentence_starters
        ):
            # Prepend appropriate starter if response doesn't already have one
            starter = sentence_starters[0]
            if not adapted_text.startswith(starter.split()[0]):
                adapted_text = f"{starter} {adapted_text.lower()}"

        return adapted_text

    def _generate_tone_aware_response(
        self, content: str, tone_config: Dict[str, Any], response_type: str
    ) -> str:
        """Generate response with tone awareness."""
        try:
            primary_tone = tone_config["primary_tone"].value
            tone_chars = tone_config.get("tone_characteristics", {})

            # Get tone-appropriate phrases
            phrases = tone_chars.get("phrases", [])
            sentence_starters = tone_chars.get("sentence_starters", [])

            prompt = f"""
            Create a {response_type} response with the following content and emotional tone.

            Content to communicate: "{content}"
            Emotional tone: {primary_tone}

            Tone characteristics to incorporate:
            - Use phrases like: {', '.join(phrases[:3]) if phrases else 'supportive, understanding phrases'}
            - Start sentences with: {', '.join(sentence_starters[:2]) if sentence_starters else 'empathetic openings'}

            Guidelines:
            - Keep the response natural and conversational
            - Match the specified emotional tone
            - Include the core content clearly
            - Be authentic and genuine
            - Keep response length appropriate (2-3 sentences)

            Response:
            """

            response = self.llm_service.generate(
                prompt=prompt, temperature=0.7, max_tokens=150
            )

            return response.content.strip()

        except Exception as e:
            logger.warning(f"Failed to generate tone-aware response: {e}")
            return f"I understand what you're sharing about {content}. I'm here to support you."

    def _analyze_message_characteristics(
        self, conversation_history: List[ConversationContext]
    ) -> Dict[str, Any]:
        """Analyze characteristics of user's messages."""
        if not conversation_history:
            return {}

        # Analyze message lengths
        message_lengths = [len(msg.message_text) for msg in conversation_history]
        avg_length = sum(message_lengths) / len(message_lengths)

        # Analyze formality (simple heuristic)
        formal_indicators = ["please", "thank you", "would you", "could you"]
        casual_indicators = ["hey", "yeah", "gonna", "wanna", "lol"]

        formal_count = 0
        casual_count = 0

        for msg in conversation_history:
            text_lower = msg.message_text.lower()
            formal_count += sum(
                1 for indicator in formal_indicators if indicator in text_lower
            )
            casual_count += sum(
                1 for indicator in casual_indicators if indicator in text_lower
            )

        # Determine style preferences
        if avg_length > 150:
            length_preference = "detailed"
        elif avg_length < 50:
            length_preference = "brief"
        else:
            length_preference = "medium"

        if formal_count > casual_count:
            formality = "formal"
        elif casual_count > formal_count:
            formality = "casual"
        else:
            formality = "moderate"

        # Analyze emotional expressiveness
        expressive_indicators = ["!", "?", "wow", "amazing", "terrible", "love", "hate"]
        expressiveness_score = 0

        for msg in conversation_history:
            text_lower = msg.message_text.lower()
            expressiveness_score += sum(
                1 for indicator in expressive_indicators if indicator in text_lower
            )

        if expressiveness_score > len(conversation_history):
            expressiveness = "high"
        elif expressiveness_score > len(conversation_history) * 0.5:
            expressiveness = "moderate"
        else:
            expressiveness = "low"

        return {
            "length_preference": length_preference,
            "formality": formality,
            "expressiveness": expressiveness,
            "avg_message_length": avg_length,
            "total_messages": len(conversation_history),
        }

    def _analyze_feedback_patterns(
        self, feedback_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze patterns in user feedback."""
        if not feedback_history:
            return {"feedback_available": False}

        # Analyze ratings
        ratings = [fb.get("rating", 3) for fb in feedback_history if "rating" in fb]
        avg_rating = sum(ratings) / len(ratings) if ratings else 3.0

        # Analyze feedback text for tone preferences
        positive_feedback = []
        negative_feedback = []

        for fb in feedback_history:
            if fb.get("rating", 3) >= 4:
                positive_feedback.append(fb.get("text", ""))
            elif fb.get("rating", 3) <= 2:
                negative_feedback.append(fb.get("text", ""))

        return {
            "feedback_available": True,
            "avg_rating": avg_rating,
            "positive_feedback_count": len(positive_feedback),
            "negative_feedback_count": len(negative_feedback),
            "total_feedback": len(feedback_history),
        }

    def _determine_preferred_tones(
        self, message_analysis: Dict[str, Any], feedback_analysis: Dict[str, Any]
    ) -> List[ToneType]:
        """Determine user's preferred tones from analysis."""
        preferred_tones = []

        # Based on formality preference
        formality = message_analysis.get("formality", "moderate")
        if formality == "formal":
            preferred_tones.append(ToneType.RESPECTFUL_NEUTRAL)
        elif formality == "casual":
            preferred_tones.append(ToneType.WARM_SUPPORTIVE)

        # Based on expressiveness
        expressiveness = message_analysis.get("expressiveness", "moderate")
        if expressiveness == "high":
            preferred_tones.append(ToneType.ENERGETIC_ENTHUSIASTIC)
        elif expressiveness == "low":
            preferred_tones.append(ToneType.GENTLE_CALMING)

        # Based on feedback (if available)
        if (
            feedback_analysis.get("feedback_available")
            and feedback_analysis.get("avg_rating", 3) >= 4
        ):
            # High satisfaction - current approach is working
            preferred_tones.append(ToneType.UNDERSTANDING_VALIDATING)

        # Default preferences if none determined
        if not preferred_tones:
            preferred_tones = [
                ToneType.WARM_SUPPORTIVE,
                ToneType.UNDERSTANDING_VALIDATING,
            ]

        return preferred_tones[:3]  # Return top 3 preferences

    def _calculate_preference_confidence(
        self, conversation_count: int, feedback_count: int
    ) -> float:
        """Calculate confidence in preference analysis."""
        # Base confidence on data availability
        conversation_factor = min(
            1.0, conversation_count / 10.0
        )  # Full confidence at 10+ conversations
        feedback_factor = min(
            1.0, feedback_count / 5.0
        )  # Full confidence at 5+ feedback items

        # Weight conversation history more heavily than feedback
        confidence = (conversation_factor * 0.7) + (feedback_factor * 0.3)

        return confidence


# Singleton instance
_emotional_tone_manager_instance = None
_emotional_tone_manager_lock = threading.Lock()


def get_emotional_tone_manager() -> EmotionalToneManager:
    """
    Get singleton emotional tone manager instance.

    Returns:
        Shared EmotionalToneManager instance
    """
    global _emotional_tone_manager_instance

    if _emotional_tone_manager_instance is None:
        with _emotional_tone_manager_lock:
            if _emotional_tone_manager_instance is None:
                _emotional_tone_manager_instance = EmotionalToneManager()

    return _emotional_tone_manager_instance
