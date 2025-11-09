"""
Emotional Handler - Integrates emotional awareness into Morgan responses

Adds emotional intelligence to conversation processing without disrupting core logic.
"""

import logging
from typing import Dict, List, Optional

from shared.models.base import Message, Response
from shared.utils.emotional import EmotionalState, EmotionType, get_emotion_detector

logger = logging.getLogger(__name__)


class EmotionalHandler:
    """Handles emotional awareness in conversations"""

    def __init__(self):
        self.detector = get_emotion_detector()
        # Track emotional states per user
        self.user_emotions: Dict[str, List[EmotionalState]] = {}
        self.max_history = 20  # Keep last 20 emotional states

    def detect_emotion(self, text: str) -> EmotionalState:
        """Detect emotion from user text"""
        return self.detector.detect(text)

    def track_user_emotion(self, user_id: str, state: EmotionalState):
        """Track emotional state for a user"""
        if user_id not in self.user_emotions:
            self.user_emotions[user_id] = []

        self.user_emotions[user_id].append(state)

        # Keep only recent states
        if len(self.user_emotions[user_id]) > self.max_history:
            self.user_emotions[user_id] = self.user_emotions[user_id][-self.max_history :]

    def get_mood_pattern(self, user_id: str) -> Dict:
        """Get mood pattern for user"""
        states = self.user_emotions.get(user_id, [])
        return self.detector.track_mood_pattern(states)

    def enhance_response_with_empathy(
        self, response_text: str, emotion: EmotionalState
    ) -> str:
        """
        Enhance response to be emotionally aware.

        Adds appropriate empathetic phrasing based on detected emotion.
        """
        # If user is showing strong emotion, acknowledge it
        if emotion.intensity > 0.7:
            if emotion.primary_emotion == EmotionType.SADNESS:
                prefix = "I understand this might be difficult. "
            elif emotion.primary_emotion == EmotionType.ANGER:
                prefix = "I hear your frustration. "
            elif emotion.primary_emotion == EmotionType.JOY:
                prefix = "I'm glad to help with this! "
            elif emotion.primary_emotion == EmotionType.FEAR:
                prefix = "I understand your concern. "
            else:
                prefix = ""

            if prefix and not response_text.startswith(prefix):
                response_text = prefix + response_text

        return response_text

    def get_emotional_context_for_llm(self, user_id: str, current_emotion: EmotionalState) -> str:
        """
        Generate emotional context to include in LLM prompt.

        This helps the LLM respond more empathetically.
        """
        mood_pattern = self.get_mood_pattern(user_id)

        context_parts = []

        # Current emotional state
        if current_emotion.intensity > 0.5:
            context_parts.append(
                f"User is expressing {current_emotion.primary_emotion.value} "
                f"(intensity: {current_emotion.intensity:.1f})"
            )

        # Mood pattern
        if mood_pattern and mood_pattern["dominant_emotion"] != "neutral":
            context_parts.append(
                f"Recent mood trend: {mood_pattern['trend']} "
                f"(dominant: {mood_pattern['dominant_emotion']})"
            )

        if context_parts:
            return (
                "Emotional Context: " + " | ".join(context_parts) + "\n"
                "Please respond with appropriate empathy and emotional awareness.\n"
            )

        return ""

    def get_stats(self) -> Dict:
        """Get emotional tracking statistics"""
        total_states = sum(len(states) for states in self.user_emotions.values())
        return {
            "tracked_users": len(self.user_emotions),
            "total_emotional_states": total_states,
            "active_users": len(
                [uid for uid, states in self.user_emotions.items() if states]
            ),
        }


# Global instance
_handler = None


def get_emotional_handler() -> EmotionalHandler:
    """Get global emotional handler instance"""
    global _handler
    if _handler is None:
        _handler = EmotionalHandler()
    return _handler
