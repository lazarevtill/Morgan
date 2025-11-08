"""
Simple Emotional Intelligence for Morgan

Detects basic emotions from text and tracks user emotional patterns.
KISS principle: Simple, effective emotion detection.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class EmotionType(Enum):
    """Basic emotion types based on Ekman's model"""

    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"


@dataclass
class EmotionalState:
    """Detected emotional state from text"""

    primary_emotion: EmotionType
    intensity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    indicators: List[str]  # Words/patterns that indicated emotion
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class SimpleEmotionDetector:
    """
    Simple emotion detector using keyword matching.

    KISS: Uses pattern matching for basic emotion detection.
    Good enough for most use cases without ML overhead.
    """

    def __init__(self):
        # Emotion keywords (simplified for efficiency)
        self.emotion_patterns = {
            EmotionType.JOY: {
                "keywords": [
                    "happy",
                    "joy",
                    "excited",
                    "love",
                    "great",
                    "awesome",
                    "wonderful",
                    "fantastic",
                    "excellent",
                    "amazing",
                    "good",
                    "nice",
                    "thanks",
                    "thank you",
                    "appreciate",
                ],
                "patterns": [r":\)", r"😊", r"😄", r"❤️", r"!+"],
            },
            EmotionType.SADNESS: {
                "keywords": [
                    "sad",
                    "unhappy",
                    "depressed",
                    "down",
                    "disappointed",
                    "sorry",
                    "miss",
                    "hurt",
                    "alone",
                    "lost",
                ],
                "patterns": [r":\(", r"😢", r"😭", r"\.\.\."],
            },
            EmotionType.ANGER: {
                "keywords": [
                    "angry",
                    "mad",
                    "furious",
                    "annoyed",
                    "frustrated",
                    "hate",
                    "stupid",
                    "terrible",
                    "awful",
                    "worst",
                ],
                "patterns": [r"!", r"!!", r"😠", r"😡"],
            },
            EmotionType.FEAR: {
                "keywords": [
                    "afraid",
                    "scared",
                    "fear",
                    "worried",
                    "nervous",
                    "anxious",
                    "panic",
                    "terrified",
                ],
                "patterns": [r"😨", r"😰"],
            },
            EmotionType.SURPRISE: {
                "keywords": [
                    "wow",
                    "amazing",
                    "incredible",
                    "unexpected",
                    "surprised",
                    "shocked",
                    "really",
                    "seriously",
                ],
                "patterns": [r"!!", r"!!!", r"😮", r"😲", r"\?!"],
            },
        }

    def detect(self, text: str) -> EmotionalState:
        """
        Detect emotion from text.

        Args:
            text: Input text to analyze

        Returns:
            EmotionalState with detected emotion
        """
        text_lower = text.lower()
        emotion_scores: Dict[EmotionType, float] = {
            emotion: 0.0 for emotion in EmotionType
        }
        indicators: Dict[EmotionType, List[str]] = {
            emotion: [] for emotion in EmotionType
        }

        # Check keywords and patterns for each emotion
        for emotion, patterns in self.emotion_patterns.items():
            score = 0.0

            # Check keywords
            for keyword in patterns["keywords"]:
                if keyword in text_lower:
                    score += 1.0
                    indicators[emotion].append(keyword)

            # Check regex patterns
            for pattern in patterns["patterns"]:
                matches = re.findall(pattern, text)
                if matches:
                    score += len(matches) * 0.5
                    indicators[emotion].extend(matches)

            emotion_scores[emotion] = score

        # Find dominant emotion
        max_score = max(emotion_scores.values())
        if max_score == 0:
            primary_emotion = EmotionType.NEUTRAL
            intensity = 0.5
            confidence = 0.3
            emotion_indicators = []
        else:
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            # Normalize intensity (cap at 1.0)
            intensity = min(max_score / 5.0, 1.0)
            # Confidence based on how clear the signal is
            total_score = sum(emotion_scores.values())
            confidence = max_score / total_score if total_score > 0 else 0.5
            emotion_indicators = indicators[primary_emotion][:5]  # Top 5 indicators

        return EmotionalState(
            primary_emotion=primary_emotion,
            intensity=intensity,
            confidence=confidence,
            indicators=emotion_indicators,
        )

    def track_mood_pattern(
        self, recent_states: List[EmotionalState], timeframe_days: int = 7
    ) -> Dict[str, any]:
        """
        Track mood patterns over recent states.

        Args:
            recent_states: List of recent emotional states
            timeframe_days: Number of days to analyze

        Returns:
            Dict with mood pattern analysis
        """
        if not recent_states:
            return {
                "dominant_emotion": EmotionType.NEUTRAL.value,
                "average_intensity": 0.5,
                "stability": 1.0,
                "trend": "stable",
            }

        # Count emotions
        emotion_counts: Dict[str, int] = {}
        intensities: List[float] = []

        for state in recent_states:
            emotion_counts[state.primary_emotion.value] = (
                emotion_counts.get(state.primary_emotion.value, 0) + 1
            )
            intensities.append(state.intensity)

        # Dominant emotion
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)

        # Average intensity
        avg_intensity = sum(intensities) / len(intensities)

        # Stability (low variance = stable)
        variance = sum((x - avg_intensity) ** 2 for x in intensities) / len(intensities)
        stability = max(0.0, 1.0 - variance)  # Higher = more stable

        # Simple trend detection
        if len(intensities) >= 3:
            recent_avg = sum(intensities[-3:]) / 3
            overall_avg = avg_intensity
            if recent_avg > overall_avg * 1.1:
                trend = "improving"
            elif recent_avg < overall_avg * 0.9:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "dominant_emotion": dominant_emotion,
            "average_intensity": round(avg_intensity, 2),
            "stability": round(stability, 2),
            "trend": trend,
            "emotion_distribution": emotion_counts,
        }


# Global instance for easy access
_detector = None


def get_emotion_detector() -> SimpleEmotionDetector:
    """Get the global emotion detector instance"""
    global _detector
    if _detector is None:
        _detector = SimpleEmotionDetector()
    return _detector
