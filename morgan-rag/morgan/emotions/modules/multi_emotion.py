"""
Multi-Emotion Detector Module.

Handles detection of multiple simultaneous emotions and their interactions.
Calculates valence and arousal dimensions.
"""

from __future__ import annotations

from typing import List, Optional

from morgan.emotions.base import EmotionModule
from morgan.emotions.types import Emotion, EmotionType


class MultiEmotionDetector(EmotionModule):
    """
    Detects and analyzes multiple simultaneous emotions.

    Handles:
    - Emotion co-occurrence
    - Valence calculation (positive/negative)
    - Arousal calculation (activation level)
    - Dominant emotion selection
    """

    def __init__(self) -> None:
        super().__init__("MultiEmotionDetector")
        self._valence_map: dict[EmotionType, float] = {}
        self._arousal_map: dict[EmotionType, float] = {}

    async def initialize(self) -> None:
        """Initialize multi-emotion detector."""
        self._load_emotion_dimensions()

    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    async def analyze_multi_emotions(
        self,
        emotions: List[Emotion],
    ) -> tuple[Optional[Emotion], float, float]:
        """
        Analyze multiple emotions.

        Args:
            emotions: List of detected emotions

        Returns:
            Tuple of (dominant_emotion, valence, arousal)
            - valence: -1 (negative) to +1 (positive)
            - arousal: 0 (calm) to 1 (excited)
        """
        await self.ensure_initialized()

        if not emotions:
            return None, 0.0, 0.0

        # Calculate dominant emotion
        dominant = self._find_dominant(emotions)

        # Calculate valence
        valence = self._calculate_valence(emotions)

        # Calculate arousal
        arousal = self._calculate_arousal(emotions)

        return dominant, valence, arousal

    def _find_dominant(self, emotions: List[Emotion]) -> Optional[Emotion]:
        """
        Find the dominant emotion.

        Dominant emotion is the one with highest (intensity * confidence).
        """
        if not emotions:
            return None

        return max(emotions, key=lambda e: float(e.intensity) * e.confidence)

    def _calculate_valence(self, emotions: List[Emotion]) -> float:
        """
        Calculate emotional valence.

        Valence represents the positivity/negativity of emotion:
        -1 = Very negative
        0 = Neutral
        +1 = Very positive
        """
        if not emotions:
            return 0.0

        weighted_valence = 0.0
        total_weight = 0.0

        for emotion in emotions:
            base_valence = self._valence_map.get(emotion.emotion_type, 0.0)
            weight = float(emotion.intensity) * emotion.confidence

            weighted_valence += base_valence * weight
            total_weight += weight

        return weighted_valence / total_weight if total_weight > 0 else 0.0

    def _calculate_arousal(self, emotions: List[Emotion]) -> float:
        """
        Calculate emotional arousal.

        Arousal represents the activation/energy level:
        0 = Very calm/low energy
        1 = Very excited/high energy
        """
        if not emotions:
            return 0.0

        weighted_arousal = 0.0
        total_weight = 0.0

        for emotion in emotions:
            base_arousal = self._arousal_map.get(emotion.emotion_type, 0.5)
            weight = float(emotion.intensity) * emotion.confidence

            weighted_arousal += base_arousal * weight
            total_weight += weight

        return weighted_arousal / total_weight if total_weight > 0 else 0.0

    def _load_emotion_dimensions(self) -> None:
        """
        Load valence and arousal values for each emotion.

        Based on Russell's circumplex model of affect.
        """
        # Valence: -1 (negative) to +1 (positive)
        self._valence_map = {
            EmotionType.JOY: 0.8,
            EmotionType.TRUST: 0.6,
            EmotionType.ANTICIPATION: 0.4,
            EmotionType.SURPRISE: 0.0,  # Can be positive or negative
            EmotionType.SADNESS: -0.8,
            EmotionType.DISGUST: -0.7,
            EmotionType.ANGER: -0.6,
            EmotionType.FEAR: -0.5,
        }

        # Arousal: 0 (calm) to 1 (excited)
        self._arousal_map = {
            EmotionType.JOY: 0.7,
            EmotionType.SURPRISE: 0.9,
            EmotionType.ANGER: 0.8,
            EmotionType.FEAR: 0.8,
            EmotionType.ANTICIPATION: 0.6,
            EmotionType.DISGUST: 0.5,
            EmotionType.TRUST: 0.4,
            EmotionType.SADNESS: 0.3,
        }
