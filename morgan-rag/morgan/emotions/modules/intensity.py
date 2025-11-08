"""
Intensity Analyzer Module.

Analyzes and refines emotion intensity based on:
1. Linguistic markers (capitalization, punctuation, repetition)
2. Contextual factors (time, frequency, topic)
3. Emotion combinations (how emotions influence each other)
4. User history (baseline emotional state)
"""

from __future__ import annotations

import re
from typing import List, Optional

from morgan.emotions.base import EmotionModule
from morgan.emotions.exceptions import EmotionAnalysisError
from morgan.emotions.types import (
    Emotion,
    EmotionContext,
    EmotionIntensity,
    EmotionType,
)


class IntensityAnalyzer(EmotionModule):
    """
    Analyzes and adjusts emotion intensity based on multiple factors.

    This module takes initial emotion classifications and refines their
    intensities based on linguistic cues, context, and emotional interactions.
    """

    def __init__(self) -> None:
        super().__init__("IntensityAnalyzer")

    async def initialize(self) -> None:
        """Initialize intensity analyzer."""
        pass  # No resources to load

    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass  # No resources to release

    async def analyze_intensity(
        self,
        emotions: List[Emotion],
        text: str,
        context: Optional[EmotionContext] = None,
    ) -> List[Emotion]:
        """
        Analyze and adjust emotion intensities.

        Args:
            emotions: Initial emotion classifications
            text: Original text
            context: Optional context for analysis

        Returns:
            Emotions with adjusted intensities

        Raises:
            EmotionAnalysisError: If analysis fails
        """
        await self.ensure_initialized()

        if not emotions:
            return []

        try:
            # Calculate intensity modifiers from text
            text_modifier = self._analyze_text_markers(text)

            # Calculate contextual modifiers
            context_modifier = self._analyze_context(context) if context else 1.0

            # Adjust each emotion
            adjusted_emotions: List[Emotion] = []

            for emotion in emotions:
                # Base intensity from classification
                base_intensity = float(emotion.intensity)

                # Apply text markers
                text_adjusted = base_intensity * text_modifier

                # Apply context
                context_adjusted = text_adjusted * context_modifier

                # Apply emotion interactions
                interaction_adjusted = self._apply_emotion_interactions(
                    emotion.emotion_type,
                    context_adjusted,
                    emotions,
                )

                # Clamp to valid range
                final_intensity = max(0.0, min(1.0, interaction_adjusted))

                # Calculate confidence adjustment
                confidence = self._calculate_confidence(
                    emotion,
                    text_modifier,
                    context_modifier,
                    context is not None,
                )

                adjusted_emotions.append(
                    Emotion(
                        emotion_type=emotion.emotion_type,
                        intensity=EmotionIntensity(final_intensity),
                        confidence=confidence,
                    )
                )

            return sorted(
                adjusted_emotions,
                key=lambda e: float(e.intensity) * e.confidence,
                reverse=True,
            )

        except Exception as e:
            raise EmotionAnalysisError(
                f"Failed to analyze intensity: {str(e)}", cause=e
            )

    def _analyze_text_markers(self, text: str) -> float:
        """
        Analyze text for intensity markers.

        Markers:
        - ALL CAPS: +0.3
        - Repeated punctuation (!!!, ???): +0.2 per repetition
        - Repeated letters (sooo, reallyyy): +0.1 per word
        - Emoji/emoticons: +0.2 (future enhancement)
        - Text length: Very short or very long can indicate intensity

        Returns:
            Intensity multiplier (0.5 to 2.0)
        """
        modifier = 1.0

        # Check for ALL CAPS
        if text.isupper() and len(text) > 5:
            modifier += 0.3
        elif any(word.isupper() and len(word) > 3 for word in text.split()):
            modifier += 0.15

        # Check for repeated punctuation
        exclamation_count = len(re.findall(r"!{2,}", text))
        question_count = len(re.findall(r"\?{2,}", text))
        modifier += (exclamation_count + question_count) * 0.2

        # Check for repeated letters
        repeated_letters = len(re.findall(r"(.)\1{2,}", text))
        modifier += repeated_letters * 0.1

        # Check for text length extremes
        word_count = len(text.split())
        if word_count <= 3:
            modifier += 0.1  # Short, intense messages
        elif word_count > 100:
            modifier += 0.2  # Long rants indicate intensity

        # Clamp modifier to reasonable range
        return max(0.5, min(2.0, modifier))

    def _analyze_context(self, context: EmotionContext) -> float:
        """
        Analyze context for intensity modifiers.

        Factors:
        - Rapid succession of messages: +0.2
        - Repeated emotions: +0.1 to +0.3 based on frequency
        - Topic sensitivity: Variable (future enhancement)

        Returns:
            Intensity multiplier (0.7 to 1.5)
        """
        modifier = 1.0

        # Check message timing
        if context.time_since_last_message is not None:
            if context.time_since_last_message < 5.0:
                # Rapid-fire messages indicate intensity
                modifier += 0.2
            elif context.time_since_last_message > 3600.0:
                # Long gap might indicate calmed down
                modifier -= 0.1

        # Check for repeated emotions in history
        if context.previous_emotions:
            recent_emotions = context.previous_emotions[-5:]  # Last 5 emotions
            emotion_counts: dict[EmotionType, int] = {}

            for emotion in recent_emotions:
                emotion_counts[emotion.emotion_type] = (
                    emotion_counts.get(emotion.emotion_type, 0) + 1
                )

            # If same emotion repeated, it's likely intensifying
            max_repetition = max(emotion_counts.values()) if emotion_counts else 0
            if max_repetition > 1:
                modifier += (max_repetition - 1) * 0.15

        # Clamp to reasonable range
        return max(0.7, min(1.5, modifier))

    def _apply_emotion_interactions(
        self,
        emotion_type: EmotionType,
        current_intensity: float,
        all_emotions: List[Emotion],
    ) -> float:
        """
        Apply emotion interaction effects.

        Some emotions amplify each other, others suppress each other:
        - Fear + Sadness = Amplify both
        - Joy + Surprise = Amplify both
        - Anger + Fear = Suppress fear (anger dominates)
        - Joy + Sadness = Suppress both (conflict)

        Returns:
            Adjusted intensity
        """
        if len(all_emotions) <= 1:
            return current_intensity

        # Get other significant emotions
        other_emotions = {
            e.emotion_type: float(e.intensity)
            for e in all_emotions
            if e.emotion_type != emotion_type and e.intensity >= 0.2
        }

        if not other_emotions:
            return current_intensity

        adjustment = 0.0

        # Define interaction rules
        amplifiers = {
            EmotionType.FEAR: [EmotionType.SADNESS],
            EmotionType.SADNESS: [EmotionType.FEAR],
            EmotionType.JOY: [EmotionType.SURPRISE, EmotionType.TRUST],
            EmotionType.SURPRISE: [EmotionType.JOY],
            EmotionType.ANGER: [EmotionType.DISGUST],
            EmotionType.DISGUST: [EmotionType.ANGER],
            EmotionType.TRUST: [EmotionType.JOY, EmotionType.ANTICIPATION],
            EmotionType.ANTICIPATION: [EmotionType.TRUST],
        }

        suppressors = {
            EmotionType.ANGER: [EmotionType.FEAR],
            EmotionType.JOY: [EmotionType.SADNESS, EmotionType.FEAR],
            EmotionType.TRUST: [EmotionType.FEAR, EmotionType.DISGUST],
        }

        # Apply amplifiers
        if emotion_type in amplifiers:
            for amplifier in amplifiers[emotion_type]:
                if amplifier in other_emotions:
                    adjustment += other_emotions[amplifier] * 0.15

        # Apply suppressors
        if emotion_type in suppressors:
            for suppressor in suppressors[emotion_type]:
                if suppressor in other_emotions:
                    adjustment -= other_emotions[suppressor] * 0.1

        return current_intensity + adjustment

    def _calculate_confidence(
        self,
        emotion: Emotion,
        text_modifier: float,
        context_modifier: float,
        has_context: bool,
    ) -> float:
        """
        Calculate confidence in the intensity measurement.

        Confidence increases when:
        - We have clear text markers
        - We have contextual information
        - The emotion is strong

        Returns:
            Confidence score (0-1)
        """
        base_confidence = emotion.confidence

        # Boost confidence if we have clear markers
        if abs(text_modifier - 1.0) > 0.3:
            base_confidence += 0.1

        # Boost confidence if we have context
        if has_context:
            base_confidence += 0.1

        # Boost confidence for strong emotions
        if emotion.intensity >= 0.7:
            base_confidence += 0.05

        return min(1.0, base_confidence)
