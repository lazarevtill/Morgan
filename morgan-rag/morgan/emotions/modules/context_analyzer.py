"""
Context Analyzer Module.

Analyzes conversational context to improve emotion detection accuracy.
Considers conversation flow, topic, and user state.
"""

from __future__ import annotations

from typing import List, Optional

from morgan.emotions.base import EmotionModule
from morgan.emotions.types import Emotion, EmotionContext, EmotionIntensity, EmotionType


class ContextAnalyzer(EmotionModule):
    """
    Analyzes context to refine emotion detection.

    Uses conversation context to:
    - Adjust emotion intensities
    - Infer unstated emotions
    - Detect emotional shifts
    """

    def __init__(self) -> None:
        super().__init__("ContextAnalyzer")

    async def initialize(self) -> None:
        """Initialize context analyzer."""
        pass

    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    async def analyze_context(
        self,
        emotions: List[Emotion],
        context: Optional[EmotionContext],
    ) -> List[Emotion]:
        """
        Analyze and adjust emotions based on context.

        Args:
            emotions: Detected emotions
            context: Conversation context

        Returns:
            Context-adjusted emotions
        """
        await self.ensure_initialized()

        if not context or not emotions:
            return emotions

        adjusted = emotions.copy()

        # Detect emotional shifts
        if context.has_history:
            adjusted = self._detect_emotional_shifts(adjusted, context)

        # Apply conversation continuity
        if context.time_since_last_message is not None:
            adjusted = self._apply_continuity(adjusted, context)

        # Infer implicit emotions
        inferred = self._infer_implicit_emotions(adjusted, context)
        adjusted.extend(inferred)

        return sorted(
            adjusted,
            key=lambda e: float(e.intensity) * e.confidence,
            reverse=True,
        )

    def _detect_emotional_shifts(
        self,
        current: List[Emotion],
        context: EmotionContext,
    ) -> List[Emotion]:
        """Detect and emphasize emotional shifts."""
        if not context.previous_emotions:
            return current

        # Get last dominant emotion
        last_emotion = max(
            context.previous_emotions,
            key=lambda e: float(e.intensity) * e.confidence,
        )

        adjusted = []

        for emotion in current:
            # If emotion is opposite to previous, it's likely a significant shift
            if emotion.emotion_type == last_emotion.emotion_type.opposite():
                # Boost confidence in the shift
                adjusted.append(
                    Emotion(
                        emotion_type=emotion.emotion_type,
                        intensity=emotion.intensity,
                        confidence=min(emotion.confidence + 0.1, 1.0),
                    )
                )
            else:
                adjusted.append(emotion)

        return adjusted

    def _apply_continuity(
        self,
        emotions: List[Emotion],
        context: EmotionContext,
    ) -> List[Emotion]:
        """Apply emotional continuity over time."""
        if context.time_since_last_message is None or not context.previous_emotions:
            return emotions

        time_gap = context.time_since_last_message

        # If rapid succession (< 10 seconds), emotions likely continue
        if time_gap < 10.0 and context.previous_emotions:
            # Carry forward some emotional momentum
            last_emotions = {e.emotion_type: e for e in context.previous_emotions}

            adjusted = []
            for emotion in emotions:
                if emotion.emotion_type in last_emotions:
                    # This emotion is continuing
                    last = last_emotions[emotion.emotion_type]
                    momentum = float(last.intensity) * 0.3  # 30% carry-forward

                    new_intensity = min(float(emotion.intensity) + momentum, 1.0)

                    adjusted.append(
                        Emotion(
                            emotion_type=emotion.emotion_type,
                            intensity=EmotionIntensity(new_intensity),
                            confidence=emotion.confidence,
                        )
                    )
                else:
                    adjusted.append(emotion)

            return adjusted

        # If long gap (> 1 hour), emotions likely reset
        elif time_gap > 3600.0:
            # Reduce influence of previous emotions
            return emotions

        return emotions

    def _infer_implicit_emotions(
        self,
        explicit: List[Emotion],
        context: EmotionContext,
    ) -> List[Emotion]:
        """Infer emotions not explicitly detected but contextually likely."""
        inferred = []

        # If someone is consistently sad, they might also have suppressed anger
        if any(
            e.emotion_type == EmotionType.SADNESS and e.intensity >= 0.6
            for e in explicit
        ):
            if context.previous_emotions and any(
                e.emotion_type == EmotionType.SADNESS
                for e in context.previous_emotions[-3:]
            ):
                # Chronic sadness might include suppressed anger
                if not any(e.emotion_type == EmotionType.ANGER for e in explicit):
                    inferred.append(
                        Emotion(
                            emotion_type=EmotionType.ANGER,
                            intensity=EmotionIntensity(0.3),
                            confidence=0.5,
                        )
                    )

        # If someone shows fear, they might also feel anticipation (of threat)
        if any(
            e.emotion_type == EmotionType.FEAR and e.intensity >= 0.6
            for e in explicit
        ):
            if not any(e.emotion_type == EmotionType.ANTICIPATION for e in explicit):
                inferred.append(
                    Emotion(
                        emotion_type=EmotionType.ANTICIPATION,
                        intensity=EmotionIntensity(0.4),
                        confidence=0.6,
                    )
                )

        return inferred
