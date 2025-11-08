"""
Emotion Aggregator Module.

Aggregates results from all emotion detection modules into a final result.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import List, Optional

from morgan.emotions.base import EmotionModule
from morgan.emotions.exceptions import EmotionAnalysisError
from morgan.emotions.types import (
    Emotion,
    EmotionContext,
    EmotionPattern,
    EmotionResult,
    EmotionTrigger,
)


class EmotionAggregator(EmotionModule):
    """
    Aggregates emotion detection results.

    Combines outputs from multiple modules into a coherent final result.
    """

    def __init__(self) -> None:
        super().__init__("EmotionAggregator")

    async def initialize(self) -> None:
        """Initialize aggregator."""
        pass

    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    async def aggregate(
        self,
        emotions: List[Emotion],
        dominant_emotion: Optional[Emotion],
        valence: float,
        arousal: float,
        triggers: List[EmotionTrigger],
        patterns: List[EmotionPattern],
        context: Optional[EmotionContext] = None,
        processing_time_ms: float = 0.0,
        warnings: Optional[List[str]] = None,
    ) -> EmotionResult:
        """
        Aggregate all results into final EmotionResult.

        Args:
            emotions: Detected emotions
            dominant_emotion: Dominant emotion
            valence: Overall valence
            arousal: Overall arousal
            triggers: Detected triggers
            patterns: Detected patterns
            context: Context used
            processing_time_ms: Processing time
            warnings: Any warnings

        Returns:
            Final EmotionResult

        Raises:
            EmotionAnalysisError: If aggregation fails
        """
        await self.ensure_initialized()

        try:
            # Sort emotions by significance
            sorted_emotions = sorted(
                [e for e in emotions if e.is_significant],
                key=lambda e: float(e.intensity) * e.confidence,
                reverse=True,
            )

            # Limit to top emotions
            top_emotions = sorted_emotions[:5]

            # Add processing warnings
            result_warnings = warnings or []

            # Warning if no significant emotions detected
            if not top_emotions:
                result_warnings.append("No significant emotions detected")

            # Warning if processing time exceeds target
            if processing_time_ms > 200.0:
                result_warnings.append(
                    f"Processing time ({processing_time_ms:.1f}ms) exceeds target (200ms)"
                )

            # Warning for conflicting emotions
            if len(top_emotions) >= 2:
                conflicts = self._detect_conflicts(top_emotions)
                if conflicts:
                    result_warnings.append(f"Conflicting emotions detected: {conflicts}")

            return EmotionResult(
                primary_emotions=top_emotions,
                dominant_emotion=dominant_emotion,
                valence=valence,
                arousal=arousal,
                triggers=triggers,
                patterns=patterns,
                context=context,
                timestamp=datetime.utcnow(),
                processing_time_ms=processing_time_ms,
                warnings=result_warnings,
            )

        except Exception as e:
            raise EmotionAnalysisError(
                f"Failed to aggregate results: {str(e)}", cause=e
            )

    def _detect_conflicts(self, emotions: List[Emotion]) -> Optional[str]:
        """
        Detect conflicting emotions.

        Returns description of conflict, or None.
        """
        # Check for opposite emotions with similar intensities
        for i, e1 in enumerate(emotions):
            for e2 in emotions[i + 1 :]:
                if e1.emotion_type == e2.emotion_type.opposite():
                    intensity_diff = abs(float(e1.intensity) - float(e2.intensity))
                    if intensity_diff < 0.2:
                        return f"{e1.emotion_type.value} vs {e2.emotion_type.value}"

        return None
