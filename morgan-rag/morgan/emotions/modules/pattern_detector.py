"""
Pattern Detector Module.

Detects patterns in emotional expressions over time:
1. Recurring patterns (same emotion repeatedly)
2. Escalating patterns (emotions intensifying)
3. Alternating patterns (flip-flopping between emotions)
4. Suppressed patterns (emotions being held back)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from morgan.emotions.base import EmotionModule
from morgan.emotions.exceptions import EmotionAnalysisError
from morgan.emotions.types import (
    Emotion,
    EmotionContext,
    EmotionPattern,
    EmotionType,
)


class PatternDetector(EmotionModule):
    """
    Detects emotional patterns over time.

    Maintains a sliding window of emotional history and identifies
    meaningful patterns that can inform response generation.
    """

    def __init__(self, window_size: int = 20, min_pattern_frequency: int = 2) -> None:
        """
        Initialize pattern detector.

        Args:
            window_size: Number of recent emotions to analyze
            min_pattern_frequency: Minimum occurrences to constitute a pattern
        """
        super().__init__("PatternDetector")
        self._window_size = window_size
        self._min_pattern_frequency = min_pattern_frequency
        self._emotion_history: Dict[str, List[tuple[Emotion, datetime]]] = defaultdict(
            list
        )

    async def initialize(self) -> None:
        """Initialize pattern detector."""
        pass

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self._emotion_history.clear()

    async def detect_patterns(
        self,
        current_emotions: List[Emotion],
        context: Optional[EmotionContext] = None,
    ) -> List[EmotionPattern]:
        """
        Detect emotional patterns.

        Args:
            current_emotions: Current detected emotions
            context: Optional context with history

        Returns:
            List of detected patterns

        Raises:
            EmotionAnalysisError: If pattern detection fails
        """
        await self.ensure_initialized()

        try:
            patterns: List[EmotionPattern] = []

            # Need context with user_id for pattern tracking
            if not context or not context.user_id:
                return patterns

            # Add current emotions to history
            self._add_to_history(context.user_id, current_emotions)

            # Get history window
            history = self._get_history_window(context.user_id)

            if len(history) < self._min_pattern_frequency:
                return patterns

            # Detect various pattern types
            patterns.extend(self._detect_recurring_patterns(history))
            patterns.extend(self._detect_escalating_patterns(history))
            patterns.extend(self._detect_alternating_patterns(history))
            patterns.extend(self._detect_suppression_patterns(history, current_emotions))

            return sorted(patterns, key=lambda p: p.confidence, reverse=True)

        except Exception as e:
            raise EmotionAnalysisError(
                f"Failed to detect patterns: {str(e)}", cause=e
            )

    def _add_to_history(self, user_id: str, emotions: List[Emotion]) -> None:
        """Add emotions to user history."""
        timestamp = datetime.utcnow()
        for emotion in emotions:
            self._emotion_history[user_id].append((emotion, timestamp))

        # Trim history to window size
        if len(self._emotion_history[user_id]) > self._window_size:
            self._emotion_history[user_id] = self._emotion_history[user_id][
                -self._window_size :
            ]

    def _get_history_window(
        self, user_id: str
    ) -> List[tuple[Emotion, datetime]]:
        """Get recent emotion history for user."""
        return self._emotion_history.get(user_id, [])

    def _detect_recurring_patterns(
        self, history: List[tuple[Emotion, datetime]]
    ) -> List[EmotionPattern]:
        """
        Detect recurring emotions.

        A pattern is "recurring" if the same emotion appears frequently
        within the window.
        """
        patterns: List[EmotionPattern] = []

        # Count emotion occurrences
        emotion_counts = Counter(emotion.emotion_type for emotion, _ in history)

        for emotion_type, count in emotion_counts.items():
            if count >= self._min_pattern_frequency:
                # Calculate frequency within window
                frequency_ratio = count / len(history)

                # Get timestamps
                timestamps = [
                    ts for emotion, ts in history if emotion.emotion_type == emotion_type
                ]

                if timestamps:
                    patterns.append(
                        EmotionPattern(
                            pattern_type="recurring",
                            emotions_involved=[emotion_type],
                            frequency=count,
                            first_seen=min(timestamps),
                            last_seen=max(timestamps),
                            confidence=min(0.5 + (frequency_ratio * 0.5), 0.95),
                            description=f"Recurring {emotion_type.value} emotion "
                            f"({count} occurrences)",
                        )
                    )

        return patterns

    def _detect_escalating_patterns(
        self, history: List[tuple[Emotion, datetime]]
    ) -> List[EmotionPattern]:
        """
        Detect escalating emotion intensity.

        A pattern is "escalating" if the same emotion's intensity
        increases over time.
        """
        patterns: List[EmotionPattern] = []

        # Group by emotion type
        by_type: Dict[EmotionType, List[tuple[float, datetime]]] = defaultdict(list)
        for emotion, timestamp in history:
            by_type[emotion.emotion_type].append((float(emotion.intensity), timestamp))

        for emotion_type, intensities in by_type.items():
            if len(intensities) < 3:
                continue

            # Check if intensities are generally increasing
            values = [intensity for intensity, _ in intensities]

            # Simple escalation check: compare first half to second half
            mid = len(values) // 2
            first_half_avg = sum(values[:mid]) / mid
            second_half_avg = sum(values[mid:]) / (len(values) - mid)

            if second_half_avg > first_half_avg * 1.2:  # 20% increase
                timestamps = [ts for _, ts in intensities]
                patterns.append(
                    EmotionPattern(
                        pattern_type="escalating",
                        emotions_involved=[emotion_type],
                        frequency=len(intensities),
                        first_seen=min(timestamps),
                        last_seen=max(timestamps),
                        confidence=min(
                            0.6 + ((second_half_avg - first_half_avg) * 0.4), 0.95
                        ),
                        description=f"Escalating {emotion_type.value} emotion "
                        f"(intensity increasing)",
                    )
                )

        return patterns

    def _detect_alternating_patterns(
        self, history: List[tuple[Emotion, datetime]]
    ) -> List[EmotionPattern]:
        """
        Detect alternating between opposite emotions.

        A pattern is "alternating" if emotions flip between opposites
        (e.g., joy <-> sadness).
        """
        patterns: List[EmotionPattern] = []

        if len(history) < 4:
            return patterns

        # Look for sequences of opposite emotions
        dominant_emotions = [
            (emotion.emotion_type, timestamp)
            for emotion, timestamp in history
            if emotion.intensity >= 0.5
        ]

        if len(dominant_emotions) < 4:
            return patterns

        # Check for alternation
        alternations = 0
        last_emotion = None

        for emotion_type, _ in dominant_emotions:
            if last_emotion and emotion_type == last_emotion.opposite():
                alternations += 1
            last_emotion = emotion_type

        if alternations >= 2:
            involved_emotions = list(
                set(emotion_type for emotion_type, _ in dominant_emotions)
            )
            timestamps = [ts for _, ts in dominant_emotions]

            patterns.append(
                EmotionPattern(
                    pattern_type="alternating",
                    emotions_involved=involved_emotions,
                    frequency=alternations,
                    first_seen=min(timestamps),
                    last_seen=max(timestamps),
                    confidence=min(0.5 + (alternations * 0.1), 0.90),
                    description=f"Alternating between {' and '.join(e.value for e in involved_emotions)} "
                    f"({alternations} switches)",
                )
            )

        return patterns

    def _detect_suppression_patterns(
        self,
        history: List[tuple[Emotion, datetime]],
        current_emotions: List[Emotion],
    ) -> List[EmotionPattern]:
        """
        Detect suppressed emotions.

        A pattern is "suppressed" if historical emotions suggest one emotion
        but current detection shows weak or absent signals.
        """
        patterns: List[EmotionPattern] = []

        if len(history) < 5:
            return patterns

        # Get recent dominant emotion from history (excluding current)
        recent_history = history[:-len(current_emotions)] if current_emotions else history
        if not recent_history:
            return patterns

        emotion_counts = Counter(
            emotion.emotion_type
            for emotion, _ in recent_history
            if emotion.intensity >= 0.5
        )

        for emotion_type, count in emotion_counts.items():
            if count < 3:
                continue

            # Check if this emotion is weak or absent in current detection
            current_intensity = next(
                (
                    float(e.intensity)
                    for e in current_emotions
                    if e.emotion_type == emotion_type
                ),
                0.0,
            )

            if current_intensity < 0.3:
                # Potentially suppressed
                timestamps = [
                    ts
                    for emotion, ts in recent_history
                    if emotion.emotion_type == emotion_type
                ]

                patterns.append(
                    EmotionPattern(
                        pattern_type="suppressed",
                        emotions_involved=[emotion_type],
                        frequency=count,
                        first_seen=min(timestamps),
                        last_seen=max(timestamps),
                        confidence=0.6,
                        description=f"Possibly suppressed {emotion_type.value} emotion "
                        f"(historically present, currently weak)",
                    )
                )

        return patterns

    async def clear_history(self, user_id: str) -> None:
        """Clear emotion history for a user."""
        if user_id in self._emotion_history:
            del self._emotion_history[user_id]
