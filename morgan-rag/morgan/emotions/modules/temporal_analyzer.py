"""
Temporal Analyzer Module.

Analyzes emotional changes over time to detect trends and trajectories.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

from morgan.emotions.base import EmotionModule
from morgan.emotions.types import Emotion, EmotionResult, EmotionType


class TemporalAnalyzer(EmotionModule):
    """
    Analyzes temporal aspects of emotions.

    Tracks:
    - Emotion trajectories (improving/worsening)
    - Velocity of emotional change
    - Cyclical patterns
    - Stability metrics
    """

    def __init__(self, window_size: int = 10) -> None:
        """
        Initialize temporal analyzer.

        Args:
            window_size: Number of recent results to analyze
        """
        super().__init__("TemporalAnalyzer")
        self._window_size = window_size
        self._timelines: Dict[str, List[tuple[EmotionResult, datetime]]] = defaultdict(
            list
        )

    async def initialize(self) -> None:
        """Initialize temporal analyzer."""
        pass

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self._timelines.clear()

    async def add_result(
        self,
        user_id: str,
        result: EmotionResult,
    ) -> None:
        """Add result to timeline."""
        self._timelines[user_id].append((result, datetime.utcnow()))

        # Trim to window size
        if len(self._timelines[user_id]) > self._window_size:
            self._timelines[user_id] = self._timelines[user_id][-self._window_size :]

    async def analyze_trajectory(
        self,
        user_id: str,
    ) -> Dict[str, any]:
        """
        Analyze emotional trajectory.

        Returns:
            Dictionary with trajectory metrics:
            - direction: "improving", "worsening", "stable"
            - velocity: Rate of change
            - volatility: How much emotions fluctuate
        """
        await self.ensure_initialized()

        timeline = self._timelines.get(user_id, [])

        if len(timeline) < 3:
            return {
                "direction": "unknown",
                "velocity": 0.0,
                "volatility": 0.0,
                "data_points": len(timeline),
            }

        # Calculate valence trajectory
        valences = [result.valence for result, _ in timeline]

        # Direction: compare first half to second half
        mid = len(valences) // 2
        first_half_avg = sum(valences[:mid]) / mid if mid > 0 else 0
        second_half_avg = sum(valences[mid:]) / (len(valences) - mid)

        direction = "stable"
        if second_half_avg > first_half_avg + 0.2:
            direction = "improving"
        elif second_half_avg < first_half_avg - 0.2:
            direction = "worsening"

        # Velocity: average change between consecutive points
        changes = [
            valences[i + 1] - valences[i]
            for i in range(len(valences) - 1)
        ]
        velocity = sum(changes) / len(changes) if changes else 0.0

        # Volatility: standard deviation of valences
        mean_valence = sum(valences) / len(valences)
        variance = sum((v - mean_valence) ** 2 for v in valences) / len(valences)
        volatility = variance ** 0.5

        return {
            "direction": direction,
            "velocity": velocity,
            "volatility": volatility,
            "data_points": len(timeline),
            "current_valence": valences[-1],
            "trend": "upward" if velocity > 0.05 else "downward" if velocity < -0.05 else "flat",
        }

    async def detect_cycles(
        self,
        user_id: str,
    ) -> Optional[Dict[str, any]]:
        """
        Detect cyclical emotional patterns.

        Returns:
            Cycle information if detected, None otherwise
        """
        timeline = self._timelines.get(user_id, [])

        if len(timeline) < 6:
            return None

        # Simple cycle detection: look for repeated peaks and valleys
        valences = [result.valence for result, _ in timeline]

        peaks = []
        valleys = []

        for i in range(1, len(valences) - 1):
            if valences[i] > valences[i - 1] and valences[i] > valences[i + 1]:
                peaks.append(i)
            elif valences[i] < valences[i - 1] and valences[i] < valences[i + 1]:
                valleys.append(i)

        if len(peaks) >= 2 and len(valleys) >= 2:
            # Calculate average cycle length
            peak_intervals = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
            valley_intervals = [
                valleys[i + 1] - valleys[i] for i in range(len(valleys) - 1)
            ]

            all_intervals = peak_intervals + valley_intervals
            avg_cycle_length = sum(all_intervals) / len(all_intervals)

            return {
                "cycle_detected": True,
                "avg_cycle_length": avg_cycle_length,
                "peaks": len(peaks),
                "valleys": len(valleys),
            }

        return None

    async def get_stability_score(
        self,
        user_id: str,
    ) -> float:
        """
        Calculate emotional stability score.

        Returns:
            Stability score (0-1, higher = more stable)
        """
        trajectory = await self.analyze_trajectory(user_id)

        if trajectory["data_points"] < 3:
            return 0.5  # Unknown

        # Stability is inverse of volatility, capped at reasonable range
        volatility = trajectory["volatility"]

        # Convert volatility (0-2+ range) to stability (0-1 range)
        stability = max(0.0, min(1.0, 1.0 - (volatility / 2.0)))

        return stability
