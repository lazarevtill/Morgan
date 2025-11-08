"""
History Tracker Module.

Maintains emotional history for users across conversations.
Provides persistence and retrieval of emotional states.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from morgan.emotions.base import EmotionModule
from morgan.emotions.exceptions import EmotionHistoryError
from morgan.emotions.types import Emotion, EmotionResult, EmotionType


class EmotionHistoryTracker(EmotionModule):
    """
    Tracks and persists emotional history.

    Maintains a timeline of emotional states for each user,
    enabling pattern detection and personalization.
    """

    def __init__(
        self,
        max_history_per_user: int = 1000,
        retention_days: int = 90,
        storage_path: Optional[Path] = None,
    ) -> None:
        """
        Initialize history tracker.

        Args:
            max_history_per_user: Maximum history entries per user
            retention_days: Days to retain history
            storage_path: Optional path for persistent storage
        """
        super().__init__("EmotionHistoryTracker")
        self._max_history = max_history_per_user
        self._retention_days = retention_days
        self._storage_path = storage_path
        self._history: Dict[str, List[EmotionResult]] = defaultdict(list)

    async def initialize(self) -> None:
        """Initialize history tracker and load persisted data."""
        if self._storage_path:
            await self._load_history()

    async def cleanup(self) -> None:
        """Cleanup and persist history."""
        if self._storage_path:
            await self._save_history()
        self._history.clear()

    async def add_result(
        self, user_id: str, result: EmotionResult
    ) -> None:
        """
        Add emotion result to history.

        Args:
            user_id: User identifier
            result: Emotion detection result
        """
        try:
            self._history[user_id].append(result)

            # Trim to max history size
            if len(self._history[user_id]) > self._max_history:
                self._history[user_id] = self._history[user_id][-self._max_history :]

            # Cleanup old entries
            await self._cleanup_old_entries(user_id)

        except Exception as e:
            raise EmotionHistoryError(f"Failed to add history: {str(e)}", cause=e)

    async def get_recent(
        self,
        user_id: str,
        limit: int = 10,
    ) -> List[EmotionResult]:
        """
        Get recent emotional history.

        Args:
            user_id: User identifier
            limit: Maximum number of results to return

        Returns:
            List of recent emotion results
        """
        history = self._history.get(user_id, [])
        return history[-limit:]

    async def get_dominant_emotions(
        self,
        user_id: str,
        days: int = 7,
    ) -> Dict[EmotionType, float]:
        """
        Get dominant emotions over a time period.

        Args:
            user_id: User identifier
            days: Number of days to analyze

        Returns:
            Dictionary mapping emotion types to average intensity
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        history = self._history.get(user_id, [])

        recent = [r for r in history if r.timestamp >= cutoff]

        if not recent:
            return {}

        # Aggregate emotions
        emotion_totals: Dict[EmotionType, List[float]] = defaultdict(list)

        for result in recent:
            for emotion in result.primary_emotions:
                emotion_totals[emotion.emotion_type].append(float(emotion.intensity))

        # Calculate averages
        averages = {
            emotion_type: sum(intensities) / len(intensities)
            for emotion_type, intensities in emotion_totals.items()
        }

        return dict(sorted(averages.items(), key=lambda x: x[1], reverse=True))

    async def get_baseline_state(
        self,
        user_id: str,
    ) -> Dict[EmotionType, float]:
        """
        Get user's baseline emotional state.

        This represents their "normal" emotional state based on
        long-term history.

        Args:
            user_id: User identifier

        Returns:
            Dictionary mapping emotion types to baseline intensity
        """
        # Use last 30 days for baseline
        return await self.get_dominant_emotions(user_id, days=30)

    async def clear_history(self, user_id: str) -> None:
        """Clear history for a user."""
        if user_id in self._history:
            del self._history[user_id]

    async def _cleanup_old_entries(self, user_id: str) -> None:
        """Remove entries older than retention period."""
        cutoff = datetime.utcnow() - timedelta(days=self._retention_days)
        history = self._history[user_id]

        self._history[user_id] = [r for r in history if r.timestamp >= cutoff]

    async def _load_history(self) -> None:
        """Load history from storage."""
        if not self._storage_path or not self._storage_path.exists():
            return

        try:
            with open(self._storage_path, "r") as f:
                data = json.load(f)

            # Deserialize history
            # This is simplified - in production, use proper serialization
            self._history = defaultdict(list, data)

        except Exception as e:
            # Log error but don't fail initialization
            print(f"Failed to load emotion history: {e}")

    async def _save_history(self) -> None:
        """Save history to storage."""
        if not self._storage_path:
            return

        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Serialize history
            # This is simplified - in production, use proper serialization
            with open(self._storage_path, "w") as f:
                json.dump(dict(self._history), f, default=str)

        except Exception as e:
            # Log error but don't fail
            print(f"Failed to save emotion history: {e}")
