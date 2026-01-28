"""
Integration tests for habits module integration.

Tests that the HabitDetector, HabitBasedAdaptation, and WellnessTracker
can be instantiated and used correctly.
"""
import pytest
from datetime import datetime, time
from unittest.mock import MagicMock, patch


class TestHabitDetectorIntegration:
    """Test HabitDetector integration."""

    def test_habit_detector_instantiation(self):
        """HabitDetector should instantiate cleanly."""
        from morgan.habits.detector import HabitDetector

        detector = HabitDetector()
        assert detector is not None

    def test_habit_detector_detect_habits(self):
        """HabitDetector.detect_habits should handle empty interactions."""
        from morgan.habits.detector import HabitDetector

        detector = HabitDetector()
        # Call with empty interactions - should return analysis with no habits
        result = detector.detect_habits(
            user_id="test_user",
            interactions=[],
        )
        assert result is not None
        assert result.user_id == "test_user"

    def test_habit_types_enum(self):
        """HabitType enum should have expected values."""
        from morgan.habits.detector import HabitType

        assert HabitType.COMMUNICATION.value == "communication"
        assert HabitType.WORK.value == "work"
        assert HabitType.LEARNING.value == "learning"
        assert HabitType.WELLNESS.value == "wellness"

    def test_habit_frequency_enum(self):
        """HabitFrequency enum should have expected values."""
        from morgan.habits.detector import HabitFrequency

        assert HabitFrequency.DAILY.value == "daily"
        assert HabitFrequency.WEEKLY.value == "weekly"
        assert HabitFrequency.MONTHLY.value == "monthly"

    def test_habit_confidence_enum(self):
        """HabitConfidence enum should have expected values."""
        from morgan.habits.detector import HabitConfidence

        assert HabitConfidence.LOW.value == "low"
        assert HabitConfidence.MEDIUM.value == "medium"
        assert HabitConfidence.HIGH.value == "high"


class TestHabitAdaptationIntegration:
    """Test HabitBasedAdaptation integration."""

    def test_habit_adaptation_instantiation(self):
        """HabitBasedAdaptation should instantiate cleanly."""
        from morgan.habits.adaptation import HabitBasedAdaptation

        adaptation = HabitBasedAdaptation()
        assert adaptation is not None


class TestWellnessTrackerIntegration:
    """Test WellnessTracker integration."""

    def test_wellness_tracker_instantiation(self):
        """WellnessHabitTracker should instantiate cleanly."""
        from morgan.habits.wellness import WellnessHabitTracker

        tracker = WellnessHabitTracker()
        assert tracker is not None
