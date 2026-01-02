"""
Habit and Routine Recognition Module for Morgan RAG.

This module provides comprehensive habit pattern detection, routine-based interactions,
intelligent reminders, habit-based adaptation, and wellness habit tracking.
"""

from .detector import HabitDetector, HabitPattern, HabitType
from .scheduler import RoutineScheduler, RoutineEvent, RoutineType
from .reminders import IntelligentReminderSystem, Reminder, ReminderType
from .adaptation import HabitBasedAdaptation, AdaptationStrategy
from .wellness import WellnessHabitTracker, WellnessMetric, WellnessGoal

__all__ = [
    "HabitDetector",
    "HabitPattern", 
    "HabitType",
    "RoutineScheduler",
    "RoutineEvent",
    "RoutineType",
    "IntelligentReminderSystem",
    "Reminder",
    "ReminderType",
    "HabitBasedAdaptation",
    "AdaptationStrategy",
    "WellnessHabitTracker",
    "WellnessMetric",
    "WellnessGoal",
]