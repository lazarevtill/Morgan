"""
Routine-Based Interaction Scheduler for Morgan RAG.

Schedules proactive interactions based on detected user habits and routines
to provide timely assistance and maintain engagement.
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
from typing import Dict, List, Optional, Callable, Any

from .detector import HabitPattern, HabitType, HabitAnalysis
from ..emotional.models import InteractionData
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RoutineType(Enum):
    """Types of routine-based interactions."""
    
    PROACTIVE_CHECK_IN = "proactive_check_in"  # Regular check-ins
    HABIT_REMINDER = "habit_reminder"  # Reminders for habits
    CONTEXT_PREPARATION = "context_preparation"  # Prepare context for expected interactions
    WELLNESS_CHECK = "wellness_check"  # Wellness-related check-ins
    LEARNING_SUPPORT = "learning_support"  # Learning-related support
    PRODUCTIVITY_BOOST = "productivity_boost"  # Productivity assistance
    SOCIAL_ENGAGEMENT = "social_engagement"  # Social interaction prompts


class RoutineStatus(Enum):
    """Status of routine events."""
    
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class RoutinePriority(Enum):
    """Priority levels for routine events."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class RoutineEvent:
    """Represents a scheduled routine-based interaction."""
    
    event_id: str
    user_id: str
    routine_type: RoutineType
    priority: RoutinePriority
    
    # Scheduling information
    scheduled_time: datetime
    estimated_duration