"""
Emotional trigger detection module.

Provides focused emotional trigger identification, sensitivity analysis,
and trigger management for enhanced emotional intelligence and user support.
"""

import threading
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
import hashlib

from morgan.config import get_settings
from morgan.utils.logger import get_logger
from morgan.emotions.memory import get_emotional_memory_storage, EmotionalMemory
from morgan.emotional.models import EmotionalState, EmotionType

logger = get_logger(__name__)


class EmotionalTrigger:
    """
    Represents an identified emotional trigger.
    
    Features:
    - Trigger pattern identification
    - Emotional response tracking
    - Sensitivity scoring
    - Context awareness
    """
    
    def __init__(
        self,
        trigger_id: str,
        user_id: str,
        trigger_pattern: s