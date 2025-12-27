"""
Communication system for emotional AI assistant.

Provides communication style adaptation, user preference learning,
emotional feedback processing, and non-verbal cue detection
for enhanced human-AI interaction.
"""

from .feedback import EmotionalFeedbackProcessor
from .nonverbal import NonVerbalCueDetector
from .preferences import UserPreferenceLearner
from .style import CommunicationStyleAdapter

__all__ = [
    "CommunicationStyleAdapter",
    "UserPreferenceLearner",
    "EmotionalFeedbackProcessor",
    "NonVerbalCueDetector",
]
