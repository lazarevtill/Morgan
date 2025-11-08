"""
Communication system for emotional AI assistant.

Provides communication style adaptation, user preference learning,
emotional feedback processing, non-verbal cue detection, and cultural
emotional awareness for enhanced human-AI interaction.
"""

from .cultural import CulturalEmotionalAwareness
from .feedback import EmotionalFeedbackProcessor
from .nonverbal import NonVerbalCueDetector
from .preferences import UserPreferenceLearner
from .style import CommunicationStyleAdapter

__all__ = [
    "CommunicationStyleAdapter",
    "UserPreferenceLearner",
    "EmotionalFeedbackProcessor",
    "NonVerbalCueDetector",
    "CulturalEmotionalAwareness",
]
