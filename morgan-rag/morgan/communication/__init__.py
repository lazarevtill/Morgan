"""
Communication system for emotional AI assistant.

Provides communication style adaptation, user preference learning,
emotional feedback processing, non-verbal cue detection, and cultural
emotional awareness for enhanced human-AI interaction.
"""

from .style import CommunicationStyleAdapter
from .preferences import UserPreferenceLearner
from .feedback import EmotionalFeedbackProcessor
from .nonverbal import NonVerbalCueDetector
from .cultural import CulturalEmotionalAwareness

__all__ = [
    'CommunicationStyleAdapter',
    'UserPreferenceLearner', 
    'EmotionalFeedbackProcessor',
    'NonVerbalCueDetector',
    'CulturalEmotionalAwareness'
]