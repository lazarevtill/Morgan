"""
Conversation intelligence system for emotional AI assistant.

Provides conversation flow management, topic preference learning,
optimal timing detection, interruption handling, and conversation
quality assessment for enhanced conversational experiences.
"""

from .flow import ConversationFlowManager
from .interruption import InterruptionHandler
from .quality import ConversationQualityAssessor
from .timing import OptimalTimingDetector
from .topics import TopicPreferenceLearner

__all__ = [
    "ConversationFlowManager",
    "TopicPreferenceLearner",
    "OptimalTimingDetector",
    "InterruptionHandler",
    "ConversationQualityAssessor",
]