"""
Learning modules package.

Contains specialized learning modules for different aspects
of the learning system.
"""

from morgan.learning.modules.adaptation_module import AdaptationModule
from morgan.learning.modules.consolidation_module import ConsolidationModule
from morgan.learning.modules.feedback_module import FeedbackModule
from morgan.learning.modules.pattern_module import PatternModule
from morgan.learning.modules.preference_module import PreferenceModule

__all__ = [
    "AdaptationModule",
    "ConsolidationModule",
    "FeedbackModule",
    "PatternModule",
    "PreferenceModule",
]
