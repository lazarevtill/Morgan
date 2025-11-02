"""
Personality and Style Modules for Morgan RAG.

Provides personality trait modeling, communication style adaptation,
humor detection and generation, formality level adjustment, and energy level matching.
"""

from .traits import PersonalityTraitModeler, PersonalityProfile
from .style import CommunicationStyleAdapter, StyleAdaptation
from .humor import HumorDetector, HumorGenerator, HumorStyle
from .formality import FormalityLevelAdjuster, FormalityLevel
from .energy import EnergyLevelMatcher, EnergyLevel

__all__ = [
    'PersonalityTraitModeler',
    'PersonalityProfile',
    'CommunicationStyleAdapter',
    'StyleAdaptation',
    'HumorDetector',
    'HumorGenerator',
    'HumorStyle',
    'FormalityLevelAdjuster',
    'FormalityLevel',
    'EnergyLevelMatcher',
    'EnergyLevel'
]