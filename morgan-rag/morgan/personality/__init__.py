"""
Personality and Style Modules for Morgan RAG.

Provides personality trait modeling, communication style adaptation,
humor detection and generation, formality level adjustment, and energy level matching.
"""

from .energy import EnergyLevel, EnergyLevelMatcher
from .formality import FormalityLevel, FormalityLevelAdjuster
from .humor import HumorDetector, HumorGenerator, HumorStyle
from .style import CommunicationStyleAdapter, StyleAdaptation
from .traits import PersonalityProfile, PersonalityTraitModeler

__all__ = [
    "PersonalityTraitModeler",
    "PersonalityProfile",
    "CommunicationStyleAdapter",
    "StyleAdaptation",
    "HumorDetector",
    "HumorGenerator",
    "HumorStyle",
    "FormalityLevelAdjuster",
    "FormalityLevel",
    "EnergyLevelMatcher",
    "EnergyLevel",
]
