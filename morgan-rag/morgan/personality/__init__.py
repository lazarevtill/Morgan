"""
Personality and Style Modules for Morgan RAG.

Provides personality trait modeling, communication style adaptation,
humor detection and generation, formality level adjustment, and energy
level matching.
"""

# Import personality modules
from .traits import PersonalityProfile, PersonalityTraitModeler
from .style import CommunicationStyleAdapter, StyleAdaptation
from .humor import HumorDetector, HumorGenerator, HumorStyle
from .formality import FormalityLevel, FormalityLevelAdjuster
from .energy import EnergyLevel, EnergyLevelMatcher

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
