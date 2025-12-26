"""
Domain Expertise Module for Morgan RAG.

Provides domain knowledge tracking, specialized vocabulary learning,
context understanding, knowledge depth assessment, and adaptive teaching strategies.
"""

from .domains import DomainKnowledgeTracker, DomainProfile
from .vocabulary import VocabularyLearner, DomainVocabulary
from .context import DomainContextEngine, DomainContext
from .depth import KnowledgeDepthAssessor, KnowledgeLevel
from .teaching import AdaptiveTeachingEngine, TeachingStrategy

__all__ = [
    "DomainKnowledgeTracker",
    "DomainProfile",
    "VocabularyLearner",
    "DomainVocabulary",
    "DomainContextEngine",
    "DomainContext",
    "KnowledgeDepthAssessor",
    "KnowledgeLevel",
    "AdaptiveTeachingEngine",
    "TeachingStrategy",
]
