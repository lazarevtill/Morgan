"""
Relationship intelligence modules for Morgan Assistant.

This module provides comprehensive relationship management capabilities:
- Relationship development and building
- Milestone detection and tracking
- Relationship timeline management
- Relationship dynamics analysis
- Relationship-based adaptation

All modules follow KISS principles with single responsibility and clean
interfaces.
"""

from .adaptation import RelationshipAdaptation
from .builder import RelationshipBuilder
from .dynamics import RelationshipDynamics
from .milestones import MilestoneDetector
from .timeline import RelationshipTimeline

__all__ = [
    "RelationshipBuilder",
    "MilestoneDetector",
    "RelationshipTimeline",
    "RelationshipDynamics",
    "RelationshipAdaptation",
]
