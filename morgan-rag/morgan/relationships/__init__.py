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

from .builder import RelationshipBuilder
from .milestones import MilestoneDetector
from .timeline import RelationshipTimeline
from .dynamics import RelationshipDynamics
from .adaptation import RelationshipAdaptation

__all__ = [
    'RelationshipBuilder',
    'MilestoneDetector',
    'RelationshipTimeline',
    'RelationshipDynamics',
    'RelationshipAdaptation'
]