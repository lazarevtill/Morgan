"""
Companion relationship management module.

This module provides functionality for building and maintaining meaningful
relationships with users over time, including profile building, conversation
adaptation, milestone tracking, and personalized interactions.
"""

from .relationship_manager import CompanionRelationshipManager
from .storage import CompanionStorage
from .schema import (
    CompanionDatabaseSchema,
    validate_companion_payload,
    validate_emotion_payload
)

__all__ = [
    'CompanionRelationshipManager',
    'CompanionStorage',
    'CompanionDatabaseSchema',
    'validate_companion_payload',
    'validate_emotion_payload'
]