"""
Morgan Storage Module

Unified data persistence for all Morgan components.
Follows KISS principles with single-responsibility storage components.

Components:
- vector.py: Vector database operations
- memory.py: Conversation and emotional memory storage
- profile.py: User profiles and preferences storage
- cache.py: Performance caching and optimization
- backup.py: Data backup and recovery operations

Requirements addressed: 23.1, 23.4, 23.5
"""

from .vector import VectorStorage
from .memory import MemoryStorage
from .profile import ProfileStorage
from .cache import CacheStorage
from .backup import BackupStorage

__all__ = [
    'VectorStorage',
    'MemoryStorage',
    'ProfileStorage',
    'CacheStorage',
    'BackupStorage'
]