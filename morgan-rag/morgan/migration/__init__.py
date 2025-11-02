"""
Migration system for Morgan knowledge bases.

Provides tools for migrating existing knowledge bases to new formats,
particularly for transitioning from legacy single-vector embeddings
to hierarchical multi-scale embeddings.

Implements requirements R10.4 and R10.5 for migration safety and rollback.
"""

from .migrator import KnowledgeBaseMigrator
from .validator import MigrationValidator
from .rollback import RollbackManager

__all__ = [
    "KnowledgeBaseMigrator",
    "MigrationValidator", 
    "RollbackManager"
]