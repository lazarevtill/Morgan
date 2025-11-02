"""
Morgan Models Module

Unified model management for local and remote AI models.
Follows KISS principles with single-responsibility components.

Components:
- manager.py: Central model coordination
- local.py: Local model integration (Ollama, Transformers)
- lazarev.py: gpt.lazarev.cloud endpoint integration
- cache.py: Model caching and optimization
- selector.py: Model selection logic

Requirements addressed: 23.1, 23.2, 23.3
"""

from .manager import ModelManager
from .local import LocalModelManager
from .lazarev import LazarevModelManager
from .cache import ModelCache
from .selector import ModelSelector

__all__ = [
    'ModelManager',
    'LocalModelManager',
    'LazarevModelManager',
    'ModelCache',
    'ModelSelector'
]