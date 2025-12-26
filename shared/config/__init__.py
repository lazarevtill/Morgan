"""
Shared configuration module for Morgan AI Assistant.

Provides base configuration classes and utilities used across all services.
"""
from shared.config.base import MorganBaseSettings
from shared.config.defaults import DEFAULTS

__all__ = [
    "MorganBaseSettings",
    "DEFAULTS",
]
