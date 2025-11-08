"""
Morgan Configuration Module

Centralized configuration management for all Morgan components.
Follows KISS principles with simple, focused configuration.

Components:
- settings.py: Core settings management and validation

Requirements addressed: 23.1, 23.4
"""

from .settings import Settings, get_settings, validate_settings

__all__ = ["get_settings", "Settings", "validate_settings"]
