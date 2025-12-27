"""
Morgan Configuration Module

Centralized configuration management for all Morgan components.
Follows KISS principles with simple, focused configuration.

Components:
- settings.py: Core settings management and validation (Pydantic-based)
- yaml_config.py: YAML-based modular configuration loader
- defaults.py: Default values for all configuration

Configuration Hierarchy (highest to lowest priority):
1. Environment variables (MORGAN_* prefix)
2. YAML config files (config/*.yaml)
3. Pydantic settings (.env file)
4. Default values (defaults.py)

Requirements addressed: 23.1, 23.4
"""

from .defaults import Defaults
from .settings import Settings, get_settings, validate_settings
from .yaml_config import YAMLConfig, get_yaml_config, reset_yaml_config

__all__ = [
    # Pydantic settings
    "get_settings",
    "Settings",
    "validate_settings",
    # YAML config
    "get_yaml_config",
    "reset_yaml_config",
    "YAMLConfig",
    # Defaults
    "Defaults",
]
