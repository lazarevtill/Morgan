# Copyright 2025 Morgan AI Assistant Contributors
# SPDX-License-Identifier: Apache-2.0

"""
YAML Configuration Loader for Morgan AI Assistant.

Loads modular YAML configuration files from the config directory.
Supports environment variable overrides and provides typed access.

Usage:
    from morgan.config.yaml_config import get_yaml_config, YAMLConfig

    # Get singleton config instance
    config = get_yaml_config()

    # Access configuration
    llm_model = config.llm.main.model
    qdrant_host = config.qdrant.connection.host

    # Or use get() with dot notation
    model = config.get("llm.main.model")
"""

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from morgan.utils.logger import get_logger

logger = get_logger(__name__)

# Try to import yaml
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not installed, YAML config loading disabled")


@dataclass
class ConfigSection:
    """
    A configuration section that provides attribute and dict-style access.
    """

    _data: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        data = super().__getattribute__("_data")
        if name in data:
            value = data[name]
            if isinstance(value, dict):
                return ConfigSection(_data=value)
            return value
        raise AttributeError(f"Config has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with dot notation support."""
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self._data.copy()


class YAMLConfig:
    """
    YAML Configuration Manager.

    Loads configuration from YAML files in the config directory
    and provides typed access with environment variable overrides.
    """

    # Config file names
    CONFIG_FILES = [
        "llm",
        "embeddings",
        "reranking",
        "qdrant",
        "redis",
        "server",
        "memory",
        "intelligence",
        "search",
        "distributed",
    ]

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize YAML config loader.

        Args:
            config_dir: Path to config directory. Defaults to project root/config.
        """
        self._config_dir = self._find_config_dir(config_dir)
        self._configs: Dict[str, ConfigSection] = {}
        self._loaded = False
        self._lock = threading.Lock()

        logger.info("YAMLConfig initialized: config_dir=%s", self._config_dir)

    def _find_config_dir(self, config_dir: Optional[str] = None) -> Path:
        """Find the config directory."""
        if config_dir:
            return Path(config_dir)

        # Check environment variable
        env_config_dir = os.environ.get("MORGAN_CONFIG_DIR")
        if env_config_dir:
            return Path(env_config_dir)

        # Try common locations
        candidates = [
            Path(__file__).parent.parent.parent.parent.parent / "config",  # Project root
            Path(__file__).parent.parent.parent / "config",  # morgan-rag/config
            Path.cwd() / "config",  # Current working directory
            Path("/app/config"),  # Docker container
        ]

        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate

        # Default to project root/config (will be created if needed)
        return Path(__file__).parent.parent.parent.parent.parent / "config"

    def _load_all(self) -> None:
        """Load all configuration files."""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            for name in self.CONFIG_FILES:
                self._load_config_file(name)

            self._loaded = True
            logger.info("All configuration files loaded")

    def _load_config_file(self, name: str) -> None:
        """Load a single configuration file."""
        file_path = self._config_dir / f"{name}.yaml"

        if not file_path.exists():
            logger.debug("Config file not found: %s, using defaults", file_path)
            self._configs[name] = ConfigSection(_data={})
            return

        if not YAML_AVAILABLE:
            logger.warning("YAML not available, skipping %s", file_path)
            self._configs[name] = ConfigSection(_data={})
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # Apply environment variable overrides
            data = self._apply_env_overrides(name, data)

            self._configs[name] = ConfigSection(_data=data)
            logger.debug("Loaded config: %s", file_path)

        except Exception as e:
            logger.error("Failed to load config %s: %s", file_path, e)
            self._configs[name] = ConfigSection(_data={})

    def _apply_env_overrides(
        self, config_name: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply environment variable overrides to config data.

        Environment variables use format: MORGAN_{CONFIG}_{KEY}_{SUBKEY}
        Example: MORGAN_LLM_MAIN_MODEL overrides llm.main.model
        """
        prefix = f"MORGAN_{config_name.upper()}_"

        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            # Convert env var name to nested path
            path_parts = key[len(prefix) :].lower().split("_")

            # Navigate to the right nested location
            current = data
            for part in path_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set the value with type conversion
            final_key = path_parts[-1]
            current[final_key] = self._convert_env_value(value)

        return data

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        if value.lower() in ("false", "no", "0", "off"):
            return False

        # Integer
        try:
            return int(value)
        except ValueError:
            pass

        # Float
        try:
            return float(value)
        except ValueError:
            pass

        # List (comma-separated)
        if "," in value:
            return [v.strip() for v in value.split(",")]

        return value

    def __getattr__(self, name: str) -> ConfigSection:
        """Get config section by name."""
        if name.startswith("_"):
            return super().__getattribute__(name)

        self._load_all()

        if name in self._configs:
            return self._configs[name]

        raise AttributeError(f"No config section named '{name}'")

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get config value with dot notation.

        Args:
            path: Dot-separated path like "llm.main.model"
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        self._load_all()

        parts = path.split(".", 1)
        if len(parts) == 1:
            # Top-level config
            return self._configs.get(parts[0], ConfigSection(_data={}))

        config_name, subpath = parts
        if config_name in self._configs:
            return self._configs[config_name].get(subpath, default)

        return default

    def reload(self) -> None:
        """Reload all configuration files."""
        with self._lock:
            self._configs.clear()
            self._loaded = False
        self._load_all()
        logger.info("Configuration reloaded")

    def to_dict(self) -> Dict[str, Any]:
        """Get all configuration as dictionary."""
        self._load_all()
        return {name: section.to_dict() for name, section in self._configs.items()}

    def validate(self) -> List[str]:
        """
        Validate configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        self._load_all()
        errors = []

        # Check LLM configuration
        llm = self._configs.get("llm", ConfigSection(_data={}))
        if not llm.get("main.endpoint"):
            errors.append("llm.main.endpoint is required")
        if not llm.get("main.model"):
            errors.append("llm.main.model is required")

        # Check Qdrant configuration
        qdrant = self._configs.get("qdrant", ConfigSection(_data={}))
        if not qdrant.get("connection.host") and not qdrant.get("connection.url"):
            errors.append("qdrant.connection.host or url is required")

        return errors


# =============================================================================
# Singleton Management
# =============================================================================

_yaml_config_instance: Optional[YAMLConfig] = None
_yaml_config_lock = threading.Lock()


def get_yaml_config(
    config_dir: Optional[str] = None,
    force_new: bool = False,
) -> YAMLConfig:
    """
    Get singleton YAML config instance.

    Args:
        config_dir: Optional config directory path
        force_new: Force create new instance

    Returns:
        Shared YAMLConfig instance
    """
    global _yaml_config_instance

    if _yaml_config_instance is None or force_new:
        with _yaml_config_lock:
            if _yaml_config_instance is None or force_new:
                _yaml_config_instance = YAMLConfig(config_dir=config_dir)

    return _yaml_config_instance


def reset_yaml_config() -> None:
    """Reset singleton instance."""
    global _yaml_config_instance

    with _yaml_config_lock:
        _yaml_config_instance = None
