"""
CLI Configuration Management.

Handles configuration loading, validation, and management for the Morgan CLI.
Supports:
- JSON configuration files
- Environment variables
- Interactive initialization
- Validation and defaults
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from morgan.core.types import ContextPruningStrategy


@dataclass
class CLIConfig:
    """
    Morgan CLI configuration.

    Configuration priority:
    1. Command-line arguments (highest)
    2. Environment variables
    3. Config file
    4. Defaults (lowest)
    """

    # Storage
    storage_path: str = field(default_factory=lambda: str(Path.home() / ".morgan"))

    # LLM Configuration
    llm_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:latest"
    llm_timeout: int = 30

    # Vector DB
    vector_db_url: str = "http://localhost:6333"
    vector_db_collection: str = "morgan_knowledge"

    # Embedding Service
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32

    # Reranking (Jina)
    reranking_enabled: bool = True
    reranking_model: str = "jina-reranker-v2-base-multilingual"
    reranking_top_n: int = 5

    # Feature Flags
    enable_emotion_detection: bool = True
    enable_learning: bool = True
    enable_rag: bool = True

    # Context Management
    max_context_tokens: int = 8000
    target_context_tokens: int = 6000
    context_pruning_strategy: str = "hybrid"

    # UI Preferences
    use_rich_formatting: bool = True
    show_emotions: bool = True
    show_sources: bool = True
    show_metrics: bool = False
    use_streaming: bool = True

    # Performance
    max_concurrent_operations: int = 10
    cache_enabled: bool = True

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    verbose: bool = False

    # Session
    default_session_id: Optional[str] = None
    save_history: bool = True
    max_history_entries: int = 1000

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> CLIConfig:
        """
        Load configuration from file and environment.

        Args:
            config_path: Path to config file (default: ~/.morgan/config.json)

        Returns:
            CLIConfig instance
        """
        # Default config path
        if config_path is None:
            config_path = Path.home() / ".morgan" / "config.json"

        # Start with defaults
        config_data: Dict[str, Any] = {}

        # Load from file if exists
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config_data = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load config from {config_path}: {e}")

        # Override with environment variables
        config_data.update(cls._load_from_env())

        # Create config instance
        return cls(**config_data)

    @staticmethod
    def _load_from_env() -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config: Dict[str, Any] = {}

        # Storage
        if val := os.getenv("MORGAN_STORAGE_PATH"):
            env_config["storage_path"] = val

        # LLM
        if val := os.getenv("MORGAN_LLM_BASE_URL"):
            env_config["llm_base_url"] = val
        if val := os.getenv("MORGAN_LLM_MODEL"):
            env_config["llm_model"] = val
        if val := os.getenv("MORGAN_LLM_TIMEOUT"):
            env_config["llm_timeout"] = int(val)

        # Vector DB
        if val := os.getenv("MORGAN_VECTOR_DB_URL"):
            env_config["vector_db_url"] = val
        if val := os.getenv("MORGAN_VECTOR_DB_COLLECTION"):
            env_config["vector_db_collection"] = val

        # Embedding
        if val := os.getenv("MORGAN_EMBEDDING_MODEL"):
            env_config["embedding_model"] = val

        # Feature flags
        if val := os.getenv("MORGAN_ENABLE_EMOTIONS"):
            env_config["enable_emotion_detection"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv("MORGAN_ENABLE_LEARNING"):
            env_config["enable_learning"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv("MORGAN_ENABLE_RAG"):
            env_config["enable_rag"] = val.lower() in ("true", "1", "yes")

        # Logging
        if val := os.getenv("MORGAN_LOG_LEVEL"):
            env_config["log_level"] = val
        if val := os.getenv("MORGAN_LOG_FILE"):
            env_config["log_file"] = val
        if val := os.getenv("MORGAN_VERBOSE"):
            env_config["verbose"] = val.lower() in ("true", "1", "yes")

        return env_config

    def save(self, config_path: Optional[Path] = None) -> None:
        """
        Save configuration to file.

        Args:
            config_path: Path to save config (default: ~/.morgan/config.json)
        """
        if config_path is None:
            config_path = Path.home() / ".morgan" / "config.json"

        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        with open(config_path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return getattr(self, key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key."""
        if not hasattr(self, key):
            raise ValueError(f"Unknown configuration key: {key}")
        setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def validate(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if valid, raises ValueError otherwise
        """
        # Validate URLs
        if not self.llm_base_url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid LLM base URL: {self.llm_base_url}")

        if not self.vector_db_url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid vector DB URL: {self.vector_db_url}")

        # Validate paths
        storage_path = Path(self.storage_path)
        if not storage_path.exists():
            storage_path.mkdir(parents=True, exist_ok=True)

        # Validate numeric ranges
        if self.llm_timeout <= 0:
            raise ValueError(f"Invalid LLM timeout: {self.llm_timeout}")

        if self.max_context_tokens <= 0:
            raise ValueError(f"Invalid max context tokens: {self.max_context_tokens}")

        if self.target_context_tokens >= self.max_context_tokens:
            raise ValueError(
                f"Target context tokens ({self.target_context_tokens}) "
                f"must be less than max ({self.max_context_tokens})"
            )

        # Validate log level
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {self.log_level}")

        return True


def get_default_config_path() -> Path:
    """Get default configuration file path."""
    return Path.home() / ".morgan" / "config.json"


def ensure_config_exists() -> CLIConfig:
    """
    Ensure configuration exists, create with defaults if not.

    Returns:
        CLIConfig instance
    """
    config_path = get_default_config_path()

    if config_path.exists():
        return CLIConfig.load(config_path)

    # Create default config
    config = CLIConfig()
    config.save(config_path)

    return config


async def interactive_init() -> CLIConfig:
    """
    Interactive configuration initialization.

    Prompts user for configuration values and creates config file.

    Returns:
        CLIConfig instance
    """
    print("Morgan CLI Configuration Setup")
    print("=" * 40)
    print()

    config = CLIConfig()

    # Storage
    storage_path = input(
        f"Storage path [{config.storage_path}]: "
    ).strip() or config.storage_path
    config.storage_path = storage_path

    # LLM
    llm_base_url = input(
        f"LLM base URL [{config.llm_base_url}]: "
    ).strip() or config.llm_base_url
    config.llm_base_url = llm_base_url

    llm_model = input(
        f"LLM model [{config.llm_model}]: "
    ).strip() or config.llm_model
    config.llm_model = llm_model

    # Vector DB
    vector_db_url = input(
        f"Vector DB URL [{config.vector_db_url}]: "
    ).strip() or config.vector_db_url
    config.vector_db_url = vector_db_url

    # Feature flags
    enable_emotions = input(
        f"Enable emotion detection? [Y/n]: "
    ).strip().lower()
    config.enable_emotion_detection = enable_emotions != "n"

    enable_learning = input(
        f"Enable learning? [Y/n]: "
    ).strip().lower()
    config.enable_learning = enable_learning != "n"

    enable_rag = input(
        f"Enable RAG? [Y/n]: "
    ).strip().lower()
    config.enable_rag = enable_rag != "n"

    # Validate and save
    try:
        config.validate()
        config.save()
        print()
        print("Configuration saved successfully!")
        print(f"Config file: {get_default_config_path()}")
        return config
    except ValueError as e:
        print(f"Configuration validation failed: {e}")
        raise
