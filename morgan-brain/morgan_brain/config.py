"""Single source of configuration. All variables are MORGAN_-prefixed.

There is exactly one settings object in the system (design principle: one config system).
Access it via ``get_settings()``.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict



class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MORGAN_", env_file=".env", extra="ignore")

    # --- Identity (single-owner now; multi-tenant-ready) ---
    owner_user_id: str = "owner"
    api_key: str = "change-me"

    # --- LLM ---
    llm_endpoint: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:7b"
    llm_fast_model: str = "qwen2.5:7b"
    embedding_model: str = "qwen3-embedding:4b"

    # --- Stores ---
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379/0"
    temporal_db_url: str = "sqlite:///./data/morgan.db"
    workspace_path: str = "./data/workspace"

    # --- Event bus ---
    event_bus: Literal["inproc", "redis"] = "inproc"

    # --- Feature flags ---
    enable_proactivity: bool = False
    enable_channels: bool = False
    enable_mcp: bool = False

    # --- Channels ---
    telegram_token: str = ""
    discord_token: str = ""

    # --- Personalization budget (fraction of context window for injected traits) ---
    personalization_budget: float = Field(default=0.15, ge=0.0, le=1.0)

    # --- Provider / role wiring (Wave 0.5a) ---
    # role_bindings: maps role name → ordered list of "provider:model" strings.
    # Default is derived from llm_model / llm_fast_model pointing at ollama.
    # Example: {"strong": ["ollama:qwen2.5:7b"], "fast": ["ollama:qwen2.5:7b"]}
    role_bindings: dict[str, list[str]] = Field(default_factory=dict)

    # providers: per-provider connection config (base_url, api_key).
    # Key is provider name (e.g. "ollama"); value is a dict with optional keys:
    #   base_url (str), api_key (str).
    # Default entry for "ollama" is derived from llm_endpoint.
    providers: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Optional path to a YAML file with extra capability overrides (provider/model key).
    models_yaml: str | None = None

    # --- Learning lifecycle ---
    learning_backend: Literal["local", "mlflow"] = "local"
    # SQLite URI for MLflow tracking store (used when learning_backend="mlflow").
    mlflow_tracking_uri: str = "sqlite:///./data/mlflow.db"

    @model_validator(mode="after")
    def _fill_provider_defaults(self) -> "Settings":
        """Populate role_bindings and providers from legacy llm_* fields if not set."""
        # providers: ensure ollama entry exists
        if "ollama" not in self.providers:
            self.providers = dict(self.providers)
            self.providers["ollama"] = {
                "base_url": self.llm_endpoint,
                "api_key": "ollama",
            }

        # role_bindings: derive from llm_model / llm_fast_model if not set
        if not self.role_bindings:
            self.role_bindings = {
                "strong": [f"ollama:{self.llm_model}"],
                "fast": [f"ollama:{self.llm_fast_model}"],
            }

        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
