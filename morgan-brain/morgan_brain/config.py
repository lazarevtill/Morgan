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
    # data_dir is the single directory every durable store derives its path from: the shared
    # SQLite database (temporal facts, vectors, FTS, entities, episodics, signals, history) lives
    # at ``{data_dir}/morgan.db`` unless temporal_db_url is overridden explicitly.
    data_dir: str = "./data"
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379/0"
    # "" → derived from data_dir (sqlite:///{data_dir}/morgan.db); see _fill_data_dir_defaults.
    temporal_db_url: str = ""
    workspace_path: str = "./data/workspace"

    # --- Event bus ---
    event_bus: Literal["inproc", "redis"] = "inproc"

    # --- Vector backend ---
    # "sqlite" → SqliteVectorIndex (default; persistent, no external deps, shares morgan.db).
    # "memory" → InMemoryVectorIndex (ephemeral; tests and scratch use only).
    # "qdrant" → QdrantVectorIndex (persistent, requires Qdrant at qdrant_url).
    vector_backend: Literal["sqlite", "memory", "qdrant"] = "sqlite"
    # Embedding vector dimension — must match the configured embedding_model output.
    # qwen3-embedding:4b → 2560; nomic-embed-text → 768; mxbai-embed-large → 1024.
    embedding_dim: int = 1024

    # --- Feature flags ---
    enable_scheduling: bool = False

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
    # Path to the golden evaluation set JSON file.
    # Empty string → use the packaged default (morgan_brain/eval/data/golden_set.json).
    eval_golden_path: str = ""

    @model_validator(mode="after")
    def _fill_data_dir_defaults(self) -> "Settings":
        """Derive temporal_db_url from data_dir when not explicitly overridden."""
        if not self.temporal_db_url:
            self.temporal_db_url = f"sqlite:///{self.data_dir}/morgan.db"
        return self

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
