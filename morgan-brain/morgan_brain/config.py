"""Single source of configuration. All variables are MORGAN_-prefixed.

There is exactly one settings object in the system (design principle: one config system).
Access it via ``get_settings()``.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
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


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
