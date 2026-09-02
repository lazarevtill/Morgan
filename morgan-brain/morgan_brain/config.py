"""Single source of configuration. All variables are MORGAN_-prefixed.

There is exactly one settings object in the system (design principle: one config system).
Access it via ``get_settings()``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def default_data_dir() -> str:
    """Where the one database lives when ``MORGAN_DATA_DIR`` is not set.

    ``$XDG_DATA_HOME/morgan`` (``~/.local/share/morgan``) -- a location that does not move
    with the working directory. The previous default, ``./data``, was relative to wherever
    the process happened to be started, which for the ``morgan`` CLI is *any* repository the
    owner is working in: every project silently got its own empty brain, and the one under
    ``morgan-brain/`` was only ever seen from ``morgan-brain/``. A brain that is supposed to
    be reachable from every project needs a home that is the same from every project.
    """
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return str(Path(base) / "morgan")


def user_config_file() -> Path:
    """The owner's persistent configuration: ``$XDG_CONFIG_HOME/morgan/.env``.

    Read before the working directory's ``.env`` (which overrides it, so a checkout of
    ``morgan-brain/`` keeps its local dev overrides). Same reason as ``default_data_dir``:
    a ``.env`` that is only found in one directory configures the CLI in exactly that one
    directory, and the CLI's whole point is running from every other one.
    """
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "morgan" / ".env"


class Settings(BaseSettings):
    # Later files win: the owner's ~/.config/morgan/.env is the baseline, a ./.env in the
    # working directory overrides it, and real environment variables override both.
    model_config = SettingsConfigDict(
        env_prefix="MORGAN_",
        env_file=(str(user_config_file()), ".env"),
        extra="ignore",
    )

    # --- Identity (single-owner now; multi-tenant-ready) ---
    owner_user_id: str = "owner"
    api_key: str = "change-me"

    # --- Inbound listeners (brain-api, and morgan-mcp's streamable-HTTP transport) ---
    # Loopback by default. The owner's real deployment binds the homelab box's overlay
    # address so the laptops can reach it over NetBird — which is exactly when an API key
    # stops being optional, so `security/network.py::assert_safe_bind` refuses to start on a
    # non-loopback host while `api_key` is unset or still the placeholder above.
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8080, gt=0, lt=65536)
    mcp_host: str = "127.0.0.1"
    mcp_port: int = Field(default=8090, gt=0, lt=65536)

    # --- LLM ---
    # Default provider is llama-server's OpenAI-compatible port (llama.cpp), not Ollama.
    # "ollama" remains a fully supported provider key — set MORGAN_PROVIDERS /
    # MORGAN_ROLE_BINDINGS explicitly to switch back.
    # The localhost default is a dev convenience for a fresh clone with zero config — the
    # owner's real deployment is a REMOTE llama-server on a homelab GPU box reached over an
    # overlay network (NetBird) from every laptop; local-loopback is the fallback topology
    # (offline laptop, or a dev machine running its own llama-server), not the baseline.
    # Both are the same OpenAI-compatible protocol — only the endpoint URL differs.
    llm_endpoint: str = "http://localhost:8081/v1"
    llm_model: str = "qwen2.5:7b"
    llm_fast_model: str = "qwen2.5:7b"
    embedding_model: str = "mxbai-embed-large"
    # Outbound API key Morgan presents TO the model server (llama-server's --api-key, or a
    # remote OpenAI-compatible provider's key). NOT the same as `api_key` above, which is the
    # INBOUND key clients present to Morgan — the two point in opposite directions. Empty by
    # default: most homelab llama-server setups run without one.
    llm_api_key: str = ""
    # Request timeout (seconds) for LLM chat + embedding calls. Sized for a network hop under
    # GPU load (remote llama-server over an overlay network), not a loopback socket — turn it
    # down for genuinely local dev if the shorter latency matters.
    llm_timeout_seconds: float = Field(default=120.0, gt=0.0)

    # --- Stores ---
    # data_dir is the single directory every durable store derives its path from: the shared
    # SQLite database (temporal facts, vectors, FTS, entities, episodics, signals, history) lives
    # at ``{data_dir}/morgan.db`` unless temporal_db_url is overridden explicitly. Defaults to
    # ``$XDG_DATA_HOME/morgan`` (see ``default_data_dir``); ``~`` is expanded.
    data_dir: str = Field(default_factory=default_data_dir)
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379/0"
    # "" → derived from data_dir (sqlite:///{data_dir}/morgan.db); see _fill_data_dir_defaults.
    temporal_db_url: str = ""

    # --- Event bus ---
    event_bus: Literal["inproc", "redis"] = "inproc"

    # --- Vector backend ---
    # "sqlite" → SqliteVectorIndex (default; persistent, no external deps, shares morgan.db).
    # "memory" → InMemoryVectorIndex (ephemeral; tests and scratch use only).
    # "qdrant" → QdrantVectorIndex (persistent, requires Qdrant at qdrant_url).
    vector_backend: Literal["sqlite", "memory", "qdrant"] = "sqlite"
    # Embedding vector dimension — must match the configured embedding_model output.
    # mxbai-embed-large → 1024; qwen3-embedding:4b → 2560; nomic-embed-text → 768.
    # composition.py probes this against a live embed() call at startup.
    embedding_dim: int = 1024
    # "provider" → call the configured embedding provider. "hash" → deterministic sha256 stub
    # (FakeEmbedder), for the CLI and acceptance tests to run without a live model.
    embedding_backend: Literal["provider", "hash"] = "provider"

    # --- Feature flags ---
    enable_scheduling: bool = False
    # Champion-preprompt self-modification gate. OFF by default: the current promotion logic
    # auto-promotes the first candidate unconditionally and thereafter uses a bare `>` on a
    # single run over a small golden set — too noisy to trust unattended. Flip only once that
    # gate has real statistical backing.
    enable_champion_promotion: bool = False

    # --- Personalization budget (fraction of context window for injected traits) ---
    personalization_budget: float = Field(default=0.15, ge=0.0, le=1.0)

    # --- Provider / role wiring (Wave 0.5a) ---
    # role_bindings: maps role name → ordered list of "provider:model" strings.
    # Default is derived from llm_model / llm_fast_model pointing at ollama.
    # Example: {"strong": ["ollama:qwen2.5:7b"], "fast": ["ollama:qwen2.5:7b"]}
    role_bindings: dict[str, list[str]] = Field(default_factory=dict)

    # providers: per-provider connection config (base_url, api_key, timeout).
    # Key is provider name (e.g. "llamacpp", "ollama"); value is a dict with optional keys:
    #   base_url (str), api_key (str), timeout (float, seconds).
    # Default entry for "llamacpp" is derived from llm_endpoint / llm_api_key / llm_timeout_seconds.
    providers: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Optional path to a YAML file with extra capability overrides (provider/model key).
    models_yaml: str | None = None

    # --- Learning lifecycle ---
    learning_backend: Literal["local", "mlflow"] = "local"
    # `mlflow_tracking_uri` used to live here. It had no reader anywhere -- the reshape's
    # diagnosis listed it among six dead settings, and the MLflow backend still raises
    # NotImplementedError. A setting that is declared, documented nowhere, and read by
    # nothing is worse than a missing one: it reads as configurable behaviour that does
    # not exist. It comes back with the code that needs it.
    # Path to the golden evaluation set JSON file.
    # Empty string → use the packaged default (morgan_brain/eval/data/golden_set.json).
    eval_golden_path: str = ""

    @model_validator(mode="after")
    def _fill_data_dir_defaults(self) -> Settings:
        """Expand ``~`` in data_dir and derive temporal_db_url from it when not overridden."""
        self.data_dir = str(Path(self.data_dir).expanduser())
        if not self.temporal_db_url:
            self.temporal_db_url = f"sqlite:///{self.data_dir}/morgan.db"
        return self

    @model_validator(mode="after")
    def _fill_provider_defaults(self) -> Settings:
        """Populate role_bindings and providers from legacy llm_* fields if not set."""
        # providers: ensure the default llamacpp entry exists
        if "llamacpp" not in self.providers:
            self.providers = dict(self.providers)
            self.providers["llamacpp"] = {
                "base_url": self.llm_endpoint,
                # llama-server without --api-key still requires SOME non-empty string for the
                # openai SDK client; llm_api_key is the real outbound credential when the
                # remote server enforces one.
                "api_key": self.llm_api_key or "llamacpp",
                "timeout": self.llm_timeout_seconds,
            }

        # role_bindings: derive from llm_model / llm_fast_model if not set.
        # All four roles must be bound — judge and reflection back the eval-gated optimize
        # loop (learning-worker); an unbound role makes RoleRouter.chat_for raise LookupError.
        if not self.role_bindings:
            self.role_bindings = {
                "strong": [f"llamacpp:{self.llm_model}"],
                "fast": [f"llamacpp:{self.llm_fast_model}"],
                "judge": [f"llamacpp:{self.llm_model}"],
                "reflection": [f"llamacpp:{self.llm_model}"],
            }

        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
