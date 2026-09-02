"""Single source of configuration. All variables are MORGAN_-prefixed.

There is exactly one settings object in the system. Access it via ``get_settings()``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def default_data_dir() -> str:
    """Where the one database lives when ``MORGAN_DATA_DIR`` is not set.

    ``$XDG_DATA_HOME/morgan`` (``~/.local/share/morgan``) -- a location that does not move
    with the working directory. A relative default gave every repository the ``morgan`` CLI
    was run from its own empty brain; a brain that is supposed to be reachable from every
    project needs a home that is the same from every project.
    """
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return str(Path(base) / "morgan")


def user_config_file() -> Path:
    """The owner's persistent configuration: ``$XDG_CONFIG_HOME/morgan/.env``.

    Read before the working directory's ``.env`` (which overrides it, so a checkout of this
    repository keeps its local dev overrides). Same reason as ``default_data_dir``: a
    ``.env`` that is only found in one directory configures the CLI in exactly that one
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
    #: INBOUND: the bearer token clients present to ``morgan-mcp --transport http``. The
    #: placeholder means "no key configured", which is allowed on loopback only -- see
    #: ``network.assert_safe_bind``.
    api_key: str = "change-me"

    # --- The MCP listener (streamable-HTTP transport; stdio needs neither) ---
    mcp_host: str = "127.0.0.1"
    mcp_port: int = Field(default=8090, gt=0, lt=65536)

    # --- The model server: any OpenAI-compatible endpoint; llama-server by default ---
    # The localhost default is a dev convenience for a fresh clone. The expected topology is
    # a remote llama-server on a GPU box reached over an overlay network from every client.
    llm_endpoint: str = "http://localhost:8081/v1"
    llm_model: str = "qwen2.5:7b"
    embedding_model: str = "mxbai-embed-large"
    #: OUTBOUND: the key Morgan presents TO the model server (llama-server's ``--api-key``).
    #: Not ``api_key`` above -- the two point in opposite directions. Empty by default.
    llm_api_key: str = ""
    #: Request timeout (seconds) for chat + embedding calls. Sized for a network hop under
    #: GPU load, not a loopback socket.
    llm_timeout_seconds: float = Field(default=120.0, gt=0.0)
    #: How structured output (fact consolidation) is requested. ``json_schema`` is native
    #: constrained decoding, which llama-server and Ollama's /v1 both support; ``json_object``
    #: for servers that only guarantee an object; ``prompted`` asks in the prompt and
    #: validates the answer.
    llm_json_mode: Literal["json_schema", "json_object", "prompted"] = "json_schema"

    # --- The one database ---
    # data_dir is the directory the SQLite database lives in: ``{data_dir}/morgan.db`` unless
    # temporal_db_url overrides it. Defaults to ``$XDG_DATA_HOME/morgan``; ``~`` is expanded.
    data_dir: str = Field(default_factory=default_data_dir)
    #: "" → derived from data_dir (sqlite:///{data_dir}/morgan.db).
    temporal_db_url: str = ""
    #: Must match the embedding model's output dimension (mxbai-embed-large → 1024,
    #: nomic-embed-text → 768). Probed against a live embed() call at startup.
    embedding_dim: int = 1024
    #: "provider" → call the configured embedding endpoint. "hash" → a deterministic sha256
    #: stub, for the memory commands to run with no model server at all.
    embedding_backend: Literal["provider", "hash"] = "provider"

    @model_validator(mode="after")
    def _fill_data_dir_defaults(self) -> Settings:
        """Expand ``~`` in data_dir and derive temporal_db_url from it when not overridden."""
        self.data_dir = str(Path(self.data_dir).expanduser())
        if not self.temporal_db_url:
            self.temporal_db_url = f"sqlite:///{self.data_dir}/morgan.db"
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
