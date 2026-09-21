"""Single source of configuration. All variables are MORGAN_-prefixed.

There is exactly one settings object in the system. Access it via ``get_settings()``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
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
    #: Where embeddings are requested. Empty means the chat endpoint above serves both,
    #: which is the common single-server case. Set it when the chat server has embeddings
    #: disabled: llama-server loads one model per process, so an embedding model is a
    #: second server at a second address, and without this there is no way to say so.
    embedding_endpoint: str = ""
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
    #: Where ``morgan snapshot`` writes its VACUUM INTO copies. "" → derived from data_dir
    #: ({data_dir}/snapshots), the same way temporal_db_url derives its path.
    snapshot_dir: str = ""
    #: PRAGMA busy_timeout (milliseconds) for a connection this process opens: how long a
    #: writer waits on another process's lock before giving up. store/db.py::open_db's own
    #: default is the same number; this is the setting a caller threads through when it wants
    #: that wait configurable -- ``morgan snapshot``'s VACUUM INTO against the live database
    #: is the first one that does.
    db_busy_timeout_ms: int = Field(default=5000, gt=0)
    #: Must match the embedding model's output dimension (mxbai-embed-large → 1024,
    #: nomic-embed-text → 768). Probed against a live embed() call at startup.
    embedding_dim: int = 1024
    #: The lowest cosine a fresh embedding may have against the active space's fingerprint,
    #: string by string, or against a stored row's vector, before the model answering is
    #: called a different one. Measured, not chosen: re-embedding 500 stored rows under seven
    #: batch, concurrency and cold-start conditions never went below 0.99820, and the worst
    #: first percentile was 0.99860 (docs/measurements/2026-09-phase0-baseline.md).
    embedding_fingerprint_tolerance: float = Field(default=0.995, gt=0.0, le=1.0)
    #: How many stored memories ride along on the first embedding request when the active space
    #: has no fingerprint yet: it is recorded only if their fresh vectors match the stored ones.
    #: At least one: zero would record whatever model happened to answer first.
    embedding_fingerprint_sample_rows: int = Field(default=5, ge=1)
    #: "provider" → call the configured embedding endpoint. "hash" → a deterministic sha256
    #: stub, for the memory commands to run with no model server at all.
    embedding_backend: Literal["provider", "hash"] = "provider"
    #: How far the best match must stand above the weaker results the same query pulled up
    #: before recall returns anything at all. ``None`` -- the default -- means no floor, and
    #: every question is answered.
    #:
    #: The right value belongs to the embedding model, and the default model is a setting, so
    #: none is shipped. For Qwen3-Embedding-0.6B it is 0.11: fitted on the bundled probes and
    #: again on ~2,000 real conversation turns with 128 labelled questions, where on the half
    #: held out of the fit it kept 24 of 25 answers and silenced 12 of 13 unanswerable
    #: questions. Another model needs its own sweep; ``pytest tests/memory_quality --live``
    #: prints what each threshold keeps and silences on the bundled probes. See
    #: memory/recall/floor.py for what the number means.
    recall_floor_margin: float | None = None

    @field_validator("recall_floor_margin", mode="before")
    @classmethod
    def _empty_means_no_floor(cls, value: object) -> object:
        """``MORGAN_RECALL_FLOOR_MARGIN=`` disables the floor rather than failing to parse.

        Writing a setting's name with nothing after it is how a reader of .env.example turns
        something off. Rejecting that as a malformed float would make an empty value crash
        every command, which is a worse answer than the one the writer plainly intended.
        """
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def _fill_data_dir_defaults(self) -> Settings:
        """Expand ``~`` in data_dir and derive temporal_db_url / snapshot_dir from it when not
        overridden."""
        self.data_dir = str(Path(self.data_dir).expanduser())
        if not self.temporal_db_url:
            # as_posix(): the path component of a URL uses forward slashes on every
            # platform. Interpolating the native path mixed separators on Windows.
            self.temporal_db_url = f"sqlite:///{(Path(self.data_dir) / 'morgan.db').as_posix()}"
        self.snapshot_dir = str(
            Path(self.snapshot_dir).expanduser()
            if self.snapshot_dir
            else Path(self.data_dir) / "snapshots"
        )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
