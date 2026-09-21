"""Single source of configuration. All variables are MORGAN_-prefixed.

A process has one settings object: its entrypoint builds it once with ``settings_for(surface)``
and passes it down, and nothing below a surface reads configuration of its own. Which ``.env``
files that reads belongs to the surface (``env_files_for``), and the environment overrides them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from pydantic import Field, PrivateAttr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


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

    Read by every surface, first. Same reason as ``default_data_dir``: a ``.env`` that is only
    found in one directory configures the CLI in exactly that one directory, and the CLI's
    whole point is running from every other one.
    """
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "morgan" / ".env"


#: Where a request comes in. Each reads its own list of ``.env`` files.
Surface = Literal["cli", "mcp"]


class EnvFileRead(TypedDict):
    """One ``.env`` file a settings object was built from, and whether it was there."""

    path: str
    present: bool


def env_files_for(surface: Surface) -> tuple[Path, ...]:
    """The ``.env`` files *surface* reads, in read order: a later file overrides an earlier
    one, and a real environment variable overrides every file.

    The CLI is run by the owner in a folder they chose, so a ``./.env`` there is theirs and
    overrides the user file -- a checkout of this repository keeps its dev overrides that way.
    ``morgan-mcp`` is started by a client in whatever folder that client has open, so a
    ``./.env`` there belongs to whatever project is open, and a stale one would move the server
    to another database: it reads the user file only.
    """
    user = user_config_file()
    return (user, Path.cwd() / ".env") if surface == "cli" else (user,)


class Settings(BaseSettings):
    # No env_file here: which files are read is the surface's, passed at construction by
    # settings_for(). A Settings built directly reads the environment only.
    model_config = SettingsConfigDict(
        env_prefix="MORGAN_",
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
    #: Request timeout (seconds) for one chat call. Sized for a network hop under GPU load,
    #: not a loopback socket. Embedding calls do not use it: they have the budgets below.
    llm_timeout_seconds: float = Field(default=120.0, gt=0.0)
    #: How long (seconds) one embedding call may keep retrying a host that answered slowly,
    #: with a server error (a 501 aside: that server serves no embeddings) or a 429, or dropped
    #: the connection -- as a cold host loading its model does -- before it fails, for a
    #: command or a tool call. A bound on the wall time of the whole call, attempts and backoff
    #: included, counted once the call's HTTP client is built.
    embedding_retry_budget_seconds: float = Field(default=60.0, gt=0.0)
    #: The same bound for an import (``morgan import``), which has thousands of calls to make
    #: and no one waiting on any single one of them.
    embedding_import_retry_budget_seconds: float = Field(default=600.0, gt=0.0)
    #: How long (seconds) an embedding call keeps retrying a host it cannot connect to at all
    #: (refused, DNS, connect timeout): the host is off or the address is wrong, and waiting
    #: on it helps no one. Counted from the call's first attempt, and only until a connection
    #: is made; after that every failure counts against the budget above.
    embedding_unreachable_budget_seconds: float = Field(default=5.0, gt=0.0)
    #: The wait (seconds) before an embedding call's first retry, doubling before each next.
    #: A retry is made only while this much of the budget is left for it -- the wait before it
    #: shrinks to leave that -- so no retry runs the call past its budget.
    embedding_retry_backoff_seconds: float = Field(default=0.5, gt=0.0)
    #: The longest (seconds) the doubling wait grows to, so a long budget -- an import's -- keeps
    #: asking at a steady rate and a host that has come up is asked again within this long.
    embedding_retry_backoff_max_seconds: float = Field(default=2.0, gt=0.0)
    #: The longest (seconds) one embedding attempt may take, and never longer than what is left
    #: of the call's budget. 50 s covers a cold host's first load from disk (43 s measured).
    embedding_timeout_seconds: float = Field(default=50.0, gt=0.0)
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

    # --- Project classification (surfaces/cli/project.py::classify). The disk walk over
    # code_roots is phase 1a; phase 0 only stores these and `doctor` names a root that does
    # not exist (Task 24). Nothing here identifies the owner -- shipped empty, so a clone of
    # this repository walks and matches nothing until its own owner sets them. ---
    #: Repository roots the phase-1a walk will scan for a project's remote and root, comma
    #: separated (``~/code,~/work``). Each entry's ``~`` is expanded eagerly, like ``data_dir``.
    code_roots: Annotated[list[str], NoDecode] = Field(default_factory=list)
    #: How many directory levels below each root that walk descends looking for a ``.git``.
    code_root_depth: int = Field(default=2, gt=0)
    #: ``fnmatch`` globs, comma separated, a remote's host or whole URL must match for
    #: ``classify`` to call the project ``work`` rather than ``personal``.
    work_remote_globs: Annotated[list[str], NoDecode] = Field(default_factory=list)

    #: Filled by settings_for(); private, so no MORGAN_ variable can set what doctor reports.
    _env_files_read: list[EnvFileRead] = PrivateAttr(default_factory=list)

    @property
    def env_files_read(self) -> list[EnvFileRead]:
        """The ``.env`` files this object was built from, in read order, each with whether it
        was there. Empty for a ``Settings`` built directly, which reads no file."""
        return self._env_files_read

    @field_validator("recall_floor_margin", mode="before")
    @classmethod
    def _empty_means_no_floor(cls, value: object) -> object:
        """``MORGAN_RECALL_FLOOR_MARGIN=`` disables the floor rather than failing to parse.

        Writing a setting's name with nothing after it is how a reader of .env.example turns
        something off. Rejecting that as a malformed float would make an empty value crash
        every command, which is a worse answer than the one the writer plainly intended.
        """
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("code_roots", "work_remote_globs", mode="before")
    @classmethod
    def _split_comma_separated(cls, value: object) -> object:
        """``MORGAN_CODE_ROOTS=~/code,~/work`` -> ``["~/code", "~/work"]``.

        Both fields carry ``NoDecode`` (see the import above): a plain ``list[str]`` field
        would otherwise have its raw env string handed to ``json.loads`` before any validator
        runs, and a comma-separated value is not JSON -- every non-empty setting would fail to
        parse before this function ever saw it. A value that is already a list (constructing
        ``Settings`` directly, as the tests do) passes through unchanged.
        """
        if isinstance(value, str):
            return [p.strip() for p in value.split(",") if p.strip()]
        return value

    @model_validator(mode="after")
    def _fill_data_dir_defaults(self) -> Settings:
        """Expand ``~`` in data_dir and derive temporal_db_url / snapshot_dir from it when not
        overridden. Also expands ``~`` in each of ``code_roots``, the same way."""
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
        self.code_roots = [str(Path(p).expanduser()) for p in self.code_roots]
        return self


def settings_for(surface: Surface) -> Settings:
    """The settings *surface* runs with: its ``.env`` files (``env_files_for``), then the
    environment, and the list of the files it read.

    The one way a process obtains its settings. Each entrypoint calls it once and passes the
    object down; shared code takes the object it is given and never picks a surface. Not
    cached: the working directory and ``XDG_CONFIG_HOME`` are read when it is called.
    """
    files = env_files_for(surface)
    # _env_file is BaseSettings.__init__'s own keyword; mypy builds a pydantic model's __init__
    # from its fields alone and does not see it.
    settings = Settings(_env_file=files)  # type: ignore[call-arg]
    settings._env_files_read = [{"path": str(f), "present": f.is_file()} for f in files]
    return settings
