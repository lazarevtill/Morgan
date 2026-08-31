"""Factory for building a PromptRegistry from Settings.

Telemetry-off enforcement
--------------------------
When ``learning_backend = "mlflow"`` the factory calls ``_force_mlflow_privacy()``
**before** any ``import mlflow`` statement.  This sets the two env vars that suppress
all MLflow telemetry (MLFLOW_DISABLE_TELEMETRY) and the generic Do-Not-Track signal
(DO_NOT_TRACK).  The guard runs even if the import later fails, so the env vars are
visible to callers that wrap the call in try/except.

Wave 1/5 note
-------------
The full MLflow-backed registry (Prompt Registry + champion aliases + GEPA optimizer)
lands in Wave 1/5.  The seam + telemetry enforcement is what ships now.
"""

from __future__ import annotations

import os
import pathlib
import sqlite3

from morgan_brain.config import Settings
from morgan_brain.learning_lifecycle.interfaces import PromptRegistry
from morgan_brain.learning_lifecycle.local import LocalPromptRegistry


def _force_mlflow_privacy() -> None:
    """Set env vars that disable MLflow telemetry before any mlflow import."""
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    os.environ["DO_NOT_TRACK"] = "true"


def build_registry(settings: Settings, *, conn: sqlite3.Connection | None = None) -> PromptRegistry:
    """Return a PromptRegistry appropriate for *settings.learning_backend*.

    Parameters
    ----------
    settings:
        A ``morgan_brain.config.Settings`` instance.
    conn:
        The shared memory-database connection (e.g. from ``composition.open_db``), when the
        caller has one. Preferred: the champion registry then lives in the same ``morgan.db``
        file as every other store -- the one-database invariant (Task 13A) that makes
        ``forget()`` a single transaction and backup/encryption a single file. Every
        production caller (``build_app_context``, ``build_worker_context``) passes this.

    Returns
    -------
    PromptRegistry
        - ``"local"``  → ``LocalPromptRegistry``, sharing *conn* when given, else its own file
          at ``{settings.data_dir}/prompts.db`` (parent directories created either way).
        - ``"mlflow"`` → telemetry env vars are set, then ``NotImplementedError`` is
          raised (full impl lands in Wave 1/5).
    """
    if settings.learning_backend == "local":
        if conn is not None:
            return LocalPromptRegistry(conn=conn)
        # No shared connection was given (e.g. building a registry standalone, without the
        # rest of the memory stack) -- fall back to its own file, still derived from
        # settings.data_dir rather than a hardcoded CWD-relative path (the original bug: the
        # CLI is the first caller to build the app context from an arbitrary cwd, where a
        # hardcoded "./data/prompts.db" raised "unable to open database file").
        db_path = f"{settings.data_dir}/prompts.db"
        pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        return LocalPromptRegistry(db_path=db_path)

    # --- mlflow branch ---
    # Telemetry MUST be disabled before any mlflow import (privacy hard rule from ADR).
    _force_mlflow_privacy()

    try:
        import mlflow as _mlflow  # noqa: F401
    except ImportError as exc:
        raise NotImplementedError(
            "MLflow registry backend lands in Wave 1/5; "
            "install with: pip install 'morgan-brain[learning]'. "
            "Telemetry env vars enforced."
        ) from exc

    raise NotImplementedError(
        "MLflow registry backend lands in Wave 1/5 (GEPA via mlflow.genai.optimize_prompts). "
        "Telemetry env vars enforced."
    )
