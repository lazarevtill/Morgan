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

from morgan_brain.config import Settings
from morgan_brain.learning_lifecycle.interfaces import PromptRegistry
from morgan_brain.learning_lifecycle.local import LocalPromptRegistry


def _force_mlflow_privacy() -> None:
    """Set env vars that disable MLflow telemetry before any mlflow import."""
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    os.environ["DO_NOT_TRACK"] = "true"


def build_registry(settings: Settings) -> PromptRegistry:
    """Return a PromptRegistry appropriate for *settings.learning_backend*.

    Parameters
    ----------
    settings:
        A ``morgan_brain.config.Settings`` instance.

    Returns
    -------
    PromptRegistry
        - ``"local"``  → ``LocalPromptRegistry`` backed by ``./data/prompts.db``
        - ``"mlflow"`` → telemetry env vars are set, then ``NotImplementedError`` is
          raised (full impl lands in Wave 1/5).
    """
    if settings.learning_backend == "local":
        return LocalPromptRegistry(db_path="./data/prompts.db")

    # --- mlflow branch ---
    # Telemetry MUST be disabled before any mlflow import (privacy hard rule from ADR).
    _force_mlflow_privacy()

    try:
        import mlflow as _mlflow  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        raise NotImplementedError(
            "MLflow registry backend lands in Wave 1/5; "
            "install with: pip install 'morgan-brain[learning]'. "
            "Telemetry env vars enforced."
        )

    raise NotImplementedError(
        "MLflow registry backend lands in Wave 1/5 (GEPA via mlflow.genai.optimize_prompts). "
        "Telemetry env vars enforced."
    )
