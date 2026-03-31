"""
Workspace path resolution and validation.

Key helpers:
- get_morgan_home()       -> ~/.morgan  (or MORGAN_HOME env override)
- get_workspace_path()    -> ~/.morgan/workspace  (or MORGAN_WORKSPACE_PATH env override)
- validate_workspace_path -> rejects relative paths and near-root paths
"""

from __future__ import annotations

import os
from pathlib import Path


def get_morgan_home() -> Path:
    """Return the Morgan home directory.

    Defaults to ``~/.morgan``.  Override with the ``MORGAN_HOME`` env var.
    """
    env = os.environ.get("MORGAN_HOME")
    if env:
        return Path(env)
    return Path.home() / ".morgan"


def get_workspace_path() -> Path:
    """Return the workspace directory path.

    Defaults to ``<morgan_home>/workspace``.  Override with the
    ``MORGAN_WORKSPACE_PATH`` env var.
    """
    env = os.environ.get("MORGAN_WORKSPACE_PATH")
    if env:
        return Path(env)
    return get_morgan_home() / "workspace"


# Paths that are too close to the filesystem root to be safe as workspaces.
_NEAR_ROOT_PATHS = frozenset({
    "/",
    "/bin",
    "/boot",
    "/dev",
    "/etc",
    "/home",
    "/lib",
    "/lib64",
    "/mnt",
    "/opt",
    "/proc",
    "/root",
    "/run",
    "/sbin",
    "/srv",
    "/sys",
    "/tmp",
    "/usr",
    "/var",
})


def validate_workspace_path(path: Path) -> None:
    """Validate that *path* is a safe workspace location.

    Raises:
        ValueError: if *path* is relative or dangerously close to the
            filesystem root.
    """
    if not path.is_absolute():
        raise ValueError(
            f"Workspace path must be absolute, got relative path: {path}"
        )

    resolved = str(path.resolve())
    if resolved in _NEAR_ROOT_PATHS:
        raise ValueError(
            f"Workspace path is too close to the filesystem root: {path}"
        )
