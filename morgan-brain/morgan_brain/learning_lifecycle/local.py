"""Dependency-light implementations of PromptRegistry and Optimizer.

LocalPromptRegistry
-------------------
SQLite-backed using stdlib ``sqlite3`` only (no ORM, no async library).  Follows the
same pattern as ``modules/memory/stores/temporal.py``: synchronous SQLite calls wrapped
in async methods (acceptable because the store is used off the hot request path).

Timestamps are injected via a ``clock`` callable (default ``datetime.utcnow``) so tests
can pin time deterministically — matching the project's deterministic-time rule.

NoopOptimizer
-------------
Safe placeholder that returns the current champion unchanged.  The real GEPA-via-MLflow
optimizer lands in Wave 1/5 (``mlflow.genai.optimize_prompts`` with
``GepaPromptOptimizer``).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import datetime
from typing import Any

from morgan_brain.learning_lifecycle.interfaces import Optimizer, PromptRegistry, PromptVersion

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prompt_versions (
    name            TEXT NOT NULL,
    version         INTEGER NOT NULL,
    body            TEXT NOT NULL,
    created_at      TEXT,
    metrics_json    TEXT NOT NULL DEFAULT '{}',
    commit_message  TEXT NOT NULL DEFAULT '',
    is_champion     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (name, version)
);
"""

_DEFAULT_CLOCK: Callable[[], datetime] = datetime.utcnow


class LocalPromptRegistry:
    """SQLite-backed PromptRegistry — no external dependencies.

    Parameters
    ----------
    db_path:
        Path to the SQLite file.  Use ``":memory:"`` (the default) for tests.
    clock:
        Zero-argument callable that returns the "current" datetime.  Injected for
        deterministic testing.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        clock: Callable[[], datetime] = _DEFAULT_CLOCK,
    ) -> None:
        self._clock = clock
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_version(self, row: sqlite3.Row) -> PromptVersion:
        return PromptVersion(
            name=row["name"],
            version=row["version"],
            body=row["body"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            metrics=json.loads(row["metrics_json"]),
            commit_message=row["commit_message"],
        )

    def _next_version(self, name: str) -> int:
        row = self._conn.execute(
            "SELECT MAX(version) AS mv FROM prompt_versions WHERE name = ?", (name,)
        ).fetchone()
        current: int | None = row["mv"]
        return 1 if current is None else current + 1

    # ------------------------------------------------------------------
    # PromptRegistry Protocol
    # ------------------------------------------------------------------

    async def register(
        self,
        name: str,
        body: str,
        *,
        commit_message: str = "",
        metrics: dict[str, float] | None = None,
    ) -> PromptVersion:
        version = self._next_version(name)
        now = self._clock()
        self._conn.execute(
            """
            INSERT INTO prompt_versions (name, version, body, created_at, metrics_json, commit_message)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                version,
                body,
                now.isoformat(),
                json.dumps(metrics or {}),
                commit_message,
            ),
        )
        self._conn.commit()
        return PromptVersion(
            name=name,
            version=version,
            body=body,
            created_at=now,
            metrics=metrics or {},
            commit_message=commit_message,
        )

    async def champion(self, name: str) -> PromptVersion | None:
        row = self._conn.execute(
            "SELECT * FROM prompt_versions WHERE name = ? AND is_champion = 1",
            (name,),
        ).fetchone()
        return self._row_to_version(row) if row else None

    async def set_champion(self, name: str, version: int) -> None:
        # Verify the version exists first.
        exists = self._conn.execute(
            "SELECT 1 FROM prompt_versions WHERE name = ? AND version = ?",
            (name, version),
        ).fetchone()
        if not exists:
            raise ValueError(f"No prompt '{name}' at version {version}")

        # Atomically clear old champion flag and set the new one.
        with self._conn:
            self._conn.execute("UPDATE prompt_versions SET is_champion = 0 WHERE name = ?", (name,))
            self._conn.execute(
                "UPDATE prompt_versions SET is_champion = 1 WHERE name = ? AND version = ?",
                (name, version),
            )

    async def list_versions(self, name: str) -> list[PromptVersion]:
        rows = self._conn.execute(
            "SELECT * FROM prompt_versions WHERE name = ? ORDER BY version ASC", (name,)
        ).fetchall()
        return [self._row_to_version(r) for r in rows]


class NoopOptimizer:
    """Safe no-op placeholder for the Optimizer seam.

    Returns the current champion unchanged so callers in the hot path are never
    broken.  The real implementation uses ``mlflow.genai.optimize_prompts`` with a
    ``GepaPromptOptimizer`` (reflection_lm = largest loadable local model) and lands in
    Wave 1/5.
    """

    def __init__(self, registry: PromptRegistry) -> None:
        self._registry = registry

    async def optimize(
        self,
        name: str,
        *,
        train: list[Any],
        scorer: Any,
        max_calls: int = 100,
        current_body: str = "",
    ) -> PromptVersion:
        """Return the current champion unchanged (Phase 3D: real GEPA via ReflectiveOptimizer)."""
        champ = await self._registry.champion(name)
        if champ is None:
            raise ValueError(
                f"NoopOptimizer: no champion set for '{name}'; register and set_champion first."
            )
        return champ


# Satisfy the Protocol at import time so mypy can validate structural compatibility.
_: PromptRegistry = LocalPromptRegistry()
__: Optimizer = NoopOptimizer(_)
