"""Protocol definitions for the learning-lifecycle seam.

These Protocols are intentionally provider-agnostic.  The real MLflow-backed
implementation lands in Wave 1/5; today's LocalPromptRegistry + NoopOptimizer are
the dependency-light defaults.

Champion alias + rollback semantics
------------------------------------
A "champion" is just a named alias pointing at a specific version number.
``set_champion(name, version)`` re-points the alias atomically — passing an older
version number achieves instant rollback without deleting any history.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class PromptVersion(BaseModel):
    """Immutable snapshot of one versioned prompt body."""

    name: str
    version: int
    body: str
    created_at: datetime | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    commit_message: str = ""


class EvalScore(BaseModel):
    """Outcome of evaluating a prompt candidate against a scorer."""

    scores: dict[str, float]
    passed: bool


@runtime_checkable
class PromptRegistry(Protocol):
    """Versioned prompt store with champion-alias support.

    All methods are async so implementations can use async I/O (e.g. SQLite via
    aiosqlite or an HTTP-backed MLflow server) without changing call sites.
    """

    async def register(
        self,
        name: str,
        body: str,
        *,
        commit_message: str = "",
        metrics: dict[str, float] | None = None,
    ) -> PromptVersion:
        """Store *body* as a new version (auto-incremented per *name*) and return it."""
        ...

    async def champion(self, name: str) -> PromptVersion | None:
        """Return the currently-championed version, or ``None`` if none set."""
        ...

    async def set_champion(self, name: str, version: int) -> None:
        """Point the champion alias at *version* (rollback = pass an older version)."""
        ...

    async def list_versions(self, name: str) -> list[PromptVersion]:
        """Return all stored versions for *name*, oldest first."""
        ...


@runtime_checkable
class Optimizer(Protocol):
    """Seam for a prompt-optimization backend (GEPA via MLflow, Wave 1/5).

    ``optimize`` returns a *candidate* ``PromptVersion`` — it is NOT auto-promoted to
    champion.  The caller is responsible for running the validation gate and calling
    ``set_champion`` if the candidate passes.
    """

    async def optimize(
        self,
        name: str,
        *,
        train: list[Any],
        scorer: Any,
        max_calls: int = 100,
        current_body: str = "",
    ) -> PromptVersion:
        """Produce a candidate optimized prompt version without promoting it.

        Args:
            name:         Prompt name (used in the returned ``PromptVersion``).
            train:        Training examples — ``list[Example]`` or ``list[dict]``.
            scorer:       Callable ``(body: str) → float`` (sync or async).
            max_calls:    Maximum number of LLM / optimization iterations allowed.
            current_body: The current champion body (baseline for comparison).
                          Defaults to ``""`` (no current champion).
        """
        ...
