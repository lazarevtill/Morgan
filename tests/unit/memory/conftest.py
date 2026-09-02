"""Shared MemoryModule factory for the memory tests.

``FakeEmbedder`` is sha256-based, so identical text embeds identically across processes --
these tests build a *fresh* MemoryModule per call to simulate a restart.
"""

from __future__ import annotations

from datetime import UTC, datetime

from morgan_brain.composition import build_memory_module as _build
from morgan_brain.memory.db import open_db
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.module import MemoryModule


def build_memory_module(path: str, *, dim: int = 4) -> MemoryModule:
    """Build a MemoryModule over the durable stack rooted at *path* (or ``:memory:``)."""
    return _build(
        open_db(path),
        embedder=FakeEmbedder(dim=dim),
        dim=dim,
        clock=lambda: datetime.now(UTC),
    )
