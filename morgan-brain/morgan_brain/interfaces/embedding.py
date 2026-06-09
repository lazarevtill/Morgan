"""Providers-layer Embedder protocol (``aembed`` batch API).

Used by ``providers/factory.py``, ``providers/adapters/``, and the factory/capability layer.
Distinct from ``modules.memory.indexing.embedder.Embedder`` (which uses ``embed``/``embed_batch``)
that is consumed by the memory-store and composition layer.
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    async def aembed(self, texts: list[str]) -> list[list[float]]: ...
