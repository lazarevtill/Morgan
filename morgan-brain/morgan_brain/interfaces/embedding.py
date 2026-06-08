"""Embedder seam — unifies the memory Embedder. Model+dim are tracked in the CapabilityDescriptor."""
from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    async def aembed(self, texts: list[str]) -> list[list[float]]: ...
