"""Reranker seam — keeps the existing multi-level fallback behind one Protocol."""

from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    async def arerank(
        self, query: str, docs: list[str], *, top_k: int | None = None
    ) -> list[tuple[int, float]]: ...
