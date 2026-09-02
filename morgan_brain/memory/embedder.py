"""Memory-layer Embedder protocol (``embed`` / ``embed_batch``) and the hash stub.

The protocol and the deterministic fake live here; the live HTTP implementation lives in
``providers/adapters/embeddings.py``, because nothing above the provider layer may talk to
a model endpoint. ``providers/factory.py::build_embedder`` is the single seam that picks
between the two.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class FakeEmbedder:
    """Deterministic hash-based embeddings. Not semantically meaningful, but stable and
    content-sensitive — enough to test storage, retrieval plumbing, and ranking determinism."""

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    async def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [digest[i % len(digest)] / 255.0 for i in range(self._dim)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]
