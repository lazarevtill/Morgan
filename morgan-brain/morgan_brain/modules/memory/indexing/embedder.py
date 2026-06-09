"""Memory-layer Embedder protocol (``embed`` / ``embed_batch`` per-text API).

Used by ``modules.memory.store``, ``composition.py``, and all integration/unit tests
that need a fake embedding backend (FakeEmbedder).
Distinct from ``interfaces.embedding.Embedder`` (``aembed`` batch API) used by the
providers/factory layer.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable

import httpx


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


class OllamaEmbedder:
    def __init__(self, endpoint: str, model: str, timeout: float = 30.0) -> None:
        self._url = endpoint.rstrip("/") + "/embeddings"
        self._model = model
        self._timeout = timeout

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(self._url, json={"model": self._model, "input": texts})
            resp.raise_for_status()
            data = resp.json()["data"]
        return [item["embedding"] for item in data]
