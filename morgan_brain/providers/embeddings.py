"""The live embedding adapter — an OpenAI-compatible ``/embeddings`` client.

This lives in ``providers/adapters/`` because that is where the invariant puts it: nothing
above the provider layer talks to a model endpoint. It previously sat in
``modules/memory/indexing/embedder.py`` under the name ``OllamaEmbedder``, holding a
module-level ``import httpx`` two layers above the seam, while a correctly-placed
``OpenAICompatEmbedder`` in ``openai_compat.py`` went uncalled from the day it was written.
One of them had to go; the one that shipped is the one that stayed.

``providers/factory.py::build_embedder`` is the only caller — the single decision point
between this and the deterministic hash stub.
"""

from __future__ import annotations

import httpx

from morgan_brain.providers.wire import ProviderUnreachable


class OpenAICompatEmbedder:
    """Embeddings over any OpenAI-compatible ``/embeddings`` endpoint.

    Works against llama-server (the default), Ollama's ``/v1`` shim, vLLM, or a hosted
    provider — the wire format is the same and only the endpoint URL differs.

    Implements the memory layer's ``Embedder`` protocol (``embed`` / ``embed_batch``), not the
    batch-only ``aembed`` shape of the deleted duplicate: the memory store embeds one memory at
    a time on the write path and a list at index-rebuild time.

    Args:
        endpoint: Base URL of the OpenAI-compatible endpoint.
        model:    Embedding model name.
        timeout:  Request timeout in seconds. Sized by default for a remote server over an
                  overlay network under GPU load, not a loopback socket.
        api_key:  Outbound bearer token, if the endpoint enforces one (llama-server's
                  ``--api-key``). ``None``/empty sends no ``Authorization`` header. This is
                  ``MORGAN_LLM_API_KEY``, never ``MORGAN_API_KEY`` — opposite directions.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        timeout: float = 120.0,
        api_key: str | None = None,
    ) -> None:
        self._url = endpoint.rstrip("/") + "/embeddings"
        self._model = model
        self._timeout = timeout
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    self._url,
                    json={"model": self._model, "input": texts},
                    headers=self._headers,
                )
        except httpx.TransportError as exc:
            # Connection refused, DNS failure, timeout: the endpoint gave no answer.
            raise ProviderUnreachable(self._url, f"{type(exc).__name__}: {exc}") from exc
        resp.raise_for_status()
        data = resp.json()["data"]
        return [item["embedding"] for item in data]
