"""The embedding adapter returns vectors in the order of its inputs, not the server's.

Each item of an OpenAI-compatible ``/embeddings`` response carries the ``index`` of the input
it embeds; the order of the list itself is not promised. The first request a process sends
carries the caller's texts and the five fingerprint strings together and is split by
position, so an adapter that trusted list order would hand the caller a fingerprint vector.
"""

from __future__ import annotations

from morgan_brain.providers.embeddings import OpenAICompatEmbedder, RetryBudget
from tests.fakes import model_server


async def test_vectors_come_back_in_input_order_when_the_server_reorders_them() -> None:
    texts = ["first", "second", "third"]
    with model_server(embedding_dim=4) as url:
        in_order = await _embedder(url).embed_batch(texts)
    with model_server(embedding_dim=4, reorder=True) as url:
        reordered = await _embedder(url).embed_batch(texts)

    assert len(set(map(tuple, in_order))) == 3
    assert reordered == in_order


def _embedder(url: str) -> OpenAICompatEmbedder:
    return OpenAICompatEmbedder(
        url,
        "m",
        budget=RetryBudget(
            seconds=10.0, unreachable_seconds=1.0, backoff_seconds=0.05, attempt_seconds=10.0
        ),
        setting="MORGAN_EMBEDDING_ENDPOINT",
        key_setting="MORGAN_LLM_API_KEY",
    )
