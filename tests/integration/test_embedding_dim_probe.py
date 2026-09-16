"""Opening memory checks the embedding model's width against ``MORGAN_EMBEDDING_DIM``.

When the two disagree, every insert into the vector table would fail, so the context refuses
to open. The refusal has to name the server that answered, which is the embedding endpoint
whenever one is configured, not the chat endpoint.
"""

from __future__ import annotations

import pytest

from morgan_brain.composition import build_memory_context
from morgan_brain.config import Settings
from tests.fakes import model_server


def test_a_width_mismatch_names_the_embedding_endpoint_that_answered(tmp_path):
    with model_server(embedding_dim=3) as embeddings:
        settings = Settings(
            data_dir=str(tmp_path),
            llm_endpoint="http://chat.invalid/v1",
            embedding_endpoint=embeddings,
            embedding_backend="provider",
            embedding_dim=1024,
        )
        with pytest.raises(RuntimeError, match="3-dimensional") as info:
            build_memory_context(settings)

    assert embeddings in str(info.value)
    assert "chat.invalid" not in str(info.value)
