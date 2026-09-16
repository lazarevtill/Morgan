"""``morgan doctor`` probes the chat server and the embedding server separately.

Every remember and recall needs embeddings; only ask and consolidate need chat. Embeddings
often have a server of their own, so a report that probed only the chat server called an
install healthy while every recall failed. Real sockets: a loopback server that answers, and
port 1, which refuses at once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from morgan_brain.config import Settings
from morgan_brain.surfaces.cli.doctor import build_doctor_report
from tests.fakes import model_server

_CLOSED = "http://127.0.0.1:1/v1"


async def _report(tmp_path: Path, **fields: Any) -> dict[str, Any]:
    settings = Settings(
        **{
            "data_dir": str(tmp_path),
            "embedding_backend": "provider",
            "embedding_endpoint": "",
            **fields,
        }
    )
    return await build_doctor_report(settings, project="p", all_projects=False)


async def test_an_embedding_server_that_is_down_is_reported_while_chat_answers(tmp_path):
    with model_server() as chat:
        report = await _report(tmp_path, llm_endpoint=chat, embedding_endpoint=_CLOSED)

    assert report["provider"] == "reachable"
    assert report["embedding_endpoint"] == _CLOSED
    assert report["embedding_provider"] == "unreachable"


async def test_a_chat_server_that_is_down_does_not_hide_working_embeddings(tmp_path):
    with model_server() as embeddings:
        report = await _report(tmp_path, llm_endpoint=_CLOSED, embedding_endpoint=embeddings)

    assert report["provider"] == "unreachable"
    assert report["embedding_endpoint"] == embeddings
    assert report["embedding_provider"] == "reachable"


async def test_embeddings_are_probed_on_the_chat_server_when_they_fall_back_to_it(tmp_path):
    """A chat server started without embeddings lists its models and answers every
    embedding request with a 501, so listing models proves nothing about recall."""
    with model_server(embeddings=False) as chat:
        report = await _report(tmp_path, llm_endpoint=chat)

    assert report["provider"] == "reachable"
    assert report["embedding_endpoint"] == chat
    assert report["embedding_provider"] == "unreachable"


async def test_the_hash_backend_uses_no_embedding_server(tmp_path):
    """A server that answers is configured, and still not reported: the hash backend never
    calls it, so reachable would describe a server recall does not use."""
    with model_server() as embeddings:
        report = await _report(
            tmp_path, embedding_backend="hash", llm_endpoint=_CLOSED, embedding_endpoint=embeddings
        )

    assert report["embedding_endpoint"] is None
    assert report["embedding_provider"] == "not used"
