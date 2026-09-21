"""A host that answered is never called unreachable.

Unreachable sends the owner to check whether a machine is on. A host that answered a 401 is on
and refused the key it was sent; one that answered a 404 or a 501 is on and is not the server
the setting should name. Each is ``refused``, with the setting to check -- the classification
``ProviderRefused`` gives an embedding call. A 429 or another 5xx is an answer a retry may turn
into a good one (a llama-server loading its model answers 503), which the embedding retry
treats as slow, and so does doctor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from morgan_brain.config import Settings
from morgan_brain.surfaces.cli.doctor import build_doctor_report
from morgan_brain.surfaces.cli.render import _render_doctor
from tests.fakes import flaky_model_server, model_server, slow_model_server

#: Port 1 refuses at once, so the probe a test does not look at contacts no server of the
#: developer's own.
_CLOSED = "http://127.0.0.1:1/v1"


async def _report(tmp_path: Path, **fields: Any) -> dict[str, Any]:
    settings = Settings(
        **{
            "data_dir": str(tmp_path),
            "embedding_backend": "provider",
            "llm_endpoint": _CLOSED,
            "embedding_endpoint": _CLOSED,
            "doctor_probe_timeout_seconds": 5.0,
            "doctor_slow_after_seconds": 2.0,
            **fields,
        }
    )
    return await build_doctor_report(settings, project="p", all_projects=False)


def _line(report: dict[str, Any], key: str) -> str:
    [line] = [ln for ln in _render_doctor(report).splitlines() if ln.startswith(f"{key}:")]
    return line


async def test_a_401_is_refused_and_names_the_key_the_request_carried(tmp_path):
    with flaky_model_server(fail_times=100, status=401) as embeddings:
        report = await _report(tmp_path, embedding_endpoint=embeddings)

    assert report["embedding_provider"] == "refused"
    assert report["embedding_probe"]["error"] == "HTTP 401; check MORGAN_EMBEDDING_API_KEY"
    assert _line(report, "embedding_provider").startswith("embedding_provider: refused (")


async def test_a_401_from_the_chat_host_serving_embeddings_names_the_chat_key(tmp_path):
    """No embedding endpoint of its own: the request went to the chat host carrying the chat
    key, so that is the key to check."""
    with flaky_model_server(fail_times=100, status=401) as chat:
        report = await _report(tmp_path, llm_endpoint=chat, embedding_endpoint="")

    assert report["provider"] == "reachable"  # its model listing answers
    assert report["embedding_provider"] == "refused"
    assert report["embedding_probe"]["error"] == "HTTP 401; check MORGAN_LLM_API_KEY"


async def test_a_404_is_refused_and_names_the_endpoint_setting(tmp_path):
    with flaky_model_server(fail_times=100, status=404) as embeddings:
        report = await _report(tmp_path, embedding_endpoint=embeddings)

    assert report["embedding_provider"] == "refused"
    assert report["embedding_probe"]["error"] == "HTTP 404; check MORGAN_EMBEDDING_ENDPOINT"


async def test_a_chat_endpoint_on_the_wrong_path_is_refused_by_its_own_setting(tmp_path):
    with model_server() as url:
        report = await _report(tmp_path, llm_endpoint=url.removesuffix("/v1") + "/wrong")

    assert report["provider"] == "refused"
    assert report["provider_probe"]["error"] == "HTTP 404; check MORGAN_LLM_ENDPOINT"


async def test_a_501_on_the_embedding_probe_is_refused(tmp_path):
    """A chat server started without embeddings: up, answering, and the wrong server for
    embeddings."""
    with model_server(embeddings=False) as chat:
        report = await _report(tmp_path, embedding_endpoint=chat)

    assert report["embedding_provider"] == "refused"
    assert report["embedding_probe"]["error"] == (
        "HTTP 501 (the server does not serve embeddings); check MORGAN_EMBEDDING_ENDPOINT"
    )


@pytest.mark.parametrize("status", [429, 503])
async def test_an_answer_a_retry_may_mend_is_slow_not_unreachable(tmp_path, status):
    with flaky_model_server(fail_times=100, status=status) as embeddings:
        report = await _report(tmp_path, embedding_endpoint=embeddings)

    assert report["embedding_provider"] == "slow"
    assert report["embedding_probe"]["error"] == f"HTTP {status}"
    # Slow by what it answered, not by the clock: the line does not claim it crossed the
    # threshold.
    line = _line(report, "embedding_provider")
    assert line.startswith("embedding_provider: slow (")
    assert line.endswith(f" s; HTTP {status})")
    assert "MORGAN_DOCTOR_SLOW_AFTER_SECONDS" not in line


async def test_a_slow_200_is_still_slow(tmp_path):
    with slow_model_server(delay=1.0) as embeddings:
        report = await _report(
            tmp_path, embedding_endpoint=embeddings, doctor_slow_after_seconds=0.5
        )

    assert report["embedding_provider"] == "slow"
    assert report["embedding_probe"]["error"] is None
    assert "MORGAN_DOCTOR_SLOW_AFTER_SECONDS=0.5" in _line(report, "embedding_provider")
