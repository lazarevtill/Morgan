"""A server that answers slowly is reported as slow, with the timeout named.

Called unreachable, it sends the owner to check a machine that is working. The embedding host
has no keep-alive by the owner's decision, so the first call after idle takes about 43 s --
slow is the normal case now, not the exception.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from morgan_brain.config import Settings
from morgan_brain.surfaces.cli.doctor import build_doctor_report
from morgan_brain.surfaces.cli.render import _render_doctor
from tests.fakes import slow_model_server

#: Port 1 refuses at once: the chat probe these tests do not look at answers nothing, and no
#: default endpoint -- which may be a real server on the developer's machine -- is contacted.
_CLOSED = "http://127.0.0.1:1/v1"


async def _report(
    tmp_path: Path,
    *,
    probe_timeout: float = 5.0,
    slow_after: float = 2.0,
    **fields: Any,
) -> dict[str, Any]:
    settings = Settings(
        **{
            "data_dir": str(tmp_path),
            "embedding_backend": "provider",
            "llm_endpoint": _CLOSED,
            "embedding_endpoint": _CLOSED,
            "doctor_probe_timeout_seconds": probe_timeout,
            "doctor_slow_after_seconds": slow_after,
            **fields,
        }
    )
    return await build_doctor_report(settings, project="p", all_projects=False)


async def test_a_slow_host_is_slow_not_unreachable(tmp_path):
    with slow_model_server(delay=3.0) as url:
        report = await _report(tmp_path, embedding_endpoint=url, probe_timeout=10.0, slow_after=2.0)

    assert report["embedding_provider"] == "slow"
    assert 3.0 <= report["embedding_probe"]["seconds"] < 10.0
    assert report["embedding_probe"]["slow_after_seconds"] == 2.0


async def test_a_host_that_misses_the_timeout_names_the_setting(tmp_path):
    with slow_model_server(delay=3.0) as url:
        report = await _report(tmp_path, embedding_endpoint=url, probe_timeout=1.0)

    assert report["embedding_provider"] == "unreachable"
    assert "MORGAN_DOCTOR_PROBE_TIMEOUT_SECONDS" in report["embedding_probe"]["error"]


async def test_a_slow_answer_prints_its_seconds_beside_the_setting_it_crossed(tmp_path):
    """The line the owner reads: how long it took, and what "slow" was measured against."""
    with slow_model_server(delay=3.0) as url:
        report = await _report(tmp_path, llm_endpoint=url, probe_timeout=10.0, slow_after=2.0)

    [line] = [ln for ln in _render_doctor(report).splitlines() if ln.startswith("provider:")]
    assert line.startswith("provider: slow (3.")
    assert line.endswith(" s; MORGAN_DOCTOR_SLOW_AFTER_SECONDS=2.0)")
    assert report["provider_probe"]["timeout_seconds"] == 10.0


async def test_a_connect_timeout_names_the_endpoint_not_the_probe_timeout(tmp_path, monkeypatch):
    """No connection within the timeout is a host that is off or an address that is wrong
    (SPEC-phase0 section 3.4): the endpoint setting is what to check, not the timeout."""
    real_client = httpx.AsyncClient

    def _never_connects(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: real_client(transport=httpx.MockTransport(_never_connects)),
    )

    report = await _report(tmp_path)

    assert report["provider"] == report["embedding_provider"] == "unreachable"
    assert report["provider_probe"]["error"] == "ConnectTimeout; check MORGAN_LLM_ENDPOINT"
    assert report["embedding_probe"]["error"] == "ConnectTimeout; check MORGAN_EMBEDDING_ENDPOINT"
