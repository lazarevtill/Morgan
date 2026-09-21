"""doctor says where the owner's text goes, by host, and never prints a key.

The chat host receives questions, recalled memories and whole batches of memories to
consolidate; the embedding host receives every memory and every query. Which host is which is
what the owner needs to know to decide what may be stored -- and a report that is pasted into
an issue or a chat must not carry the credential that opens either host.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from morgan_brain.config import Settings
from morgan_brain.surfaces.cli.doctor import build_doctor_report
from morgan_brain.surfaces.cli.render import _render_doctor
from tests.fakes import flaky_model_server, model_server

#: Port 1 refuses at once, so no default endpoint -- which may be a real server on the
#: developer's machine -- is ever contacted.
_CLOSED = "http://127.0.0.1:1/v1"


async def _report(tmp_path: Path, **fields: Any) -> dict[str, Any]:
    settings = Settings(
        **{
            "data_dir": str(tmp_path),
            "embedding_backend": "provider",
            "llm_endpoint": _CLOSED,
            "embedding_endpoint": _CLOSED,
            "doctor_probe_timeout_seconds": 5.0,
            **fields,
        }
    )
    return await build_doctor_report(settings, project="p", all_projects=False)


async def test_the_data_flow_lines_name_hosts_and_never_keys(tmp_path):
    report = await _report(tmp_path, llm_api_key="secret-key")

    flows = report["data_flow"]
    assert any("ask" in f["carries"] and "consolidate" in f["carries"] for f in flows)
    assert any("recall" in f["carries"] for f in flows)
    assert "secret-key" not in repr(report)


async def test_each_endpoint_is_named_by_its_host_alone(tmp_path):
    """Host only: no scheme, port or path -- and so no userinfo either, the one part of a URL
    that can carry a credential."""
    report = await _report(
        tmp_path,
        llm_endpoint="http://chat.invalid:8081/v1",
        embedding_endpoint="http://embed.invalid:9000/v1",
    )

    by_host = {f["host"]: f for f in report["data_flow"]}
    assert set(by_host) == {"chat.invalid", "embed.invalid"}
    assert set(by_host["chat.invalid"]["carries"]) == {"ask", "consolidate"}
    assert set(by_host["embed.invalid"]["carries"]) >= {"remember", "recall", "import"}
    assert by_host["chat.invalid"]["settings"] == ["MORGAN_LLM_ENDPOINT"]
    assert by_host["embed.invalid"]["settings"] == ["MORGAN_EMBEDDING_ENDPOINT"]


async def test_one_server_for_both_is_one_line_carrying_everything(tmp_path):
    report = await _report(tmp_path, llm_endpoint="http://both.invalid/v1", embedding_endpoint="")

    [flow] = report["data_flow"]
    assert flow["host"] == "both.invalid"
    assert set(flow["carries"]) >= {"ask", "consolidate", "remember", "recall", "import"}


async def test_the_hash_backend_sends_no_text_to_an_embedding_host(tmp_path):
    report = await _report(
        tmp_path, embedding_backend="hash", embedding_endpoint="http://embed.invalid/v1"
    )

    assert [f["host"] for f in report["data_flow"]] == ["127.0.0.1"]


async def test_no_key_appears_in_the_report_or_its_rendering(tmp_path):
    """Every key set, and an embedding host that refuses with the key it was sent written
    back into its answer, as a careless gateway does: the refusal is reported by its status
    and the setting to check, never by quoting what the server said."""
    keys = {
        "llm_api_key": "chat-secret-key-1",
        "embedding_api_key": "embed-secret-key-2",
        "api_key": "inbound-secret-key-3",
    }
    with (
        model_server() as chat,
        flaky_model_server(fail_times=100, status=401, echo=lambda t: f"refused {t}") as embed,
    ):
        report = await _report(tmp_path, llm_endpoint=chat, embedding_endpoint=embed, **keys)

    assert report["provider"] == "reachable"
    assert report["embedding_provider"] == "refused"
    assert "HTTP 401" in report["embedding_probe"]["error"]
    assert "MORGAN_EMBEDDING_API_KEY" in report["embedding_probe"]["error"]
    printed = "\n".join([repr(report), json.dumps(report, default=str), _render_doctor(report)])
    for key in keys.values():
        assert key not in printed
