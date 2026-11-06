"""Local chat proposals remain bounded, unapproved and separate from memory writes."""

import io
import json

import httpx
import pytest

from morgan_brain import note_proposal_cli
from morgan_brain.providers.note_probe import ProbeFailure, propose_chat


def _transport(choice):
    """Respond with one explicit model identity and a configurable chat choice."""

    def handler(request):
        """Validate the schema request and return no real model output."""
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "granite"}]})
        payload = json.loads(request.content)
        assert payload["max_tokens"] == 128
        assert payload["response_format"]["type"] == "json_schema"
        return httpx.Response(200, json={"model": "granite", "choices": [choice]})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_chat_proposal():
    """Structured chat produces the exact selected source line."""
    choice = {"finish_reason": "stop", "message": {"content": '{"status":"ok","lines":[2]}'}}
    result = await propose_chat(
        "http://127.0.0.1:11436", "granite", "Done.\nWater plants.", transport=_transport(choice)
    )
    assert result == ("Water plants.",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "choice,reason",
    [
        ({"finish_reason": "length"}, "incomplete"),
        ({"finish_reason": "stop", "message": {"content": "{}"}}, "invalid_response"),
        ({"finish_reason": "stop", "message": {"content": "{}", "tool_calls": [{}]}}, "refused"),
        (
            {"finish_reason": "stop", "message": {"content": '{"status":"ok","lines":[9]}'}},
            "invalid_response",
        ),
    ],
)
async def test_rejected_chat(choice, reason):
    """Incomplete, tool-calling or invented-line output never becomes a proposal."""
    with pytest.raises(ProbeFailure, match=reason):
        await propose_chat(
            "http://127.0.0.1:11436", "granite", "Water plants.", transport=_transport(choice)
        )


@pytest.mark.asyncio
async def test_wrong_model():
    """A different inventory model cannot silently substitute for the selected one."""
    with pytest.raises(ProbeFailure, match="model_not_listed"):
        await propose_chat(
            "http://127.0.0.1:11436", "other", "Water plants.", transport=_transport({})
        )


def test_cli_private_unapproved_output(tmp_path, monkeypatch, capsys):
    """Real stdin content is saved privately only after a successful explicit proposal."""

    async def fake(*args, **kwargs):
        """Return a proposed exact line without network access."""
        return ("Water plants.",)

    output = tmp_path / "private" / "proposal.json"
    monkeypatch.setattr(note_proposal_cli, "propose_chat", fake)
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(b"Water plants.")))
    monkeypatch.setattr(
        "sys.argv",
        [
            "proposal",
            "--endpoint",
            "http://127.0.0.1:11436",
            "--model",
            "granite",
            "--output",
            str(output),
        ],
    )
    note_proposal_cli.main()
    data = json.loads(output.read_text())
    assert data["source"] == "agent_inferred"
    assert data["approved"] is False
    assert data["review_required"] is True
    assert output.stat().st_mode & 0o777 == 0o600
    assert "Water plants" not in capsys.readouterr().out


def test_failure_creates_no_output(tmp_path, monkeypatch, capsys):
    """Unavailable inference yields a sanitized result with no persisted draft."""

    async def unavailable(*args, **kwargs):
        """Simulate a model outage."""
        raise ProbeFailure("unavailable")

    output = tmp_path / "private" / "proposal.json"
    monkeypatch.setattr(note_proposal_cli, "propose_chat", unavailable)
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(b"Private note")))
    monkeypatch.setattr(
        "sys.argv",
        [
            "proposal",
            "--endpoint",
            "http://127.0.0.1:11436",
            "--model",
            "granite",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit):
        note_proposal_cli.main()
    assert not output.parent.exists()
    assert json.loads(capsys.readouterr().out)["reason"] == "unavailable"
