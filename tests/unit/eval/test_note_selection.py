"""Offline contract tests; no model service, personal data or task execution."""

import asyncio
import json

import httpx
import pytest

from morgan_brain.eval.note_selection import CASES, evaluate, score
from morgan_brain.providers.note_probe import ProbeFailure, probe, select_lines, validate_endpoint


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_goldens(case):
    """Every hand-labelled fixture is representable and exactly scored."""
    selected = select_lines(case.note, json.dumps({"status": "ok", "lines": case.expected}))
    assert score(case, selected) == {"exact": True, "tp": len(case.expected), "fp": 0, "fn": 0}


@pytest.mark.parametrize("lines", [[True], ["1"], [0], [-1], [2], [1, 1], list(range(1, 10))])
def test_bad_references(lines):
    """No coercion, fabricated references or duplicate selections are admitted."""
    with pytest.raises(ValueError):
        select_lines("One line", json.dumps({"status": "ok", "lines": lines}))


def test_refusal_and_extra_fields():
    """A refusal cannot smuggle tasks and added model text is forbidden."""
    with pytest.raises(ProbeFailure, match="refused"):
        select_lines("One line", '{"status":"refused","lines":[]}')
    with pytest.raises(ValueError):
        select_lines("One line", '{"status":"refused","lines":[1]}')
    with pytest.raises(ValueError):
        select_lines("One line", '{"status":"ok","lines":[1],"execute":true}')


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.com:11434",
        "http://8.8.8.8:11434",
        "http://169.254.169.254:80",
        "http://user:pass@127.0.0.1:11434",
        "http://127.0.0.1:11434/path",
        "http://127.0.0.1:11434?token=x",
        "http://127.0.0.1:11434#x",
        "http://127.0.0.1",
    ],
)
def test_endpoint_restrictions(endpoint):
    """Only explicit permitted IP endpoints can receive the synthetic probe."""
    with pytest.raises(ValueError):
        validate_endpoint(endpoint)


@pytest.mark.parametrize(
    "endpoint", ["http://127.0.0.1:11434", "http://192.168.100.101:11434", "http://[::1]:11434"]
)
def test_local_endpoint(endpoint):
    """Loopback and the requested private-network endpoint are supported."""
    assert validate_endpoint(endpoint) == endpoint


def transport_for(result, *, models=None):
    """Serve a listed model and one bounded generation response without network I/O."""

    def handler(request):
        """Check the wire request does not enable streaming or unlimited output."""
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": models if models is not None else [{"name": "tiny"}]}
            )
        body = json.loads(request.content)
        assert body["stream"] is False
        assert body["options"]["num_predict"] == 256
        assert "keep_alive" not in body
        return httpx.Response(200, json=result)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_valid_probe():
    """One listed model produces verbatim source lines, never an executable task."""
    result = {
        "model": "tiny",
        "done": True,
        "done_reason": "stop",
        "response": '{"status":"ok","lines":[2]}',
    }
    assert await probe(
        "http://127.0.0.1:11434", "tiny", "Done.\nBuy bread.", transport=transport_for(result)
    ) == ("Buy bread.",)


@pytest.mark.parametrize(
    "result,reason",
    [
        ({"done": False}, "incomplete"),
        ({"done": True, "done_reason": "length"}, "incomplete"),
        (
            {"model": "other", "done": True, "done_reason": "stop", "response": "{}"},
            "invalid_response",
        ),
        (
            {"model": "tiny", "done": True, "done_reason": "stop", "response": "not JSON"},
            "invalid_response",
        ),
        (
            {
                "model": "tiny",
                "done": True,
                "done_reason": "stop",
                "response": '{"status":"refused","lines":[]}',
            },
            "refused",
        ),
    ],
)
@pytest.mark.asyncio
async def test_generation_failures(result, reason):
    """Incomplete, malformed, mismatched and refused answers yield no selections."""
    with pytest.raises(ProbeFailure, match=reason):
        await probe("http://127.0.0.1:11434", "tiny", "Buy bread.", transport=transport_for(result))


@pytest.mark.parametrize(
    "models,reason",
    [([], "model_not_listed"), ([{"name": "tiny", "remote_host": "remote"}], "remote_model")],
)
@pytest.mark.asyncio
async def test_no_unlisted_or_remote_model(models, reason):
    """Inventory failure stops before a generation request is sent."""
    with pytest.raises(ProbeFailure, match=reason):
        await probe(
            "http://127.0.0.1:11434",
            "tiny",
            "Buy bread.",
            transport=transport_for({}, models=models),
        )


@pytest.mark.parametrize(
    "status,reason", [(302, "http_error"), (403, "refused"), (500, "http_error")]
)
@pytest.mark.asyncio
async def test_no_redirects_or_error_body(status, reason):
    """Redirects are not followed and remote error text is not exposed."""
    calls = []

    def handler(request):
        """Return an error with a deliberately sensitive-looking body."""
        calls.append(request)
        return httpx.Response(status, headers={"location": "http://8.8.8.8"}, text="secret")

    with pytest.raises(ProbeFailure) as caught:
        await probe(
            "http://127.0.0.1:11434", "tiny", "Buy bread.", transport=httpx.MockTransport(handler)
        )
    assert str(caught.value) == reason
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_whole_request_timeout():
    """The outer deadline works even when a transport ignores httpx read timeouts."""

    async def handler(request):
        """Simulate a stalled peer without opening a socket."""
        await asyncio.sleep(1)
        return httpx.Response(200, json={})

    with pytest.raises(ProbeFailure, match="timeout"):
        await probe(
            "http://127.0.0.1:11434",
            "tiny",
            "Buy bread.",
            seconds=0.01,
            transport=httpx.MockTransport(handler),
        )


@pytest.mark.asyncio
async def test_response_cap():
    """Oversized responses are rejected before JSON interpretation."""
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"x" * 262145))
    with pytest.raises(ProbeFailure, match="oversized"):
        await probe("http://127.0.0.1:11434", "tiny", "Buy bread.", transport=transport)


@pytest.mark.asyncio
async def test_unavailable_is_unmeasured(monkeypatch):
    """Infrastructure failure is visible, stops repeats and cannot masquerade as a score."""

    async def unavailable(*args, **kwargs):
        """Return the same sanitized failure as a refused TCP connection."""
        raise ProbeFailure("unavailable")

    monkeypatch.setattr("morgan_brain.eval.note_selection.probe", unavailable)
    result = await evaluate("http://127.0.0.1:11434", "tiny")
    assert result["cases_valid"] == 0
    assert result["cases_unattempted"] == 5
    assert result["exact_accuracy_valid_only"] is None
    assert result["tp"] is None
    assert result["approved"] is False


def test_negative_case_false_positive():
    """A grounded quote can still be a wrong task, and scoring detects it."""
    case = CASES[2]
    assert score(case, (case.note.splitlines()[0],)) == {"exact": False, "tp": 0, "fp": 1, "fn": 0}


@pytest.mark.asyncio
async def test_streaming_deadline_and_close():
    """A trickling body cannot extend the whole-call budget and its stream is closed."""
    closed = []

    class SlowBody(httpx.AsyncByteStream):
        """Supply bytes slowly enough to evade only a per-chunk timeout."""

        async def __aiter__(self):
            """Trickle incomplete JSON until the enclosing deadline cancels consumption."""
            for _ in range(100):
                await asyncio.sleep(0.005)
                yield b" "

        async def aclose(self):
            """Observe cleanup after cancellation."""
            closed.append(True)

    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=SlowBody()))
    with pytest.raises(ProbeFailure, match="timeout"):
        await probe(
            "http://127.0.0.1:11434", "tiny", "Buy bread.", seconds=0.02, transport=transport
        )
    assert closed == [True]


@pytest.mark.asyncio
async def test_cancellation_propagates():
    """Caller cancellation remains cancellation instead of becoming a retryable result."""

    async def handler(request):
        """Simulate cancellation during inventory retrieval."""
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await probe(
            "http://127.0.0.1:11434", "tiny", "Buy bread.", transport=httpx.MockTransport(handler)
        )
