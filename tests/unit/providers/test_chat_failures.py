"""Chat failures cross the real HTTP/SDK boundary with safe, actionable errors."""

from __future__ import annotations

import json
import threading
import traceback
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote

import pytest

from morgan_brain.providers.openai_compat import OpenAICompatAdapter
from morgan_brain.providers.wire import ChatMessage, ProviderRefused, ProviderUnreachable
from tests.fakes import raw_model_server, silent_model_server

_KEY = "test-only-key/never-print-this"
_MESSAGES = [ChatMessage(role="user", content="synthetic question")]


@contextmanager
def _failed_stream_server(failure, partial):
    """Send real SSE headers, then fail before or after one complete text delta."""
    calls = []
    stopped = threading.Event()
    payload = b""
    if partial:
        payload += (
            b'data: {"id":"synthetic","object":"chat.completion.chunk","created":0,'
            b'"model":"synthetic","choices":[{"index":0,"delta":{"content":"Partial"},'
            b'"finish_reason":null}]}\n\n'
        )
    if failure == "sse_error":
        error = {"error": {"message": f"stream failed {_KEY} {quote(_KEY, safe='')}"}}
        payload += f"data: {json.dumps(error)}\n\n".encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            calls.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            # A dropped connection must be distinguishable from a complete empty stream.
            length = len(payload) + (0 if failure == "sse_error" else 1024)
            self.send_header("Content-Length", str(length))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
            if failure == "timeout":
                stopped.wait(5)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls
    finally:
        stopped.set()
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize("failure", ["sse_error", "timeout", "disconnect"])
@pytest.mark.parametrize("partial", [False, True])
async def test_stream_read_failures_are_safe_without_replaying_partial_output(failure, partial):
    chunks = []
    with _failed_stream_server(failure, partial) as (url, calls):
        adapter = OpenAICompatAdapter(
            base_url=url,
            api_key=_KEY,
            provider="synthetic",
            setting="MORGAN_LLM_ENDPOINT",
            timeout=0.25,
        )
        try:
            with pytest.raises(ProviderUnreachable) as caught:
                async for chunk in adapter.astream(_MESSAGES, model="synthetic"):
                    chunks.append(chunk)
        finally:
            await adapter._client.close()

    assert [(chunk.kind, chunk.text) for chunk in chunks] == (
        [("text_delta", "Partial")] if partial else []
    )
    assert calls == ["/v1/chat/completions"]
    assert caught.value.outcome == "slow"
    assert url in str(caught.value)
    assert "MORGAN_LLM_ENDPOINT" in str(caught.value)
    assert "unreachable" not in str(caught.value)
    logged = "".join(traceback.format_exception(caught.value))
    assert _KEY not in logged and quote(_KEY, safe="") not in logged


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("status", [301, 400, 401, 403, 404, 501, 429, 500, 503])
async def test_chat_status_names_endpoint_and_setting_without_echoing_credentials(status, stream):
    calls = []
    body = json.dumps({"error": f"refused {_KEY} {quote(_KEY, safe='')}"}).encode()

    def reply(request):
        calls.append(request)
        return (
            f"HTTP/1.1 {status} scripted\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )

    expected = ProviderUnreachable if status in (429, 500, 503) else ProviderRefused
    with raw_model_server(reply) as url:
        adapter = OpenAICompatAdapter(
            base_url=url, api_key=_KEY, provider="synthetic", setting="MORGAN_LLM_ENDPOINT"
        )
        try:
            with pytest.raises(expected) as failure:
                if stream:
                    async for _ in adapter.astream(_MESSAGES, model="synthetic"):
                        pytest.fail("A failed request must yield nothing")
                else:
                    await adapter.agenerate(_MESSAGES, model="synthetic")
        finally:
            await adapter._client.close()

    message = str(failure.value)
    assert url in message
    assert f"HTTP {status}" in message
    assert (
        "MORGAN_LLM_API_KEY" in message
        if status in (401, 403)
        else ("MORGAN_LLM_ENDPOINT" in message)
    )
    assert _KEY not in message and quote(_KEY, safe="") not in message
    logged = "".join(traceback.format_exception(failure.value))
    assert _KEY not in logged and quote(_KEY, safe="") not in logged
    assert len(calls) == 1  # Chat has no automatic retries, including transient failures.
    if isinstance(failure.value, ProviderUnreachable):
        assert failure.value.outcome == "slow"
        assert "unreachable" not in message


@pytest.mark.parametrize("stream", [False, True])
async def test_chat_timeout_reports_slow_instead_of_unreachable(stream):
    with silent_model_server() as url:
        adapter = OpenAICompatAdapter(
            base_url=url,
            api_key=_KEY,
            provider="synthetic",
            setting="MORGAN_LLM_ENDPOINT",
            timeout=0.1,
        )
        try:
            with pytest.raises(ProviderUnreachable) as failure:
                if stream:
                    async for _ in adapter.astream(_MESSAGES, model="synthetic"):
                        pytest.fail("A timed out request must yield nothing")
                else:
                    await adapter.agenerate(_MESSAGES, model="synthetic")
        finally:
            await adapter._client.close()
    assert failure.value.outcome == "slow"
    assert url in str(failure.value)
    assert "MORGAN_LLM_ENDPOINT" in str(failure.value)
    assert "unreachable" not in str(failure.value)


@pytest.mark.parametrize("stream", [False, True])
async def test_malformed_http_does_not_log_a_raw_exception_chain(stream):
    def reply(request):
        return f"malformed HTTP header Bearer {_KEY}\r\n\r\n".encode()

    with raw_model_server(reply) as url:
        adapter = OpenAICompatAdapter(
            base_url=url, api_key=_KEY, provider="synthetic", setting="MORGAN_LLM_ENDPOINT"
        )
        try:
            with pytest.raises(ProviderUnreachable) as failure:
                if stream:
                    async for _ in adapter.astream(_MESSAGES, model="synthetic"):
                        pytest.fail("Malformed HTTP must yield nothing")
                else:
                    await adapter.agenerate(_MESSAGES, model="synthetic")
        finally:
            await adapter._client.close()
    assert _KEY not in "".join(traceback.format_exception(failure.value))
    assert url in str(failure.value)


@pytest.mark.parametrize("stream", [False, True])
async def test_malformed_refusal_body_does_not_escape_with_a_credential_echo(stream):
    calls = []

    def reply(request):
        calls.append(request)
        return (
            b"HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
            + f"invalid-chunk-{_KEY}-{quote(_KEY, safe='')}\r\n".encode()
        )

    with raw_model_server(reply) as url:
        adapter = OpenAICompatAdapter(
            base_url=url, api_key=_KEY, provider="synthetic", setting="MORGAN_LLM_ENDPOINT"
        )
        try:
            with pytest.raises(ProviderUnreachable) as failure:
                if stream:
                    async for _ in adapter.astream(_MESSAGES, model="synthetic"):
                        pytest.fail("A malformed refusal must yield nothing")
                else:
                    await adapter.agenerate(_MESSAGES, model="synthetic")
        finally:
            await adapter._client.close()

    assert calls and len(calls) == 1
    assert url in str(failure.value)
    assert "MORGAN_LLM_ENDPOINT" in str(failure.value)
    logged = "".join(traceback.format_exception(failure.value))
    assert _KEY not in logged and quote(_KEY, safe="") not in logged


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("status", [307, 308])
async def test_redirects_never_forward_a_prompt_to_another_endpoint(status, stream):
    destination_requests = []

    def destination(request):
        destination_requests.append(request)
        # Even another failure must not let a redirect silently move the prompt.
        return b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n"

    with raw_model_server(destination) as target:

        def redirect(request):
            return (
                f"HTTP/1.1 {status} Redirect\r\n"
                f"Location: {target}/chat/completions\r\nContent-Length: 0\r\n\r\n"
            ).encode()

        with raw_model_server(redirect) as url:
            adapter = OpenAICompatAdapter(
                base_url=url, api_key=_KEY, provider="synthetic", setting="MORGAN_LLM_ENDPOINT"
            )
            try:
                with pytest.raises(ProviderRefused) as failure:
                    if stream:
                        async for _ in adapter.astream(_MESSAGES, model="synthetic"):
                            pytest.fail("A redirect must yield nothing")
                    else:
                        await adapter.agenerate(_MESSAGES, model="synthetic")
            finally:
                await adapter._client.close()

    assert destination_requests == []
    assert failure.value.status == status
    assert failure.value.endpoint == url
