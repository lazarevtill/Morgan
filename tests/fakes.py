"""Deterministic test doubles. Nothing here ships."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections import deque
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from morgan_brain.providers.wire import ChatMessage, ChatResult, StreamDelta, ToolSpec


def _unit_vector(text: str, dim: int) -> list[float]:
    """A deterministic unit vector for *text*. Every component is at least 1/256 before
    normalising, so no text maps to the zero vector."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [(digest[i % len(digest)] + 1) / 256.0 for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


class Calls:
    """The requests a model server has received, of any kind: ``total``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total = 0

    def count(self) -> None:
        with self._lock:
            self.total += 1


@contextmanager
def counting_model_server(
    *, embeddings: bool = True, embedding_dim: int = 1024, reorder: bool = False
) -> Iterator[tuple[str, Calls]]:
    """``model_server``, counting every request it receives; yields its ``/v1`` URL and the
    count. Every request, not only embedding ones: a cold embedding host pays for any request
    that loads its model, so "no request" is what an open that must not wait on it asserts."""
    calls = Calls()
    with _model_server(
        embeddings=embeddings, embedding_dim=embedding_dim, reorder=reorder, calls=calls
    ) as url:
        yield url, calls


@contextmanager
def model_server(
    *, embeddings: bool = True, embedding_dim: int = 1024, reorder: bool = False
) -> Iterator[str]:
    """An OpenAI-compatible model server on a free loopback port; yields its ``/v1`` URL.

    It lists no models and embeds every input as a unit vector ``embedding_dim`` wide derived
    from a hash of the text: the same text gets the same vector in every process, so a
    fingerprint recorded by one is matched by the next, and no vector is the zero vector that
    ``fingerprint.cosine`` rightly refuses. With ``embeddings=False`` it answers embedding
    requests with a 501, as a llama-server started without ``--embeddings`` does. With
    ``reorder=True`` it lists its answers last input first, each still carrying the
    ``index`` of its input: the order of the list is not part of the protocol. Real sockets,
    so a probe is tested the way it runs.
    """
    with _model_server(
        embeddings=embeddings, embedding_dim=embedding_dim, reorder=reorder, calls=None
    ) as url:
        yield url


@contextmanager
def _model_server(
    *, embeddings: bool, embedding_dim: int, reorder: bool, calls: Calls | None
) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if calls is not None:
                calls.count()
            if self.path == "/v1/models":
                self._reply(200, {"object": "list", "data": []})
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self) -> None:
            if calls is not None:
                calls.count()
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path != "/v1/embeddings":
                self._reply(404, {"error": "not found"})
            elif not embeddings:
                self._reply(501, {"error": "this server was started without --embeddings"})
            else:
                texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
                vectors = [
                    {"index": i, "embedding": _unit_vector(text, embedding_dim)}
                    for i, text in enumerate(texts)
                ]
                if reorder:
                    vectors.reverse()
                self._reply(200, {"object": "list", "data": vectors})

        def _reply(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:
            """The suite's output is not a request log."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


class FakeChatClient:
    """A scripted chat client. ``replies`` are consumed one per call; when exhausted the last
    one repeats. Records the last prompt so a test can assert what reached the model."""

    def __init__(self, reply: str = "", replies: list[str] | None = None) -> None:
        self._queue: deque[str] = deque(replies if replies is not None else [reply])
        self._last_reply = self._queue[-1] if self._queue else reply
        self.calls = 0
        self.last_messages: list[ChatMessage] = []
        self.last_model = ""
        self.last_response_format: dict[str, Any] | None = None

    def _next_reply(self) -> str:
        if self._queue:
            self._last_reply = self._queue.popleft()
        return self._last_reply

    async def agenerate(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        self.calls += 1
        self.last_messages = messages
        self.last_model = model
        self.last_response_format = response_format
        return ChatResult(text=self._next_reply(), model=model)

    async def _astream_impl(
        self, messages: list[ChatMessage], *, model: str, tools: list[ToolSpec] | None = None
    ) -> AsyncIterator[StreamDelta]:
        self.last_messages = messages
        yield StreamDelta(kind="text_delta", text=self._next_reply())
        yield StreamDelta(kind="finish", finish_reason="stop")

    def astream(
        self, messages: list[ChatMessage], *, model: str, tools: list[ToolSpec] | None = None
    ) -> AsyncIterator[StreamDelta]:
        return self._astream_impl(messages, model=model, tools=tools)
