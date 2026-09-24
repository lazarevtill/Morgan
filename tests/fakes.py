"""Deterministic test doubles. Nothing here ships."""

from __future__ import annotations

import hashlib
import json
import math
import socket
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from morgan_brain.memory import fingerprint
from morgan_brain.providers.wire import ChatMessage, ChatResult, StreamDelta, ToolSpec


def _unit_vector(text: str, dim: int) -> list[float]:
    """A deterministic unit vector for *text*. Every component is at least 1/256 before
    normalising, so no text maps to the zero vector."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [(digest[i % len(digest)] + 1) / 256.0 for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


class Calls:
    """The requests a model server has received, of any kind: ``total``, and ``times``, when
    each arrived (``time.monotonic``). ``inputs`` records how many texts each request's
    ``input`` carried, in arrival order, and is always the same length as ``times`` -- a
    non-embeddings request (a GET, say) records ``0`` -- so a test can tell an import's
    ordinary memory requests apart from an import canary's five-string one, or a retry from a
    fresh request, without guessing from the count alone. ``bodies`` holds every POST body as
    parsed JSON, in arrival order, so a test can assert what text reached the server."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total = 0
        self.times: list[float] = []
        self.inputs: list[int] = []
        self.bodies: list[dict[str, Any]] = []

    def count(self, inputs: int = 0, body: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.total += 1
            self.times.append(time.monotonic())
            self.inputs.append(inputs)
            if body is not None:
                self.bodies.append(body)


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
def vector_audit_server(
    *, wrong: dict[str, Any] | None = None, embedding_dim: int = 1024
) -> Iterator[str]:
    """A model server for ``doctor --vectors``: answers exactly like ``model_server`` -- the
    same deterministic unit vector per input text -- except for entries named in *wrong*.
    Yields its ``/v1`` URL.

    *wrong* maps a text to a spec (every client's request for that text gets the scripted
    answer), or a client label (``"client-1"``, ``"client-2"``, ... -- read from the
    ``X-Morgan-Audit-Client`` header ``doctor``'s multi-client audit tags each of its
    requests with) to a nested ``{text: spec}``, so only that one client gets it; every other
    client, and every other text, gets the normal vector. A spec is ``"wrong"`` (the exact
    negation of the normal vector -- as far from it as a vector of the same length can be,
    so a comparison never has to hope a hash landed far enough by chance), ``"zero"`` (the
    zero vector, which ``fingerprint.cosine`` refuses to compare), ``"nan"`` (a vector of
    ``NaN``s, which it also refuses) -- the two answers an audit must report as failing rather
    than crash on -- or ``"overflow"`` (every component ``1e200``: individually finite, so
    ``cosine`` does not refuse it, but its dot product and norm overflow to infinity against
    another huge vector, and ``inf / inf`` is ``nan`` -- the case ``cosine`` itself does not
    catch, and the caller comparing two clients' answers must).
    """
    wrong = wrong or {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path != "/v1/embeddings":
                self._reply(404, {"error": "not found"})
                return
            label = self.headers.get("X-Morgan-Audit-Client", "")
            texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
            vectors = [
                {"index": i, "embedding": _scripted_vector(text, label, wrong, embedding_dim)}
                for i, text in enumerate(texts)
            ]
            self._reply(200, {"object": "list", "data": vectors})

        def _reply(self, status: int, payload: dict[str, Any]) -> None:
            # allow_nan: a "nan" spec answers with a bare NaN token, which Python's json
            # module writes and reads as an extension of the format -- the live adapter
            # parses a real server's malformed answer the same way.
            data = json.dumps(payload, allow_nan=True).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:
            """The suite's output is not a request log."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    ).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


def _scripted_vector(text: str, label: str, wrong: dict[str, Any], dim: int) -> list[float]:
    spec = wrong.get(text)
    if spec is None:
        per_client = wrong.get(label)
        if isinstance(per_client, dict):
            spec = per_client.get(text)
    normal = _unit_vector(text, dim)
    if spec == "wrong":
        return [-x for x in normal]
    if spec == "zero":
        return [0.0] * dim
    if spec == "nan":
        return [math.nan] * dim
    if spec == "overflow":
        return [1e200] * dim
    return normal


@contextmanager
def drifting_model_server(*, embedding_dim: int = 1024, drift_after: int) -> Iterator[str]:
    """A model server that answers correctly until *drift_after* memory pieces have been
    embedded, then answers every input -- the import canary's own fingerprint strings
    included -- with the exact negation of the normal vector, the way a model having a bad
    moment does: nothing about the request tells it apart from a memory's, so it does not
    single the canary's own request out. Yields its ``/v1`` URL.

    Counts only texts that are not one of the five fingerprint strings
    (``memory.fingerprint.STRINGS``), matching what `morgan import` itself counts toward its
    canary interval: a canary's own small request never moves the drift point, and neither
    does a piece the importer skipped because it was already stored (it is never sent here).
    """
    seen = 0
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/v1/models":
                self._reply(200, {"object": "list", "data": []})
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self) -> None:
            nonlocal seen
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path != "/v1/embeddings":
                self._reply(404, {"error": "not found"})
                return
            texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
            with lock:
                seen += sum(1 for t in texts if t not in fingerprint.STRINGS)
                drifted = seen > drift_after
            vectors = [
                {"index": i, "embedding": _drifted_vector(text, embedding_dim, drifted)}
                for i, text in enumerate(texts)
            ]
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
    threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    ).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


def _drifted_vector(text: str, dim: int, drifted: bool) -> list[float]:
    normal = _unit_vector(text, dim)
    return [-x for x in normal] if drifted else normal


class Headers:
    """The headers of the last request a ``header_recording_server`` received, lower-cased."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.last: dict[str, str] = {}

    def record(self, headers: Any) -> None:
        with self._lock:
            self.last = {str(k).lower(): str(v) for k, v in headers.items()}


@contextmanager
def header_recording_server(*, embedding_dim: int = 1024) -> Iterator[tuple[str, Headers]]:
    """A model server that answers every request normally -- chat's ``/models`` and
    embeddings alike -- and records the headers of the last one it received; yields its
    ``/v1`` URL and a ``Headers`` whose ``last`` holds them, lower-cased. For asserting which
    key, if any, a request carried: a wrong key is never why the call fails here, so the test
    is free to check what was sent rather than only what came back.
    """
    headers = Headers()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            headers.record(self.headers)
            if self.path == "/v1/models":
                self._reply(200, {"object": "list", "data": []})
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            headers.record(self.headers)
            if self.path != "/v1/embeddings":
                self._reply(404, {"error": "not found"})
                return
            texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
            vectors = [
                {"index": i, "embedding": _unit_vector(text, embedding_dim)}
                for i, text in enumerate(texts)
            ]
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
    threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    ).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", headers
    finally:
        server.shutdown()
        server.server_close()


@contextmanager
def flaky_model_server(
    fail_times: int,
    status: int | None,
    *,
    after: int = 0,
    fail_for: float | None = None,
    echo: Callable[[str], str] | None = None,
    embedding_dim: int = 1024,
    calls: Calls | None = None,
) -> Iterator[str]:
    """``model_server``, failing *fail_times* embedding requests in a row; yields its URL.

    The first *after* embedding requests are answered, the next *fail_times* fail, and every
    one after that is answered again. With *fail_for*, every embedding request that arrives
    within that many seconds of the first one fails too: a host loading its model for a while.
    A failure answers with *status* -- a 503 is what llama-server says while it loads a model,
    a 401 a key it refused -- or, with ``status=None``, closes the connection without a word: a
    dropped connection, as a host that goes to sleep mid-request gives one. With *echo* the
    failure's body is, byte for byte, what *echo* makes of the bearer token the request carried:
    a careless gateway repeating a credential back, in whatever encoding it writes.
    *calls*, when given, counts every request.
    """
    failures = _Failures(after=after, times=fail_times, status=status, seconds=fail_for, echo=echo)
    with _model_server(
        embeddings=True, embedding_dim=embedding_dim, reorder=False, calls=calls, fail=failures
    ) as url:
        yield url


@contextmanager
def silent_model_server(*, trickle_every: float | None = None) -> Iterator[str]:
    """A server that accepts every connection and never answers; yields a ``/v1`` URL.

    A cold host loading its model looks like this from the client until the model is loaded:
    the connection is made, the request is sent, and nothing comes back. The connections are
    held open, unread, until the context exits. With *trickle_every*, each connection is sent
    the head of a 200 and then one byte of its body every that many seconds, for at most five
    seconds: an answer that is always arriving and never arrives.
    """
    listener = socket.create_server(("127.0.0.1", 0))
    listener.settimeout(0.05)
    held: list[socket.socket] = []
    stop = threading.Event()

    def trickle(conn: socket.socket, every: float) -> None:
        head = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1000\r\n\r\n"
        deadline = time.monotonic() + 5.0
        try:
            conn.sendall(head)
            while not stop.is_set() and time.monotonic() < deadline:
                conn.sendall(b" ")
                stop.wait(every)
        except OSError:
            return

    def accept() -> None:
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            held.append(conn)
            if trickle_every is not None:
                threading.Thread(target=trickle, args=(conn, trickle_every), daemon=True).start()

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}/v1"
    finally:
        stop.set()
        thread.join()
        for conn in held:
            conn.close()
        listener.close()


def _content_length(head: bytes) -> int:
    """What the request head declares it will send, or 0 when it declares nothing."""
    for line in head.split(b"\r\n"):
        name, _, value = line.partition(b":")
        if name.strip().lower() == b"content-length":
            return int(value)
    return 0


def _whole_request(conn: socket.socket) -> bytes | None:
    """Every byte of one HTTP request -- the head, then the body its ``Content-Length``
    declares -- or None when the client closed before the request was complete.

    Read whole before anything is answered: closing a socket with unread input resets the
    connection, and the client would see that instead of the answer.
    """
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = conn.recv(65536)
        if not chunk:
            return None
        data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    while len(body) < _content_length(head):
        chunk = conn.recv(65536)
        if not chunk:
            return None
        body += chunk
    return head + b"\r\n\r\n" + body


@contextmanager
def raw_model_server(reply: Callable[[bytes], bytes]) -> Iterator[str]:
    """A server that reads each request whole and answers with exactly the bytes *reply* makes
    of it, then closes; yields a ``/v1`` URL. For the answers no well-behaved server gives: a
    malformed head, which httpx reports by quoting the line it could not parse."""
    listener = socket.create_server(("127.0.0.1", 0))
    listener.settimeout(0.05)
    stop = threading.Event()

    def answer(conn: socket.socket) -> None:
        with conn:
            conn.settimeout(5.0)
            try:
                request = _whole_request(conn)
                if request is None:
                    return
                conn.sendall(reply(request))
                conn.shutdown(socket.SHUT_WR)
            except OSError:
                return

    def accept() -> None:
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            threading.Thread(target=answer, args=(conn,), daemon=True).start()

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}/v1"
    finally:
        stop.set()
        thread.join()
        listener.close()


class _Failures:
    """Which embedding requests ``flaky_model_server`` fails: numbers *after* + 1 to *after* +
    *times*, counted across every connection, and any that arrives within *seconds* of the
    first. ``fail()`` counts one request and says whether it is one of them."""

    def __init__(
        self,
        *,
        after: int,
        times: int,
        status: int | None,
        seconds: float | None = None,
        echo: Callable[[str], str] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._seen = 0
        self._first: float | None = None
        self._after = after
        self._times = times
        self._seconds = seconds
        self.status = status
        self.echo = echo

    def fail(self) -> bool:
        with self._lock:
            now = time.monotonic()
            if self._first is None:
                self._first = now
            self._seen += 1
            in_window = self._seconds is not None and now - self._first < self._seconds
            return in_window or self._after < self._seen <= self._after + self._times


@contextmanager
def slow_model_server(delay: float, *, embedding_dim: int = 1024) -> Iterator[str]:
    """``model_server``, answering every request *delay* seconds after it arrives; yields its
    ``/v1`` URL. A host loading its model, as the embedding host does on its first request
    after idle, answers like this: late, and correctly. A request still waiting when the
    context exits is dropped unanswered, so a client that gave up early costs the suite
    nothing more at teardown."""
    with _model_server(
        embeddings=True, embedding_dim=embedding_dim, reorder=False, calls=None, delay=delay
    ) as url:
        yield url


@contextmanager
def _model_server(
    *,
    embeddings: bool,
    embedding_dim: int,
    reorder: bool,
    calls: Calls | None,
    fail: _Failures | None = None,
    delay: float = 0.0,
) -> Iterator[str]:
    #: Set when the context exits: a handler still waiting out *delay* stops waiting and
    #: answers nothing, instead of holding ``server_close`` (which joins every handler).
    stopping = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if calls is not None:
                calls.count()
            if delay and stopping.wait(delay):
                return
            if self.path == "/v1/models":
                self._reply(200, {"object": "list", "data": []})
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if calls is not None:
                raw_input = body.get("input")
                if isinstance(raw_input, list):
                    calls.count(len(raw_input), body=body)
                else:
                    calls.count(1 if raw_input is not None else 0, body=body)
            if delay and stopping.wait(delay):
                return
            if self.path != "/v1/embeddings":
                self._reply(404, {"error": "not found"})
            elif fail is not None and fail.fail():
                if fail.status is None:
                    # Nothing is written: the handler returns and the server closes the socket.
                    self.close_connection = True
                elif fail.echo is not None:
                    token = self.headers.get("Authorization", "").removeprefix("Bearer ")
                    self._send(fail.status, fail.echo(token).encode())
                else:
                    self._reply(fail.status, {"error": f"scripted failure {fail.status}"})
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
            self._send(status, json.dumps(payload).encode())

        def _send(self, status: int, data: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:
            """The suite's output is not a request log."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    # shutdown() waits for the loop's next poll: the default half second, paid by every test
    # that starts a server, is most of the time such a test takes.
    threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    ).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        stopping.set()
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
