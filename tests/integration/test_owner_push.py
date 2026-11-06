"""Synthetic provider responses exercise durable sends without contacting a push service."""

import asyncio
import json
import plistlib
from contextlib import closing
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from morgan_brain.launch_agent_plan import render
from morgan_brain.local_notes import NoteWorkspace, StopRequested
from morgan_brain.owner_push import OwnerPush, Push, send
from morgan_brain.tasks import TaskConflict, TaskKey


@pytest.mark.parametrize("mode", ["accepted", "redirect", "wrong_topic", "lost", "oversize"])
@pytest.mark.asyncio
async def test_provider_receipts(tmp_path: Path, mode: str, caplog) -> None:
    """Only an exact receipt proves server acceptance; every uncertain send is fenced."""
    calls = []
    push = Push(server="https://push.example", topic="private-demo", text="Синтетический план")

    async def provider(request: httpx.Request) -> httpx.Response:
        """Emulate provider acceptance or a response lost after receiving the request."""
        calls.append(request)
        assert request.headers["Authorization"] == "Bearer test-only-credential"
        assert request.content == push.text.encode()
        if mode == "lost":
            raise httpx.ReadError("synthetic lost response test-only-credential")
        if mode == "redirect":
            return httpx.Response(302, headers={"Location": "https://other.example"})
        if mode == "oversize":
            return httpx.Response(200, content=b"x" * 16385)
        return httpx.Response(
            200,
            json={
                "id": "receipt-1",
                "event": "message",
                "topic": "wrong" if mode == "wrong_topic" else push.topic,
                "message": push.text,
            },
        )

    with closing(NoteWorkspace(tmp_path / "outbox")) as workspace:
        adapter = OwnerPush(
            workspace, SecretStr("test-only-credential"), transport=httpx.MockTransport(provider)
        )
        key = TaskKey("demo", "personal", "one")
        if mode == "accepted":
            result = await send(workspace, key, push, adapter)
            assert json.loads(result.result) == {
                "server_accepted": True,
                "message_id": "receipt-1",
                "device_delivered": None,
                "read": None,
            }
        else:
            with pytest.raises(RuntimeError, match="outcome is unknown") as error:
                await send(workspace, key, push, adapter)
            assert "test-only-credential" not in str(error.value)
        replay = await send(workspace, key, push, adapter)
        assert replay.status == ("succeeded" if mode == "accepted" else "outcome_unknown")
        assert len(calls) == 1
        with pytest.raises(TaskConflict):
            await send(workspace, key, push.model_copy(update={"topic": "different"}), adapter)
        with closing(workspace.journal()) as journal:
            assert "test-only-credential" not in repr(journal.get(key))
        assert "test-only-credential" not in caplog.text


@pytest.mark.parametrize("during", [True, False])
@pytest.mark.asyncio
async def test_stop(tmp_path: Path, during: bool) -> None:
    """STOP prevents a first send or cancels waiting without pretending to retract it."""
    calls = []
    cancelled = asyncio.Event()
    with closing(NoteWorkspace(tmp_path / "outbox")) as workspace:

        async def provider(request: httpx.Request) -> httpx.Response:
            """Model a received request whose response has not arrived."""
            calls.append(request)
            workspace.stop()
            try:
                await asyncio.sleep(30)
            finally:
                cancelled.set()
            return httpx.Response(200)

        if not during:
            workspace.stop()
        key = TaskKey("demo", "personal", "stop")
        adapter = OwnerPush(
            workspace, SecretStr("test-only-credential"), transport=httpx.MockTransport(provider)
        )
        with pytest.raises(StopRequested):
            await send(
                workspace,
                key,
                Push(server="https://push.example", topic="demo", text="Synthetic"),
                adapter,
            )
        assert len(calls) == int(during)
        assert cancelled.is_set() == during
        with closing(workspace.journal()) as journal:
            task = journal.get(key)
            assert (task.status if task else None) == ("outcome_unknown" if during else None)


@pytest.mark.parametrize(
    "server",
    [
        "http://host",
        "https://user:password@host",
        "https://host/x",
        "https://host/?token=x",
        "https://host/#x",
    ],
)
def test_destination_refusal(server: str) -> None:
    """Credentials, cleartext and extra paths cannot be smuggled into the destination."""
    with pytest.raises(ValueError):
        Push(server=server, topic="demo", text="Synthetic").endpoint()


def test_launchd_plan(tmp_path: Path) -> None:
    """The rendered plan is disabled, fixed-command, bounded and contains no grant or secrets."""
    plan = plistlib.loads(render(Path("/opt/morgan/bin/python"), tmp_path))
    assert plan["Disabled"] is True
    assert plan["KeepAlive"] is False
    assert plan["RunAtLoad"] is True
    assert plan["ProgramArguments"] == [
        "/opt/morgan/bin/python",
        "-m",
        "morgan_brain.local_notes_cli",
        str(tmp_path),
        "run",
        "--actions",
        "1",
        "--cycles",
        "1",
        "--seconds",
        "10",
    ]
    assert plan["StartCalendarInterval"] == [{"Minute": n} for n in (0, 15, 30, 45)]
    assert "EnvironmentVariables" not in plan
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError):
        render(Path("relative-python"), tmp_path)


@pytest.mark.asyncio
async def test_real_http_loopback(tmp_path: Path) -> None:
    """Exercise real HTTP framing locally; this does not certify external TLS or a phone."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    received = []

    class Handler(BaseHTTPRequestHandler):
        """Receive only synthetic data and return the ntfy acknowledgement shape."""

        def do_POST(self):
            """Record the actual request bytes without emitting access logs."""
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append(body)
            data = json.dumps(
                {"id": "local-1", "event": "message", "topic": "demo", "message": body.decode()}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            """Keep synthetic credentials and destinations out of test logs."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    class LocalTransport(httpx.AsyncHTTPTransport):
        """The test-only seam redirects the fixed synthetic origin to the local server."""

        async def handle_async_request(self, request):
            """Use real socket I/O but never contact the configured public test hostname."""
            request.url = request.url.copy_with(
                scheme="http", host="127.0.0.1", port=server.server_port
            )
            return await super().handle_async_request(request)

    try:
        with closing(NoteWorkspace(tmp_path / "outbox")) as workspace:
            task = await send(
                workspace,
                TaskKey("demo", "personal", "http"),
                Push(server="https://push.example", topic="demo", text="Synthetic"),
                OwnerPush(workspace, SecretStr("test-only-credential"), transport=LocalTransport()),
            )
            assert json.loads(task.result)["server_accepted"] is True
            assert received == [b"Synthetic"]
    finally:
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        thread.join(timeout=2)


def test_generated_worker_restart(tmp_path: Path) -> None:
    """Run the exact generated argv twice, then with STOP, using temporary finite approval."""
    import subprocess
    import sys

    from morgan_brain.local_notes import ACTION, approve
    from morgan_brain.tasks import TaskRequest

    with closing(NoteWorkspace(tmp_path / "notes")) as workspace:
        key = TaskKey("demo", "personal", "restart")
        approve(workspace, [TaskRequest(key, ACTION, "Synthetic restart note")], ttl_seconds=60)
        plan = plistlib.loads(render(Path(sys.executable), workspace.path))
        first = subprocess.run(plan["ProgramArguments"], capture_output=True, timeout=15)
        assert first.returncode == 0, first.stderr.decode()
        second = subprocess.run(plan["ProgramArguments"], capture_output=True, timeout=15)
        assert second.returncode == 0, second.stderr.decode()
        with closing(workspace.journal()) as journal:
            task = journal.get(key)
            assert task.status == "succeeded"
            attempt = task.attempt_id
        workspace.stop()
        stopped = subprocess.run(plan["ProgramArguments"], capture_output=True, timeout=15)
        assert stopped.returncode == 2
        with closing(workspace.journal()) as journal:
            assert journal.get(key).attempt_id == attempt
