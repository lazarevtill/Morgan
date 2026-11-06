"""Real process lifecycle checks using a tiny trusted Unix-socket model fixture."""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from morgan_brain.local_notes import NoteWorkspace, StopRequested
from morgan_brain.model_cycle import ModelCycle, managed_model, run_cycle
from morgan_brain.providers.note_probe import ProbeFailure

SERVER = """
import http.server,json,os,socketserver,sys
from pathlib import Path
args=sys.argv; sock=args[args.index('--host')+1]
Path(sock+'.pid').write_text(str(os.getpid()))
class Handler(http.server.BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def do_GET(self):
  body=json.dumps({'data':[{'id':'morgan-cycle'}]}).encode()
  self.send_response(200);self.end_headers();self.wfile.write(body)
 def do_POST(self):
  self.rfile.read(int(self.headers['Content-Length']))
  body=json.dumps({'model':'morgan-cycle','choices':[{'finish_reason':'stop','message':{'content':'{"status":"ok","lines":[1]}'}}]}).encode()
  self.send_response(200);self.end_headers();self.wfile.write(body)
class Server(socketserver.UnixStreamServer): pass
Server(sock,Handler).serve_forever()
"""


@pytest.fixture
def fixture(tmp_path):
    """Create a trusted test executable and a fake weight file in private scratch space."""
    server = tmp_path / "server"
    server.write_text(f"#!{sys.executable}\n" + SERVER)
    server.chmod(0o700)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fixture")
    workspace = NoteWorkspace(tmp_path / "workspace")
    yield workspace, ModelCycle(server, model, seconds=5)
    workspace.close()


async def test_publish_replay_and_cleanup(fixture):
    """Publish once, reap the native child, and recover the same artifact without inference."""
    workspace, config = fixture
    first = await run_cycle(workspace, config, "Water plants.")
    assert first["reason"] == "written"
    assert json.loads(workspace.read("proposal.json", 16384))["approved"] is False
    config.server.unlink()
    assert (await run_cycle(workspace, config, "Water plants."))["reason"] == "already_written"
    with pytest.raises(ValueError):
        await run_cycle(workspace, config, "Different note.")


async def test_stop_before_launch(fixture):
    """Persistent STOP prevents both scheduling and model launch."""
    workspace, config = fixture
    workspace.stop()
    with pytest.raises(StopRequested):
        async with managed_model(workspace, config):
            pytest.fail("must not launch")


async def test_stop_during_session(fixture):
    """A running model observes STOP and the guardian releases its lock after cleanup."""
    workspace, config = fixture
    with pytest.raises(StopRequested):
        async with managed_model(workspace, config) as session:
            workspace.stop()
            await session.select("Water plants.")
    with workspace.locked():
        pass


async def test_contention(fixture):
    """A second cycle cannot start another model in the same workspace."""
    workspace, config = fixture
    async with managed_model(workspace, config):
        with pytest.raises(BlockingIOError):
            async with managed_model(workspace, config):
                pytest.fail("second cycle entered")


async def test_cancel_reaps_model(fixture):
    """Caller cancellation cleans up its guardian and native server."""
    workspace, config = fixture
    entered = asyncio.Event()

    async def work():
        """Hold the session until the outer task cancels it."""
        async with managed_model(workspace, config):
            entered.set()
            await asyncio.sleep(10)

    task = asyncio.create_task(work())
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with workspace.locked():
        pass


async def test_failed_start_releases_lock(fixture):
    """Model crashes are visible and never trigger an unbounded restart loop."""
    workspace, config = fixture
    config.server.write_text(f"#!{sys.executable}\nraise SystemExit(1)\n")
    with pytest.raises(ProbeFailure, match="model_start_failed"):
        async with managed_model(workspace, config):
            pytest.fail("failed model entered")
    with workspace.locked():
        pass

    config.server.write_text(f"#!{sys.executable}\n" + SERVER)
    assert (await run_cycle(workspace, config, "Water plants."))["reason"] == "written"


def test_guardian_expires_without_caller(fixture, tmp_path):
    """An independent guardian enforces its own deadline even with no live caller."""
    workspace, config = fixture
    temporary = tempfile.TemporaryDirectory(prefix="mc-test-")
    socket = Path(temporary.name) / "short.sock"
    command = [
        sys.executable,
        "-I",
        "-m",
        "morgan_brain.model_guardian",
        str(workspace.path),
        str(config.server),
        str(config.model),
        str(socket),
        "5",
        "0",
        "1",
    ]
    child = subprocess.Popen(command)
    try:
        started = time.monotonic()
        child.wait(timeout=9)
        assert time.monotonic() - started >= 4.5
        assert child.returncode == 0
        pid = int(Path(str(socket) + ".pid").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert not socket.exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=2)
        temporary.cleanup()


def test_invalid_limits(fixture):
    """Reject unbounded resources or schedules before any child starts."""
    _, config = fixture
    for kwargs in ({"threads": 5}, {"seconds": 121}, {"delay": 90}, {"threads": True}):
        with pytest.raises(ValueError):
            ModelCycle(config.server, config.model, **kwargs)


async def test_caller_hard_kill_still_expires(fixture):
    """A killed CLI leaves a guardian lock until the independently bounded model is reaped."""
    workspace, config = fixture
    script = """
import asyncio,sys
from pathlib import Path
from morgan_brain.model_cycle import ModelCycle,managed_model
from morgan_brain.local_notes import NoteWorkspace
async def main():
 ws=NoteWorkspace(Path(sys.argv[1]))
 async with managed_model(ws,ModelCycle(Path(sys.argv[2]),Path(sys.argv[3]),seconds=5)) as s:
  print(s.socket,flush=True)
  await asyncio.sleep(30)
asyncio.run(main())
"""
    caller = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-c",
        script,
        str(workspace.path),
        str(config.server),
        str(config.model),
        stdout=asyncio.subprocess.PIPE,
    )
    try:
        socket = Path((await asyncio.wait_for(caller.stdout.readline(), 5)).decode().strip())
        pid = int(await asyncio.to_thread(Path(str(socket) + ".pid").read_text))
        caller.kill()
        await caller.wait()
        with pytest.raises(BlockingIOError), workspace.locked():
            pass
        async with asyncio.timeout(8):
            while True:
                try:
                    with workspace.locked():
                        break
                except BlockingIOError:
                    await asyncio.sleep(0.05)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert not await asyncio.to_thread(socket.exists)
        await asyncio.to_thread(Path(str(socket) + ".pid").unlink)
        socket.parent.rmdir()
    finally:
        if caller.returncode is None:
            caller.kill()
            await caller.wait()


async def test_delayed_launch_stop(fixture):
    """STOP during a scheduled delay prevents a native server from ever starting."""
    workspace, config = fixture
    delayed = ModelCycle(config.server, config.model, seconds=5, delay=3)
    task = asyncio.create_task(run_cycle(workspace, delayed, "Water plants."))
    await asyncio.sleep(0.3)
    workspace.stop()
    with pytest.raises(StopRequested):
        await task
    assert not (workspace.path / "proposal.json").exists()
    with workspace.locked():
        pass


async def test_native_ignores_term(fixture):
    """Guardian escalates to kill and reaps a native child that ignores termination."""
    workspace, config = fixture
    config.server.write_text(
        f"#!{sys.executable}\nimport signal\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        + SERVER
    )
    async with managed_model(workspace, config) as session:
        pid = int(await asyncio.to_thread(Path(str(session.socket) + ".pid").read_text))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    with workspace.locked():
        pass


async def test_invalid_input_never_launches(fixture):
    """Input limits apply to the SDK as well as the CLI."""
    workspace, config = fixture
    with pytest.raises(ValueError):
        await run_cycle(workspace, config, "x" * 4097)
    assert not (workspace.path / ".supervisor.lock").exists()


@pytest.mark.parametrize("payload", ["[]", "null"])
async def test_corrupt_recovery_refused(fixture, payload):
    """A matching hash without the proposal schema cannot masquerade as completed work."""
    workspace, config = fixture
    workspace.publish("proposal.json", payload)
    with pytest.raises(TypeError, match="malformed"):
        await run_cycle(workspace, config, "Water plants.")


async def test_stop_interrupts_inflight_inference(fixture):
    """STOP terminates the model while a request is waiting and reports cancellation as STOP."""
    workspace, config = fixture
    slow = SERVER.replace(
        " def do_POST(self):", " def do_POST(self):\n  import time;time.sleep(10)"
    )
    config.server.write_text(f"#!{sys.executable}\n" + slow)
    async with managed_model(workspace, config) as session:
        request = asyncio.create_task(session.select("Water plants."))
        await asyncio.sleep(0.2)
        workspace.stop()
        with pytest.raises(StopRequested):
            await asyncio.wait_for(request, 3)
    assert not (workspace.path / "proposal.json").exists()
