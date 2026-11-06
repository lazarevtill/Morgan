"""Finite background admission, recovery and ownership against isolated owner journals."""

import asyncio
import json
import time
from contextlib import closing

import pytest

from morgan_brain import owner_input
from morgan_brain.local_notes import NoteWorkspace
from morgan_brain.model_cycle import proposal_filename
from morgan_brain.owner_loop import OwnerLoop, read_state
from morgan_brain.owner_model import OwnerModel
from tests.integration.test_owner_model import model as model_fixture

model = model_fixture


async def until(predicate):
    async with asyncio.timeout(12):
        while not predicate():  # noqa: ASYNC110 -- observe another process
            await asyncio.sleep(0.05)


def observed(root):
    with closing(NoteWorkspace(root, create=False)) as workspace:
        return read_state(workspace)


@pytest.mark.asyncio
async def test_intake_draft_restart_and_separate_review(model):
    loop = OwnerLoop(model.path, model.config, seconds=30, poll=0.05)
    running = asyncio.create_task(loop.run())
    await until(lambda: observed(model.path)["state"] == "waiting_for_owner")
    for identity, note in (("a", "Return book."), ("b", "Полить цветы.")):
        await asyncio.to_thread(owner_input.request, model.path, ["submit", identity], text=note)
        await until(lambda note=note: (model.path / proposal_filename(note)).exists())
    loop.shutdown.set()
    assert await running == "shutdown"
    result = await asyncio.to_thread(owner_input.request, model.path, ["status"])
    assert result["plan"]["tasks"] == []
    assert not result["stopped"]
    assert all(m["state"] == "needs_review" for m in result["messages"])
    expires = result["expires_at"]
    restored = OwnerLoop(model.path, model.config, seconds=30, poll=0.05)
    again = asyncio.create_task(restored.run())
    await until(lambda: observed(model.path)["launches"] == 2)
    await asyncio.sleep(0.15)
    assert restored.task is None
    restored.shutdown.set()
    assert await again == "shutdown"
    assert observed(model.path)["expires_at"] == expires
    assert len(list(model.path.glob("proposal-*.json"))) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["stop", "shutdown", "expiry"])
async def test_inflight_cleanup_keeps_shared_locks_and_stop_semantics(model, mode):
    model.config.server.write_text(
        model.config.server.read_text().replace(
            " def do_POST(self):", " def do_POST(self):\n  import time; time.sleep(10)"
        )
    )
    await asyncio.to_thread(owner_input.request, model.path, ["submit", "a"], text="Return book.")
    loop = OwnerLoop(model.path, model.config, seconds=30, poll=0.05)
    running = asyncio.create_task(loop.run())
    await until(lambda: loop.task is not None)
    await asyncio.sleep(0.2)
    # Closing a separate read/review window must not cancel the background model.
    OwnerModel(model.path, model.config, time.monotonic() + 30).close()
    assert not (model.path / "STOP").exists()
    if mode == "stop":
        await asyncio.to_thread(owner_input.request, model.path, ["stop"])
    elif mode == "shutdown":
        loop.shutdown.set()
    else:
        with closing(NoteWorkspace(model.path, create=False)) as workspace:
            payload = json.loads(workspace.read("owner-input.json", 262144))
            payload["expires_at"] = "2000-01-01T00:00:00+00:00"
            workspace.publish("owner-input.json", json.dumps(payload), replace=True)
    assert (
        await asyncio.wait_for(running, 8)
        == {"stop": "stopped", "shutdown": "shutdown", "expiry": "expired"}[mode]
    )
    assert not (model.path / proposal_filename("Return book.")).exists()
    with closing(NoteWorkspace(model.path, create=False)) as workspace, workspace.locked():
        pass
    assert (model.path / "STOP").exists() == (mode == "stop")
    assert (model.path / ("attempt-" + proposal_filename("Return book."))).exists()


@pytest.mark.asyncio
async def test_failure_is_not_retried_and_parallel_loop_refused(model):
    model.config.server.write_text("#!/bin/sh\nexit 1\n")
    await asyncio.to_thread(owner_input.request, model.path, ["submit", "a"], text="Return book.")
    loop = OwnerLoop(model.path, model.config, seconds=30, poll=0.05)
    running = asyncio.create_task(loop.run())
    await until(lambda: loop.task is not None)
    with pytest.raises(BlockingIOError):
        await OwnerLoop(model.path, model.config, seconds=30).run()
    await until(lambda: loop.task is None)
    loop.shutdown.set()
    await running
    restored = OwnerLoop(model.path, model.config, seconds=30, poll=0.05)
    again = asyncio.create_task(restored.run())
    await until(lambda: observed(model.path)["launches"] == 2)
    await asyncio.sleep(0.2)
    assert restored.task is None
    restored.shutdown.set()
    await again


@pytest.mark.asyncio
async def test_launch_budget_and_corrupt_state_are_not_reset(model):
    with closing(NoteWorkspace(model.path, create=False)) as workspace:
        workspace.publish("owner-loop.json", '{"version":1,"launches":16}')
    assert await OwnerLoop(model.path, model.config, seconds=30).run() == "launch_limit"
    assert observed(model.path)["launches"] == 16
    with closing(NoteWorkspace(model.path, create=False)) as workspace:
        workspace.publish("owner-loop.json", '{"version":1,"launches":-1}', replace=True)
    with pytest.raises(ValueError, match="invalid loop state"):
        await OwnerLoop(model.path, model.config, seconds=30).run()


def test_render_is_disabled_exact_and_has_no_shell(tmp_path):
    import plistlib

    from morgan_brain.owner_loop_plan import render

    paths = [tmp_path / name for name in ("python", "workspace", "server", "weights")]
    payload = plistlib.loads(render(*paths))
    assert payload["Disabled"] is True
    assert payload["ProgramArguments"][:5] == [
        str(paths[0]),
        "-I",
        "-m",
        "morgan_brain.owner_loop",
        str(paths[1]),
    ]
    assert payload["ProgramArguments"][-2:] == ["--seconds", "604800"]
    assert payload["KeepAlive"] == {"SuccessfulExit": False}
    assert not any(p.exists() for p in paths)
    with pytest.raises(ValueError):
        render(paths[0], paths[1] / "..", *paths[2:])


def test_real_process_sigterm_then_restart_preserves_expiry(model):
    import signal
    import subprocess
    import sys

    command = [
        sys.executable,
        "-m",
        "morgan_brain.owner_loop",
        str(model.path),
        "--server",
        str(model.config.server),
        "--model",
        str(model.config.model),
        "--seconds",
        "30",
        "--poll",
        "0.05",
    ]
    expiry = None
    for launches in (1, 2):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                snapshot = observed(model.path)
                if snapshot["launches"] == launches and snapshot["state"] == "waiting_for_owner":
                    break
                assert process.poll() is None
                time.sleep(0.05)
            else:
                pytest.fail("loop did not become ready")
            if expiry is not None:
                assert snapshot["expires_at"] == expiry
            expiry = snapshot["expires_at"]
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=8)
            assert process.returncode == 0, stderr
            assert json.loads(stdout) == {"reason": "shutdown"}
            assert not (model.path / "STOP").exists()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
