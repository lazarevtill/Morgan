"""Real bounded child processes, crashes, recovery, contention and shutdown."""

import asyncio
import json
import os
import sys

import pytest

from morgan_brain import local_supervisor as supervisor
from morgan_brain.local_notes import ACTION, NoteWorkspace, approve
from morgan_brain.local_supervisor import SupervisorLimits, supervise
from morgan_brain.local_worker_state import scoped_counts
from morgan_brain.tasks import TaskKey, TaskRequest


@pytest.fixture
def workspace(tmp_path):
    """Create a private scratch workspace and close its descriptor after the test."""
    value = NoteWorkspace(tmp_path / "agent")
    yield value
    value.close()


def command(script):
    """Build isolated Python argv for test-only fault injection."""
    return [sys.executable, "-I", "-c", script]


async def phase(workspace, expected):
    """Wait for a real persisted transition with a bounded observation timeout."""
    async with asyncio.timeout(5):
        while True:
            try:
                state = json.loads(workspace.read(".supervisor.json", 16384))
            except FileNotFoundError:
                state = {}
            if state.get("phase") == expected:
                return state
            await asyncio.sleep(0.01)


@pytest.mark.parametrize("after_effect", [True, False])
async def test_signal_crash_recovery_never_repeats_effect(workspace, monkeypatch, after_effect):
    """Kill a real worker around publication and verify receipt recovery never replays it."""
    approve(
        workspace,
        [TaskRequest(TaskKey("owner", "personal", "one"), ACTION, "# Private note")],
        ttl_seconds=60,
    )
    script = """
import asyncio, os, signal, sys
from pathlib import Path
from morgan_brain.local_notes import LocalNoteAdapter, NoteWorkspace, run_local
workspace = NoteWorkspace(Path(sys.argv[1]))
original = LocalNoteAdapter.execute
async def crash(self, task, control):
    if sys.argv[2] == 'after':
        await original(self, task, control)
    os.kill(os.getpid(), signal.SIGKILL)
LocalNoteAdapter.execute = crash
result = asyncio.run(run_local(workspace))
sys.exit(0 if result.reason == 'cycle_limit' else 2)
"""
    monkeypatch.setattr(
        supervisor,
        "_worker_command",
        lambda ws, limits: [*command(script), str(ws.path), "after" if after_effect else "before"],
    )
    result = await supervise(workspace, limits=SupervisorLimits(seconds=5, restart_delay=0.05))
    assert result.starts == 2
    assert result.reason == ("complete" if after_effect else "worker_stopped")
    assert len(list(workspace.path.glob("note-*.md"))) == int(after_effect)
    assert scoped_counts(workspace, user_id="owner", project="personal") == {
        "succeeded" if after_effect else "outcome_unknown": 1,
    }
    assert "# Private note" not in workspace.read(".supervisor.json", 16384)


async def test_crash_loop_exhausts_budget(workspace, monkeypatch):
    """Repeated signal crashes consume exactly the configured process-start budget."""
    monkeypatch.setattr(
        supervisor,
        "_worker_command",
        lambda ws, limits: command("import os, signal; os.kill(os.getpid(), signal.SIGKILL)"),
    )
    result = await supervise(
        workspace, limits=SupervisorLimits(max_starts=3, seconds=5, restart_delay=0.05)
    )
    assert result.reason == "restart_limit" and result.starts == 3
    assert result.last_returncode == -9


@pytest.mark.parametrize("exit_code", [0, 1, 2])
async def test_normal_and_failure_exits_do_not_restart(workspace, monkeypatch, exit_code):
    """Ordinary completion and failure exit codes terminate supervision immediately."""
    monkeypatch.setattr(
        supervisor, "_worker_command", lambda ws, limits: command(f"raise SystemExit({exit_code})")
    )
    result = await supervise(workspace)
    assert result.starts == 1 and result.last_returncode == exit_code


async def test_stop_before_launch_starts_nothing(workspace, monkeypatch):
    """A pre-existing STOP marker prevents even the first process launch."""
    workspace.stop()

    def forbidden(*args):
        """Fail the test if STOP incorrectly reaches the command-construction seam."""
        pytest.fail("must not launch")

    monkeypatch.setattr(supervisor, "_worker_command", forbidden)
    assert (await supervise(workspace)).starts == 0


async def test_stop_during_backoff_prevents_restart(workspace, monkeypatch):
    """STOP interrupts the crash delay before another child is created."""
    monkeypatch.setattr(
        supervisor,
        "_worker_command",
        lambda ws, limits: command("import os, signal; os.kill(os.getpid(), signal.SIGKILL)"),
    )
    running = asyncio.create_task(supervise(workspace, limits=SupervisorLimits(restart_delay=2)))
    await phase(workspace, "backoff")
    workspace.stop()
    result = await running
    assert result.reason == "stopped" and result.starts == 1


async def test_deadline_escalates_and_reaps_unresponsive_child(workspace, monkeypatch):
    """Escalate ignored SIGTERM to SIGKILL and verify the owned child was reaped."""
    script = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
    monkeypatch.setattr(supervisor, "_worker_command", lambda ws, limits: command(script))
    running = asyncio.create_task(
        supervise(workspace, limits=SupervisorLimits(seconds=0.5, stop_grace=0.1))
    )
    pid = (await phase(workspace, "running"))["child_pid"]
    result = await running
    assert result.reason == "deadline" and result.forced_kills == 1 and result.starts == 1
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert (workspace.path / "STOP").exists()


async def test_competing_supervisor_and_resume_are_refused(workspace, monkeypatch):
    """Hold supervisor ownership through monitoring and exclude resume until shutdown."""
    monkeypatch.setattr(
        supervisor, "_worker_command", lambda ws, limits: command("import time;time.sleep(30)")
    )
    running = asyncio.create_task(supervise(workspace))
    await phase(workspace, "running")
    with pytest.raises(BlockingIOError):
        await supervise(workspace)
    with pytest.raises(BlockingIOError):
        workspace.resume()
    workspace.stop()
    assert (await running).reason == "stopped"
    workspace.resume()


async def test_cancellation_during_spawn_reaps_child(workspace, monkeypatch):
    """Cancel inside the spawn window and retain the child handle through cleanup."""
    monkeypatch.setattr(
        supervisor, "_worker_command", lambda ws, limits: command("import time;time.sleep(30)")
    )
    original = asyncio.create_subprocess_exec
    created = []
    ready = asyncio.Event()

    async def delayed(*args, **kwargs):
        """Expose the real child before returning to simulate cancellation during spawn."""
        child = await original(*args, **kwargs)
        created.append(child)
        ready.set()
        await asyncio.sleep(0.05)
        return child

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed)
    running = asyncio.create_task(supervise(workspace))
    await asyncio.wait_for(ready.wait(), 5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert created[0].returncode is not None
    assert (workspace.path / "STOP").exists()
    with workspace.locked(supervisor=True):
        pass


async def test_real_fixed_worker_end_to_end(workspace):
    """Run the installed worker twice and retain exactly one approved Markdown effect."""
    approve(
        workspace,
        [TaskRequest(TaskKey("owner", "personal", "one"), ACTION, "# One")],
        ttl_seconds=60,
    )
    result = await supervise(workspace)
    assert result.reason == "complete" and result.starts == 1
    assert (await supervise(workspace)).reason == "complete"
    assert [p.read_text() for p in workspace.path.glob("note-*.md")] == ["# One"]


async def test_cli_sigterm_stops_worker_then_exits(workspace):
    """Signal the actual supervisor CLI and verify sequential graceful child shutdown."""
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-m",
        "morgan_brain.local_notes_cli",
        str(workspace.path),
        "supervise",
        "--cycles",
        "60",
        "--poll",
        "0.1",
        "--seconds",
        "10",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        await phase(workspace, "running")
        child.terminate()
        output, error = await asyncio.wait_for(child.communicate(), 5)
        assert child.returncode == 2, error
        result = json.loads(output)
        assert result["reason"] == "stopped" and result["forced_kills"] == 0
        state = supervisor.supervisor_status(workspace)
        assert state["active"] is False and state["last"]["observed_phase"] == "stopped"
        with workspace.locked():
            pass
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()


def test_observation_validation_and_interruption(workspace):
    """Report an inactive live-phase record as interrupted and reject arbitrary phase text."""
    supervisor._Observation(workspace, SupervisorLimits(), supervisor.RunLimits())
    assert supervisor.supervisor_status(workspace)["last"]["observed_phase"] == "interrupted"
    raw = json.loads(workspace.read(".supervisor.json", 16384))
    raw["phase"] = "private payload"
    workspace.publish(".supervisor.json", json.dumps(raw), replace=True)
    with pytest.raises(ValueError):
        supervisor.supervisor_status(workspace)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_starts": 0},
        {"max_starts": 6},
        {"seconds": float("inf")},
        {"seconds": float("nan")},
        {"stop_grace": 0},
        {"restart_delay": 0},
    ],
)
def test_unbounded_supervision_is_refused(kwargs):
    """Reject configurations that remove finite start, duration, or backoff bounds."""
    with pytest.raises(ValueError):
        SupervisorLimits(**kwargs)


async def test_observation_failure_after_spawn_reaps_child(workspace, monkeypatch):
    """A failed running-state write must still terminate and reap the spawned child."""
    original_spawn = asyncio.create_subprocess_exec
    original_publish = NoteWorkspace.publish
    created = []

    async def capture(*args, **kwargs):
        """Retain the real process handle so cleanup can be asserted directly."""
        child = await original_spawn(*args, **kwargs)
        created.append(child)
        return child

    def fail(self, name, text, *, replace=False):
        """Inject a metadata failure only after the supervisor owns a live child."""
        if name == ".supervisor.json" and json.loads(text)["phase"] == "running":
            raise OSError("recording failed")
        return original_publish(self, name, text, replace=replace)

    monkeypatch.setattr(
        supervisor, "_worker_command", lambda ws, limits: command("import time;time.sleep(30)")
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    monkeypatch.setattr(NoteWorkspace, "publish", fail)
    with pytest.raises(OSError, match="recording failed"):
        await supervise(workspace)
    assert len(created) == 1 and created[0].returncode is not None


def test_copied_supervisor_observation_is_refused(workspace, tmp_path):
    """Do not attribute another workspace's saved supervisor state to this directory."""
    supervisor._Observation(workspace, SupervisorLimits(), supervisor.RunLimits())
    other = NoteWorkspace(tmp_path / "other")
    try:
        other.publish(".supervisor.json", workspace.read(".supervisor.json", 16384))
        with pytest.raises(ValueError):
            supervisor.supervisor_status(other)
    finally:
        other.close()
