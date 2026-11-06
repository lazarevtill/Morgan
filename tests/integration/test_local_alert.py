"""Exercise real journal fencing with a controlled subprocess instead of desktop UI."""

import asyncio
import json
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from morgan_brain import local_alert
from morgan_brain.local_notes import NoteWorkspace, StopRequested
from morgan_brain.tasks import Task, TaskConflict, TaskJournal, TaskKey, run_task


@pytest.mark.parametrize("response", ["acknowledged", "dismissed", "timed_out", "unexpected"])
@pytest.mark.asyncio
async def test_responses_and_no_replay(tmp_path: Path, monkeypatch, response: str) -> None:
    """Known results remain distinct; unknown responses and duplicate runs never redisplay."""
    monkeypatch.setattr(local_alert.sys, "platform", "darwin")
    process = AsyncMock()
    process.returncode = 0
    process.communicate.return_value = (response.encode(), None)
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(local_alert.asyncio, "create_subprocess_exec", spawn)
    key = TaskKey("owner", "personal", "one")
    with closing(NoteWorkspace(tmp_path / "alerts")) as workspace:
        journal = TaskJournal(tmp_path / "journal.db")
        payload = local_alert.Alert(text='" & do shell script "touch /tmp/no"').model_dump_json()
        journal.create(key, action=local_alert.ACTION, payload=payload)

        def authorize(task: Task) -> None:
            """Authorize only the exact account-selected request, never remembered text."""
            assert (task.key, task.action, task.payload) == (key, local_alert.ACTION, payload)

        async def run() -> Task:
            """Use the production journal and execution path."""
            return await run_task(
                journal,
                key,
                handlers={local_alert.ACTION: local_alert.LocalAlert(workspace)},
                timeout_seconds=2,
                authorize=authorize,
            )

        if response == "unexpected":
            with pytest.raises(RuntimeError):
                await run()
            assert journal.get(key).status == "outcome_unknown"
        else:
            result = await run()
            assert json.loads(result.result) == {
                "response": response,
                "acknowledged": response == "acknowledged",
            }
        with pytest.raises(TaskConflict):
            await run()
        assert spawn.await_count == 1
        assert spawn.call_args.args[2] == json.loads(payload)["text"]
        assert "shell" not in spawn.call_args.kwargs
        journal.close()


@pytest.mark.parametrize("during", [False, True])
@pytest.mark.asyncio
async def test_stop(tmp_path: Path, monkeypatch, during: bool) -> None:
    """STOP prevents launch or kills an in-flight child without claiming non-delivery."""
    monkeypatch.setattr(local_alert.sys, "platform", "darwin")
    process = AsyncMock()
    process.returncode = None
    process.kill = lambda: setattr(process, "returncode", -9)
    with closing(NoteWorkspace(tmp_path / "alerts")) as workspace:

        async def communicate(script: bytes) -> tuple[bytes, None]:
            """Simulate a visible dialog followed by STOP before a receipt."""
            workspace.stop()
            await asyncio.sleep(10)
            return b"acknowledged", None

        process.communicate.side_effect = communicate
        spawn = AsyncMock(return_value=process)
        monkeypatch.setattr(local_alert.asyncio, "create_subprocess_exec", spawn)
        journal = TaskJournal(tmp_path / "journal.db")
        key = TaskKey("owner", "personal", "stop")
        journal.create(
            key,
            action=local_alert.ACTION,
            payload=local_alert.Alert(text="Synthetic").model_dump_json(),
        )
        if not during:
            workspace.stop()
        with pytest.raises(StopRequested):
            await run_task(
                journal,
                key,
                handlers={local_alert.ACTION: local_alert.LocalAlert(workspace)},
                timeout_seconds=2,
                authorize=lambda task: workspace.check_stop(),
            )
        assert spawn.await_count == int(during)
        assert journal.get(key).status == ("outcome_unknown" if during else "pending")
        if during:
            assert process.returncode == -9
        journal.close()


@pytest.mark.asyncio
async def test_lost_commit_acknowledgement(tmp_path: Path, monkeypatch) -> None:
    """A visible effect followed by journal failure is fenced and never shown again."""
    from morgan_brain.local_alert_cli import send

    calls = []

    async def visible(self, task, control):
        """Simulate the effect without claiming a real desktop acknowledgement."""
        control.check_ready()
        calls.append(task.key)
        return '{"response":"acknowledged","acknowledged":true}'

    def fail(*args, **kwargs):
        """Lose persistence after the external effect."""
        raise OSError("synthetic persistence failure")

    monkeypatch.setattr(local_alert.LocalAlert, "__call__", visible)
    monkeypatch.setattr(TaskJournal, "finish", fail)
    with closing(NoteWorkspace(tmp_path / "alerts")) as workspace:
        key = TaskKey("owner", "personal", "crash")
        alert = local_alert.Alert(text="Synthetic")
        with pytest.raises(OSError):
            await send(workspace, key, alert)
        recovered = await send(workspace, key, alert)
        assert recovered.status == "outcome_unknown"
        assert recovered.result is None
        assert len(calls) == 1
        with pytest.raises(TaskConflict):
            await send(workspace, key, local_alert.Alert(text="Changed"))


@pytest.mark.asyncio
async def test_authorization_required(tmp_path: Path, monkeypatch) -> None:
    """The SDK's optional authorization default cannot silently enable desktop effects."""
    monkeypatch.setattr(local_alert.sys, "platform", "darwin")
    spawn = AsyncMock()
    monkeypatch.setattr(local_alert.asyncio, "create_subprocess_exec", spawn)
    with (
        closing(NoteWorkspace(tmp_path / "alerts")) as workspace,
        closing(workspace.journal()) as journal,
    ):
        key = TaskKey("owner", "personal", "no-grant")
        journal.create(
            key,
            action=local_alert.ACTION,
            payload=local_alert.Alert(text="Synthetic").model_dump_json(),
        )
        with pytest.raises(ValueError, match="explicit authorizer"):
            await run_task(
                journal,
                key,
                handlers={local_alert.ACTION: local_alert.LocalAlert(workspace)},
                timeout_seconds=2,
            )
        spawn.assert_not_called()


@pytest.mark.parametrize("stop", [True, False])
@pytest.mark.asyncio
async def test_real_child_cleanup(tmp_path: Path, monkeypatch, stop: bool) -> None:
    """STOP and an outer timeout reap a real sleeping child and fence the task."""
    import sys

    real_spawn = asyncio.create_subprocess_exec
    children = []
    started = asyncio.Event()
    monkeypatch.setattr(local_alert.sys, "platform", "darwin")

    async def spawn(*args, **kwargs):
        """Replace only the native GUI executable with a real isolated Python child."""
        child = await real_spawn(sys.executable, "-c", "import time; time.sleep(30)", **kwargs)
        children.append(child)
        started.set()
        return child

    monkeypatch.setattr(local_alert.asyncio, "create_subprocess_exec", spawn)
    with (
        closing(NoteWorkspace(tmp_path / "alerts")) as workspace,
        closing(workspace.journal()) as journal,
    ):
        key = TaskKey("owner", "personal", "child")
        journal.create(
            key,
            action=local_alert.ACTION,
            payload=local_alert.Alert(text="Synthetic").model_dump_json(),
        )

        async def request_stop() -> None:
            """Wait for the effect boundary before writing the control marker."""
            await started.wait()
            if stop:
                workspace.stop()

        controller = asyncio.create_task(request_stop())
        try:
            with pytest.raises(StopRequested if stop else TimeoutError):
                await run_task(
                    journal,
                    key,
                    handlers={local_alert.ACTION: local_alert.LocalAlert(workspace)},
                    timeout_seconds=0.5,
                    authorize=lambda task: workspace.check_stop(),
                )
            assert len(children) == 1
            assert children[0].returncode is not None
            assert journal.get(key).status == "outcome_unknown"
        finally:
            await controller
