"""Explicit bounded macOS dialogue adapter; no service or inferred authorization."""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import suppress

from pydantic import BaseModel, ConfigDict, Field

from morgan_brain.local_notes import NoteWorkspace
from morgan_brain.tasks import Task, TaskControl

ACTION = "local_alert_v1"
_SCRIPT = """on run argv
set messageText to item 1 of argv
set waitSeconds to (item 2 of argv) as integer
try
set response to display dialog messageText with title "Morgan" ¬
buttons {"Dismiss", "Acknowledge"} default button "Dismiss" ¬
giving up after waitSeconds
if gave up of response then return "timed_out"
if button returned of response is "Acknowledge" then return "acknowledged"
return "dismissed"
on error number -128
return "dismissed"
end try
end run
"""


class Alert(BaseModel):
    """Exact display content and a finite interaction window, carried as task data."""

    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=2000)
    seconds: int = Field(default=15, ge=1, le=30)


class LocalAlert:
    """Trusted handler for run_task with an explicit caller-supplied authorizer.

    The account may acknowledge a dialogue; this is not proof of identity or reading.
    STOP and cancellation terminate the child but cannot retract an already seen effect.
    """

    def __init__(self, workspace: NoteWorkspace) -> None:
        """Reuse the existing private workspace and its STOP marker."""
        self.workspace = workspace

    def _ready(self, control: TaskControl) -> None:
        """Check both current permission and STOP before launch and while waiting."""
        if control.authorize is None:
            raise ValueError("local alerts require an explicit authorizer")
        self.workspace.check_stop()
        control.check_ready()

    async def __call__(self, task: Task, control: TaskControl) -> str:
        """Return a known OS response, or raise so the journal fences an unknown outcome."""
        if sys.platform != "darwin" or task.action != ACTION:
            raise ValueError("local alerts require macOS and the registered action")
        alert = Alert.model_validate_json(task.payload)
        if "\x00" in alert.text:
            raise ValueError("alert text cannot contain NUL")
        self._ready(control)
        process = await asyncio.create_subprocess_exec(
            "/usr/bin/osascript",
            "-",
            alert.text,
            str(alert.seconds),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        communication = asyncio.create_task(process.communicate(_SCRIPT.encode()))
        try:
            async with asyncio.timeout(alert.seconds + 5):
                while not communication.done():
                    self._ready(control)
                    await asyncio.wait({communication}, timeout=0.1)
                output, _ = communication.result()
                self._ready(control)
            response = output.decode("ascii").strip()
            if process.returncode != 0 or response not in {
                "acknowledged",
                "dismissed",
                "timed_out",
            }:
                raise RuntimeError("local alert outcome is unknown")
            return json.dumps({"response": response, "acknowledged": response == "acknowledged"})
        finally:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
            await process.wait()
            if not communication.done():
                communication.cancel()
            with suppress(asyncio.CancelledError):
                await communication
