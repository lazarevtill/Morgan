"""Explicit finite supervision of the fixed local-note worker; never a system service."""

from __future__ import annotations

import asyncio
import json
import signal
import sys
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from morgan_brain.local_notes import NoteWorkspace, RunLimits, StopRequested


@dataclass(frozen=True)
class SupervisorLimits:
    """Bound process starts, elapsed time and cooperative/forced shutdown waits."""

    max_starts: int = 3
    seconds: float = 60
    stop_grace: float = 2
    restart_delay: float = 0.1

    def __post_init__(self) -> None:
        """Reject unlimited starts, nonfinite durations and busy restart loops."""
        if isinstance(self.max_starts, bool) or not isinstance(self.max_starts, int):
            raise TypeError("max_starts must be an integer")
        if not 1 <= self.max_starts <= 5 or not 0 < self.seconds <= 3600:
            raise ValueError("supervision requires 1..5 starts and at most 3600 seconds")
        if not 0.05 <= self.stop_grace <= 10 or not 0.05 <= self.restart_delay <= 60:
            raise ValueError("shutdown grace must be 0.05..10s and restart delay 0.05..60s")


@dataclass(frozen=True)
class Supervision:
    """Result of this invocation, without child output or task content."""

    reason: str
    starts: int
    last_returncode: int | None
    forced_kills: int


class _Observation:
    """Keep only one atomic private supervisor record, with no captured child output."""

    def __init__(
        self, workspace: NoteWorkspace, limits: SupervisorLimits, worker_limits: RunLimits
    ) -> None:
        """Persist start before launching a child."""
        self.workspace, self.limits = workspace, limits
        self.worker_limits = worker_limits
        self.run_id = uuid4().hex
        self.starts = self.forced_kills = 0
        self.last_returncode: int | None = None
        self.save("starting")

    def save(self, phase: str, *, pid: int | None = None) -> None:
        """Record transition metadata; a PID is informational and never used for later control."""
        self.workspace.publish(
            ".supervisor.json",
            json.dumps(
                {
                    "version": "morgan.local-supervisor.v1",
                    "workspace": self.workspace.provider_id,
                    "run_id": self.run_id,
                    "updated_at": datetime.now(UTC).isoformat(),
                    "phase": phase,
                    "starts": self.starts,
                    "last_returncode": self.last_returncode,
                    "forced_kills": self.forced_kills,
                    "child_pid": pid,
                    "limits": asdict(self.limits),
                    "worker_limits": asdict(self.worker_limits),
                }
            ),
            replace=True,
        )

    def result(self, reason: str) -> Supervision:
        """Persist the final reason before returning a bounded result."""
        self.save(reason)
        return Supervision(reason, self.starts, self.last_returncode, self.forced_kills)


def _worker_command(workspace: NoteWorkspace, limits: RunLimits) -> list[str]:
    """Use the installed worker with isolated Python imports; never accept shell/program input."""
    return [
        sys.executable,
        "-I",
        "-m",
        "morgan_brain.local_notes_cli",
        str(workspace.path),
        "run",
        "--actions",
        str(limits.max_actions),
        "--cycles",
        str(limits.max_cycles),
        "--seconds",
        str(limits.seconds),
        "--poll",
        str(limits.poll_seconds),
    ]


def _stop_reason(workspace: NoteWorkspace, deadline: float) -> str | None:
    """Check owner STOP before deadline; neither authorizes a restart."""
    try:
        workspace.check_stop()
    except StopRequested:
        return "stopped"
    if asyncio.get_running_loop().time() >= deadline:
        return "deadline"
    return None


async def _shutdown(child: asyncio.subprocess.Process, grace: float) -> bool:
    """Terminate only the owned child, escalate after grace, and reap before returning."""
    if child.returncode is not None:
        await child.wait()
        return False
    with suppress(ProcessLookupError):
        child.terminate()
    try:
        await asyncio.wait_for(child.wait(), grace)
    except TimeoutError:
        with suppress(ProcessLookupError):
            child.kill()
        await asyncio.wait_for(child.wait(), grace)
        return True
    return False


async def _launch(
    workspace: NoteWorkspace, worker_limits: RunLimits, grace: float
) -> asyncio.subprocess.Process:
    """Retain ownership if cancellation arrives while asyncio is creating the child."""
    pending = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *_worker_command(workspace, worker_limits),
            stdout=asyncio.subprocess.DEVNULL,
        )
    )
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        child = await pending
        with suppress(OSError):
            workspace.stop()
        await _shutdown(child, grace)
        raise


def _exit_reason(returncode: int | None) -> str | None:
    """Only selected crash signals permit restart; ordinary failure and termination stop."""
    if returncode == 0:
        return "complete"
    if returncode in (2, -signal.SIGTERM, -signal.SIGINT):
        return "worker_stopped"
    if returncode in (-signal.SIGKILL, -signal.SIGSEGV, -signal.SIGABRT):
        return None
    return "worker_failed"


async def _watch(
    child: asyncio.subprocess.Process, workspace: NoteWorkspace, deadline: float
) -> str | None:
    """Wait for child exit or cooperative control, without buffering child output."""
    while child.returncode is None:
        reason = _stop_reason(workspace, deadline)
        if reason is not None:
            return reason
        await asyncio.sleep(0.05)
    await child.wait()
    return None


async def _backoff(workspace: NoteWorkspace, deadline: float, delay: float) -> str | None:
    """Remain responsive to STOP and the total deadline between crash restarts."""
    until = min(deadline, asyncio.get_running_loop().time() + delay)
    while asyncio.get_running_loop().time() < until:
        reason = _stop_reason(workspace, deadline)
        if reason is not None:
            return reason
        await asyncio.sleep(min(0.05, max(0, until - asyncio.get_running_loop().time())))
    return _stop_reason(workspace, deadline)


async def supervise(
    workspace: NoteWorkspace,
    *,
    worker_limits: RunLimits = RunLimits(),
    limits: SupervisorLimits = SupervisorLimits(),
) -> Supervision:
    """Restart only signal-crashed workers within finite start/time budgets.

    Normal exits, policy/receipt failures and explicit stop never restart. Each child still
    requires exact current grants and reconciles uncertain tasks without repeating effects.
    """
    with workspace.locked(supervisor=True):
        observation = _Observation(workspace, limits, worker_limits)
        deadline = asyncio.get_running_loop().time() + limits.seconds
        child: asyncio.subprocess.Process | None = None
        try:
            while observation.starts < limits.max_starts:
                reason = _stop_reason(workspace, deadline)
                if reason is not None:
                    return observation.result(reason)
                observation.starts += 1
                observation.save("launching")
                child = await _launch(workspace, worker_limits, limits.stop_grace)
                observation.save("running", pid=child.pid)
                reason = await _watch(child, workspace, deadline)
                if reason is not None:
                    workspace.stop()
                    observation.save("stopping", pid=child.pid)
                    observation.forced_kills += await _shutdown(child, limits.stop_grace)
                    observation.last_returncode = child.returncode
                    return observation.result(reason)
                observation.last_returncode = child.returncode
                exit_reason = _exit_reason(child.returncode)
                if exit_reason is not None:
                    return observation.result(exit_reason)
                if observation.starts >= limits.max_starts:
                    return observation.result("restart_limit")
                observation.save("backoff")
                reason = await _backoff(workspace, deadline, limits.restart_delay)
                if reason is not None:
                    return observation.result(reason)
            return observation.result("restart_limit")
        except BaseException:
            # Stop/reap our child even if recording or cancellation failed. Never restart it.
            if child is not None and child.returncode is None:
                with suppress(OSError):
                    workspace.stop()
                observation.forced_kills += await _shutdown(child, limits.stop_grace)
                observation.last_returncode = child.returncode
            with suppress(OSError):
                observation.save("aborted")
            raise


def supervisor_status(workspace: NoteWorkspace) -> dict[str, object]:
    """Read validated metadata and lock state; never signal a PID saved in an old record."""
    active = False
    try:
        with workspace.locked(supervisor=True):
            pass
    except BlockingIOError:
        active = True
    try:
        raw = json.loads(workspace.read(".supervisor.json", 16384))
    except FileNotFoundError:
        return {"active": active, "last": None}
    expected = {
        "version",
        "workspace",
        "run_id",
        "updated_at",
        "phase",
        "starts",
        "last_returncode",
        "forced_kills",
        "child_pid",
        "limits",
        "worker_limits",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("invalid supervisor observation fields")
    if raw["workspace"] != workspace.provider_id:
        raise ValueError("supervisor observation belongs to another workspace")
    _validate_observation(raw)
    last = {key: value for key, value in raw.items() if key not in ("version", "workspace")}
    last["observed_phase"] = (
        "interrupted"
        if not active
        and raw["phase"] in ("starting", "launching", "running", "stopping", "backoff")
        else raw["phase"]
    )
    return {"active": active, "last": last}


def _validate_observation(raw: dict[str, object]) -> None:
    """Reject corrupt or unexpected fields instead of exposing arbitrary stored text."""
    if raw["version"] != "morgan.local-supervisor.v1":
        raise ValueError("unsupported supervisor observation")
    identity, timestamp = raw["run_id"], raw["updated_at"]
    if not isinstance(identity, str) or UUID(identity).hex != identity:
        raise ValueError("invalid supervisor run ID")
    if not isinstance(timestamp, str) or len(timestamp) > 64:
        raise ValueError("invalid supervisor timestamp")
    if datetime.fromisoformat(timestamp).utcoffset() is None:
        raise ValueError("supervisor timestamp requires a timezone")
    if raw["phase"] not in (
        "starting",
        "launching",
        "running",
        "stopping",
        "backoff",
        "complete",
        "worker_stopped",
        "worker_failed",
        "restart_limit",
        "stopped",
        "deadline",
        "aborted",
    ):
        raise ValueError("invalid supervisor phase")
    for name, lower, upper in (
        ("starts", 0, 5),
        ("forced_kills", 0, 5),
        ("last_returncode", -255, 255),
        ("child_pid", 1, 2**31),
    ):
        value = raw[name]
        if value is None and name in ("last_returncode", "child_pid"):
            continue
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
            raise ValueError("invalid supervisor numeric metadata")
    limits, worker = raw["limits"], raw["worker_limits"]
    if not isinstance(limits, dict) or not isinstance(worker, dict):
        raise TypeError("invalid supervisor limits")
    SupervisorLimits(**limits)
    RunLimits(**worker)
