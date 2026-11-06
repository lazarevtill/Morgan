"""One optional finite local-model operation behind the owner UI; no execution authority."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from contextlib import closing
from dataclasses import replace
from pathlib import Path

from morgan_brain import owner_input
from morgan_brain.local_notes import NoteWorkspace, StopRequested
from morgan_brain.model_cycle import ModelCycle, _check_existing, proposal_filename, run_cycle
from morgan_brain.providers.note_probe import ProbeFailure


class OwnerModel:
    """Share one guardian lock across messages and retain only grounded, unapproved drafts."""

    def __init__(self, owner_workspace: Path, config: ModelCycle, deadline: float) -> None:
        """Use the private owner workspace and its shared STOP; start() is always explicit."""
        self.owner_workspace = owner_workspace
        self.path = owner_workspace
        self.config, self.deadline = config, deadline
        self.thread: threading.Thread | None = None
        self.failure: str | None = None
        self.current_note: str | None = None
        with closing(NoteWorkspace(self.path)):
            pass

    def status(self, note: str) -> dict[str, object]:
        """Recover exact drafts or report an active/interrupted operation without retrying it."""
        with closing(NoteWorkspace(self.path, create=False)) as workspace:
            try:
                draft = json.loads(workspace.read(proposal_filename(note), 16384))
            except FileNotFoundError:
                pass
            else:
                _check_existing(draft, hashlib.sha256(note.encode()).hexdigest(), note)
                return {"state": "ready", "lines": draft["selected_lines"]}
            if self.thread is not None and self.thread.is_alive():
                return {"state": "running"}
            try:
                with workspace.locked(supervisor=True), workspace.locked():
                    pass
            except BlockingIOError:
                return {"state": "running"}
            try:
                workspace.check_stop()
            except StopRequested:
                return {"state": "stopped"}
            if self.failure is not None and self.current_note == note:
                return {"state": "failed", "reason": self.failure}
            try:
                workspace.read("attempt-" + proposal_filename(note), 1024)
            except FileNotFoundError:
                return {"state": "not_started"}
            return {"state": "interrupted"}

    def start(self, note: str) -> None:
        """An explicit owner action starts one bounded draft operation; STOP is never cleared."""
        with closing(NoteWorkspace(self.owner_workspace, create=False)) as owner:
            owner.check_stop()
        if owner_input.request(self.path, ["status"])["expired"]:
            raise ValueError("seven-day owner pilot expired")
        remaining = min(self.config.seconds, int(self.deadline - time.monotonic()))
        if remaining < 5:
            raise ValueError("not enough foreground time for a model cycle")
        state = self.status(note)["state"]
        if state == "ready":
            return
        if state == "running":
            raise ValueError("a previous model operation is still running")
        config = replace(self.config, seconds=remaining, delay=0)
        with closing(NoteWorkspace(self.path, create=False)) as workspace:
            workspace.check_stop()
            workspace.publish(
                "attempt-" + proposal_filename(note), "Explicit model start.\n", replace=True
            )
        self.failure = None
        self.current_note = note
        self.thread = threading.Thread(target=self._run, args=(note, config), daemon=True)
        self.thread.start()

    def _run(self, note: str, config: ModelCycle) -> None:
        """Run the existing guardian lifecycle; never write a commitment or confirm a receipt."""
        try:
            with closing(NoteWorkspace(self.path, create=False)) as workspace:
                asyncio.run(run_cycle(workspace, config, note, cache_by_source=True))
        except (OSError, ValueError, TypeError, ProbeFailure, StopRequested):
            self.failure = "model operation failed or stopped; explicit retry required"

    def stop(self) -> None:
        """Persist cooperative STOP for this model workspace, including a surviving guardian."""
        with closing(NoteWorkspace(self.path, create=False)) as workspace:
            workspace.stop()

    def close(self) -> None:
        """Request cancellation on UI exit; the independent guardian retains its finite expiry."""
        if self.thread is not None and self.thread.is_alive():
            self.stop()
        if self.thread is not None:
            self.thread.join(timeout=6)
