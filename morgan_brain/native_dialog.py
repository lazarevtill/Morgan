"""Finite foreground macOS dialogs; text is data, never AppleScript or shell source."""

from __future__ import annotations

import json
import subprocess  # nosec B404 -- fixed native dialog executable, never a supplied shell
import sys
import time
from pathlib import Path
from typing import Literal, Protocol

from morgan_brain.local_notes import NoteWorkspace

Kind = Literal["text", "choose", "confirm", "info"]

_SCRIPT = """function run(argv) {
    var p = JSON.parse(argv[0]);
    var app = Application.currentApplication();
    app.includeStandardAdditions = true;
    try {
        if (p.kind === 'choose') {
            var chosen = app.chooseFromList(p.options, {withTitle:'Morgan',
                withPrompt:p.prompt, okButtonName:'Выбрать', cancelButtonName:'Отмена',
                multipleSelectionsAllowed:false, emptySelectionAllowed:false});
            return JSON.stringify({value:chosen ? chosen[0] : null});
        }
        var options = {withTitle:'Morgan', givingUpAfter:p.seconds,
            buttons:['Отмена', p.kind === 'confirm' ? 'Подтвердить' : 'Продолжить'],
            defaultButton:'Отмена', cancelButton:'Отмена'};
        if (p.kind === 'text') options.defaultAnswer = p.initial;
        var answer = app.displayDialog(p.prompt, options);
        if (answer.gaveUp) return JSON.stringify({value:null});
        var value = p.kind === 'text' ? answer.textReturned : answer.buttonReturned;
        return JSON.stringify({value:value});
    } catch (error) {
        if (error.errorNumber === -128) return JSON.stringify({value:null});
        throw error;
    }
}
"""


class DialogClosed(Exception):
    """The foreground session expired or was stopped, without an implied confirmation."""


class Dialogs(Protocol):
    """A narrow presentation seam; implementations cannot write task state."""

    def show(
        self,
        kind: Kind,
        prompt: str,
        *,
        initial: str = "",
        options: tuple[str, ...] = (),
        allow_stopped: bool = False,
    ) -> str | None:
        """Return an explicit response or cancellation; never synthesize approval."""
        ...


class NativeDialogs:
    """Own one osascript child at a time and enforce STOP/deadline while awaiting UI."""

    def __init__(self, workspace: Path, deadline: float) -> None:
        """Use the existing private workspace and a finite monotonic session deadline."""
        if sys.platform != "darwin":
            raise ValueError("native owner UI requires macOS")
        self.workspace = workspace
        self.deadline = deadline

    def check(self, allow_stopped: bool) -> None:
        """Refuse late responses and cooperate with the persistent workspace STOP marker."""
        if time.monotonic() >= self.deadline:
            raise DialogClosed("UI session deadline reached")
        if not allow_stopped:
            workspace = NoteWorkspace(self.workspace, create=False)
            try:
                workspace.check_stop()
            finally:
                workspace.close()

    def show(
        self,
        kind: Kind,
        prompt: str,
        *,
        initial: str = "",
        options: tuple[str, ...] = (),
        allow_stopped: bool = False,
    ) -> str | None:
        """Pass JSON as an argv value to a fixed JXA program; never interpolate owner text."""
        self.check(allow_stopped)
        expires = min(self.deadline, time.monotonic() + 120)
        payload = json.dumps(
            {
                "kind": kind,
                "prompt": prompt,
                "initial": initial,
                "options": options,
                "seconds": max(1, int(expires - time.monotonic())),
            }
        )
        process = subprocess.Popen(  # noqa: S603 # nosec B603 -- fixed executable/script, JSON data argv
            ["/usr/bin/osascript", "-l", "JavaScript", "-", payload],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        script: bytes | None = _SCRIPT.encode()
        try:
            while True:
                self.check(allow_stopped)
                if time.monotonic() >= expires:
                    return None
                try:
                    output, _ = process.communicate(script, timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    script = None
            self.check(allow_stopped)
            if time.monotonic() >= expires:
                return None
            if process.returncode != 0:
                raise ValueError("native dialog failed; no response accepted")
            value = json.loads(output)["value"]
            if value is not None and not isinstance(value, str):
                raise ValueError("invalid native dialog response")
            if kind == "choose" and value is not None and value not in options:
                raise ValueError("dialog returned an unknown selection")
            if kind == "confirm" and value not in (None, "Подтвердить"):
                raise ValueError("dialog did not explicitly confirm")
            return value
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
