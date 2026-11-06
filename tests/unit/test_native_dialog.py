"""Controlled child processes verify native adapter parsing and cooperative shutdown."""

import json
import subprocess
import sys
import time

import pytest

from morgan_brain import native_dialog
from morgan_brain.local_notes import NoteWorkspace, StopRequested


@pytest.fixture
def native(tmp_path, monkeypatch):
    workspace = NoteWorkspace(tmp_path / "workspace")
    workspace.close()
    monkeypatch.setattr(native_dialog.sys, "platform", "darwin")
    return native_dialog.NativeDialogs(tmp_path / "workspace", time.monotonic() + 10)


def child(monkeypatch, script):
    original = subprocess.Popen
    captured = []

    def launch(args, **kwargs):
        captured.append(args)
        process = original([sys.executable, "-c", script], **kwargs)
        captured.append(process)
        return process

    monkeypatch.setattr(native_dialog.subprocess, "Popen", launch)
    return captured


@pytest.mark.parametrize(
    "kind,value",
    [("text", "typed"), ("choose", "one"), ("confirm", "Подтвердить"), ("confirm", None)],
)
def test_fixed_script_uses_json_data_and_validates_response(native, monkeypatch, kind, value):
    captured = child(monkeypatch, f"import json; print(json.dumps({{'value':{value!r}}}))")
    hostile = 'quote " and run shell $(touch /tmp/not-executed)'
    assert native.show(kind, hostile, options=("one",)) == value
    argv = captured[0]
    assert argv[:4] == ["/usr/bin/osascript", "-l", "JavaScript", "-"]
    assert json.loads(argv[4])["prompt"] == hostile
    assert hostile not in native_dialog._SCRIPT


@pytest.mark.parametrize("payload", [{"value": "OK"}, {"value": True}, {"value": 42}])
def test_unknown_confirmation_is_not_authority(native, monkeypatch, payload):
    child(monkeypatch, f"print({json.dumps(payload)!r})")
    with pytest.raises(ValueError):
        native.show("confirm", "Review")


def test_deadline_reaps_child(native, monkeypatch):
    processes = child(monkeypatch, "import time; time.sleep(10)")
    native.deadline = time.monotonic() + 0.15
    with pytest.raises(native_dialog.DialogClosed):
        native.show("confirm", "Review")
    assert processes[1].poll() is not None


def test_external_stop_reaps_child(native, monkeypatch):
    marker = str(native.workspace / "STOP")
    script = f"from pathlib import Path; import time; Path({marker!r}).touch(); time.sleep(10)"
    processes = child(monkeypatch, script)
    with pytest.raises(StopRequested):
        native.show("confirm", "Review")
    assert processes[1].poll() is not None
