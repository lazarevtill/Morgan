"""Integrated owner/model lifecycle with real bounded fixture subprocesses and stored receipts."""

import json
import sys
import time
from contextlib import closing

import pytest

from morgan_brain import owner_input
from morgan_brain.local_notes import NoteWorkspace, StopRequested
from morgan_brain.model_cycle import ModelCycle, proposal_filename
from morgan_brain.owner_model import OwnerModel
from morgan_brain.owner_ui import OwnerUI
from tests.integration.test_model_cycle import SERVER
from tests.integration.test_owner_ui import DialogFixture, initialize, state


@pytest.fixture
def model(tmp_path):
    root = tmp_path / "owner"
    initialize(root)
    server = tmp_path / "server"
    server.write_text(f"#!{sys.executable}\n" + SERVER)
    server.chmod(0o700)
    weights = tmp_path / "weights"
    weights.write_bytes(b"fixture")
    controller = OwnerModel(root, ModelCycle(server, weights, seconds=5), time.monotonic() + 60)
    yield controller
    controller.close()


def finish(model):
    assert model.thread is not None
    model.thread.join(timeout=10)
    assert not model.thread.is_alive()


def test_one_shared_model_lane_multiple_messages_and_restart(model):
    for note in ("Return book.", "Water plants."):
        assert model.status(note)["state"] == "not_started"
        model.start(note)
        with pytest.raises(ValueError, match="still running"):
            model.start("Another request.")
        finish(model)
        assert model.status(note) == {"state": "ready", "lines": [note]}
    restored = OwnerModel(model.path, model.config, model.deadline)
    assert restored.thread is None
    assert restored.status("Return book.")["state"] == "ready"
    assert len(list(model.path.glob("proposal-*.json"))) == 2
    assert state(model.path)["plan"]["tasks"] == []
    restored.close()
    assert not state(model.path)["stopped"]


def test_shared_owner_stop_blocks_model_and_requires_idle_resume(model):
    owner_input.request(model.path, ["stop"])
    with pytest.raises(StopRequested):
        model.start("Do not start.")
    assert model.thread is None
    owner_input.request(model.path, ["resume"])
    with closing(NoteWorkspace(model.path, create=False)) as workspace, workspace.locked():
        assert model.status("Pending.")["state"] == "running"
        with pytest.raises(ValueError):
            model.start("Pending.")
        with pytest.raises(BlockingIOError):
            owner_input.request(model.path, ["resume"])


def test_interrupted_attempt_and_invalid_cache_never_auto_run(model):
    note = "Return book."
    with closing(NoteWorkspace(model.path, create=False)) as workspace:
        workspace.publish("attempt-" + proposal_filename(note), "interrupted")
        assert model.status(note)["state"] == "interrupted"
        assert model.thread is None
        workspace.publish(proposal_filename(note), json.dumps({"approved": True}))
    with pytest.raises(ValueError):
        model.status(note)
    assert model.thread is None


def test_foreground_expiry_prevents_start(model):
    model.deadline = time.monotonic() + 1
    with pytest.raises(ValueError, match="not enough"):
        model.start("Return book.")
    assert model.thread is None


def test_owner_stop_interrupts_running_inference(model):
    server = model.config.server
    server.write_text(
        server.read_text().replace(
            " def do_POST(self):", " def do_POST(self):\n  import time; time.sleep(10)"
        )
    )
    model.start("Return book.")
    time.sleep(0.2)
    owner_input.request(model.path, ["stop"])
    finish(model)
    assert model.status("Return book.")["state"] == "stopped"
    assert not (model.path / proposal_filename("Return book.")).exists()


def test_ui_model_result_still_requires_separate_task_confirmation(model):
    note = "Нужно вернуть книгу."

    def ready():
        finish(model)
        return "Обновить состояние"

    dialogs = DialogFixture(
        [
            ("choose", "Новое сообщение"),
            ("text", note),
            ("choose", "Предложение локальной модели"),
            ("confirm", "Подтвердить"),
            ("choose", ready),
            ("choose", "Заполнить задачу вручную"),
            ("text", "Вернуть книгу"),
            ("text", "10"),
            ("choose", "Обычный"),
            ("text", "2026-11-06"),
            ("confirm", None),
            ("choose", "Выход"),
        ]
    )
    OwnerUI(model.path, dialogs, model.deadline, model).run()
    assert not dialogs.answers
    assert state(model.path)["plan"]["tasks"] == []
    assert state(model.path)["messages"][0]["state"] == "reviewed"
    assert any("не подтверждено" in item["prompt"] for item in dialogs.seen)
    assert model.status(note)["state"] == "ready"


def test_ui_cancel_model_requests_shared_stop(model):
    dialogs = DialogFixture(
        [
            ("confirm", "Подтвердить"),
            ("choose", "STOP: остановить модель и ввод"),
        ]
    )
    assert not OwnerUI(model.path, dialogs, model.deadline, model).model_review("Return book.")
    finish(model)
    assert state(model.path)["stopped"]
    assert state(model.path)["plan"]["tasks"] == []


def test_close_stops_owned_cycle_and_preserves_shared_stop(model):
    model.start("Return book.")
    model.close()
    finish(model)
    assert state(model.path)["stopped"]


def test_failed_process_is_visible_and_restart_never_retries(model):
    model.config.server.write_text(f"#!{sys.executable}\nraise SystemExit(1)\n")
    model.start("Return book.")
    finish(model)
    assert model.status("Return book.")["state"] == "failed"
    assert model.status("Other message.")["state"] == "not_started"
    restored = OwnerModel(model.path, model.config, model.deadline)
    assert restored.status("Return book.")["state"] == "interrupted"
    assert restored.thread is None
    restored.close()


def test_expired_inbox_cannot_start_inference(model):
    with closing(NoteWorkspace(model.path, create=False)) as workspace:
        data = json.loads(workspace.read("owner-input.json", 262144))
        data["expires_at"] = "2000-01-01T00:00:00+00:00"
        workspace.publish("owner-input.json", json.dumps(data), replace=True)
    with pytest.raises(ValueError, match="expired"):
        model.start("Return book.")
    assert model.thread is None
