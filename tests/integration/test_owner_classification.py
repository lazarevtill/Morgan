"""Explicit reversible non-task classification preserves sources and attempt fencing."""

import json
from contextlib import closing

import pytest

from morgan_brain import owner_input
from morgan_brain.local_notes import NoteWorkspace, StopRequested
from morgan_brain.model_cycle import proposal_filename
from morgan_brain.owner_loop import OwnerLoop
from tests.integration.test_owner_model import model as model_fixture
from tests.integration.test_owner_ui import drive, initialize, state

model = model_fixture


def test_source_retained_duplicate_delivery_cannot_reopen_and_restore_is_explicit(tmp_path):
    root = tmp_path / "pilot"
    initialize(root)
    note = "Я уже вернул книгу. Это не новая задача."
    owner_input.request(root, ["submit", "a"], text=note)
    before = state(root)
    for _ in range(2):
        result = owner_input.request(root, ["dismiss", "a"])
        assert not result["execution_authorized"]
    owner_input.request(root, ["submit", "a"], text=note)
    after = state(root)
    assert after["state"] == "ready"
    assert after["messages"][0]["state"] == "dismissed"
    assert after["messages"][0]["text"] == note
    assert {k: v for k, v in after["plan"].items() if k != "as_of"} == {
        k: v for k, v in before["plan"].items() if k != "as_of"
    }
    assert after["expires_at"] == before["expires_at"]
    owner_input.request(root, ["restore", "a"])
    assert state(root)["messages"][0]["state"] == "needs_review"


@pytest.mark.parametrize("operation", ["dismiss", "restore"])
@pytest.mark.parametrize("saved_state", ["reviewed", "applying", "applied"])
def test_classification_cannot_hide_reviewed_or_attempted_work(tmp_path, operation, saved_state):
    root = tmp_path / "pilot"
    initialize(root)
    owner_input.request(root, ["submit", "a"], text="Synthetic note.")
    with closing(NoteWorkspace(root)) as workspace:
        inbox = json.loads(workspace.read("owner-input.json", 262144))
        inbox["messages"][0]["state"] = saved_state
        workspace.publish("owner-input.json", json.dumps(inbox), replace=True)
    before = (root / "owner-input.json").read_bytes()
    with pytest.raises(ValueError, match="only unreviewed"):
        owner_input.request(root, [operation, "a"])
    assert (root / "owner-input.json").read_bytes() == before


def test_stop_expiry_quota_and_review_boundaries(tmp_path):
    root = tmp_path / "pilot"
    initialize(root)
    for index in range(14):
        owner_input.request(root, ["submit", str(index)], text="Not a task.")
        owner_input.request(root, ["dismiss", str(index)])
    with pytest.raises(ValueError, match="limit"):
        owner_input.request(root, ["submit", "overflow"], text="Not a task.")
    with pytest.raises(ValueError):
        owner_input.request(root, ["review", "0", "--new", "--due", "2026-11-06"])
    owner_input.request(root, ["stop"])
    with pytest.raises(StopRequested):
        owner_input.request(root, ["restore", "0"])
    owner_input.request(root, ["resume"])
    with closing(NoteWorkspace(root)) as workspace:
        inbox = json.loads(workspace.read("owner-input.json", 262144))
        inbox["expires_at"] = "2000-01-01T00:00:00+00:00"
        workspace.publish("owner-input.json", json.dumps(inbox), replace=True)
    with pytest.raises(ValueError, match="expired"):
        owner_input.request(root, ["restore", "0"])


@pytest.mark.parametrize("note", ["Already done, no new task.", "Уже сделано, новой задачи нет."])
def test_dialog_dismiss_and_restore_show_source_and_require_confirmation(tmp_path, note):
    root = tmp_path / "pilot"
    initialize(root)
    dialogs = drive(
        root,
        [
            ("choose", "Новое сообщение"),
            ("text", note),
            ("choose", "Не задача"),
            ("confirm", "Подтвердить"),
            ("info", "Продолжить"),
            ("choose", "Проверить сообщения"),
            ("info", "Продолжить"),
            ("choose", "Не задачи"),
            ("choose", 0),
            ("confirm", None),
            ("choose", "Выход"),
        ],
    )
    assert state(root)["messages"][0]["state"] == "dismissed"
    assert all(note in item["prompt"] for item in dialogs.seen if item["kind"] == "confirm")
    drive(
        root,
        [
            ("choose", "Не задачи"),
            ("choose", 0),
            ("confirm", "Подтвердить"),
            ("info", "Продолжить"),
            ("choose", "Выход"),
        ],
    )
    assert state(root)["messages"][0]["state"] == "needs_review"
    assert state(root)["plan"]["tasks"] == []


def test_background_skips_dismissed_and_restoration_preserves_attempt_fence(model):
    note = "Synthetic no-task input."
    owner_input.request(model.path, ["submit", "a"], text=note)
    owner_input.request(model.path, ["dismiss", "a"])
    loop = OwnerLoop(model.path, model.config, seconds=30)
    with closing(NoteWorkspace(model.path)) as workspace:
        loop.admit(workspace, 30)
        assert loop.task is None
        workspace.publish("attempt-" + proposal_filename(note), "previous attempt")
    owner_input.request(model.path, ["restore", "a"])
    with closing(NoteWorkspace(model.path)) as workspace:
        loop.admit(workspace, 30)
        assert loop.task is None
