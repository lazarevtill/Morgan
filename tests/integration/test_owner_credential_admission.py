"""Refused paired captures must not strand the finite owner inbox in an attempted state."""

import json

import pytest

from morgan_brain.owner_input import Inbox, _digest
from tests.integration.test_owner_input import (
    confirm,
    good,
    initial,
    initialize,
    review_new,
    run,
)


def credential_shape():
    """Use a synthetic token shape, never a real provider credential."""
    return "gh" + "p_" + "A" * 36


@pytest.mark.parametrize("field", ["source", "title"])
def test_review_refuses_capture_before_journalling_review(tmp_path, field):
    initialize(tmp_path)
    source = credential_shape() if field == "source" else "Rotate the test token."
    good(run(tmp_path, "submit", "first", text=source))
    refused = run(
        tmp_path,
        "review",
        "first",
        "--new",
        "--title",
        credential_shape() if field == "title" else "Rotate the token",
        "--minutes",
        "5",
        "--priority",
        "normal",
        "--due",
        "2026-11-06",
    )
    assert refused.returncode == 2 and "Credential-shaped text refused" in refused.stderr
    assert credential_shape() not in refused.stderr and not refused.stdout
    status = good(run(tmp_path, "status"))
    assert status["messages"][0]["state"] == "needs_review"
    assert status["plan"]["tasks"] == []
    good(run(tmp_path, "dismiss", "first"))
    good(run(tmp_path, "submit", "clean", text="Return the book."))


@pytest.mark.parametrize("field", ["source", "title"])
def test_older_saved_review_refuses_before_applying_and_preserves_memory(tmp_path, field):
    initialize(tmp_path)
    good(run(tmp_path, "submit", "first", text="Return the book."))
    review_new(tmp_path)
    path = tmp_path / "session/owner-input.json"
    inbox = Inbox.model_validate_json(path.read_text())
    message = inbox.messages[0]
    # Simulate a review saved before admission preflight existed. This workflow journal
    # is not memory and can retain refused source text for explicit owner inspection.
    if field == "source":
        message.text = credential_shape()
    else:
        message.capture.commitment.title = credential_shape()
    message.digest = _digest(inbox, message)
    path.write_text(inbox.model_dump_json())
    database = tmp_path / "session/memory/morgan.db"
    before = database.read_bytes()
    refused = confirm(tmp_path, message.model_dump(mode="json"))
    assert refused.returncode == 2 and "Credential-shaped text refused" in refused.stderr
    assert credential_shape() not in refused.stderr and not refused.stdout
    assert database.read_bytes() == before
    assert json.loads(path.read_text())["messages"][0]["state"] == "reviewed"
    good(run(tmp_path, "submit", "clean", text="A different safe message."))
    if field == "title":
        corrected = review_new(tmp_path)
        good(confirm(tmp_path, corrected))
        assert len(good(run(tmp_path, "status"))["plan"]["tasks"]) == 1


def test_refused_renewal_leaves_current_task_and_guided_review_unchanged(tmp_path):
    initial(tmp_path)
    before = good(run(tmp_path, "status"))["plan"]["tasks"]
    good(run(tmp_path, "submit", "renewal", text=credential_shape()))
    refused = run(
        tmp_path,
        "review",
        "renewal",
        "--target",
        before[0]["evidence_id"],
        "--due",
        "2026-11-07",
    )
    assert refused.returncode == 2 and "Credential-shaped text refused" in refused.stderr
    after = good(run(tmp_path, "status"))
    assert after["plan"]["tasks"] == before and after["plan"]["state"] == "ready"
    assert after["messages"][-1]["state"] == "needs_review"
    good(run(tmp_path, "dismiss", "renewal"))


def test_historical_applying_capture_replays_without_rescanning_or_rewriting(tmp_path):
    initialize(tmp_path)
    good(run(tmp_path, "submit", "first", text="Return the book."))
    review_new(tmp_path)
    path = tmp_path / "session/owner-input.json"
    inbox = Inbox.model_validate_json(path.read_text())
    message = inbox.messages[0]
    message.text = credential_shape()
    message.digest = _digest(inbox, message)
    path.write_text(inbox.model_dump_json())
    # A previous runtime accepted both events then died before acknowledging them. Only
    # this synthetic subprocess disables screening to construct that historical state.
    script = """
import os
from morgan_brain import owner_input
from morgan_brain.memory import module
owner_input.refuse_credentials = lambda text: None
module.refuse_credentials = lambda text: None
original = owner_input._add
async def lost_acknowledgement(*args):
    await original(*args)
    os._exit(24)
owner_input._add = lost_acknowledgement
owner_input.main()
"""
    saved = message.model_dump(mode="json")
    assert confirm(tmp_path, saved, script=script).returncode == 24
    database = tmp_path / "session/memory/morgan.db"
    before = database.read_bytes()
    assert good(confirm(tmp_path, saved))["message"]["state"] == "applied"
    assert database.read_bytes() == before
    assert good(confirm(tmp_path, saved))["message"]["state"] == "applied"
    assert database.read_bytes() == before
