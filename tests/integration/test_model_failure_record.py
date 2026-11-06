"""Persist sanitized model failures without retrying or accepting drafts."""

import asyncio
import json
from contextlib import closing

import pytest

from morgan_brain import model_cycle
from morgan_brain.local_notes import NoteWorkspace, StopRequested
from morgan_brain.owner_model import OwnerModel
from morgan_brain.providers.note_probe import ProbeFailure
from tests.integration.test_owner_model import model as model_fixture

model = model_fixture


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ProbeFailure("timeout"), "timeout"),
        (ProbeFailure("secret payload"), "cycle_failed"),
        (ValueError("private note"), "cycle_failed"),
        (StopRequested(), "stopped"),
    ],
)
def test_failure_vocabulary_never_copies_payload(error, code):
    assert model_cycle.failure_code(error) == code


def test_failed_child_is_durable_visible_and_not_a_draft(model):
    model.config.server.write_text("#!/bin/sh\nexit 1\n")
    note = "Private synthetic source."
    with closing(NoteWorkspace(model.path, create=False)) as workspace:
        with pytest.raises(ProbeFailure, match="model_start_failed"):
            asyncio.run(model_cycle.run_cycle(workspace, model.config, note, cache_by_source=True))
        filename = model_cycle.proposal_filename(note)
        failure = workspace.read("failure-" + filename, 1024)
        assert note not in failure
        assert json.loads(failure)["code"] == "model_start_failed"
        assert not (model.path / filename).exists()
    restored = OwnerModel(model.path, model.config, model.deadline)
    assert restored.status(note) == {"state": "failed", "reason": "model_start_failed"}
    assert restored.thread is None


@pytest.mark.parametrize("code", ["private provider text", [], None])
def test_corrupt_failure_record_is_not_displayed(model, code):
    note = "Synthetic."
    with closing(NoteWorkspace(model.path, create=False)) as workspace:
        workspace.publish(
            "failure-" + model_cycle.proposal_filename(note),
            json.dumps({"version": 1, "code": code}),
        )
    with pytest.raises(ValueError, match="invalid model failure record"):
        model.status(note)
