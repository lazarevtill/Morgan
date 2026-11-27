"""Reject new forbidden captures before persisting an unrecoverable guided attempt."""

import json

import pytest

from tests.integration.test_guided_availability import busy_answers, state
from tests.integration.test_guided_plan import inputs, run
from tests.integration.test_guided_renewal import initial, renewal
from tests.integration.test_guided_status import status
from tests.integration.test_owner_credential_admission import credential_shape


def test_selection_preflights_every_task_before_first_capture(tmp_path):
    note = "Return the book.\n" + credential_shape()
    args = inputs(tmp_path, note, [])
    refused = run(tmp_path, "1 2\n10 normal 2026-11-06\n10 normal 2026-11-06\nCONFIRM\n", args)
    assert refused.returncode == 2 and "Credential-shaped text refused" in refused.stderr
    assert credential_shape() not in refused.stderr
    assert state(tmp_path)["captures"] == []
    assert not (tmp_path / "session/memory").exists()
    corrected = run(tmp_path, "1\n10 normal 2026-11-06\nCONFIRM\nquit\n")
    assert corrected.returncode == 0, corrected.stderr
    assert len(json.loads(status(tmp_path).stdout)["tasks"]) == 1


@pytest.mark.parametrize("operation", ["renewal", "availability"])
def test_refused_source_does_not_latch_pending_review(tmp_path, operation):
    assert initial(tmp_path).returncode == 0
    before = json.loads(status(tmp_path).stdout)["tasks"]
    answers = (
        renewal(statement=credential_shape())
        if operation == "renewal"
        else busy_answers(statement=credential_shape())
    )
    refused = run(tmp_path, answers)
    assert refused.returncode == 2 and "Credential-shaped text refused" in refused.stderr
    saved = state(tmp_path)
    assert saved["pending_renewal"] is None and saved["pending_availability"] is None
    assert saved["unavailability_ids"] == []
    after = json.loads(status(tmp_path).stdout)
    assert after["state"] == "ready" and after["tasks"] == before
    # A fresh, valid owner review works on restart without editing the journal or
    # overriding an uncertain prior attempt.
    corrected = run(tmp_path, renewal() if operation == "renewal" else busy_answers() + "quit\n")
    assert corrected.returncode == 0, corrected.stderr


def test_completion_of_legacy_title_cannot_latch_a_rejected_new_revision(tmp_path):
    # Construct one historically accepted task. Current admission must still apply to
    # a new completion event, while reading the existing task remains supported.
    script = """
from morgan_brain import guided_plan
from morgan_brain.memory import module
guided_plan.refuse_credentials = lambda text: None
module.refuse_credentials = lambda text: None
guided_plan.main()
"""
    created = run(
        tmp_path,
        "1\n10 normal 2026-11-06\nCONFIRM\nquit\n",
        inputs(tmp_path, credential_shape(), []),
        script=script,
    )
    assert created.returncode == 0, created.stderr
    before = json.loads(status(tmp_path).stdout)["tasks"]
    refused = run(tmp_path, "done 1\n")
    assert refused.returncode == 2 and "Credential-shaped text refused" in refused.stderr
    assert state(tmp_path)["pending_done"] is None
    after = json.loads(status(tmp_path).stdout)
    assert after["state"] == "ready" and after["tasks"] == before
    assert run(tmp_path, "quit\n").returncode == 0
