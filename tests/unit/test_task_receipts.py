"""Adapter evidence must match the attempt, and lookup never replays an external action."""

import asyncio
import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from examples.synthetic_receipts import SyntheticReceiptProvider
from morgan_brain.task_batches import TaskPermissionDenied, TaskPermissions, TaskPermit
from morgan_brain.task_receipts import ReceiptMismatch, TaskReceipt, receipt_request, reconcile_task
from morgan_brain.tasks import TaskConflict, TaskJournal, TaskKey, TaskRequest

KEY = TaskKey("owner", "synthetic", "one")


@pytest.fixture
def setup(tmp_path):
    journal = TaskJournal(tmp_path / "tasks.db")
    provider = SyntheticReceiptProvider(tmp_path / "provider.db")
    spec = TaskRequest(KEY, "uppercase-v1", "hello")
    journal.create(KEY, action=spec.action, payload=spec.payload)
    task = journal.claim(KEY, action=spec.action)
    journal.mark_unknown(KEY, attempt_id=task.attempt_id)
    policy = TaskPermissions()
    policy.grant(TaskPermit(spec, "synthetic-approval", datetime.now(UTC) + timedelta(hours=1)))
    yield journal, provider, task, policy
    provider.close()
    journal.close()


async def reconcile(journal, provider, task, policy, seconds=1):
    return await reconcile_task(
        journal,
        KEY,
        expected_attempt_id=task.attempt_id,
        reader=provider,
        authorize=policy.check,
        timeout_seconds=seconds,
    )


async def test_provider_idempotency_and_receipt_persistence(setup, tmp_path):
    journal, provider, task, policy = setup
    first = await provider.execute(task)
    second = await provider.execute(task)
    assert first == second and provider.count_effects() == 1
    confirmed = await reconcile(journal, provider, task, policy)
    assert confirmed.outcome == "confirmed"
    assert confirmed.task.status == "succeeded" and confirmed.task.result == "HELLO"
    saved = json.loads(confirmed.task.receipt_json)
    assert saved["basis"] == "trusted_adapter_report"
    assert saved["request"]["attempt_id"] == task.attempt_id
    assert saved["receipt_id"] == first.receipt_id
    again = await reconcile(journal, provider, task, policy)
    assert again.outcome == "already_confirmed" and again.task == confirmed.task
    assert (
        journal.confirm_receipt(
            KEY,
            attempt_id=task.attempt_id,
            result=confirmed.task.result,
            receipt_json=confirmed.task.receipt_json,
        )
        == confirmed.task
    )
    with pytest.raises(TaskConflict):
        journal.confirm_receipt(
            KEY,
            attempt_id=task.attempt_id,
            result="different",
            receipt_json="different",
        )
    assert journal.get(KEY) == confirmed.task
    reopened = TaskJournal(tmp_path / "tasks.db")
    try:
        assert reopened.get(KEY) == confirmed.task
        with pytest.raises(TaskConflict):
            reopened.claim(KEY, action=task.action)
    finally:
        reopened.close()


async def test_reusing_idempotency_key_with_changed_input_is_refused(setup):
    _, provider, task, _ = setup
    await provider.execute(task)
    changed = replace(task, payload="changed")
    old_query = receipt_request(task, provider_id=provider.provider_id)
    changed_query = receipt_request(changed, provider_id=provider.provider_id)
    assert old_query.idempotency_key == changed_query.idempotency_key
    assert old_query.specification_sha256 != changed_query.specification_sha256
    with pytest.raises(ReceiptMismatch):
        await provider.execute(changed)
    assert provider.count_effects() == 1


async def test_absent_receipt_is_unresolved_not_permission_to_retry(setup):
    journal, provider, task, policy = setup
    result = await reconcile(journal, provider, task, policy)
    assert result.outcome == "unresolved"
    assert journal.get(KEY).status == "outcome_unknown" and provider.count_effects() == 0
    with pytest.raises(TaskConflict):
        journal.claim(KEY, action=task.action)


@pytest.mark.parametrize(
    "field", ["provider", "owner", "project", "task", "attempt", "key", "spec"]
)
async def test_mismatched_receipt_never_confirms(setup, field):
    journal, provider, task, policy = setup
    expected = receipt_request(task, provider_id=provider.provider_id)
    changes = {
        "provider": {"provider_id": "other"},
        "owner": {"key": TaskKey("other", KEY.project, KEY.task_id)},
        "project": {"key": TaskKey(KEY.user_id, "other", KEY.task_id)},
        "task": {"key": TaskKey(KEY.user_id, KEY.project, "other")},
        "attempt": {"attempt_id": "other"},
        "key": {"idempotency_key": "wrong"},
        "spec": {"specification_sha256": "wrong"},
    }

    class Reader:
        provider_id = provider.provider_id

        async def lookup(self, request):
            return TaskReceipt(replace(expected, **changes[field]), "receipt", "result")

    with pytest.raises(ReceiptMismatch):
        await reconcile(journal, Reader(), task, policy)
    assert journal.get(KEY).status == "outcome_unknown"
    assert journal.get(KEY).receipt_json is None


@pytest.mark.parametrize("failure", ["exception", "timeout", "malformed", "revoked"])
async def test_lookup_failure_or_revoked_permission_leaves_unknown(setup, failure):
    journal, provider, task, policy = setup
    receipt = await provider.execute(task)

    class Reader:
        provider_id = provider.provider_id

        async def lookup(self, request):
            if failure == "exception":
                raise OSError("synthetic provider unavailable")
            if failure == "timeout":
                await asyncio.sleep(5)
            if failure == "malformed":
                return {"success": True}
            policy.revoke(KEY)
            return receipt

    error = {
        "exception": OSError,
        "timeout": TimeoutError,
        "malformed": ReceiptMismatch,
        "revoked": TaskPermissionDenied,
    }[failure]
    with pytest.raises(error):
        await reconcile(journal, Reader(), task, policy, seconds=0.02)
    assert journal.get(KEY).status == "outcome_unknown"
    assert journal.get(KEY).receipt_json is None and provider.count_effects() == 1


async def test_permission_denied_before_lookup_makes_no_provider_call(setup):
    journal, provider, task, policy = setup
    policy.revoke(KEY)

    class Reader:
        provider_id = provider.provider_id

        async def lookup(self, request):
            pytest.fail("lookup ran without authority")

    with pytest.raises(TaskPermissionDenied):
        await reconcile(journal, Reader(), task, policy)


async def test_running_attempt_requires_explicit_fence(tmp_path):
    journal = TaskJournal(tmp_path / "tasks.db")
    provider = SyntheticReceiptProvider(tmp_path / "provider.db")
    try:
        journal.create(KEY, action="uppercase-v1", payload="hi")
        task = journal.claim(KEY, action="uppercase-v1")
        with pytest.raises(TaskConflict, match="fence"):
            await reconcile_task(
                journal,
                KEY,
                expected_attempt_id=task.attempt_id,
                reader=provider,
                authorize=lambda task: None,
                timeout_seconds=1,
            )
    finally:
        provider.close()
        journal.close()


async def test_wrong_expected_attempt_is_refused_before_lookup(setup):
    journal, provider, _, policy = setup
    with pytest.raises(TaskConflict):
        await reconcile_task(
            journal,
            KEY,
            expected_attempt_id="wrong",
            reader=provider,
            authorize=policy.check,
            timeout_seconds=1,
        )
    assert provider.count_effects() == 0


async def test_receipt_and_state_rollback_together_on_commit_failure(setup):
    journal, provider, task, policy = setup
    await provider.execute(task)

    def deny_commit(action, argument, _second, _db, _trigger):
        if action == sqlite3.SQLITE_TRANSACTION and argument == "COMMIT":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    journal._conn.set_authorizer(deny_commit)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            await reconcile(journal, provider, task, policy)
    finally:
        journal._conn.set_authorizer(None)
    assert journal.get(KEY).status == "outcome_unknown" and journal.get(KEY).receipt_json is None
    assert (await reconcile(journal, provider, task, policy)).outcome == "confirmed"
    assert provider.count_effects() == 1


async def test_already_confirmed_does_not_claim_a_different_provider(setup):
    journal, provider, task, policy = setup
    await provider.execute(task)
    await reconcile(journal, provider, task, policy)

    class WrongReader:
        provider_id = "other"

        async def lookup(self, request):
            pytest.fail("cached evidence should be checked locally")

    with pytest.raises(ReceiptMismatch):
        await reconcile(journal, WrongReader(), task, policy)
