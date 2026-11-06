# Bounded task batches and exact execution permits

**Status:** Accepted for the opt-in SDK; no service or provider deployment.
**Date:** 2026-11-06

## Context and decision

The [single-attempt task SDK](explicit-task-execution.md) prevents automatic replay of started
tasks. The next useful step is atomic submission of a small worklist and explicit permission
checks before dispatch. It must not silently continue after an uncertain earlier action.

Add `TaskJournal.create_many` and `task_batches.run_batch`. Both accept 1 to 16 distinct task
keys in one owner/project scope. Submission is one SQLite transaction: an invalid request or
a conflict with an existing task specification rolls back every newly inserted row. Exact
submission replay returns existing records without resetting their attempt state.

Execution takes an explicit ordered list, a trusted handler mapping, a live `TaskPermissions`
registry and one cooperative time budget. There is no queue scan, persistent batch identity,
dependency graph, scheduler or worker. The caller must supply the intended ordering again on
restart. Removing an unresolved predecessor from the list bypasses that worklist boundary;
this API does not infer or enforce dependencies across separate invocations.

## Permission boundary

The registry denies everything initially. Each immutable `TaskPermit` binds a `TaskRequest`:
owner, project, task ID, action name and exact input text. It also requires an approval label
and timezone-aware expiration. Grants may be replaced or revoked by trusted application code.
An expired, absent or mismatched permit is denied before claim, leaving the task pending.

Permissions are checked again after claim and before invoking the handler. Handlers must call
`TaskControl.check_ready()` immediately before each effect; that checks cancellation, attempt
ownership and the current permission. Revocation or expiration after a started attempt is
conservatively outcome-unknown, even if no effect happened. This avoids false certainty about
partially executed handlers.

The registry is deliberately not stored in the journal: reopening pending tasks grants no
permission to execute them. The application must obtain current authority again. An approval
ID is an audit label supplied by the caller, not a signature or proof. The permission clock and
issuer are trusted. The application must never construct permits from model output, memory
or checkpoints; the SDK does not authenticate the issuer.

This is an in-process policy boundary, not authentication or a sandbox against malicious Python
code. The existing low-level `run_task` remains a trusted-caller primitive with an optional
authorization callback; batch execution always supplies its permission check. A handler can
ignore cooperative checks, and revocation can race with an action after its last check. Real
provider integrations still need narrowly scoped credentials and provider-side guarantees.

## Stop and restart semantics

| Observed boundary | Batch outcome | Remaining tasks |
| --- | --- | --- |
| Already succeeded | Read result and continue; no handler invocation | Eligible for later dispatch |
| Running or outcome unknown | `needs_reconciliation` | Untouched |
| Cancelled | `cancelled` | Untouched |
| Missing task or handler | `missing_task` or `missing_handler` | Untouched |
| Permission denied before claim | `permission_denied` | Pending and unclaimed |
| Handler error, timeout or permission lost after claim | `needs_reconciliation` | Untouched |
| Concurrent claim/state conflict | `needs_reconciliation` for unresolved state; otherwise `state_changed` | Untouched |
| Budget exhausted before dispatch | `deadline` | Unclaimed |

`BatchResult.observed` contains records read through the stopping point, including the stopping
record when it exists; `stopped_at` and `reason` explain why processing ended. These are
observations over time, not one global transaction snapshot. Storage errors propagate rather
than being treated as ordinary handler failures. Caller cancellation also propagates; the
single-task runner records an interrupted started attempt as unknown when storage works.

The timeout is a shared cooperative execution budget, not a fresh allowance per task. Blocking
Python/SQLite work or handlers suppressing cancellation can overrun it. It is not a spending
cap, nor a promise to terminate a remote request. No automatic retry or manual declaration of
remote success is added. An unknown result still requires independent receipt reconciliation;
regranting permission does not make that task runnable again.

## Alternatives and limits

Persisting permissions alongside tasks would make restart convenient but risks treating an
old approval as current authority. A durable queue/lease system would add implicit scheduling
and stale-worker semantics before there is a real external-action contract. This stage keeps
permission refresh and task selection explicit, with no schema migration or new service.

Submission atomicity does not make the external actions of a batch atomic. A later failure does
not undo earlier successes. Stable task IDs, journal retention and restore discipline remain
necessary: new IDs or rolled-back database copies can duplicate effects. The separate journal
is still outside memory `forget` and snapshots; its lifecycle must be established before real
private data is stored.

## Validation and example

```sh
PYTHONPATH=. python examples/task_batch.py /tmp/new-batch.db --owner demo
```

The example atomically creates two tasks and issues short-lived synthetic permits only for a
pure uppercase handler. It uses no keys, network or model. Repeating it reads both saved results.

Unit tests cover late submission conflict rollback, exact grants, scope boundaries, expiry,
revocation before/between/during handlers, cancellation, a shared deadline, storage errors and
restart with an empty permission registry. Spawned-process tests exit after a synthetic fsynced
effect or between two tasks. They prove no duplicate effect and no continuation across an
unresolved result. They do not establish power-loss durability or remote exactly-once execution.

Validation on Python 3.12.13: **1365 passed, 4 live tests skipped** in 86.13 seconds.
All 65 task tests passed, including 34 new batch/permission tests. Ruff lint/format, strict
mypy and Bandit passed. Repeating the synthetic example returned identical saved attempts.

## Next integration requirement

The [receipt reconciliation stage](task-receipt-reconciliation.md) now exercises lookup and
binding checks with an isolated synthetic provider. It does not add a real provider or retry.

Choose one provider/action with an idempotency key or receipt lookup and an authoritative source
of current approvals. Define revocation, result reconciliation, credential scope, spending/call
limits and journal retention for that exact integration. Validate remote acceptance followed by
a lost reply before adding any supervised background worker. Nothing here enables one.
