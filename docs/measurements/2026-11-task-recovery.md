# Task execution and recovery review

Reviewed on 2026-11-06 against main `436cc00f34a58fd8179f4e8a3a9dec3d60671199`.
The product direction is a personal agent available 24/7. This review tests the persistence
boundary that an executor would depend on; it does not claim a deployed task executor.

## What runs today

`app/chat.py` performs one requested turn: recall, generate, and atomically store its two
memories and history entries. Provider calls precede the write transaction. Existing
`tests/unit/app/test_chat_atomic.py` covers embedding failures, partial writes, evidence
changes and forgetting during generation. A failed call is not a durable pending job.

`memory/checkpoints.py` and `MemoryGate.put_checkpoint/get_checkpoint` store bounded,
source-backed summaries with compare-and-swap against the previous fact ID. They preserve
historical versions and refuse stale writers. `completed` is an agent report, not an execution
receipt. A current checkpoint establishes an eligible source basis, not authorization or proof
that an external action happened. There is no task queue, worker lease, scheduler, tool executor,
automatic restart or external-action reconciliation in current main.

## Reproduced defect and fix

Both transaction helpers caught exceptions from the body but called `commit()` outside that
handler. A rejected COMMIT could therefore return an error while leaving the connection inside
its transaction. Reproduction with a deferred SQLite foreign-key constraint left one pending
row and `in_transaction=True`. A later request could join the abandoned transaction as a
savepoint, and other writers could remain blocked.

Move COMMIT into the existing rollback-protected block in `read_transaction` and
`write_transaction`. The exception still reaches the caller; no automatic retry or replay is
introduced. This changes neither schemas nor successful transaction semantics. Rollback itself
can still fail on a broken connection/storage device; such an error is propagated and callers
must not assume the connection is reusable.

## New regression checks

`tests/unit/memory/test_transaction_recovery.py` checks:

- A real deferred-constraint failure at COMMIT removes pending rows, ends the transaction,
  releases the writer lock to a second connection, and permits a subsequent valid write.
- SQLite authorizer rejection of COMMIT closes an owned read or write transaction; a subsequent
  transaction succeeds after the injected rejection is removed.

`tests/integration/test_checkpoint_process_recovery.py` runs a separate spawned process against
a temporary database. It exits via `os._exit` after a checkpoint update, bypassing Python
finalizers and connection cleanup. It exercises two deterministic boundaries:

| Exit point | Reopened checkpoint | Recovery decision |
| --- | --- | --- |
| Inside an outer transaction, before commit | Previous active state; one fact | Update using the previous fact ID succeeds |
| After commit, before communicating its result | New paused state; two historical facts | Replay using the previous fact ID raises `StaleCheckpoint` |

Both cases check database integrity, owner isolation and preservation of historical evidence.
These are process-exit tests, not power-loss, disk-full, distributed execution or hardware
durability measurements. A checkpoint cannot make a separate external side effect atomic.

Local validation on macOS with Python 3.12.13: `pytest -q --tb=short` completed with
**1300 passed, 4 skipped** in 86.46 seconds. The four live tests were not enabled. The full
suite needed permission for its temporary loopback fake servers; the restricted first run
could not bind them. `ruff check .`, `ruff format --check .`, `mypy morgan_brain` and
`bandit -c pyproject.toml -r morgan_brain` passed, with zero Bandit findings.

## Required contract before a 24/7 executor

Keep the executor a separate consumer of the memory gate. Start with one bounded, manually
invoked synthetic task, then establish these behaviours before scheduling real work:

1. Persist task identity, explicit authorization scope and execution attempts separately from
   inferred memory. Define pending, running, succeeded, failed, cancelled and outcome-unknown.
2. After restart, read durable state before deciding what to do. An interrupted external action
   needs a provider idempotency key or a receipt lookup. Without either, surface outcome-unknown
   and reconcile; never infer that retry is safe from a missing local completion record.
3. Bound attempts, runtime and spending independently. Timeouts and request byte/token controls
   do not constitute an account spending cap. Record cancellation and check it before each
   external action; test cancellation/retry races and concurrent ownership before adding leases.
4. Test an interrupted action before submission, after remote acceptance, before local receipt,
   and after receipt. A synthetic executor must show no duplicate irreversible action before a
   real provider is enabled. Keep health reporting distinct from evidence of task completion.

No executor, background loop or paid provider request was added during this review.

## Launch prerequisites

A concrete launch proposal must name the endpoint/model, credential source, hard spending
limit and period, maximum calls, exact task/action allowlist, isolated data directory and stop
condition. Begin with a single explicit invocation using synthetic data and the hash embedding
backend. A persistent service additionally requires a chosen executor, service identity,
restart policy, cancellation path and the reconciliation tests above. The incoming MCP bearer
key and outbound model credentials serve different purposes. No deployment is part of this fix.
