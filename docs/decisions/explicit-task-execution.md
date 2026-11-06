# ADR: Explicit single-attempt task execution

**Status:** Accepted for the opt-in SDK stage; no deployment or autonomous scheduling.
**Date:** 2026-11-06
**Deciders:** Repository owner requested staged implementation and validation of task execution.

## Context

The product direction is a personal agent available 24/7. The existing memory core and typed
checkpoints preserve information but do not execute tasks. The
[recovery review](../measurements/2026-11-task-recovery.md) identified the missing boundary:
a crash after an external action but before its local receipt must not trigger blind replay.
An inferred checkpoint, including a reported `completed` state, is not an execution receipt
or permission to run a tool.

## Decision

Add `morgan_brain.tasks`, an opt-in SDK for explicitly running one task with one attempt.
The caller supplies an isolated journal path, owner/project/task ID, a named action and bounded
text input. `run_task` requires an explicit mapping of action names to trusted async handlers
and a cooperative timeout of at most one hour. There is no dynamic import, shell command,
provider client, memory-to-task conversion, background loop, scheduler or default wiring.

The journal uses SQLite WAL with `synchronous=FULL`, a format/application marker and a version.
It requires an on-disk `Path` and refuses `:memory:` or existing foreign databases before
changing their journal mode. Writes use Morgan's
existing `BEGIN IMMEDIATE` transaction primitive. A claim commits a random attempt ID before
the handler is invoked. Other connections cannot claim that task again. Payload and result are
limited to 16,384 UTF-8 bytes each. Error strings are not persisted or logged by this SDK.

Task identity is `(user_id, project, task_id)`. Repeating `create` with exactly the same action
and payload returns existing state; changing the specification under the same ID is refused.
There is no update-input, reset, retry, automatic expiry or reclaim operation. Use stable task
IDs chosen by trusted application code. A different ID or a copied/rolled-back journal can
duplicate an external action; this ledger is not an external idempotency service.

| Current state | Operation | Result |
| --- | --- | --- |
| Absent | Explicit create | Pending |
| Pending | Claim matching action | Running with one attempt ID |
| Pending | Cancel | Cancelled; handler cannot run |
| Running | Cancel | Running with cancellation requested |
| Running | Handler returns bounded text and receipt commits | Succeeded |
| Running | Exception, timeout, cancellation or invalid result | Outcome unknown when storage works |
| Running | Explicit mark-unknown for matching attempt | Outcome unknown; no new attempt |
| Any started/terminal state | Claim/re-run | Refused |

Cancellation is cooperative. A handler should call `TaskControl.check_cancelled()` before each
effect. Cancellation can race with an effect after that check. Running cancellation therefore
does not imply that nothing happened: a normal return may still persist success with the
cancellation flag, while an interrupted handler is marked outcome unknown. Async timeout is
not a process sandbox or hard spending deadline; a blocking handler or one suppressing
cancellation violates the trusted-handler contract.

## Recovery and evidence

On restart, `get` reads the durable state without running anything. A remaining `running`
record means an unresolved attempt, not proof that a worker is alive or that work failed.
The caller can mark that exact attempt unknown. This fences local late completion but cannot
stop a remote operation or old process. Neither running nor unknown is runnable again.

If result persistence fails, the runner tries to mark an outstanding attempt unknown and
propagates the error. If the result already committed but its acknowledgement failed, it keeps
the durable success. If the database cannot be read or written, state may remain running;
the caller must reopen and inspect it before any further decision. Even a failed claim commit
can leave an unresolved running record without a handler invocation. Conservative false
uncertainty is preferable to silently repeating a possibly completed action.

`succeeded` means that the trusted handler returned and the local result committed. It does
not independently verify a remote effect. This stage intentionally has no API to declare a
remote operation definitely failed or to reconcile an unknown result; the next integration
must define provider receipt lookup and idempotency semantics first.

## Options considered

| Option | Complexity/cost | Benefit | Limitation |
| --- | --- | --- | --- |
| Reuse inferred checkpoint facts | Low; existing storage | Compact remembered progress | Confuses reports with execution; memory erasure could remove duplicate protection |
| Isolated explicit journal, one attempt | Small SQLite SDK; no service or model cost | Testable crash boundary and conservative replay refusal | No automatic retry, scheduler or cross-system atomicity |
| Full worker/lease/queue framework now | Larger service and operations surface | Scheduling and throughput | Premature recovery/permission assumptions without a real provider contract |

Choose the isolated journal for this first stage. The memory core retains one database and one
gate; the executor ledger is an explicit separate artifact with a different lifecycle.

## Consequences and access boundary

- This SDK trusts its in-process caller and registered handlers. IDs and action names are
  labels, not authentication or permission checks. Do not expose it directly to untrusted model
  output, an MCP client or a network request. The calling application must authorize the exact
  action/input using current user authority before creating/running it.
- No memory, checkpoint or recalled instruction grants execution authority. A handler that
  needs memory must use `MemoryGate`; this module does not read memory itself.
- Journal payloads/results may be private. Use a protected directory and do not put credentials
  in them. `morgan forget`, memory migrations, snapshots and restore do not cover this separate
  file. Retention, erasure, backup and permissions must be explicitly designed before storing
  real user data. Deleting execution history or restoring an old copy can remove replay guards.
- One `TaskJournal` connection belongs to one thread. Use separate instances for concurrent
  processes; SQLite serializes state changes. There is no distributed coordination across
  different files, database copies or hosts.
- Process-exit tests do not establish power-loss or filesystem durability. No exactly-once
  claim is made for external actions; handlers may perform multiple effects themselves.

## Validation and example

The synthetic example requires an explicit journal filename and performs only uppercase text:

```sh
PYTHONPATH=. python examples/single_task.py /tmp/new-tasks.db --owner demo --task-id sample
```

Running the same command reads the committed result without executing again. It launches no
service and reads no API key. Use a new isolated file, never the memory database.

Tests cover exact create replay, conflicting specifications, owner/project isolation, concurrent
claims, cancellation before/during a handler, caller cancellation, exceptions, invalid results,
timeouts, lost result acknowledgements, format rejection and reopening. Spawned processes exit
without cleanup before claim, after claim, after a synthetic fsynced external effect, and after
result commit. The receipt remains a single effect; unresolved attempts cannot be rerun.

The initial full run on Python 3.12.13 completed with **1328 passed, 4 live tests skipped**
in 86.42 seconds. After adding explicit rejection of ephemeral journal paths, all **31 task
tests passed**. They use isolated data and synthetic handlers. Ruff lint/format, strict mypy
and Bandit passed. Repeating the example returned the identical persisted attempt and result.

## Next stages

1. Choose one real provider/action with a documented receipt lookup or idempotency key. Test
   remote acceptance followed by a lost response and cancellation races in an isolated sandbox.
2. Define explicit authorization and revocation, data retention/backup, bounded call counts and
   enforceable spending limits. Add reconciliation evidence without turning it into auto-retry.
3. Only then consider a supervised worker, health reporting and scheduling. Require bounded
   restart attempts, a stop control and tests proving an old worker cannot duplicate effects.

These are future integration criteria, not services enabled by this change.
