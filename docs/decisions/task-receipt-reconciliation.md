# Receipt reconciliation without replaying uncertain effects

**Status:** Accepted for the SDK and synthetic provider; no real provider or service enabled.
**Date:** 2026-11-06

## Decision and trust boundary

Extend the [task SDK](explicit-task-execution.md) with an explicit, read-only provider lookup.
`task_receipts.reconcile_task` can confirm an unknown attempt using a matched `TaskReceipt`.
It never invokes a task handler, submits an action, retries an effect or continues a batch.

The trusted application supplies `ReceiptAccess`, pairing its selected adapter with a mandatory
authorization callback.
Authority is checked before lookup and again before recording a returned completion receipt.
The callback must cover the selected provider, lookup and local state update; a closure can
bind those details. `TaskPermissions.check` is suitable for the synthetic example's exact task
grant, but does not itself authenticate a real provider or grant network access. Never derive
authority, an adapter implementation or its endpoint from recalled text or model output.

A receipt is an affirmative completion report from the selected adapter. The adapter must
establish final completion under its provider's actual contract before emitting one. Pending,
missing, expired or otherwise non-definitive provider records yield `None`; transport failures
raise. The SDK cannot independently prove an external effect. It stores the basis explicitly
as `trusted_adapter_report`. Field equality and hashes detect mismatched requests; they are
not signatures, provider authentication or protection against a compromised adapter/database.

## Request and receipt binding

`receipt_request` derives a stable idempotency key from a versioned namespace, provider ID,
owner/project/task IDs and the attempt ID. A separate SHA-256 digest binds the exact action
and input. Separating the key from the specification lets a provider reject the same key used
with changed input instead of treating that change as a fresh operation.

A returned receipt must match the complete expected request: provider, scope, task, attempt,
idempotency key and specification digest. Its ID is bounded and nonblank, its result is at most
16,384 UTF-8 bytes, and the persisted JSON evidence is limited to 32,768 bytes. Nothing is
silently truncated. The proof contains the request binding, receipt ID, result, version and
adapter-report basis; raw task input is not copied into it. Digests are not secret storage.

The same key must be used when submitting an action and later looking it up. The SDK does not
assume that any real provider implements this contract. Key length, namespace, retention window,
receipt stability, completion semantics and payload-conflict handling must be established for
the specific integration. A different task ID, attempt, provider ID or restored old journal can
produce a different key and remove duplicate protection.

## State and failure semantics

| Situation | Result |
| --- | --- |
| Running attempt after crash | Caller must explicitly fence that exact attempt as unknown first |
| Missing/incorrect expected attempt | Refused before lookup |
| Unknown attempt, no definitive receipt | `unresolved`; no transition or retry |
| Wrong provider/task/attempt/input receipt | `ReceiptMismatch`; no transition |
| Lookup failure, timeout or revoked authority | Error propagates; no confirmation |
| Matched completion receipt and current authority | Receipt JSON and `succeeded` commit atomically |
| Same confirmation replay | Existing evidence is preserved; no new attempt |
| Already receipt-confirmed task | Validate cached binding locally and return `already_confirmed` |

An already-confirmed response uses saved evidence, not a fresh provider query. Conflicting
confirmation data cannot overwrite the original. The low-level `TaskJournal.confirm_receipt`
method trusts its caller; use `reconcile_task` for binding checks. This is not a security boundary
against arbitrary in-process Python code. On any uncertain local write result, reopen and read
the journal before deciding what to do.

Marking an attempt unknown fences its local completion but cannot stop an old process or remote
action. A missing receipt might reflect delayed visibility or an action still in flight. It is
never proof that replay is safe. This stage adds no negative-result retry, reset or rescheduling
API, even for the synthetic idempotent provider. Cancellation/revocation can race with a prior
external effect; recording its actual completion is distinct from authorizing another effect.

Lookup timeout is cooperative and capped at 60 seconds. It is not a hard process deadline or
spending cap. Real adapters must enforce their own transport limits. An unresolved response is
an observation of the attempt; concurrent callers may have changed durable state since that
observation. Subsequent actions must read current state and obtain current authority.

## Additive journal upgrade

The task journal moves from version 1 to version 2 by adding nullable `receipt_json`. Existing
task fields and attempt states are preserved. Schema change and version marker share one writer
transaction; a rejected commit rolls both back. Unknown future versions and foreign databases
remain refused. Older version-1 SDK code will refuse a version-2 file rather than ignore receipt
evidence. This upgrade is only for the explicitly opened task journal; memory schema and working
services are unchanged. Its existing separate retention, backup and restore obligations remain.

## Synthetic provider and validation

```sh
PYTHONPATH=. python examples/synthetic_receipts.py /tmp/new-receipt-demo --owner demo
```

The example uses two isolated SQLite files: the task journal and a synthetic provider ledger.
The provider supports only `uppercase-v1`; its entire effect is one durable result row. Effect
and receipt commit together on the provider side. The same key/specification returns the same
receipt; changed input with the same key is refused. Lookup only reads. This models an
idempotent service contract, not arbitrary filesystem, email, payment or model API behaviour.

The demo simulates a lost response after provider commit and recovers by lookup. Repeating the
demo retains one provider effect. Tests additionally cover mismatched bindings, absent receipts,
permission refusal/revocation, timeout, local commit rollback, cached-provider mismatch,
idempotent confirmation and non-overwrite. Spawned processes exit after provider commit and
after local confirmation without cleanup; reopening recovers the one effect. Upgrade tests
preserve all version-1 fields and verify rollback of a failed schema commit.

Validation on Python 3.12.13: **1388 passed, 4 live tests skipped** in 87.29 seconds.
All 88 task tests passed, including 23 new receipt/upgrade/recovery tests. Ruff lint/format,
strict mypy and Bandit passed. The repeated demo retained one effect and identical saved evidence.

Before enabling a real action, choose its provider and completion contract, authenticate the
adapter/endpoint, define current authorization and narrowly scoped credentials, and validate its
receipt lookup/idempotency retention against lost responses and concurrent callers. No real
keys, paid requests, retries or background processes are enabled here.
