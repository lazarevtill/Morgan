# Explicit cancellation of a personal commitment

A task that is no longer wanted must leave the focus plan without being reported
as completed. `owner_input review MESSAGE --cancel CURRENT_ID` and the native
`Отменить задачу` action prepare one exact owner-selected transition from `open`
to `cancelled`. The original statement, current target, unchanged task fields and
explicit confirmation remain required. Message wording never selects a target or
authorizes cancellation.

Cancellation reuses `RenewalUpdate` and its paired source/revision capture and
recovery. A `guided-cancel-` identity permits only the status transition; title,
date, estimate and priority are unchanged. Existing renewal/completion payloads
and digests are unchanged. STOP, expiry, stale-parent checks and exact replay
still apply. A terminal task cannot be renewed, completed or cancelled again via
this path. The legacy guided completion command likewise refuses closed tasks.

`Commitment.status` now accepts `cancelled` alongside `open` and `done`. Only open
commitments enter focus plans. Excluded entries and source lookup preserve the
actual terminal status: cancellation is not evidence of performance. This does
not cancel an external booking, send a message, erase evidence, release the pilot
quota, or add reopening. `cancel N` in the legacy guided dialogue still refers to
availability; task cancellation uses the explicit owner-input/native surface.

## Compatibility and deployment

No database migration is required; the new status is stored in an immutable
revision. Old open/done records and pending reviews remain readable. Older
runtimes cannot parse a newly stored cancelled commitment or pending cancellation.
All readers of an affected pilot (worker, UI, status/resume shortcuts) must use a
compatible pinned runtime before the feature is exposed. Do not downgrade an
updated journal after cancellation has been used, delete its receipt, or restore
an older database alone. Preserve the complete data and deploy a compatible
runtime; the ordinary evidence snapshots are not a whole-workspace rollback.

Validation includes RU/EN owner statements, declining/resuming a review, failures
after either paired write, idempotent acknowledgement, STOP, stale parents,
terminal-state rejection, exact provenance, and exclusion from the plan. Scripted
native-dialog tests are not physical GUI or human usability measurements.
