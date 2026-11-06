# Read a guided session without changing it

The owner or an already authorized local assistant can now ask what remains in a specific
session without entering the interactive dialogue, running a model, or creating another plan:

```sh
python -m morgan_brain.guided_plan /private/session --status
```

The command returns one JSON object, version `morgan.guided-status.v1`. It uses the existing
private workspace and a shared nonblocking session lock, validates source/proposal hashes,
and reads current evidence through MemoryGate. It never treats an old Markdown draft as truth.
It does not initialize a missing session or lock, change inputs, capture memory, acknowledge
pending writes, publish a draft, or call a model or embedding provider.

SQLite's read-only WAL access may create/update its `-wal`/`-shm` coordination files, as already
documented by `open_readonly`. This is not an immutable-filesystem mode. Durable memory data,
the session journal, inputs and draft files are not rewritten by this query.

## Interpret the response before answering the user

- `owner` and `project` identify the exact selected scope. Do not substitute another workspace.
- `state: ready` and `plan_available: true` mean this session has no pending review. They do
  not mean the owner is free, all obligations are known, or any action has been authorized.
- `state: needs_review` includes `pending_reviews`: task selection/confirmation, completion,
  or unavailability confirmation. No task/constraint plan is returned in this state. An earlier
  write may already have committed; querying must not silently settle that uncertainty.
- `planning_day` is the fixed session date, not automatically today. `day_relation` is `past`,
  `today`, or `future` in the explicit planning timezone, or `unknown_timezone` when no zone was
  selected. Do not present an old or undated-zone session as today's schedule.
- `as_of` identifies the current evidence snapshot. `tasks` and `unavailability` include exact
  current evidence IDs and structured fields; the resolution maps retain the original IDs.
- `selected_ids`, `deferred_ids`, `excluded_ids`, and `blocked_ids` separate focus tasks, budget
  deferral, completion/future dates, and unavailability. `planned_minutes` is not elapsed work.
- `scope_complete` and `execution_authorized` are always false. This is neither a whole-life
  calendar nor a grant to send, book, pay, deliver, or run any listed task.

Exit 0 returns the snapshot, including a legitimate `needs_review` result. Exit 2 refuses
missing/changed inputs, an active interactive dialogue, invalid overrides, or another handled
read error. Do not reuse a cached successful answer after a refusal. The lock is nonblocking;
retry only after the interactive dialogue exits. Status accepts no other session-input flags.

## User channel and control boundary

This is a small query interface for the current local CLI or an existing authorized assistant,
not a new messaging service. A connected assistant may format the result for the same owner,
retaining the planning date, limitations and evidence IDs. It must not automatically approve
pending operations or forward private records to another recipient/provider. Changes remain
in the reviewed interactive dialogue; status never runs commands from remembered text.

No daemon, secret access, notification, listener, recipient or background polling is configured.
A genuinely proactive personal agent still needs a chosen event source, a user-visible delivery
channel with acknowledgement, and explicit lifecycle/stop behavior. This query makes reliable
on-demand answers possible without pretending those missing pieces are implemented.

## Verification

Subprocess tests verify current completion/cancellation, misleading old draft rejection,
unchanged durable files, no model/capture calls, pending states, missing sessions/locks,
active-dialogue refusal, modified sources, input overrides and timezone-aware day freshness.
The ordinary interactive flow uses the same read-plan function, so the two views cannot drift
into separate planning implementations.

A synthetic completed session with a cancelled interval returned its current state in one
launch and zero answer submissions. This is a scripted functional check, not a human usability
study. [Measurement](../measurements/2026-11-guided-status.json).
