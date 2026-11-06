# One-shot local delivery windows

## Practical gap and scope

A reviewed personal focus plan can already become an exact approved Markdown delivery.
Previously a worker delivered it immediately: there was no way to authorize delivery later,
stop overnight, and resume without losing the intended earliest time. This change adds one
optional time window to the existing approval file and worker, without another scheduler,
database, model, or service.

`approve --not-before TIMESTAMP` approves the exact note for delivery at or after that instant
and strictly before the existing expiry. The timestamp must include a UTC offset; it is stored
in UTC. `--ttl` still starts at approval time, not delivery time, and remains at most 86,400
seconds. An empty window is rejected before replacing the old approval. A past earliest time
is allowed explicitly, meaning deliver on the next invocation while the grant is valid.

## Example: deliver a reviewed plan in five minutes

Use a fresh private directory; `approve` replaces all grants in that workspace. The note below
is deliberately synthetic. The default interactive prompt shows the exact note and timing.
`--yes` is reserved for explicitly trusted automation; it is not proof of human confirmation.

```sh
printf '# Focus plan\n- Water the plants\n' > /tmp/morgan-plan.md
DUE=$(python3 -c 'from datetime import UTC, datetime, timedelta; print(
    (datetime.now(UTC) + timedelta(minutes=5)).isoformat())')
python -m morgan_brain.local_notes_cli /tmp/morgan-scheduled-demo approve \
  --user owner --project personal --task morning-plan \
  --note /tmp/morgan-plan.md --not-before "$DUE" --ttl 900
python -m morgan_brain.local_notes_cli /tmp/morgan-scheduled-demo status
python -m morgan_brain.local_notes_cli /tmp/morgan-scheduled-demo run \
  --cycles 60 --poll 10 --seconds 600 --actions 1
```

The finite worker polls the persisted window. It creates no task attempt before the earliest
time. Status reports `approval: scheduled` plus `not_before` and `expires_at` without note text.
Late delivery is allowed only inside the window; an expired grant returns `permission_denied`.
No recurring occurrence or new approval is inferred from the note contents.

## Stop, resume, and missed delivery

```sh
python -m morgan_brain.local_notes_cli /tmp/morgan-scheduled-demo stop
python -m morgan_brain.local_notes_cli /tmp/morgan-scheduled-demo status
python -m morgan_brain.local_notes_cli /tmp/morgan-scheduled-demo resume
python -m morgan_brain.local_notes_cli /tmp/morgan-scheduled-demo run
```

STOP persists across invocations. `resume` removes it only while worker and supervisor locks
are free; it does not start a process or extend the grant. Invoke `run` again explicitly.
`revoke` removes authority while preserving receipts, notes and STOP. To reschedule, explicitly
replace approval with the same exact request and a new window. A completed task ID still means
completed: a genuinely new delivery needs a new task ID. Changing a pending task's payload
under its old ID remains a conflict.

A crash while waiting loses no schedule. A crash around publication follows existing receipt
reconciliation: a matching note confirms completion; an unverified effect remains unresolved
and is not repeated. Later invocation uses the same identity, so completed delivery is skipped.
`enqueue` refuses any scheduled batch rather than silently changing its time or scope. Use a
separate workspace for independently timed notes, or explicitly replace the whole approval.

## Compatibility and limits

Immediate grants retain `morgan.local-approval.v1`; scheduled grants use the strict v2 shape
with required `not_before`. Older workers reject v2 instead of executing it early. Scheduling
covers the whole approved batch (1 to 16 exact notes), not independent per-item times.
The policy is re-read at dispatch and immediately before publication, including after wall-clock
changes. OS wall time is trusted; clock jumps can change when a task becomes due or expires.
Polling, OS sleep, and filesystem calls can delay delivery. There is no hard real-time guarantee.

This is one-shot scheduling for a local file, not a calendar, recurring automation, notification
channel, authenticated approval UI, or installed 24/7 service. The worker stays finite, with
existing action/cycle/runtime limits. macOS sleep or process exit needs a later explicit run;
no launchd/cron configuration is created. Same-account software can edit the approval file;
private permissions and digests do not authenticate a human against that account.
No credentials, network calls, model calls, or memory writes are added. The measured model
extraction quality remains 16/20; reviewed note content must still be chosen explicitly.

## Validation

Tests cover boundary times, explicit offsets, malformed windows, legacy immediate approval,
STOP and missed expiry, revocation, clock jumps at the effect boundary, and append refusal.
A real CLI process is killed while waiting, restarted, and delivers a Russian synthetic plan
once. SIGKILL before/after publication verifies unresolved/completed receipt recovery for both
immediate and scheduled grants. All data and processes are isolated test fixtures.
