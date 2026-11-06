# Finite recurring local notes

## Scenario and choice

An owner can approve the same break reminder three times during a work session, then stop
Morgan and resume without producing a burst of old reminders. `local_notes_cli repeat` expands
that intent into exact occurrences in the existing approval file. The same finite worker,
STOP marker, exclusive lock and receipt journal execute them. There is no scheduler database,
persistent cursor, installed service, model call, or implicit renewal.

A repeat has 2 to 16 occurrences, an interval of 60 to 86,400 seconds, and a delivery window
of 1 second up to the interval. Its final window must end in the future within seven days of
approval; this is an explicit finite authorization, separate from the legacy 24-hour TTL.
The entire plan is shown before interactive APPROVE, including every request, ID, UTC due time
and expiry. `--yes` is for trusted, already-authorized automation, not human authentication.

## Example: three hourly breaks

```sh
printf '# Break\nTake a short screen break.\n' > /tmp/morgan-break.md
START=$(python3 -c 'from datetime import UTC, datetime; print(datetime.now(UTC).isoformat())')
python -m morgan_brain.local_notes_cli /tmp/morgan-repeat-demo repeat \
  --user owner --project personal --task work-breaks --note /tmp/morgan-break.md \
  --start "$START" --interval 3600 --count 3 --window 300
python -m morgan_brain.local_notes_cli /tmp/morgan-repeat-demo status
python -m morgan_brain.local_notes_cli /tmp/morgan-repeat-demo run
```

`run` remains finite. Invoke it again during subsequent windows, or explicitly run a bounded
polling session with the existing `--cycles`, `--poll`, `--seconds`, and `--actions` limits.
This example does not install a background process and does not promise a macOS notification:
the approved effect is an immutable local Markdown file. Persistent process hosting remains a
separate deployment decision.

## Time and identity

The start requires an explicit UTC offset, and all occurrences are normalized to UTC. Intervals
are elapsed time, not local calendar recurrence: 86,400 seconds across a daylight-saving boundary
can shift the local hour. No zone name, ambiguous local time, cron expression, or DST rule is
inferred. Calendar recurrence at a fixed local hour is not implemented.

Each occurrence ID hashes the owner, project, explicit series task ID and UTC due instant.
Content, count, and window length are excluded: extending an identical plan preserves existing
IDs; changing the content at an old instant encounters the existing task conflict. A new series
ID or changed due instant explicitly creates a different occurrence and may authorize another
delivery. There is no attempt to infer that two separately authorized series mean the same thing.

Windows are half-open: `due <= now < expiry`. Missed windows are skipped, with no catch-up queue.
During gaps the worker idles. `status` adds `next_due` and `elapsed_windows`; the latter counts
elapsed windows, not missed or completed deliveries. Use journal diagnostics for outcome counts.
Polling, machine sleep and OS clock changes affect delivery time; there is no real-time guarantee.
The final authorization check re-reads the policy and time before publication.

## Stop, recovery and changes

`stop`, `resume`, `revoke`, bounded supervisor limits and worker locking remain unchanged.
STOP persists. Resume does not extend windows or start the worker. Revocation removes all future
authority while preserving files and journal. `repeat` explicitly replaces the whole approval;
`enqueue` cannot append work to a recurring plan. No renewal happens automatically.

Completed occurrences are not repeated after restart or clock rollback. Within an active window,
an interrupted attempt follows receipt reconciliation: a matching published file confirms the
outcome, and an unverified effect is never retried. If a prior attempt remains running/unknown
outside its window, later occurrences still listed in the same approval stop with
`needs_reconciliation`. Expiry does not erase uncertainty or grant authority to inspect an effect.
Resolve it under explicit authority using the existing receipt workflow, or explicitly replace
or revoke the plan after review. There is no automatic reset of the journal.

The strict `morgan.local-approval.v3` file contains explicit occurrences. Older workers reject it
instead of running all repeats immediately. Immediate v1 and one-shot v2 approvals still work.
The owning OS account is trusted and can edit these private files; mode checks are not a sandbox.

## Verification and remaining gaps

Tests use frozen clocks for exact boundaries, UTC offsets and a DST-crossing interval. They
cover missed windows, STOP/resume, expiry, stable IDs, content conflicts, malformed policies,
clock rollback and an expired uncertain attempt blocking the next occurrence. A real sequence
of CLI processes approves a Russian synthetic break plan, delivers the first, stops through
the second, resumes the third and verifies no duplicate. Process crash tests exercise SIGKILL
before and after publication for immediate, one-shot and recurring grants.

Still absent: an authenticated approval UI, recurring calendar rules, unattended OS hosting,
notifications and external action adapters. Model extraction remains the measured 16/20 and is
not treated as production-quality authority. No personal data, secrets or paid APIs are needed
for this stage.
