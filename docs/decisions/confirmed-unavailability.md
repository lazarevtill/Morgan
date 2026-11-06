# Confirmed unavailability in daily plans

A remembered sentence about being unavailable was previously searchable but had no effect
on a focus plan. The explicit daily-brief workflow now accepts reviewed unavailability evidence.
No location, travel, appointment, notification, or permission is inferred from that evidence.

## Scope and conservative behavior

The current planner assigns a budget to a day, not start times to tasks. If any selected active
interval overlaps that day, all otherwise eligible tasks are deferred with a distinct reason.
A one-minute overlap therefore defers the day's tasks too. This deliberately sacrifices useful
free time: it does not claim that the whole day is unavailable or schedule tasks into an unknown
free period. Completed and future-dated tasks remain excluded as before.

Only explicitly selected constraint IDs are checked. Omitting a constraint is not evidence that
the user is available. `guided_plan` does not yet collect or forward constraints; use the explicit
`daily_brief_cli` path below for this feature. There is no automatic calendar discovery or service.

## Reviewed record and source provenance

An account-authorized client first captures the actual source statement through normal memory
capture. After reviewing the exact interval and timezone, it captures this typed JSON through
`morgan remember`, with `--source user_stated`, the reported author, a stable `--event-id`, and
an explicit `--effective-at`. This reuses the existing single write path; there is no new database.
Do not mark inferred model output as user-stated without actual confirmation of the interval.

Synthetic example, not dates or statements from the owner:

```json
{
  "version": "morgan.unavailability.v1",
  "starts_at": "2027-04-10T00:00:00+02:00",
  "ends_at": "2027-04-12T00:00:00+02:00",
  "timezone": "Europe/Paris",
  "status": "active",
  "source_ids": ["source-message-example"]
}
```

`source_ids` contains one to eight exact supporting records in the same owner/project scope.
The planner requires current, unambiguous, user-stated episodic evidence for both the typed
assertion and its sources. Missing, quarantined, inferred, instruction-like, superseded or
conflicted sources refuse the plan. A source reference proves provenance, not that software
under the same OS account cannot forge confirmation. The client must faithfully represent
what the user confirmed. Sources are not automatically parsed into intervals.

There is no location field. Dates alone, relative phrases and missing zones must be resolved
explicitly before capture. For inclusive whole-day absence through April 11, the exclusive end
is April 12 at midnight in the reviewed zone, as in the example.

## Time and revisions

Intervals are half-open `[start, end)`, with explicit offset-bearing endpoints and an IANA zone.
Each endpoint's wall time and offset must agree with that zone. Nonexistent spring-forward
wall times are rejected. During a repeated autumn hour, the explicit offsets distinguish the
instants. Duration must be positive and at most 366 real days. No system timezone is inherited.

The planning day has its own explicit IANA timezone. Day boundaries are converted to UTC,
so 23-hour and 25-hour days and a different planning zone are handled by actual instants.
An interval ending exactly at local midnight does not block the following day. A calendar day
skipped entirely by a zone transition is rejected. The machine's installed timezone database
supplies the rules; no timezone data is fetched or updated by the planner.

Correct or cancel an assertion with a new immutable event, using `--revises-event-id` for the
previous assertion and the normal known-source/author/effective-time requirements. A cancellation
keeps the same interval fields and uses `status: cancelled`. Constraint IDs always follow one
current, unambiguous revision, even when task revision following was not requested. Conflicting
branches refuse planning; they never silently release the restriction. A revised supporting
statement requires a new reviewed constraint pointing to its current evidence.

Retry the exact event ID and fields after acknowledgement loss. The existing capture path
provides immutable retry semantics; drafting never mutates memory. Reopening the database and
reusing the original constraint ID preserves a correction or cancellation. A saved draft is an
as-of snapshot, not a live execution guard: regenerate/review before any later delivery or action.

## CLI

After capturing the reviewed task, supporting statement and typed constraint in a private memory:

```sh
python -m morgan_brain.daily_brief_cli \
  --memory-db /private/memory/morgan.db --user owner --project personal \
  --task morning-plan --evidence task-event --follow-revisions \
  --unavailability constraint-event --timezone Europe/Paris \
  --day 2027-04-10 --budget 30 --output /private/drafts/plan.md
```

Repeat `--unavailability` for up to eight distinct constraints. A missing planning timezone
refuses the request. The JSON result includes `blocked` and `resolved_unavailability`; the
Markdown shows interval endpoints, zones, assertion IDs, source IDs, followed revisions and the
conservative deferral reason. It grants no delivery or execution permission.

## Verification and remaining work

Isolated SQLite tests cover start/end boundaries, DST folds/gaps, timezone differences,
partial-day deferral, source and owner isolation, missing/untrusted evidence, correction,
cancellation, conflicts, database reopen and the actual CLI. Examples and fixtures are synthetic;
no personal chat data is committed. Existing task selection remains unchanged without constraints.

This is a useful day-level restriction, not a calendar scheduler. Next useful work is a reviewed
intake interface for intervals and a connected personal workflow that carries selected constraints
into every relevant plan. Slot scheduling, automatic timezone/date inference, notifications and
persistent service operation are intentionally not claimed by this change.
