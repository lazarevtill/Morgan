# Reviewed unavailability in the guided conversation

`guided_plan` now carries the reviewed intervals from
[confirmed unavailability](confirmed-unavailability.md) into every subsequent plan in its session.
No separate memory ID selection or JSON authoring is needed for these intervals.

## Add, review, change and cancel

After the ordinary task review, the menu accepts:

- `busy` / `недоступность`: add an interval.
- `change N` / `изменить N`: replace the displayed constraint U<number> with a reviewed version.
- `cancel N` / `отменить N`: explicitly cancel that constraint.
- Existing `done N` / `готово N` and `quit` / `выход` retain their task meanings.

Task indices and constraint indices are separate. Enter `change 1` for U1; a bare number is
not accepted. Up to eight constraint families are retained, including cancelled ones. Changing a
cancelled family can reactivate it, but still requires a new statement and explicit confirmation.

Adding the first interval asks for four answers after `busy`:

1. Your own statement describing the unavailability.
2. An explicit IANA planning timezone, such as `Europe/Paris`.
3. Start and exclusive end with UTC offsets, separated by a space, for example
   `2027-04-10T09:00+02:00 2027-04-10T10:00+02:00`.
4. `CONFIRM` or `ПОДТВЕРЖДАЮ` after reviewing the statement, owner/project, endpoints,
   timezone, status and source identity. Any other answer cancels without writing those records.

Examples are synthetic. The planning day already belongs to the session. The chosen timezone is
persisted and reused, never inferred from machine settings or the person's location. Later adds
or changes ask for the statement and endpoints, without asking for the zone again. A cancellation
asks for the user's cancellation statement and confirmation, and preserves the previous endpoints.
To change the planning zone or day, start a new session; changing either silently on resume is not
supported. Offsets must match the chosen zone, including DST. Nonexistent times are rejected;
explicit offsets distinguish repeated hours. Typos return to the same prompt, with bounded retries.

The interface states that unavailability does not imply location. The source is the exact statement
entered by the user in this dialogue, not a model-generated explanation or an assertion about
travel or an empty home. The statement and interval share a digest and event time, with distinct
immutable IDs. If the operator is testing a scenario, use an isolated test owner/project rather
than attributing test statements to the real user.

## Plan behavior and coverage

Before any interval is recorded, the interface explicitly says that availability is not verified.
It does not claim that the person is free. Every subsequent plan includes all constraint families
saved in the session, following their current unambiguous versions. The display lists U1, U2, etc.,
status, interval, zone, record and source IDs. Deferred tasks name unavailability as the reason.
The Markdown plan keeps the same detailed provenance and revision mappings as the explicit CLI.

This remains a daily focus plan, not a timetable. Any active overlap conservatively defers all
otherwise eligible tasks for the day. A one-hour absence does not imply all-day unavailability,
but no task is placed into an assumed free window. Useful partial-day scheduling would require
explicit working windows and assigned task times, which this session does not have.

Only the session's selected tasks and constraints are considered. It is not a complete calendar,
and it does not import constraints from other databases or discover earlier chat statements.
The optional local model is used only for the original task proposal; interval input, correction,
cancellation and planning make no model calls or external requests.

## Recovery and trust boundary

Confirmed source and interval identities, exact fields, event time and parent are journaled
before either memory write. The source quote is captured first, then its typed assertion, through
the existing MemoryGate write path. Only after both succeed does the session acknowledge the
update and clear the pending journal. The same session lock prevents overlapping dialogues.

If a process dies after either write, resuming shows the exact pending statement and fields and
requires confirmation again before retrying the same immutable IDs. Declining a pending retry
stops the session and keeps the pending entry: it is not safe to assume the earlier write failed.
This also applies to corrections and cancellations. A completed retry does not add duplicate
constraint families. Changed immutable data, conflicting revisions or unavailable sources stop
planning; they are not silently discarded to release the restriction.

Older v1 sessions without availability fields can be resumed. New sessions and availability
updates use v2. Older executables reject the new fields rather than silently ignoring constraints;
use the updated executable to resume. The private workspace retains the statements, memory,
journal and all generated plan drafts. It remains trusted against other code under the same OS
account: the confirmation phrase is not authentication against that code.

## Verification

Subprocess tests cover English and Russian menu commands, confirmation rejection, changes,
cancellation/reactivation, multiple constraints, exact end boundaries, DST rejection, source
provenance, old-session compatibility and crash recovery after each source/assertion write in
all three operations. Existing task completion and session-lock tests still apply.

This adds no service, notification, credential access or execution permission. Saved plans are
snapshots, not live execution guards; refresh/review before any later action.

A scripted two-language pilot used one launch and 17 answer submissions for task intake,
adding an interval, moving it, cancelling it, completing the task and exiting. The first
interval needs four answers after its menu command, a change three, and cancellation two.
Each language produced planned minutes `10 → 0 → 10 → 10 → 0`; a separate resume retained
completion and cancellation. Runtime was 0.324 s (English) and 0.288 s (Russian), excluding
human time and model inference. These are synthetic, scripted decisions, not a usability study.
Setup paths/flags are excluded.
[Archived results](../measurements/2026-11-guided-unavailability.json).
