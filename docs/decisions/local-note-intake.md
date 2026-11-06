# Local note intake and lifecycle

Status: implemented, 2026-11-06. Extends the concrete
[account-approved note workflow](local-approved-notes.md).

## Problem and resulting behavior

Previously `approve` replaced the whole worklist, so submitting another note could remove an
undelivered note's authority. A worker without an approval file exited immediately. The new
`enqueue` command appends an exact owner-approved note, and a finite worker can wait for it in
an initially empty workspace. This is trusted account-operated intake, not automatic approval
of an email, remembered instruction, model proposal or arbitrary watched file.

The separate private `.approval.lock` serializes approve/enqueue/revoke operations. Writers
wait on that lock for their short read/validate/atomic-publish operation; process death releases
it. The worker never takes that lock and continues reading complete atomic policy snapshots.
As with filesystem I/O generally, a stuck writer/filesystem can delay another writer; no hard
intake deadline is claimed. Worker invocation bounds remain unchanged.

An identical queued request is a no-op, including after an acknowledgement was lost. A changed
payload/action under an existing ID fails without modifying the policy. The bound remains 16
notes in one user/project scope, including completed entries until the owner explicitly replaces
or revokes the worklist. Overflow fails; nothing is silently evicted. `approve` remains the
explicit replacement operation, and its help text now says it replaces all grants.

## Expiration and revocation

The existing policy format has one expiry for its whole worklist. Enqueue conservatively uses
the earlier of the existing expiry and the new request's TTL. It never extends older authority;
a shorter new TTL also shortens the old grants. Exact duplicate enqueue does not change expiry.
An expired or malformed policy is not revived: inspect and explicitly replace it with `approve`.
This avoids a new policy schema or an implicit renewal mechanism for this small bounded workflow.

`revoke` removes active grants, preserving notes, task history and STOP. Before an effect, missing
authority is a permission denial. When idle, the missing file simply means there is no work;
the worker may observe a later explicit grant within its remaining cycle/runtime budget.
Revocation is cooperative and cannot retract an effect after its final permission check.

## Lifecycle controls

`status` reports worker lock ownership, STOP presence, approval validity/expiry and approved
worklist size without exposing note text or opening/migrating the task journal. It may create
the private workspace/lock metadata. Counts are approved entries, not pending or completed task
counts. This is a momentary observation, not a reservation; another process can change state
immediately afterward. Malformed or unsafe policy files fail visibly.

SIGTERM to the explicitly invoked CLI runner requests persistent STOP through the event loop.
The next cooperative boundary ends the invocation with reason `stopped` and exit code 2.
The signal handler is removed on exit. Polling latency and blocking filesystem operations still
apply; failure to persist STOP is reported by the event loop, and finite invocation limits remain.
No process is killed by a stale PID file, no daemon is installed, and no stopped worker is
restarted automatically. `resume` still requires an inactive worker and only removes STOP.

Process death releases both locks. Existing recovery still reconciles a started delivery against
the real Markdown file; absent or mismatched evidence blocks replay. Enqueue/revoke never erase
that history. A later explicit reauthorization cannot reset an uncertain execution attempt.

## Reproducible live intake

In an isolated scratch directory, start a finite worker, then submit two approved notes:

```bash
scratch=$(mktemp -d)
printf '# Morning\n\nReview priorities.\n' > "$scratch/morning.md"
printf '# Evening\n\nWrite a short retrospective.\n' > "$scratch/evening.md"
python -m morgan_brain.local_notes_cli "$scratch/agent" run \
  --cycles 60 --seconds 30 --poll 0.5 --actions 2 &
worker_pid=$!
python -m morgan_brain.local_notes_cli "$scratch/agent" enqueue \
  --user demo --project personal --task morning --note "$scratch/morning.md" --yes
python -m morgan_brain.local_notes_cli "$scratch/agent" enqueue \
  --user demo --project personal --task evening --note "$scratch/evening.md" --yes
python -m morgan_brain.local_notes_cli "$scratch/agent" status
# Inspect the two note-*.md files after a polling cycle, then request stop:
kill -TERM "$worker_pid"
wait "$worker_pid"  # normally exit 2 after cooperative STOP
python -m morgan_brain.local_notes_cli "$scratch/agent" status
python -m morgan_brain.local_notes_cli "$scratch/agent" resume
python -m morgan_brain.local_notes_cli "$scratch/agent" run  # zero new actions if both finished
```

Stopping immediately after submission can prevent delivery; inspect files or wait for them
before the demonstration's signal. Retain the PID only for this child invocation, never as a
persistent control mechanism. The integration test synchronizes on actual worker/file state.

## Evidence and remaining boundary

Tests launch eight concurrent producer processes and verify all eight requests survive. Another
process exits after publishing a policy; resubmission leaves exactly one grant and unchanged
expiry. The CLI lifecycle test starts empty, accepts two tasks, observes actual notes, sends
SIGTERM, verifies inactive/STOP state, resumes and verifies zero duplicate deliveries. Unit tests
cover capacity, conflicting IDs, scope, expiry preservation, revocation before effects, missing
policy, live intake, status and idle deadlines. Existing crash-before/after-effect tests remain.

The OS account remains trusted. This is not isolation from a planner running with the same UID.
Continuous service supervision, untrusted proposal intake and planner isolation are still future
work requiring a concrete deployment boundary. No credentials or external API are involved.
