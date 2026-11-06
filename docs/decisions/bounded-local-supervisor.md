# Finite local worker supervision

Status: implemented as an explicitly invoked POSIX command, 2026-11-06.

## Concrete purpose

The local-note worker already reconciles interrupted tasks but previously required an operator
to start the next process. `supervise` now owns a fixed worker child, detects selected signal
crashes, and starts a replacement within finite start/time budgets. Tests exercise real process
death before and after the actual Markdown effect; this is more than a simulated retry callback.

This does not install launchd/systemd, create a background service, survive machine reboot, or
connect credentials/models/providers. The command exits. Run it only in a selected private
workspace with the installed Morgan package; the child uses the current Python executable with
`-I -m morgan_brain.local_notes_cli`, ignoring import overrides from the working directory and
Python environment variables. There is no caller-configurable executable or shell command.

## Restart and action budgets

Defaults: at most **3 starts** including the initial start, **60 seconds** of supervision,
**2 seconds** shutdown grace and **0.1 seconds** between crash restarts. Hard configuration bounds:
1..5 starts, up to 3600 seconds, 0.05..10 seconds grace, 0.05..60 seconds backoff.

The ordinary worker limits remain per child. `--actions` is explicitly a **per-worker** dispatch
budget; the whole invocation can charge at most `--starts * --actions` attempts (default 3,
maximum 80). Each child can perform at most 16 actions and 60 cycles. Recovery lookups do not
repeat the effect and do not count as new dispatch. These are invocation budgets, not a durable
quota across separate operator invocations. No paid calls are present.

Only SIGKILL, SIGSEGV and SIGABRT exits are eligible for restart. A normal exit, action/cycle
completion, permission failure, unresolved receipt, startup error, SIGTERM or SIGINT ends
supervision. Normal finite completion never silently becomes a periodic service. Start capacity
is consumed before spawning, so failed launches cannot generate uncounted retry loops.

## Ownership and sequential shutdown

A private `.supervisor.lock` excludes another cooperating supervisor for the entire invocation,
including backoff. The child's existing `.worker.lock` independently excludes concurrent task
execution. A direct worker already holding that lock makes the new child fail; the supervisor
stops instead of replacing or killing the unrelated worker. `resume` now requires both locks,
so it cannot clear STOP during a supervised backoff gap.

STOP is checked before launch, during child monitoring and during backoff. SIGTERM to the
supervisor CLI creates STOP through its event loop. On STOP or total deadline with a live child,
the supervisor persists STOP, sends SIGTERM to its owned child, waits for grace, then uses SIGKILL
if needed and waits for reaping. It never restarts after this shutdown. It holds its lock until
shutdown completes. Cancellation during child creation is shielded long enough to retain the
child handle and terminate/reap it; cancellation is then propagated.

Only the live subprocess object is used for signals. Saved PIDs are diagnostic values, never
restart/kill authority. The fixed worker does not spawn grandchildren; this is not a general
process-tree sandbox. If a post-kill reap exceeds another grace interval, shutdown fails visibly
and no new child is started. Uninterruptible OS/filesystem operations remain outside a hard
wall-clock guarantee; ordinary deadline shutdown can add up to two grace waits.

Errors saving supervisor metadata after a child was spawned also cause owned-child cleanup.
Failure to persist STOP still attempts termination. Diagnostic failures do not authorize another
execution. Child stdout is discarded rather than retained as an unbounded buffer; stderr stays
on the invoking terminal. No child output or exception strings are persisted by the supervisor.

## Recovery without duplicate effects

The replacement is the same fixed worker with current account-issued grants. It takes the
worker lock, fences the old running attempt, and reads the actual Markdown file. A matched file
confirms success; absence or mismatch blocks the task. The supervisor never resets task state,
changes task IDs, extends grants or interprets a crash as proof that an effect did not happen.

A test kills a child with SIGKILL after atomic note publication, before journal success; the
replacement confirms the receipt and leaves exactly one file. A corresponding pre-effect kill
leaves an unknown task and zero files, and the replacement exits without retrying that action.

## Observability

A private, atomic `.supervisor.json` keeps only the latest invocation: random run ID, timestamp,
transition phase, start/forced-kill counts, last return code, child PID and configured limits.
It contains no notes, task IDs, captured output or credentials. `diagnose` includes this validated
record and sampled supervisor lock ownership alongside existing worker/journal diagnostics.

An inactive record in a live phase is labelled `interrupted`; this is only an observation.
The supervisor could die while its worker remains alive. Separate lock/metadata reads are not
one atomic snapshot, and stored PID values can become stale. Diagnosis does not kill or adopt
an old process. Journal/receipt evidence, not these counters, determines task completion.

## Reproduce an explicit finite run

After installing this checkout into an isolated Python environment, reuse a scratch workspace
with notes approved through the intake example:

```bash
python -m morgan_brain.local_notes_cli "$scratch/agent" supervise \
  --starts 3 --supervisor-seconds 30 --stop-grace 2 --restart-delay 0.5 \
  --actions 2 --cycles 60 --seconds 25 --poll 0.5
python -m morgan_brain.local_notes_cli "$scratch/agent" diagnose \
  --user demo --project personal
```

From another terminal, explicit `stop` requests sequential shutdown. After termination, `resume`
clears STOP but starts nothing. A normally completed worker ends supervision even if start
capacity remains. To demonstrate crashes, use the isolated automated process tests:

```bash
pytest -q tests/integration/test_local_supervisor.py
```

They inject crash scripts only by monkeypatching a private test seam. Production accepts no
alternative command or crash flag.

## Evidence and remaining gaps

Tests cover real SIGKILL before/after delivery, exhausted restart budgets, non-restarting ordinary
exits, STOP before launch and in backoff, ignored SIGTERM followed by kill/reap, competing
supervisors, denied resume, cancellation during spawn, observation failure after spawn, the
actual fixed worker, CLI SIGTERM, copied/corrupt observations and invalid limits.

This remains a trusted-account local workflow. An uncatchable supervisor SIGKILL or host crash
can prevent cleanup; a surviving child retains its own finite limits and worker lock. A new
supervisor does not adopt or kill it by stale PID, and a competing child fails safely. A future
OS supervisor must define orphan/reboot handling and durable restart/rate budgets explicitly.

Continuous 24/7 deployment still needs that operator-selected supervisor policy, health alerts,
boot/session lifecycle, and isolation of any untrusted planner/proposal source from approvals.
No such service or permanent secret access is silently enabled here.
