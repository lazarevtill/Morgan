# Local worker diagnostics without replay authority

Status: implemented, 2026-11-06. Builds on
[local intake](local-note-intake.md) and [receipt recovery](local-approved-notes.md).

## Concrete problem

After an interrupted invocation, `status` could report an inactive lock but could not explain
whether a run had finished or died during delivery. Operators also lacked a scoped view of
pending/running/unknown tasks without opening the database themselves.

The explicit `diagnose --user OWNER --project PROJECT` command now returns three blocks:
control state, the latest invocation observation, and journal status counts for that exact
owner/project. It reads no note text, task IDs, action payloads, result strings or receipt bodies.
There is no external telemetry endpoint, extra database, service, model or credential.

## One bounded observation

The worker atomically saves a private `.run.json` under its existing exclusive lock. It contains
only a random invocation ID, UTC timestamps, a fixed state/phase/reason vocabulary, action and
recovery counters, configured finite limits, and the workspace namespace. Updates occur at
start, polling, dispatch, reconciliation and exit. The file records only the latest invocation;
it is overwritten on the next invocation and is not an append-only activity archive.

`starting`, `polling`, `dispatching`, `reconciling` and `finished` describe cooperative boundaries.
The timestamp is not an independent heartbeat: a blocked filesystem call can stop updates while
the process still holds its lock. There is no timeout-based lock stealing or guessed PID cleanup.
Clock changes can affect timestamps; they never affect execution authority or recovery decisions.

Normal exits preserve the existing finite reason (`stopped`, `deadline`, `action_limit`, etc.).
Exceptions are reduced to a fixed category such as `storage_error`, `receipt_mismatch` or
`cancelled`; exception strings, private paths and SQL are never written to the observation.
Persisted metadata is bounded and validated before display, including vocabulary and limits.

Action counters count charged dispatch attempts, not independently verified filesystem effects.
During a crash a record can lag a committed result or precede an actual claim. Recovery counters
likewise reflect the last recorded boundary. Journal state and receipt lookup remain authoritative.

## Interruption and controlled recovery

When a running observation is read while the sampled worker lock is inactive, diagnosis labels
it `interrupted`. It retains `recorded_state=running` so the distinction is explicit. Lock state,
observation and journal counts are separate snapshots, not an atomic global transaction. A new
worker can start between reads. Diagnosis never fences tasks, changes grants, clears STOP,
reclaims a lock, or repeats an effect.

The operator can inspect the scoped counts and invoke the existing finite `run` command when
ready. Only that runner holds the worker lock and applies the existing exact-approval receipt
reconciliation. Missing/mismatched notes still block unknown attempts; no retry/reset was added.
A new run replaces the previous observation, while the durable task journal preserves execution
state and saved receipt evidence.

## Diagnostic storage failures

Start metadata is saved before opening the journal. Failure there prevents dispatch. An error
at a later boundary stops the invocation visibly. Failure saving the final observation may occur
**after** a note and its journal success committed: diagnosis can then say interrupted even
though the scoped journal count says succeeded. A repeat invocation skips that completed task.

When an execution error is already propagating, a secondary filesystem failure while recording
it cannot mask the primary error. Observability is not a distributed transaction with effects;
it cannot establish rollback or absence of an effect. Original CLI execution errors still
propagate normally, while the observation contains only the fixed failure category.

## Read-only scoped journal inspection

`TaskJournal.status_counts` uses SQLite URI `mode=ro`, validates the application/version, and
selects only status/count grouped within the explicitly supplied user/project. It accepts known
journal versions without migrating them, rejects foreign/future formats, and never creates a
missing database. The local adapter validates private regular files and SQLite sidecars first.
SQLite may manage WAL reader coordination metadata; this is not an immutable-file forensic read.

The latest invocation record describes the workspace runner, not a particular project. Scoped
counts describe only the requested project. Approved worklist size in the controls block is not
a pending/completed count. Existing same-account and trusted-filesystem assumptions still apply.

## Reproduce

Use the scratch workspace from the intake example:

```bash
python -m morgan_brain.local_notes_cli "$scratch/agent" diagnose \
  --user demo --project personal
python -m morgan_brain.local_notes_cli "$scratch/agent" stop
python -m morgan_brain.local_notes_cli "$scratch/agent" run  # reports stopped, exit 2
python -m morgan_brain.local_notes_cli "$scratch/agent" diagnose \
  --user demo --project personal
```

The latest run shows `finished` / `stopped`. To continue, use explicit `resume` and finite `run`.
A diagnosis for another owner/project returns only that scope's journal counts. Commands do not
send note data anywhere and do not enable a permanent process.

## Validation and next boundary

Process crash tests now inspect `interrupted` plus a running journal task before recovery, then
verify the final recovered count or blocked unknown outcome. CLI integration checks scoped
counts without note text. Tests cover STOP, cancellation, isolated scopes, missing/foreign/v1
journals, symlinks, malformed metadata, failed diagnostic writes before dispatch and after a
completed effect, and preservation of the original exception when its diagnostic write fails.

Remaining work for continuous operation is still an explicit supervisor/health policy and an
isolated planner/proposal intake. These observations are useful to such an operator but do not
silently activate or implement that service.
