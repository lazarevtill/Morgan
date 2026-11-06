# Account-approved local notes

Status: implemented as an opt-in, bounded POSIX workflow on 2026-11-06.

## Concrete scenario

The owner approves an exact Markdown note, for example today's priorities. Morgan delivers it
into a private local folder. If the process dies after delivery, the next invocation reads that
actual file and reconciles the journal without writing it again. This exercises real authority,
execution, restart and stop controls without credentials, a model, or an external service.

This is a useful execution stage toward a personal agent. It is not yet an always-on assistant:
there is no planner, launchd service, untrusted inbox ingestion, notification delivery, or paid
provider. Existing memory CLI/MCP behavior is unchanged.

## Trust and authority

`NoteWorkspace` requires a directory owned by the current effective UID with no group/other
permissions. It pins the directory by descriptor; its device/inode identifies the receipt and
approval namespace. Regular files are checked for ownership/private modes and opened without
following symlinks. The journal and SQLite sidecars are checked before opening SQLite.

The owner invokes `approve` explicitly. The private `.approval.json` contains exact scoped task
IDs, action, payload and expiration, bound to this workspace. The worker reloads it before each
effect and reconciliation. Memory, the task journal and planner output never create grants.
The CLI displays the exact request and asks for `APPROVE`; `--yes` is explicit authorization for
trusted automation. Approving replaces the active worklist; the Python API accepts up to 16
notes belonging to one user/project. Each approval lasts at most 24 hours when issued by the API.

The **OS account is the trust boundary**, not a cryptographic identity or proof that a human
clicked approval. Another process with the same UID can replace the policy, alter a note, bypass
the lock, or call the SDK directly. Do not run untrusted planner code as this account and claim
isolation. Trusted local filesystem ancestors are required: SQLite opens by path, and this is
not protection against malicious ancestor renames, privileged users or filesystem tampering.
A copied workspace has another identity and requires new explicit approval; historical receipts
must not be silently rebound after relocation or restore.

## Execution and recovery

- `.worker.lock` uses a kernel advisory lock throughout an invocation. A second cooperating
  worker and `resume` are refused while it is held. Process death releases the lock.
- `.tasks.db` is an isolated task journal, never Morgan's memory database. Grants are not read
  from it. Changing an existing task ID's input fails atomically.
- Only `write-markdown-v1` is registered. No shell, dynamic plugin, model or network call exists.
- The note filename is derived from the exact provider/task/attempt idempotency key. UTF-8
  contents are limited to 16 KiB. A private temporary file is flushed and fsynced, then linked
  into its final name without overwrite, followed by directory fsync. Existing entries are
  verified rather than replaced. Interrupted temporary files are retained, not auto-pruned.
- On restart, holding the lock proves no prior cooperating worker still owns this workspace.
  Approved `running` tasks are fenced as unknown. Receipt lookup reads the actual Markdown file
  and binds its contents to the original specification before committing success/evidence.
- Missing or modified files do not authorize a retry. An unknown outcome blocks later work.
  A completed task is never replayed, even if someone later deletes its file. After inspecting
  the situation, the owner can explicitly approve a new task ID; Morgan does not do this itself.

Receipt evidence is a trusted local adapter's observation, not independent proof against the
owning account. `fsync` provides the operating system's durability semantics, not a guarantee
against every hardware/filesystem failure. Locks and recovery assume all journal users follow
this worker protocol; do not mix direct SDK writers with a running local worker.

## Limits and stop control

Defaults: one new action, one cycle, ten seconds. Explicit bounds allow at most 16 new actions,
60 cycles, 3600 seconds, and polling between 0.05 and 60 seconds. Limits apply per invocation,
not as a global spending quota. There are no paid calls. Recovery lookups do not count as new
actions, but remain within the bounded worklist/cycles and runtime checks.

Any `STOP` directory entry prevents further dispatch. `stop` persists it; `resume` explicitly
clears it while the worker is inactive and does not start work. During polling it is noticed at
the next cycle (at most the chosen interval under normal operation). Checks and deadlines are
cooperative. Filesystem operations can block, and a STOP arriving after the final permission
check cannot retract an already-started publication. No rollback or hard process deadline is
claimed. SIGKILL recovery is tested; an ordinary CLI interruption also releases the lock.

Worker JSON includes reason/action/recovery counts without note contents. `cycle_limit` means
all selected cycles finished, not an assertion that every historical task in the journal is done.
`action_limit` leaves further tasks pending. STOP/denial/deadline/unresolved states exit CLI code 2;
malformed input, storage errors or mismatched receipts fail visibly instead of being swallowed.

## Reproduce end to end

Use a scratch parent you trust and the installed repository Python environment:

```bash
scratch=$(mktemp -d)
printf '# Today\n\nReview priorities and take a walk.\n' > "$scratch/today.md"
python -m morgan_brain.local_notes_cli "$scratch/agent" approve \
  --user demo --project personal --task today --note "$scratch/today.md" --yes
python -m morgan_brain.local_notes_cli "$scratch/agent" run
python -m morgan_brain.local_notes_cli "$scratch/agent" run
cat "$scratch"/agent/note-*.md
python -m morgan_brain.local_notes_cli "$scratch/agent" stop
python -m morgan_brain.local_notes_cli "$scratch/agent" run  # exit 2, stopped
python -m morgan_brain.local_notes_cli "$scratch/agent" resume
```

The first run reports one action; the second reports zero. The output file contains exactly the
approved Markdown. For bounded polling, explicitly use `run --cycles 10 --seconds 60 --poll 2`.
The worker can observe replacement approvals during those cycles. No background service is
installed and the invocation always has finite limits.

## Validation and next boundary

Unit tests cover exact delivery, restart without rewrite, permissions/expiry/revocation, copied
policy rejection, STOP/resume, competing locks, path traversal and symlinks, changed inputs,
missing/mismatched effects, action limits and deadline checks. Process tests kill a child before
and after the real effect, then verify no replay and persisted reconciliation evidence. A CLI
integration test runs approve, delivery, repeat, stop and resume in an isolated directory.

Owner-controlled `enqueue` intake and lifecycle observation are described in
[local note intake](local-note-intake.md). Next: a separately isolated planner that cannot write
approvals. Continuous operation also needs a chosen supervisor, health reporting,
and explicit operational/budget policy for any real provider. None is silently activated here.
