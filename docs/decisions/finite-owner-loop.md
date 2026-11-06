# Finite local owner drafting loop

`owner_ui` is interactive and cannot itself provide unattended operation. The explicitly
invoked `python -I -m morgan_brain.owner_loop WORKSPACE --server PATH --model PATH`
now watches the existing owner inbox and prepares one local draft for each unreviewed,
previously unattempted source. It never confirms a task or executes an obligation.

The owner chooses this processing scope by starting the loop. Only messages submitted
through `owner_input` / `owner_ui` enter it; no directories, email or conversation archives
are scanned. The existing pilot permits 14 messages, eight task families and seven days
from initialization. The loop reads the persisted UTC expiry on every poll and never
extends it on restart. Its invocation also has a monotonic limit (default one hour;
maximum seven days). STOP, expiry and SIGTERM cancel the owned model operation.

The implementation reuses the model-cycle guardian, private Unix socket, four CPU threads,
90-second model budget, source-keyed caches and shared model locks. One automatic attempt
marker per source prevents crash retries; an interrupted draft requires explicit owner
review/retry. No model remains loaded while idle. A separate loop lock prevents duplicate
pollers; a persisted cap of 16 launches bounds process restarts. Corrupt state requires
manual recovery and is preserved. Normal completion, STOP, expiry, launch-limit and known
configuration failures exit successfully so launchd does not restart them indefinitely.
Unexpected crashes can be restarted, within that cap. A crash before durable state can be
written is not covered by the counter; launchd throttling is not a total restart cap.

Polling reads validated local session/inbox files without opening memory or a database.
`owner-loop.json` records state changes and a 30-second heartbeat without source text.
`--status` reports **last observed state**, not verified process liveness. Draft failures
retain their attempt markers and are visible in the existing model-review UI as interrupted;
they do not silently become task changes. Raw messages and drafts remain private local files.
Trust is the current OS account, not protection from another process with that account.

## Owner interaction and deployment boundary

Initialize an owner inbox explicitly, then submit one selected UTF-8 line:

```sh
"$python" -I -m morgan_brain.owner_input "$workspace" submit message-1 < selected-message.txt
"$python" -I -m morgan_brain.owner_loop "$workspace" --status
"$python" -I -m morgan_brain.owner_ui "$workspace" --server "$server" --model "$model" --seconds 600
```

In the UI choose a saved message and **Предложение локальной модели** to inspect the ready
draft, then review and separately confirm task fields. This is a pull interface: no automatic
notification, phone delivery or read receipt is claimed. Explicit completion changes the
current plan through the existing MemoryGate path. Automatic draft creation does not solve
known language/ownership extraction errors or make arbitrary task execution safe.

`owner_loop_plan --python PATH --workspace PATH --server PATH --model PATH` prints a
**disabled, uninstalled** `org.morgan.owner-pilot` plist with fixed arguments, `-I`, private
logs, `RunAtLoad`, unsuccessful-exit restart and a 60-second throttle. It does not call
launchctl, install a service, create credentials or change power settings. First verify a
new pinned runtime with `runtime_check --expected-revision SHA`; never use an editable
checkout for deployment. Install/enable only after the concrete pilot authorization arrives.

A user LaunchAgent operates after login, not before login, while logged out or while the
Mac sleeps. SIGTERM cleanup is tested separately; actual logout/reboot/sleep availability is
not asserted. On waking, the next check observes the original expiry. The model guardian
provides a separate finite process deadline; sleep can delay scheduling and cleanup.

Stop processing with `owner_input WORKSPACE stop`. For an installed, explicitly authorized
pilot, disable and boot out `gui/$(id -u)/org.morgan.owner-pilot`, then remove only its plist
from `~/Library/LaunchAgents/`. Preserve the workspace, receipts and pinned runtime for
review; rollback does not erase personal data. `resume` alone does not restart an exited
loop; restarting is explicit and does not reset expiry, attempts or launch count.

No installed service or seven-day endurance claim follows from these tests. This is a
bounded unattended drafting stage with manual review, not a complete 24/7 personal agent.

## End of the installed pilot

The rendered login plan now passes `--login-agent`. This explicit macOS-only mode requires
its fixed 604800-second invocation bound. After cleanup, `expired`, `finished` or
`launch_limit` invokes only `/bin/launchctl disable gui/<current uid>/org.morgan.owner-pilot`,
with a five-second timeout. It does not delete the plist, workspace or runtime. Ordinary
STOP/logout does not disable the job, so explicit resume remains possible within the
original lifetime. Foreground loops without the flag never call launchctl.

If the Mac sleeps or is powered off at expiry, disabling occurs on the next execution
that can observe expiry; no model admission is allowed after that check. A successful
`disable` prevents subsequent login launches. Failure is printed as `autostart_disabled:
false`, without retries; the finite loop still exits and manual rollback remains available.
This is a current-account job control, not a credential or a new service.

An actual deployment on macOS stalled in Python path initialization when the runtime was
under Documents: launchd reported a PID but no application heartbeat appeared. Moving the
owned runtime and empty workspace into a private Application Support pilot directory
resolved startup without granting Documents access or Full Disk Access. The file-open
stack was observed; its underlying OS authorization cause was not conclusively diagnosed.
Use a normal application-data directory for a login deployment and verify a fresh
`waiting_for_owner` observation and clean STOP/restart, not merely a running launchd PID.
