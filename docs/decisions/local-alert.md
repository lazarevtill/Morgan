# Explicit local macOS alert

Morgan can display one account-approved message through the existing macOS `osascript`
executable. It installs no service, permission grant, shortcut, app or credential. It makes
no model or network call. This is a foreground delivery adapter, not continuous operation.

## Choice of channel

| Existing facility | Available on this Mac | Receipt limitation |
| --- | --- | --- |
| Markdown workspace | Yes | File existence proves storage, not attention |
| AppleScript notification banner | Yes | Submission does not prove visibility or reading |
| AppleScript dialog | Yes | Returns timeout, dismissal or explicit button response |
| UserNotifications app | Framework available | Requires an app integration/authorization; delivered list only covers notifications still present |
| Shortcuts CLI | Yes | Requires a separately configured shortcut; no inherent read receipt |
| terminal-notifier | Not installed | Additional dependency; not a human read receipt |

Use a bounded native dialog for this small pilot. It interrupts the desktop, so it is only
suitable for explicitly requested alerts. Default button is **Dismiss**, not Acknowledge.
The button records interaction reported by the OS, not understanding, authenticated owner
identity, task completion, or permission to execute another action. No sound is requested.

Apple references, checked 2026-11-06:

- [Displaying notifications](https://developer.apple.com/library/archive/documentation/LanguagesUtilities/Conceptual/MacAutomationScriptingGuide/DisplayNotifications.html)
- [AppleScript commands and dialog results](https://developer.apple.com/library/archive/documentation/AppleScript/Conceptual/AppleScriptLangGuide/reference/ASLR_cmds.html)
- [Notifications still in Notification Center](https://developer.apple.com/documentation/usernotifications/unusernotificationcenter/getdeliverednotifications(completionhandler:))

## One explicit invocation

Use a separate private workspace, never the memory database. The local caller approves the
exact stdin text, owner, project, task ID and duration by invoking `send`. The task ledger
is data, not a standing authorization. Do not expose this command as an unreviewed model tool.

```sh
printf '%s' 'Synthetic reminder: review your local plan.' |
  python -m morgan_brain.local_alert_cli /private/path/alerts send \
    --user demo --project personal --task-id reminder-1 --seconds 15
python -m morgan_brain.local_alert_cli /private/path/alerts stop
python -m morgan_brain.local_alert_cli /private/path/alerts resume
```

The duration is 1–30 seconds plus up to five seconds for process startup/response. The caller
must supply an available GUI session. Local text is passed as an argv value to a fixed script,
never interpolated into source or a shell; it can be visible to local process inspection and
on the desktop. Do not include secrets. Text is bounded to 2,000 characters. The private task
journal retains the exact text and response; memory retention/forget does not govern it.

A `succeeded` journal state means the handler returned a known response. Read the result:
`acknowledged: false` with `timed_out` or `dismissed` is **not confirmed delivery to a person**.
Only `response: acknowledged` reports the explicit button response. No real human
acknowledgement was measured in this pilot.

Same-ID exact replay returns the previous state without displaying again. Changed content
under the same ID is rejected. A running or unknown attempt is never retried automatically;
there is no native lookup that can safely settle a missing response. Do not generate a new ID
just to bypass uncertainty. The operator must decide whether a new alert is warranted.

STOP before launch prevents the effect. While waiting it is checked every 0.1 seconds;
STOP/cancellation kills and reaps the child, and the journal records uncertainty. This cannot
retract what was already seen. Resume clears STOP while holding the existing locks; it sends
nothing. Concurrent cooperating invocations share the workspace lock. SIGKILL of the parent
can leave the dialog until its native timeout; the running journal entry prevents replay.
There is no claim of hard real-time shutdown or a complete GUI-process sandbox.

## Evidence and limits

On macOS 15.8.2 the initial native probe returned `timed_out` in 3.113 seconds. The complete
production CLI/adapter/journal path then returned `succeeded`, `timed_out`, `acknowledged:false`
in 2.207 seconds for a two-second synthetic dialog. Same-ID replay returned the stored result;
STOP refused a different task before launch. No real personal information was used.

Automated tests use a controlled subprocess seam for acknowledgement/dismissal, STOP during
an attempt, invalid output, missing authority, duplicate IDs and failure after display before
journal commit. These are simulations of UI responses, not measurements of a user reading.
The native timed-out run confirms OS execution and return handling, not screen visibility.

There is still no independently running 24/7 process, outbound chat channel, phone delivery,
quiet-hours policy or automatic mapping from model plans to authorized alerts. Integrating
those requires a concrete delivery/authorization policy; this adapter does not supply one.
