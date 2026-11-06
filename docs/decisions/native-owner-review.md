# Bounded native owner review on macOS

Date: 2026-11-06. This is an opt-in foreground interface to `owner_input`,
not an always-running personal agent or a reminder delivery channel.

## Decision and alternatives

| Surface | Benefit | Cost / limitation |
| --- | --- | --- |
| Existing owner-input CLI | Scriptable, exact receipts, already tested | Owner copies identities and review digests across commands |
| Existing guided-plan dialogue | Existing text conversation and completion support | Requires terminal interaction; model drafts require separate validation |
| macOS standard dialogs (selected) | Built into the target Mac, no listener or extra UI dependency | Sequential fields, one-line input, macOS only, limited layout |
| Browser form | Better multiline layout | Adds local server lifecycle, request authentication and CSRF protection |
| Tk / SwiftUI application | Richer native forms | Additional UI packaging and maintenance for this small pilot |

Use Apple's existing [text dialogs](https://developer.apple.com/library/archive/documentation/LanguagesUtilities/Conceptual/MacAutomationScriptingGuide/PromptforText.html)
and [list selection](https://developer.apple.com/library/archive/documentation/LanguagesUtilities/Conceptual/MacAutomationScriptingGuide/ManipulateListsofItems.html).
The fixed JXA program receives JSON data, never interpolated source or shell commands.
The Python controller calls the same owner-input command implementation as the CLI;
there is no second journal or task executor. Without optional model flags it performs no inference.
[The optional integrated model flow](integrated-owner-model.md) keeps draft generation separate
from task confirmation.

## Run

Build/install a trusted revision containing `morgan_brain.owner_ui` using
[the pinned-runtime procedure](pinned-runtime.md). Verify its expected SHA first.
With that runtime's absolute Python path:

```sh
/path/to/runtime/bin/python -I -m morgan_brain.owner_ui /absolute/private/pilot \
  --init-user OWNER --init-project personal --init-timezone Europe/Belgrade
```

Initialization creates a new private pilot dated today in the chosen timezone with a
30-minute planning budget. Omit all three init flags to reopen the same pilot. Existing
pilots are never reset by initialization. Do not use a stale editable installation.

Choose **Новое сообщение**, enter a short message, explicitly choose new task, renewal or completion,
review fields and press **Подтвердить** on the complete source/owner/project/action preview.
Renewal requires selection of an exact open task. Ambiguous text does not choose a task.
**Проверить сообщения** resumes saved messages and exact unfinished reviews.
**План** displays current tasks. To mark a task done, submit the completion statement, choose
**Завершить задачу**, select one exact open task and confirm the complete review. Completion
changes only status; title, date, priority and minutes stay fixed. Cancel leaves it open.
The saved review can be resumed through **Проверить сообщения** after interruption.
Completion reuses the paired source/revision write path, not a new executor or database.
`owner_input review ID --complete CURRENT_TASK_ID` exposes the same explicit CLI review.
Neither a model draft nor completion-like text selects or completes a task automatically.

A completion uses a `guided-complete-` event identity in the existing transition envelope.
Legacy renewal digests and payloads are unchanged. Older runtimes cannot read new completion
reviews; after using this feature, do not downgrade the runtime against that journal.

## Authority, cancellation and recovery

- The local OS account and private workspace are the trust boundary. UI text and digests
  are not authentication against malicious processes running as the same account.
- The complete persisted review is shown before its digest is submitted. A review changed
  by another process is rejected; task target staleness and immutable replay stay in the core.
- Default/cancel is **Отмена**, never approval. Closing a form retains its submitted message;
  unsaved field edits may be lost. Reopening never auto-applies pending work.
- A partial apply requires explicit reconfirmation of the exact saved review, using the
  existing idempotent source/task writes. A lost acknowledgement does not create another task.
- **STOP** in the menu or a standalone submitted STOP/СТОП persists the existing stop marker.
  An external `owner_input WORKSPACE stop` interrupts a pending form. Resume needs an explicit
  confirmation and does not execute tasks. Menu/status remain available while stopped.
- Only one UI session holds `.owner-ui.lock`; the existing short command lock still protects
  shared state. No database write transaction remains open while a person reads a dialog.
- Default lifetime is 600 seconds; accepted range 5–1800, maximum 30 menu iterations and
  120 seconds per dialog. Timeouts and late responses do not authorize writes.
- Dialog children are terminated on ordinary timeout, STOP and cancellation. A hard SIGKILL
  of the parent can leave a list dialog visible; it has no remaining controller to commit
  a response. This is not a crash-proof desktop service.
- Text is visible on the local screen and in the native child argv to processes permitted
  to inspect it. Use a trusted OS session. No HTTP listener, credentials, external sends,
  notification permissions or paid calls are introduced.

The existing pilot limits remain: seven days, fourteen messages and eight task families.
The planning day is fixed, not a rolling scheduler. This UI does not extract fields with an
LLM by default, infer ownership, deliver reminders, install launchd jobs or demonstrate 24/7 availability.

## Evidence

See [the measured synthetic workflow](../measurements/2026-11-06-native-owner-review.md).
