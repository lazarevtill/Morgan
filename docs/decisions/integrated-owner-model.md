# One foreground owner/model cycle

Date: 2026-11-06. Extends the [native owner review](native-owner-review.md), without
installing a service or asserting 24/7 availability.

## Problem and behavior

The owner previously ran `model_cycle` separately, found its proposal file, then returned
to owner-input review. `owner_ui --server ABSOLUTE_BINARY --model ABSOLUTE_GGUF` now offers
**Предложение локальной модели** for a saved message in the same review flow. No model
starts just because the interface opens or receives text. The owner explicitly confirms
starting a local operation; its progress menu offers refresh or **STOP: остановить модель
и ввод**. Cancel also requests STOP. Completed drafts appear inside the review flow.
Empty model selection is explicitly described as fallible, not proof of no obligations.

After reading the result, the owner may fill the task manually or leave the message for
later. The existing exact task review and confirmation are still separate. Model output
never chooses a renewal target, grants execution authority or commits a task. Current
one-line message and seven-day/fourteen-message/eight-task limits still apply. An expired
inbox cannot start inference. This is not general chat or autonomous task execution.

## Reused lifecycle and recovery

`owner_model` starts one daemon Python thread to await the existing asynchronous
`model_cycle`/guardian lifecycle, leaving the native UI free to receive STOP. It introduces
no model server implementation, scheduler, network listener or additional memory store.
All messages share the owner workspace's supervisor/worker locks and STOP marker, so
`owner_input WORKSPACE stop` also reaches a guardian surviving a crashed UI. The owner
must explicitly resume after STOP; starting a model never clears STOP. Only one model
can run in that workspace, including after a UI crash. This is not a machine-wide lock:
independent workspaces remain independent workloads.

A cycle uses the smaller of 90 seconds and the remaining foreground session budget,
with at least five seconds required to start. Four CPU threads, zero GPU layers,
1024 context and batch 64 remain the existing defaults. No API keys or downloads are
introduced. These are requested resource limits, not hard RSS or CPU percentage quotas.
An open native dialog still has its own finite timeout; progress refresh is manual.

`run_cycle(cache_by_source=True)` stores an immutable `proposal-SHA256.json`, keyed by
exact source bytes; default standalone CLI behavior remains `proposal.json`.
Results are validated against the exact note and must remain grounded and unapproved.
They are reused on restart without inference. A changed message cannot reuse another
source's result. Cached proposals are historical drafts, not re-evaluations by a changed
model; choosing different weights does not silently invalidate or recompute a saved draft.
An attempt marker with no result and no active lock is shown as interrupted. There is no
automatic retry. A running operation is identified by locks, never by trusting a saved PID.

Ordinary UI exit requests STOP if an operation is still active, then waits up to six
seconds for its thread while retaining the exclusive UI-session lock. A new UI cannot
start between releasing the old session and its cleanup. A stopped or failed operation never
authorizes a task. The existing
independent guardian retains its finite expiry if the UI is forcibly killed. Killing both
UI and guardian can still orphan the native model; external OS supervision is not installed.
Foreground deadlines are not a tested wall-clock guarantee across Mac sleep. Resume,
logout/reboot recovery and continuous headless reception remain separate deployment work.

## Launch

Use a verified installed revision containing `owner_model`, with the existing private owner
workspace. Add `--init-user OWNER --init-project PROJECT --init-timezone IANA_ZONE` only
when creating a new pilot. Keep server/model paths explicit and trusted:

```sh
/absolute/runtime/bin/python -I -m morgan_brain.owner_ui /absolute/private/pilot \
  --server /absolute/trusted/llama-server --model /absolute/existing/weights.gguf --seconds 600
```

No cloud fallback exists. Without both model flags, the old manual interface is unchanged.
One shared stop command is sufficient:

```sh
/absolute/runtime/bin/python -I -m morgan_brain.owner_input /absolute/private/pilot stop
```

Readiness remains limited to the logged-in foreground desktop. There is no persistent
credential, installed LaunchAgent, phone delivery or incoming-message service.

Measured real-model and synthetic lifecycle evidence: [pilot measurement](../measurements/2026-11-06-integrated-owner-model.md).

The owner menu reports pending review and saved, unapproved draft counts. The
pending-message list distinguishes a ready draft, a failure, an interrupted attempt,
and an unreadable cache without starting inference. A busy shared model lane does
not claim that every listed message is running. Dismissed/applied messages do not
inflate pending counts. These are on-open snapshots, not push notifications or proof
that the background process is alive; no retry or task confirmation is implied.

The separate `Читать черновики` action reads validated saved drafts for messages
still awaiting review, including under STOP or expiry. It offers no start, retry,
resume or confirmation action. An empty model selection is explicitly fallible;
a saved proposal is not presented as the current verified state of a task.
Unreadable caches are counted and excluded. Cancelling returns to the menu.
