# Decisions and contracts

Use these records for behavior, authority, recovery and compatibility details. Implementation
status is stated in each record; a recorded test or deployment describes its own date and scope.
The author/scope fact-key expansion in decision 0001 remains deferred, and the bounded personal
pilot is an acceptance plan. Neither is a claim that all proposed behavior is implemented.

For setup, start with [wiring](../WIRING.md). For results, use the
[measurement index](../measurements/README.md).

## Memory, provenance and recovery

- [0001: The fact key, and how far `forget` reaches](0001-fact-key-and-forget-reach.md)
- [Prepared stores and committed erasure](0010-prepared-store-erasure-generation.md)
- [Consolidation proposals carry an explicit preparation basis](0012-consolidation-preparation-basis.md)
- [Atomic native chat turns](atomic-chat-turns.md)
- [Explicit revisions and caller-declared consolidation request bytes](consolidation-revision-byte-limit.md)
- [Conservative credential admission](credential-admission.md)
- [Explicit conversation export memory](explicit-conversation-export.md)
- [Optional resumable checkpoints](optional-checkpoints.md)
- [Default chat stops at unresolved revision forks](default-chat-conflict-guard.md)

## Model and reader contracts

- [Opt-in adapter request controls](adapter-request-controls.md)
- [Optional counted model capability](counted-model-capability.md)
- [Opt-in strict native context](strict-native-context.md)
- [Explicit reader status contract](reader-response-contract.md)
- [Synthetic note selection before autonomous task intake](synthetic-note-selection.md)
- [Missed reminders: extraction regression gate](missed-reminder-evaluation.md)
- [A finite, scheduled local model cycle](bounded-model-cycle.md)

## Trusted tasks and local delivery

- [Explicit single-attempt task execution](explicit-task-execution.md)
- [Bounded task batches and exact execution permits](bounded-task-batches.md)
- [Receipt reconciliation without replaying uncertain effects](task-receipt-reconciliation.md)
- [Account-approved local notes](local-approved-notes.md)
- [Local note intake and lifecycle](local-note-intake.md)
- [One-shot local delivery windows](scheduled-local-notes.md)
- [Finite recurring local notes](finite-recurring-notes.md)
- [Local worker diagnostics without replay authority](local-worker-diagnostics.md)
- [Finite local worker supervision](bounded-local-supervisor.md)

## Plans and reviewed owner input

- [Memory-backed personal focus plan](memory-backed-daily-brief.md)
- [Personal notes, reviewed commitments and a daily plan](personal-note-intake.md)
- [Review a proposal before it becomes a commitment](reviewed-proposal-intake.md)
- [Vertical pilot: note → reviewed memory → plan → done → current plan](current-personal-plan-pilot.md)
- [Confirmed unavailability in daily plans](confirmed-unavailability.md)
- [One foreground conversation for a personal plan](guided-personal-plan.md)
- [Reviewed unavailability in the guided conversation](guided-unavailability.md)
- [Read a guided session without changing it](guided-status.md)
- [Reviewed notes without open tasks](reviewed-empty-plan.md)
- [Renew the selected task, without guessing its identity](reviewed-task-renewal.md)
- [Local owner input: one-shot commands, explicit decisions](local-owner-input.md)
- [Bounded native owner review on macOS](native-owner-review.md)
- [One foreground owner/model cycle](integrated-owner-model.md)
- [Finite local owner drafting loop](finite-owner-loop.md)
- [Explicit cancellation of a personal commitment](reviewed-task-cancellation.md)
- [Explicitly reopen a mistakenly completed task](reopen-completed-task.md)
- [Verified task state, separate from generated answers](verified-task-state.md)

## Channels and deployment

- [Explicit local macOS alert](local-alert.md)
- [Owner communication and macOS hosting preparation](owner-channel-hosting.md)
- [Owner push shutdown and restart pilot](owner-push-shutdown-pilot.md)
- [A pinned, non-editable local runtime](pinned-runtime.md)
- [Seven-day personal-agent pilot: readiness and acceptance plan](bounded-personal-pilot.md)
