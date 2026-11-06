# Explicitly reopen a mistakenly completed task

A mistaken completion previously left an owner correction pending: renewal selected only open
tasks. Creating a replacement task produced two unrelated families and ambiguous title lookup.
The owner can now select **Вернуть выполненную задачу в работу**, choose one completed task,
review a due date and explicitly confirm. The confirmation shows the old task/source and says
that its history is retained. Candidate labels identify completed tasks as completed.

The one-shot CLI exposes the same reviewed operation:

```sh
python -m morgan_brain.owner_input WORKSPACE review MESSAGE_ID \
  --reopen CURRENT_COMPLETED_ID --due YYYY-MM-DD
python -m morgan_brain.owner_input WORKSPACE confirm MESSAGE_ID --digest REVIEW_DIGEST
```

The message must first be submitted using the existing input command. Its text does not choose
a target or authorize a transition. Reopening accepts only `done -> open` plus an explicitly
reviewed date. Title, priority and minutes stay unchanged. Open and cancelled tasks are rejected;
restoring a cancelled obligation is outside this operation. No model or external action runs.

A `guided-reopen-` event revises the exact current completed event, using the existing sourced
paired write and confirmation digest. Existing `_check_renewal` rejects changed/conflicted
parents. Exact retries reuse the same source/revision IDs, preserving one task family and all
previous completion evidence. STOP and expiry still block confirmation. This is a plan-state
correction, not an instruction to repeat an external side effect or deliver a notification.

## Reader compatibility: do not deploy piecemeal

No database schema migration or existing-record rewrite is needed. However, this extends the
persisted `RenewalUpdate` transition contract. Older owner inbox readers reject a retained reopen
review, including after it is applied. Older guided-session readers also reject a pending reopen.
Do not enable this action until **all** owner UI, CLI, worker and guided-session readers are on
compatible pinned runtimes. The installed pilot is not updated by this change.

Probes against the immediately preceding revision `369737c` and the new reader show:

| Saved state | Old owner reader | Old guided reader | New readers |
| --- | --- | --- | --- |
| Before reopen review | accepts | accepts | accept |
| Reviewed | rejects | accepts | accept |
| Interrupted after source write | rejects | rejects | accept |
| Applied | rejects | accepts | accept |

The owner UI calls owner_input and the worker uses its load_pilot validator, so both share the
owner-reader restriction. These are isolated loader probes, not installed GUI/service tests.
All 16 probes preserve durable inbox/session/database bytes. Do not delete a review, reinterpret
it as renewal, or restore a pre-operation backup merely to make an old reader accept it: that
would discard provenance or later writes. Keep compatible readers and the original STOP/expiry.
There is no lossless downgrade procedure after new reopen records have been written.

## Validation

New tests exercise RU/EN controller journeys, exact current-parent and digest checks, wrong
starting status, STOP/expiry, and rejection of unrelated field changes. Six real child-process
exits cover the inbox attempt marker, session pending marker, each paired write, session cleanup
and final inbox acknowledgement. Reconfirmation recovers one current family with three linked
sources; repeating confirmation adds no revision. The test dialogs are scripted: physical native
GUI interaction, user acceptance and unattended service deployment are not claimed.
