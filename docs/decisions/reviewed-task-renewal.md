# Renew the selected task, without guessing its identity

The local model missed a renewed request in the previous note-selection pilot. A new
candidate-aware experiment correctly interpreted concrete updates but selected arbitrary
identities for ambiguous, third-party and quoted messages. Exact quotes did not fix that.

Morgan now offers a narrow explicit path in the existing guided dialogue:

```text
renew 1 / повторить 1
<the owner's actual statement>
2027-02-02
CONFIRM / ПОДТВЕРЖДАЮ
```

The displayed task number selects its current scoped evidence ID. The review shows owner,
project, target, original statement and old/new date. Only the date of an open task changes.
The task stays open in the same revision family; no duplicate task or notification is created.
Completed tasks cannot be reopened through renewal. The date is supplied explicitly, not
inferred from "again", the clock, or a model. This is a date-only plan, not a timed reminder.

## Evidence and recovery

Before writing, the exact source, selected parent, before/after fields and event time are
journaled as one pending review. The normal MemoryGate capture path stores the statement as
`EVENT_ID-source`, then the task revision as `EVENT_ID`. These are generated paired identities,
not inferred conversation ancestry. Both carry `guided-renewal-v1` and the same digest of the
review plus owner/project. Existing task record schemas and initial capture remain unchanged.

The statement is an audit of the reviewed text, not an automatic proof of meaning or execution
permission. The explicit command and confirmation select the task and new date. The local OS
account remains the trust boundary; neither the digest nor a typed owner label authenticates
an external sender. No service, credential or external send is added.

Interrupted writes retain their exact IDs and require confirmation again. Replay is idempotent;
the original task root continues to resolve to its current revision. A changed parent or
ambiguous family blocks replay. Parent state is checked again after writes. A concurrent writer
can produce an explicit conflicting branch, never silent last-writer-wins; pending review then
remains blocked for manual reconciliation. Source and revision are separate transactions, so
an interrupted source-only write may leave an audit statement without an applied renewal.

## Frozen automatic-matching evaluation: rejected

Thirty new synthetic EN/RU cases were fixed and hashed before the candidate-aware prompt was
run. They are independent of PR #99's cases, but were authored by the same investigator;
this is not a blind third-party benchmark. No prompt tuning followed these results.

[Corpus](../../tests/fixtures/task_updates/frozen-en-ru.json) SHA-256:
`bfde6da49204c2dce168002a3930f8550a24b65421b848c82b9af7b9a26553cd`

The protocol allowed one update to one supplied open task; multi-target messages must abstain.
Acceptance required zero false positives, at least 90% actionable recall and exact valid quotes.
The local model received task IDs/titles and the source message, and proposed action, target ID,
source ID and a verbatim quote. It never wrote state.

| Measure | Result |
| --- | --- |
| Correct actionable updates (TP) | 18 / 18 |
| Missed actionable updates (FN) | 0 |
| Wrong accepted updates under the protocol (FP) | 10 / 12 negative cases |
| Invalid outputs, rejected by validation | 2 |
| Exact cases | 18 / 30 |
| Actionable recall | 100% |
| Precision among valid proposed updates | 64.3% |
| Acceptance | **Rejected** |

Eight false positives chose ambiguous, foreign-owner or quoted tasks. Two chose only one action
from a multi-target message, violating this experiment's explicit abstention contract rather
than necessarily asserting that that individual action was false. All ten had verbatim quotes:
quote fidelity is not identity, ownership, or semantic correctness.
See [exact prompt and per-case outputs](../measurements/2026-11-task-update-matching.json).
The request used `local`; the single explicitly configured native server advertised
`morgan-cycle`. No remote model, personal message or paid endpoint was used in this experiment.

## Verified manual path and its limits

The reviewed path has a separate predeclared acceptance contract: no wrong-target or
unconfirmed changes, no duplicate families, and persistence after restart. Integration tests
cover EN/RU commands, identical titles, missing target/date, refusal, completed targets,
both write crash boundaries, stale/conflicted parents, scope changes and a concurrent writer.
Unit tests reject hidden changes to title, status or estimate and malformed pending provenance.
These are protocol and recovery tests, not evidence of improved language-model accuracy.
Automatic natural-language matching coverage of this path is **zero**: the person selects it.

A local synthetic end-to-end run renewed one task in five answer submissions (including exit),
kept exactly one open task and preserved the result after restarting the process. It made no
model calls and no notifications. The fixture operator supplied the answers; this was not a
live owner usability study. Actual repeated messages without a confirmed target and new date
still require review, so the original unattended-matching limitation remains open.

Continuous operation additionally needs a selected input source, an enforced total trial budget,
a verified owner-facing channel, and a concrete approved runtime/LaunchAgent installation.
Finite cycles and STOP support do not establish a running 24/7 assistant or phone delivery.
