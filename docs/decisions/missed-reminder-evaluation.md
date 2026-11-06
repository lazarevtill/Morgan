# Missed reminders: extraction regression gate

A local evaluation found an explicit repeated reminder missing from the model's suggestions
when the author reported not having done the action. Negated completion must not become
completion, and a missing suggestion must not close an existing obligation.

The existing guided dialogue already shows all source lines and lets the reviewer select a
line the model omitted. That review remains necessary. This change adds six synthetic probes
and publishes a controlled comparison; it does not change the runtime prompt or add a service.

## Measured decision

Granite 4.1 8B was evaluated over a private Unix socket, temperature zero, four threads,
finite batches of at most 120 seconds. The dataset combines the existing 20 personal-note
cases with six new Russian/English reminder cases: renewed, cancelled and third-party tasks.
One run per variant is a regression observation, not a statistical quality guarantee.

| Variant | Exact cases / 26 | True positives | False positives | False negatives |
| --- | --- | --- | --- | --- |
| Existing prompt | 22 | 14 | 1 | 3 |
| Broader reminder instruction, rejected | 20 | 15 | 4 | 2 |
| Narrower non-completion instruction, rejected | 21 | 13 | 2 | 4 |

See [per-case measurements](../measurements/2026-11-reminder-regression.json).
The broader instruction selected reminder requests as tasks and explicitly treated missed
or negated completion as unfinished. The narrower instruction applied that rule only when
the author reported their own task unfinished and requested another reminder, retaining
cancellation precedence. Both regressed on the synthetic comparison and were rejected.

A separate authorized private replay found the original request but missed its subsequent
renewal: TP=1, FP=0, FN=1 over two positive opportunities and two negative messages. The broader
candidate corrected that tiny sample, but failed the wider gate. Private texts, dates,
identifiers, model outputs and memory artifacts are not part of this repository.

The local replay preserved exact user statements separately from analyst-inferred
observations, checked effective-time reads and read the updated observation after reopening
the process. It did not claim automatic task identity resolution, invent a new deadline,
mark the request done, or execute it. Those are separate requirements from line extraction.

## Remaining gate to unattended use

Do not admit proposals without review or close tasks merely because a later extraction omits
them. A renewed request needs source attribution, an unambiguous task match and a reviewed
new time. Model quality and that end-to-end lifecycle remain unproven for unattended operation.
The finite local worker, STOP controls and disabled LaunchAgent plan do not establish a
running personal agent after reboot or a verified notification channel.

The six new cases live in `morgan_brain.eval.personal_notes.REMINDER_CASES` and can be scored
with the existing `score(case, selected)` function and bounded `managed_model` API. They were
used during this investigation; they are not a hidden holdout set. No paid provider,
credentials, permanent process or external delivery is needed for this regression exercise.
