# Reviewed notes without open tasks

A note can contain completed work and an availability constraint but no open commitment.
Previously, leaving task selection empty exited the dialogue, so recording availability
required selecting an unrelated line as a task. That was a product error.

The guided planner now accepts `none` or `нет задач`, followed by explicit `CONFIRM` /
`ПОДТВЕРЖДАЮ`. This records a reviewed zero-task session and opens its existing availability
menu. Blank input still exits. Refusal or EOF at confirmation leaves the session unreviewed
and creates no memory database. Resuming a reviewed empty session does not repeat selection.

The core planner accepts zero selected evidence IDs. Zero means no selected commitments,
never an implicit search of memory. Availability still requires exact scoped source reads and
follows the same immutable revision and confirmation rules as a task-bearing session.
The memory database is initialized through the existing composition path after confirmation;
there is no new store or write bypass. Status remains read-only.

`--status --human` shows reviewed availability, its exclusive end, current status and whether
it overlaps the explicit planning day, even without tasks. A cancelled interval does not block
a day. The display does not infer location. A zero-task session does not establish that the
owner has no other obligations: this remains a limited selection, not a complete agenda.

## Local validation

A bounded native local-model pilot on authorized owner statements produced zero proposed
open tasks, matching operator review. It then admitted one sourced availability constraint,
restarted, and showed that constraint without inventing an obligation. Model plus initial
review took 21.603 seconds; update 0.249 seconds; restart 0.224 seconds; readable status
0.215 seconds. Status and a boundary-day preview preserved durable-file hashes.

The inputs, dates, message identifiers and private artifacts remain outside this repository.
The operator supplied review decisions under delegated permission; this was not a new live
owner interaction or a usability study. Ten scripted answer submissions covered initial
review, availability entry and restart. No external inference, notifications, credentials or
permanent process were used. This tiny observation is not an extraction-quality benchmark.

Synthetic integration tests cover both selection languages, refusal, EOF, blank exit,
zero-evidence lookup, availability persistence, exclusive end and read-only status.
Existing crash-recovery and completion tests continue to cover task-bearing sessions.

## Limits

Availability bounds are still explicitly entered and reviewed. The model proposes task lines;
it does not interpret and commit calendar intervals. Already completed work is not reopened
just to demonstrate task completion. This change supports a useful empty agenda; it does not
supply a phone channel, automatic ingestion, hourly scheduling or a continuously running agent.
