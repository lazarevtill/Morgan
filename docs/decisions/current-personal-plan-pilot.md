# Vertical pilot: note → reviewed memory → plan → done → current plan

## Product assessment

Morgan is not a continuously operating personal assistant. Its local worker/supervisor have
finite invocation limits (at most one hour); delivery means a local Markdown file. Daily briefs
require selected evidence IDs and manually supplied priorities, dates and time estimates.
The model proposes lines but does not reliably decide whose commitments they are. A working
recurrence loop does not remove these product gaps.

The concrete gap addressed here is task completion in a repeated plan. Previously, after a
user corrected or completed a selected commitment, the original evidence ID became stale.
The planner refused it, requiring the caller to find and substitute the new ID manually.
`daily_brief_cli --follow-revisions` now follows each explicitly selected commitment family
to its one current version. The same source selection can reflect done tasks and changed
estimates. This is a bounded improvement to a useful loop, not automatic task discovery.

## Behavior and safety

Strict exact-ID planning remains the default. With `--follow-revisions`, both the source and
resolved records must be scoped, user-stated commitments. Missing, quarantined, instruction-like,
future, conflicting or truncated sources fail closed. The resolved leaf must still belong to
the same family and be its unique active version; a correction between the two reads refuses
the draft. Selecting both an ancestor and its leaf counts the task once.

The output contains `resolved_sources`, and Markdown shows changed source IDs. Completed or
future-dated commitments are excluded; corrected dates, priority and duration are applied.
This remains a dated evidence snapshot, not a live calendar. No memory is written or migrated,
no model is called, and no delivery is approved by planning. All evidence reads use MemoryGate.

## Reproducible sequence

1. Supply an explicitly chosen note to the existing bounded local-model cycle. Retain the
   original note and unapproved proposal. Do not accept arbitrary model output automatically.
2. Use `morgan review-proposal` for each selected line; inspect the complete source, author,
   project, due date, duration and priority before supplying its review digest.
3. Save the chosen event IDs once, and prepare the initial plan with `daily_brief_cli`.
4. Record an explicit user completion as a new immutable commitment revision. Existing CLI:

```sh
morgan commitment 'Нужно полить растения.' \
  --due 2026-11-06 --minutes 10 --priority normal --status done \
  --source user_stated --author-id pilot --project personal \
  --event-id plants-done --revises-event-id plants --effective-at 2026-11-02T08:00:00Z
```

5. Reuse the original IDs, adding `--follow-revisions`, and use a new output file:

```sh
python -m morgan_brain.daily_brief_cli \
  --memory-db /path/to/pilot/memory/morgan.db --user pilot --project personal \
  --task updated-plan --evidence plants --evidence library \
  --day 2026-11-06 --budget 30 --follow-revisions --output /path/to/pilot/after.md
```

6. Inspect the new plan before a separate explicit local-note approval. STOP prevents delivery;
   resume plus a new finite run delivers once, and another run does not repeat the effect.

Dates and identities above are synthetic fixtures, not instructions to backdate real events.
`tests/integration/test_personal_plan_pilot.py` executes this whole sequence in fresh CLI
processes, isolated memory and a clean hash-embedding environment. Its automatic confirmations
assert the exact known synthetic fixture first; they are not a general approval automation.

## Measured usefulness and quality — 2026-11-06

A fresh CPU run of the already installed Granite 4.1 8B produced the correct two proposals for:

```text
Нужно полить растения.
Я уже купил хлеб.
Надо позвонить в библиотеку.
```

The actual proposal was reused in the vertical pilot. Before completion: two open tasks,
20 planned minutes. After explicit completion of plants: one open task, 10 minutes, one excluded
completed task. Both original IDs were reused without discovering a replacement manually.
STOP: zero deliveries. Resume: one delivery. Replay: zero further deliveries.

The post-model pipeline took 2.068 seconds across **14 CLI invocations**, excluding human review.
This exposes the remaining interaction burden; no user time saving or usability success is
claimed. The four-note model smoke check took 32.989 seconds including 15.333 seconds to load.
Three of four selections were exact. The failure selected Anton's bicycle task instead of the
speaker's parcel task. One successful case repeats the known workflow fixture; this is not a
new independent benchmark. The earlier 16/20 result remains a separate small evaluation.
Neither measurement supports unattended acceptance. Scoring normalized the API tuple to a list;
the archived raw selections, not another inference run, were used for that correction.

Machine-readable evidence:
[model check](../measurements/2026-11-current-plan-model-check.json),
[pilot results](../measurements/2026-11-current-plan-pilot.json).

## Safe next pilot, with real data only after concrete selection

Proposed scope: one owner-selected UTF-8 note of at most 4 KiB, at most two reviewed commitments,
one isolated private project/database, one local model cycle capped at 90 seconds/four CPU
threads, and one explicitly approved local Markdown delivery. Use no API keys or cloud inference.
The model file is capped at 8 GB by the runner; this is not a hard RSS limit. No RAM-peak claim
was measured here. The memory hash backend is for this bounded pilot, not semantic-search quality.

The next decision needs the exact input file, output/database directory and owner/project to
use. The user must review each proposed commitment and confirm its planning fields. Approval to
store a commitment is distinct from approval to deliver a note. A claim that a task is done must
come from the user, not from absence in model output. No broad folder scan or background read of
conversations is part of this pilot. Retention/deletion of this private pilot directory should
also be agreed before copying real data.

Result channel: show the Markdown in the current user session. There is no push-notification
adapter. A desktop notification would require an explicitly chosen channel, message visibility
policy and adapter/permission test; it is not simulated as a delivered notification here.

## Hosting options before any permanent launch

| Option | Fit for the next pilot | Remaining work |
| --- | --- | --- |
| Explicit foreground run | Recommended for one supervised note-to-plan session | User review and selected files |
| Existing macOS launchd | Candidate for later finite wake-ups, not a new custom scheduler | Exact agent label, argv, paths, schedule, stop/unload and sleep tests |
| cron | No advantage for this Mac pilot | Sleep/missed-run semantics and separate lifecycle controls |

Apple's archived guidance distinguishes launchd calendar jobs that resume on wake from cron
jobs missed during sleep. This is deployment evidence, not proof of behavior on this Mac;
verify current OS behavior before installing anything. See
[Apple: Scheduling Timed Jobs](https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/ScheduledJobs.html)
and [Apple launchd manual source](https://github.com/apple-oss-distributions/launchd/blob/main/man/launchd.plist.5).
No launch agent, cron job, notification permission, permanent secret or paid process was created.
The earlier rejected unspecified permanent launch has not been repeated.
