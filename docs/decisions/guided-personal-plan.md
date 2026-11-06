# One foreground conversation for a personal plan

The previous vertical pilot required 14 CLI invocations. `python -m morgan_brain.guided_plan`
now exposes its useful core in one foreground conversation: select your tasks, review their
planning fields, confirm, view a plan, report completion and see the updated plan. It reuses
the bounded local-model runner, normal MemoryGate capture and current-revision daily brief.
It adds no scheduler, daemon, provider credentials or notification adapter.

## First session

Choose an existing trusted local model/server and one source note. Paths, owner/project, date
and budget are first-run setup; they are not inferred from private folders or environment keys.
The workspace must be new/private and belongs to this one planning session.

```sh
python -m morgan_brain.guided_plan /path/to/private-session \
  --note /path/to/note.txt \
  --server /Applications/Ollama.app/Contents/Resources/llama-server \
  --model /absolute/path/to/already-installed-model.gguf \
  --user owner --project personal --day 2026-11-06 --budget 30
```

The date is an example: choose the real planning day. The model run is capped at 90 seconds
and four CPU threads; it ends before the interactive review. Ctrl-C stops the foreground flow.
A saved unapproved local-model proposal can be supplied with `--proposal /path/to/proposal.json`
instead of `--server` and `--model`; this avoids rerunning inference.

For the synthetic note “water plants; bread already bought; call library”, the interaction is:

1. See every source line, with `[model]` beside suggestions. Enter `1 3` to select only your tasks.
2. Enter `10 normal 2026-11-06` for the first task: minutes, priority, due date.
3. Enter the same planning fields for the second task.
4. Review both titles, dates, durations, status, owner and project. Enter `CONFIRM` or
   `ПОДТВЕРЖДАЮ` only if they are your commitments. Any other answer cancels the write.
5. Read the plan. Enter `done 1` or `готово 1` to report that the displayed task is complete.
6. Read the refreshed plan and enter `quit` or `выход`.

Russian priorities `высокий`, `обычный`, `низкий` are also accepted. Each source selection,
planning-field entry, confirmation and menu command is one answer submission. The model does
not supply ownership, deadlines or priority by authority. Its suggestions may be wrong.

## Ownership error: deliberate user control

All source lines remain available, not just model-selected ones. In a fresh test Granite chose
Anton's bicycle task rather than the speaker's parcel task in Russian; in English it selected
both. The user can choose line 2 and reject line 1, even if line 2 was omitted by the model.
The entire note is shown again at confirmation. Nothing becomes `user_stated` before explicit
confirmation. This is a human verification boundary, not an automated ownership classifier.

A confirmed line is captured with the existing write path, explicit author, stable event ID,
and a digest binding the saved source/proposal and planning fields. Titles are treated as text,
not tool instructions. Terminal/bidi controls are escaped. Same-account software can modify or
pipe inputs into this flow: the confirmation phrase is not authentication against that account.

## Stop, resume and completion

```sh
python -m morgan_brain.guided_plan /path/to/private-session
```

Resume supplies only the workspace: changing scope/input flags requires a new session.
The source and proposal hashes are checked before reuse. The local session journal is saved
before memory writes. If capture or completion succeeds and the process dies before acknowledgement,
resume shows the pending assertion for confirmation and retries the same immutable ID. It does
not silently reapprove pending work. Cancelling an unstarted review clears the candidate choices
so they can be selected again; a partially written batch retains its pending exact fields.

A separate nonblocking session lock prevents two dialogues writing the same journal. Process
exit releases it. At most 20 menu iterations occur in one invocation. Completion is a user-stated
revision; current memory drives the next plan. A conflict or changed source stops the operation
rather than overwriting history. Every displayed plan is a new private Markdown draft, and no
local-note delivery grant is created. Existing files and memory remain after exit.

An interrupted first-time setup before `session.json` is published may leave source artifacts;
use a new workspace rather than deleting or overwriting uncertain setup data. This limited
setup recovery is separate from the tested capture/completion recovery path.

## Measurements — 2026-11-06

| Scenario | CLI launches | Answer submissions | Reviewed outcome |
| --- | --- | --- | --- |
| Two tasks, Russian | 1 | 6 | 20 minutes → 10 after one completion |
| Two tasks, English | 1 | 6 | 20 minutes → 10 after one completion |
| Ownership correction, Russian | 1 | 5 | Only the speaker's parcel task, then done |
| Ownership correction, English | 1 | 5 | Only the speaker's parcel task, then done |

These are scripted synthetic decisions, not a human usability study. Setup paths/flags are
excluded. Four fresh real-model proposals took 39.239 seconds in one bounded CPU session.
Only 2/4 model selections were exact; the reviewed workflows were 4/4 correct because fixture
answers explicitly corrected ownership. This does not improve or supersede earlier 16/20 and
3/4 measurements. No automatic acceptance should be inferred from the test's piped answers.

A separate direct-mode run included model startup and the entire two-task dialogue in one
command: 19.343 seconds, six scripted answers, 20 planned minutes becoming 10. Human thinking
and typing time were excluded. The previous 14 CLI calls and current six answer submissions
are different units; the valid command comparison is 14 launches versus one after setup.

Evidence: [four workflows](../measurements/2026-11-guided-plan.json),
[direct model run](../measurements/2026-11-guided-direct.json).
Tests cover both languages, missed/foreign model suggestions, cancellation before memory,
source tampering, partial capture and completion acknowledgement loss, and concurrent sessions.

## Limits and next real-data pilot

This is an operator-driven personal planning conversation, not an autonomous 24/7 agent.
Task selection and planning fields remain explicit. There is no conversational free-form editing,
calendar integration, desktop notification, or automatic discovery of all commitments. The day
and budget belong to this session; changing them currently needs a new session or existing brief CLI.
Memory uses explicit workspace-local 1024-dimensional hash embeddings. It does not claim
semantic-search quality. No configured cloud embedding/provider destination is used.

The workspace retains the original note, proposal, session journal, memory and plan drafts.
Before real data, choose the exact input file, private destination, owner/project, and retention
policy. One note is capped at 4 KiB/32 lines; at most eight tasks can be selected. The model file
limit is 8 GB, not a hard RSS quota. Real data and each task/completion remain the user's choice.
No permanent access, paid process or background service is installed by this command.
