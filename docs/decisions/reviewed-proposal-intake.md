# Review a proposal before it becomes a commitment

The next functional bottleneck is user confirmation. Model extraction reached only 16/20 exact
cases, including a third-party task and missed obligations. A recurring timer would repeat those
mistakes more frequently. This change therefore connects the existing proposals to a meaningful
review step before expanding scheduling.

| Capability | Current boundary | Remaining gap |
| --- | --- | --- |
| Model lifecycle | Finite delay, expiry guardian, STOP and bounded resources | Persistent schedule and OS supervision |
| Task management | Scoped immutable commitments, completion revisions and daily plans | Calendar integration and richer task UX |
| Confirmation | Source-bound preview and explicit acceptance added here | Trusted interactive UI and reviewer authentication |
| Execution | Explicit local-note grants, receipts and finite worker | Broader adapters with their own authority boundaries |
| Extraction | 16/20 exact on a small synthetic set | Reliable real-world evaluation; human review remains required |

## Preview

Select the private proposal file and its original source note. Both are read locally and bounded;
no model is called and no memory context is opened. The proposal must have the existing strict
unapproved schema. The note must match its SHA-256 and contain the selected line. The original
note is shown in full (at most 4 KiB), so later completion/cancellation lines are visible even if
the model omitted them.

```bash
morgan review-proposal /path/to/private/proposal.json \
  --note /path/to/original-note.txt --selection 1 \
  --due 2026-11-06 --minutes 10 --priority normal \
  --author-id OWNER --event-id plants-reviewed-20261106 \
  --effective-at 2026-11-06T07:00:00Z --project personal
```

The example values are illustrative: supply the actual reviewer statement time and intended
planning fields. `--selection` is the one-based position in `selected_lines`, not the line number
in the original note. `--title` optionally supplies a reviewed correction. The new commitment
has status `open`; completion and later corrections use the existing revision workflow.

The readable preview displays source context, original line, reviewed title, date, duration,
priority, owner, project, destination database, reported author and event identity/time. Control
characters are escaped; normal Unicode stays readable. `--json` returns the full machine-readable
review and `review_sha256`. Preview creates neither an input directory nor a database. Missing,
symlinked, oversized or malformed proposal files fail before memory access. The original note
must be an explicitly selected regular file; symlinks and FIFOs are refused.

## Accept exactly what was reviewed

After checking the entire preview, repeat the same command with its displayed digest:

```text
--accept REVIEW_SHA256
```

The digest binds all displayed values, the complete proposal bytes and original source context,
including configured owner and resolved database path. If the file, source note, selected line,
reviewed wording, planning fields, provenance or destination changes, acceptance fails and a new
preview is required. Preview and acceptance each read the proposal only once, and the captured
claim is derived from those same bytes. No later reread can swap the accepted content.

Acceptance uses the existing `cmd_remember` and `MemoryGate` path to store one `user_stated`
commitment. The explicit event ID and timestamp retain immutable replay semantics. The memory
records client `reviewed-proposal-v1` and the review digest as its session ID. Keep the original
proposal and source note for later audit; this command does not duplicate them into memory.
The proposal itself remains unchanged and unapproved. The new confirmed memory is a separate
assertion, not an approval marker written back into model output.

Acceptance may call the configured embedding backend exactly as normal `remember` does; preview
never does. The selected commitment text, not the full source-note preview, is sent to that
backend. An isolated hash-backend database supports exact-ID planning without model calls, but
must not be used as a silent replacement for an existing semantic embedding space.

The digest is a consistency check, not a secret or proof of human identity. A same-account
program can compute it. A trusted UI/harness must show the review and obtain the user's actual
confirmation; it must not automatically feed the digest back as acceptance. `--author-id` is
reported provenance, not authentication. Nothing here grants tool execution, starts a model or
installs a schedule. The result explicitly reports `execution_authorized: false`.

## Recovery and limits

Accepting the same unchanged review again returns the same memory event without duplication.
A reused event ID with changed content or review digest is rejected by the existing immutable
capture contract, even when the reviewed title and planning fields are identical.
If a proposal or note is lost, this command cannot replay its confirmation; retrieve the stored
memory by event ID instead. Each invocation accepts one item, avoiding hidden partial batches.
Declining a proposal means not accepting it; no rejection history is yet maintained.

The accepted event ID can be passed to `daily_brief_cli`; delivery still requires its independent
local-note permission. No claim is made that an activity in the plan has been performed.
This stage improves review integrity and usability; it does not improve the 16/20 extraction
measurement or make Morgan a continuously running agent.

## Validation on 2026-11-06

The archived synthetic Russian proposal from the real Granite run was reused with its exact
original note. Preview created no memory database. After explicit fixture review, acceptance
stored one commitment; repeating the same confirmation returned the same event ID. The existing
planner then produced a one-item, 10-minute focus plan. No new inference, real personal data or
external action was involved in this replay.

The tests run preview/accept in separate CLI processes and cover stale source/proposal files,
all editable planning/provenance fields, changed owner/database/project, corrected wording,
missing/symlinked/malformed inputs, exact replay, planner integration and safe readable output.
The final combined CI suite passed 1,586 tests with four live tests skipped (152.27 seconds).
All 25 focused review tests also passed locally.
