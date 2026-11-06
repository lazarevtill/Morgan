# Memory-backed personal focus plan

Status: implemented, 2026-11-06. A concrete memory → plan → approved task → delivered note path.

## User outcome

Given explicitly selected personal commitments and a time budget, Morgan prepares a focus plan
with source references, selected items, deferred items and completed/future exclusions. The owner
reviews the exact Markdown and approves its delivery through the existing local-note workflow.
The controlled task writes one private note; a repeated worker invocation does not duplicate it.

This stage uses a **controlled deterministic planner**, not an LLM. It sorts eligible open items
by declared priority, due date and stable source ID, then greedily includes items that fit the
remaining budget. It does not invent commitments, infer duration, optimize a calendar or execute
the checklist. A high-priority item that does not fit is visibly deferred; smaller items may fit.

A bounded local-model experiment on the validation host found a cached Granite model but did
not receive a response within its inference deadline. The temporary loopback server was stopped.
There is no automatic model fallback or paid provider in this implementation. A future model
could propose structured commitments for review; it must not create execution grants.

## Memory contract

`prepare_brief` reads only through `MemoryGate.evidence`: one owner/project, 1..8 distinct explicit
source IDs, a stated planning day and 1..480 minutes. It does not scan personal memory, ask an
embedding endpoint, or silently select another project's data. Missing evidence refuses the
whole draft instead of making an incomplete plan look complete.

Each source must be a current, unambiguous, stored episodic record attributed as `user_stated`,
not marked instruction-like. The existing evidence eligibility rules reject quarantined, revised,
conflicted and future-effective records. These labels describe provenance, not cryptographic
proof of human authorship or permission. Account-level trust boundaries remain unchanged.

The record content is explicit planning data, with extra fields rejected:

```json
{
  "version": "morgan.commitment.v1",
  "title": "Prepare meeting questions",
  "due": "2026-11-06",
  "minutes": 30,
  "priority": "high",
  "status": "open"
}
```

Titles are bounded printable lines; minutes are integers 1..240. Priorities are high/normal/low;
status is open/done. The due date is a calendar date supplied by the caller, with no guessed
personal timezone. A source is limited to 4096 UTF-8 bytes. Source text is escaped as literal
Markdown, never parsed as commands or execution instructions. The resulting task payload retains
the existing 16 KiB limit.

Free-form conversation notes are deliberately not auto-converted in this version. The owner
selects current structured sources. This is a useful small planning capability, not a claim of
understanding all personal history or finding every obligation automatically.

## Preparation is separate from authority

`daily_brief_cli` requires an explicit existing memory database path. The new
`open_evidence_context(path)` uses the existing source-only gate without reading environment
settings, `.env` files, API keys, model configuration or indexes. It opens sources read-only,
validates their schema, and performs no migration.

The CLI atomically creates a new private Markdown draft in an owner-private parent directory,
refusing overwrite and unsafe paths. Its JSON response says `approved: false`. Preparation
neither enqueues work nor changes `.approval.json`. Only the existing explicit approve/enqueue
operation grants authority for the exact Markdown, scoped task ID and expiry. Approval authorizes
writing the plan note, not performing its checklist items.

The plan is an evidence snapshot with an explicit timestamp. Later corrections/forget operations
do not alter already copied drafts, approvals or delivered notes. These private derived copies
have their own lifecycle; the owner must remove them separately when erasing copied content.
A changed source should produce a newly reviewed draft and a new task identity. The worker does
not reinterpret a saved note against a changing database.

## Reproduce with synthetic data

The example creates a **new** private directory and refuses an existing one. It stores five
synthetic commitments through the real gate in an isolated SQLite database. A deterministic hash
embedder is used only to build that demo database; evidence reading and planning perform no
embedding or model call. It is not a semantic retrieval evaluation.

```bash
python examples/daily_brief.py /path/to/new-private-demo
```

The fixture plans two items totaling 50 minutes, defers one for insufficient time, excludes one
completed and one future item, then explicitly approves only that checked synthetic fixture.
First delivery reports one action; the repeated delivery reports zero and retains one file.
`draft.md` is a readable result, not just a test log.

For a separately reviewed draft from existing selected structured sources:

```bash
python -m morgan_brain.daily_brief_cli \
  --memory-db /explicit/private/memory.db --user OWNER --project personal \
  --task focus-2026-11-06 --evidence SOURCE_A --evidence SOURCE_B \
  --day 2026-11-06 --budget 50 --output /private/drafts/focus.md
# Read focus.md, then explicitly approve its exact contents:
python -m morgan_brain.local_notes_cli /private/delivery approve \
  --user OWNER --project personal --task focus-2026-11-06 --note /private/drafts/focus.md
python -m morgan_brain.local_notes_cli /private/delivery run
```

The source records must already exist; this command does not initialize or rewrite a real memory
store with the demo hash embedder. The standard remember/gate write path remains the only writer.

## Validation and next useful capability

Tests use persistent SQLite memory and actual managed note delivery. They cover priority/budget
selection, explicit approval, repeat suppression, owner/project isolation, missing/inferred/
quarantined/instruction-like sources, corrected evidence, malformed durations and fields,
literal rendering, read-only source access, and private CLI drafts without authority/overwrite.

Next useful work is reviewed extraction of commitments from ordinary notes, discovery of relevant
sources, and a reliable local model deployment with bounded inference. Continuous operation still
needs the separately documented supervisor/boot/orphan/alert policy. None of these capabilities
is implied by this controlled planner, and no permanent process or paid provider is enabled.
