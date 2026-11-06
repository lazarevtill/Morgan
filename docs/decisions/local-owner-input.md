# Local owner input: one-shot commands, explicit decisions

The guided planner required a foreground stdin conversation. A local caller can now submit
one ordinary message and receive a persistent JSON review receipt without answering prompts:
`python -m morgan_brain.owner_input WORKSPACE COMMAND`. There is no listener, watcher, model,
provider request, notification, scheduler or installed service. This is a usable local input
boundary, not an autonomous 24/7 assistant or a fix for the failed automatic matching experiment.

## Interaction

Use a fresh private workspace. The examples use synthetic data and explicit fixture decisions.
A human must inspect the review output before supplying its digest; do not make a harness
blindly extract and confirm that digest.

```sh
python -m morgan_brain.owner_input ./owner-pilot init --user owner --project personal-pilot \
  --day 2026-11-06 --budget 30 --timezone Europe/Belgrade
printf '%s' 'I need to return the book.' | \
  python -m morgan_brain.owner_input ./owner-pilot submit note-1
python -m morgan_brain.owner_input ./owner-pilot review note-1 --new \
  --title 'Return the book' --minutes 10 --priority normal --due 2026-11-06
# Inspect exact source, owner/project, capture and digest, then replace REVIEW_DIGEST:
python -m morgan_brain.owner_input ./owner-pilot confirm note-1 --digest REVIEW_DIGEST
python -m morgan_brain.owner_input ./owner-pilot status
```

Submission alone returns `needs_review` and writes no task. There is no keyword-based or model
interpretation. To renew, submit another message, inspect `status`, choose the exact current
`evidence_id`, then review and confirm:

```sh
printf '%s' 'I did not return it. Remind me again.' | \
  python -m morgan_brain.owner_input ./owner-pilot submit note-2
python -m morgan_brain.owner_input ./owner-pilot review note-2 \
  --target CURRENT_EVIDENCE_ID --due 2026-11-07
python -m morgan_brain.owner_input ./owner-pilot confirm note-2 --digest REVIEW_DIGEST
python -m morgan_brain.owner_input ./owner-pilot stop
python -m morgan_brain.owner_input ./owner-pilot status
```

A standalone `STOP` or `СТОП` piped to `submit` also latches STOP, including while a cooperating
writer owns the guided lock. Text embedded inside another message is not a command. `resume`
explicitly removes STOP but processes nothing and does not renew the pilot expiry. STOP is
cooperative: it cannot undo a committed write or preempt an in-flight SQLite transaction.
The new path checks it before each paired memory write. The legacy interactive planner is
not universally STOP-aware; do not use it to bypass the stopped input path.

## Authority, provenance and bounded state

The local OS account is the trust boundary. Typed owner/project fields and digests are not
identity authentication. The workspace and child files are private and reject symlinks;
commands share the existing `.guided.lock` with the interactive planner. No endpoint grants
remote access. JSON escapes terminal controls. Submit reads at most 1002 characters from a
finite stdin stream before taking the lock; this is not a daemon consuming an open pipe.

The pilot expires seven days after init, accepts at most 14 unique message IDs and eight task
families. No automatic reset/eviction occurs. Submission IDs deduplicate exact text; reuse
with changed text fails. The fixed planning day remains explicit, and status reports freshness
rather than silently rolling it forward. Expired work remains inspectable; unfinished writes
are preserved for deliberate reconciliation, not retried automatically.

`owner-input.json` is a finite workflow journal, not another memory index. Raw messages stay
there pending review. The reviewed source and new commitment are then written through the
existing scoped MemoryGate path with stable generated IDs and a shared digest; session capture
roots feed the existing planner. Renewals reuse `RenewalUpdate` and `_apply_renewal`: exact current
parent, open status and only a changed due date. Source text never chooses an action or target.
The displayed digest binds owner/project, message ID, source and exact proposed fields. Changing
review invalidates the old digest. A reviewed date is a planning date, **not a delivered reminder**.

Confirmation marks the receipt `applying` before writes. After a crash, it remains visibly
pending and blocks mutations for other messages. The caller must submit the same explicit
confirmation again; replay reuses the exact source/task IDs. A completed receipt replay is a
read-only acknowledgement of the prior operation, not a claim that the task is still current.
Paired memory writes are separate transactions: source-only partial state is possible.
Concurrent uncooperative memory writers can cause explicit conflicts; no automatic winner or
reconciliation is provided. A stale target known before writing is refused without poisoning
the guided pending state. Already attempted reviews cannot be edited.

## Evidence and remaining work

`tests/integration/test_owner_input.py` exercises actual subprocesses for EN/RU messages,
explicit confirmation, identical task titles, replay IDs, invalid/changed digest, stale targets,
crash after each paired write, STOP between writes, contention, scope mismatch, private files,
expiry, and message/task limits. These are fixture decisions, not model-quality or human-usability
measurements. See `docs/measurements/2026-11-owner-input.json` for a local synthetic command run.

The previous 30-case automatic matching candidate (TP18/FP10/FN0 plus two invalid outputs)
remains rejected. No claim of reliable linguistic renewal matching follows from this interface.
All messages, including ambiguous ones, need explicit owner review. New task titles and dates
are supplied by that review, not extracted or verified semantically by Morgan.

A real continuous pilot still needs an owner-facing UI/channel and a noninteractive caller that
preserves message IDs, obtains genuine review and handles expiry/STOP/conflicts. `owner_input`
exits after one command; putting it in KeepAlive would not create a useful always-on agent.
No LaunchAgent or credentials were installed. Local JSON output is not phone delivery.

## Reversible non-task review

`dismiss ID` marks an unreviewed message as `dismissed`; `restore ID` returns it to
`needs_review`. Both require the local account, an unexpired unstopped pilot and the
existing dialogue lock. Only `needs_review` and `dismissed` receipts are eligible:
reviewed, applying and applied task changes cannot be hidden this way. The source,
identity, model attempt/cache and expiry remain intact. Duplicate delivery of the same
ID does not undo dismissal. Dismissed messages still count toward the fourteen-message
limit; this is classification, not deletion or quota recycling.

The native review offers «Не задача» with the complete source and confirmation. The
«Не задачи» menu lets the owner read and explicitly restore a source. Pending review
omits dismissed messages. Restoration may admit a first background draft if none was
attempted; it never clears an existing attempt marker or confirms a task. Dismissal
is not cancellation of an already running model: that operation may finish its
unapproved draft. Use STOP to interrupt the model and intake.

Tests cover EN/RU source display, cancellation, restart/replay, retained expiry and
quota, STOP, forbidden reviewed/attempted states and background attempt fencing.
These are scripted interface checks, not native button or usability measurements.

Deploy readers and writers together as one pinned revision. Older runtimes reject the
new `dismissed` state; do not downgrade a journal that contains it or erase receipts to
make an older reader accept it. Normal restore retains STOP/expiry restrictions. This
change does not automatically upgrade an installed pilot or mutate its existing inputs.

## Literal source lookup

`find QUERY` searches a case-insensitive literal fragment in this pilot's at most
fourteen retained messages and verified current task titles. It is not semantic QA,
whole-memory search or evidence that an absent match means no obligation exists.
The native «Найти источник» entry shows selectable results with exact quotations,
source identifiers, review status and an as-of timestamp. No model, embedding or
external endpoint is used; nothing is captured or confirmed by searching.

For applied receipts the original companion source and current task revision are
read through the scoped MemoryGate. Completed and renewed tasks are resolved from
memory rather than the stale captured fields in the workflow receipt. Missing,
conflicted, mismatched or ineligible sources are counted as unavailable and omitted;
the source text is never recovered from the applied journal as a fallback. Pending,
reviewed, applying and dismissed entries are explicitly inbox receipts with unknown
current task state, not verified task facts. This bounded lookup does not include
unrelated memories or sources outside the pilot.

CLI lookup remains available after STOP or expiry without clearing either. Native
search is offered in the active menu and is subject to the existing GUI session bound.
Tests cover current completion/renewal, source quotes, owner isolation, missing memory,
conflicting revisions, query bounds, unconfirmed/dismissed sources and unchanged storage.

## Multiline source context

The inbox accepts literal LF line breaks within its existing 1..1000-character message
limit. It preserves the exact source, including quote boundaries; tabs, carriage returns,
terminal escapes and other non-printable characters remain rejected. Delivery replay and
review digests bind the entire unchanged text. Availability statements remain single-line.

The native review leaves the task title empty for a multiline message, so quoted context
is not silently copied into a task title. The owner supplies the task fields and confirms
them with the full source visible. Every displayed source line has a `│ ` prefix so
quoted field labels cannot impersonate selected task fields; stored text is unchanged.
Confirmation explicitly distinguishes the provided text
from the selected owner's task: it does not establish the author of embedded quotations.
The source memory records what the account supplied, not a speaker annotation for every
sentence. Imported archive text still has unknown authorship; no import promotion is added.

Renewal/completion/cancellation journals also accept LF in their exact source text, allowing
review to resume after restart. Older runtimes reject these pending multiline revisions:
update every reader before allowing this input in a deployed workspace. This change does
not update or restart an existing pilot automatically.

Eight new paired EN/RU mixed-context inputs previously failed at submission; all eight now
complete scripted source-preserving review, rejection, resume and explicit confirmation.
The first review still has eight dialog responses from the main menu through confirmation
(no new dialog); the task title is deliberately entered by the owner. This is a controller
measurement, not a physical macOS GUI trial or evidence of improved model extraction.
