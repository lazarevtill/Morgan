# Using the bounded inspection core in an always-on application

Morgan's merged inspection core is a durable, scoped source store and exact context
inspector. A caller can use it without a chat model. It is not a complete personal
agent: source capture, relevant-ID selection, scheduling, interpretation, permissions
and action execution remain the caller's responsibility. Keeping a Mac powered on
makes these APIs available to your application; this example installs no background
service and does not import anyone's memory.

## A small offline example

The measured core is pinned to commit `1b9c1559f50eee8555708301bc4d10468a6fbec1`
(package metadata version `0.2.0`, Python 3.12+). The integration example accompanies
PR62; use its merged public main revision rather than assuming a packaged release
contains this example. This is a tested source revision, not a new version/tag release.


From a Morgan checkout with its existing dependencies, run:

```sh
PYTHONPATH=. python examples/context_inspect_zero_model.py /tmp/morgan-synthetic-demo
```

Use a Python environment containing Morgan's dependencies; an isolated `uv tool`
installation does not put those dependencies into every system Python environment.
`PYTHONPATH=.` selects this checkout's code. The example was also executed under
guards denying socket/DNS access and chat/live-model construction: six offline hash
embeddings, zero network attempts and zero model-construction attempts.

The directory must be new. The example uses only synthetic records, one explicit
owner/project and the deterministic hash embedding backend at width 1024. Hash
vectors exercise storage plumbing; they do not provide semantic retrieval. Explicit
settings select this isolated SQLite file without loading a user `.env` file.

`MemoryGate.store` returns each stored source ID. The example retains those IDs in
`source-ids.json`, closes the writer, opens `build_evidence_context`, and requests
only its named sources at a fixed aware cutoff. `context.json` contains original
source content/provenance and source lifecycle states:

- The corrected plan is inactive; its Sunday replacement is active.
- Both unresolved alternatives are conflicted, and the selected constraint section
  is `contested` rather than silently choosing one.
- The missing ID remains missing and its proposed question section is `unknown`.
- Eligible caller selections are `unverified`, including the selection placed in
  `current_facts`. The selection under `completed_progress` is an `unverified_report`:
  classifying a proposed sentence there proves no completion.

`action_authority` is always `none`. A stored preference, plan or instruction-like
sentence never grants permission to act. An `active` event means eligible in its
explicit correction lineage at the cutoff; it does not establish real-world truth.

The same saved database can be inspected by the CLI. Use the generated IDs from
`source-ids.json` as positional arguments:

```sh
MORGAN_DATA_DIR=/tmp/morgan-synthetic-demo \
MORGAN_TEMPORAL_DB_URL=sqlite:////tmp/morgan-synthetic-demo/morgan.db \
MORGAN_OWNER_USER_ID=synthetic-demo-owner \
python -m morgan_brain.surfaces.cli context inspect SOURCE_ID_1 SOURCE_ID_2 \
  --project synthetic-demo --effective-at 2026-10-01T00:00:00+00:00 --json
```

For a CLI call using `--selections /tmp/morgan-synthetic-demo/selections.json`, include
all source IDs named by those selections plus `synthetic-missing-source`. The existing
MCP `inspect_context` tool accepts the same `ids`, `project`, `effective_at` and
`selections` fields; its owner comes from server settings. The bounded test report
measures a real in-memory MCP ClientSession, not stdio/HTTP transport deployment.

## Integration contract

Persist the source IDs alongside your task's own continuation state. For every
inspection specify one owner/project, 1–16 distinct IDs and at most 16 exact Unicode
spans of up to 240 characters. A proposed span must match the original source text.
Inspection returns only requested sources, capped at 16,384 UTF-8 bytes. Missing
support/branch IDs are reported; the caller must deliberately request them next.
A large record may exceed the complete-output budget and be refused; the API does
not silently truncate a quotation or guarantee that every 16-ID request fits.

Corrections must explicitly name their parents, asserted time, compatible source,
author and scope. Parallel unresolved corrections remain contested. Temporal facts
use `upsert_fact`/`close_fact` and half-open validity intervals; they are separate from
caller-selected text in `current_facts`. Existing support metadata can invalidate a
fact when its supporting events are revised or conflicted. No semantic truth oracle
or automatic conflict resolution is supplied.

Use fresh inspection contexts for read-only continuation. Keep ordinary writes through
`MemoryGate`, and close contexts after use. The core uses SQLite snapshots to keep one
inspection internally consistent; caller-defined lifecycle cutoffs and ingestion
snapshot visibility are distinct. A caller processing work across `forget` can capture
an erasure generation and pass `expected_generation` to `store`, handling
`StoreInterruptedByForget` with an explicit decision to retry.

Before destructive SDK operations create a snapshot through the existing snapshot
API; CLI `forget` takes one itself. Forget erases the named scope from the live core,
but retained snapshots/backups can contain it. This is functional deletion, not a
certification of physical erasure from storage media. Snapshot retention needs an
explicit owner policy.

This example gives a usable zero-model building block for scoped continuation.
Its lifecycle checks and scale measurements do not establish autonomous-agent answer
quality. PR59's independent semantic quality gate remains failed and unmerged.
