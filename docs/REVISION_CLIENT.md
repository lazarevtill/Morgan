# Explicit source corrections

An ordinary `remember` call remains compatible. Outside a repository, omitting the project
uses personal memory. A correction appends immutable source evidence; it does not rewrite
the old event. Reported source and author do not authenticate a person or authorize action.

```sh
morgan remember "Read 20 minutes" --event-id reading-a --source user_stated --author-id person:owner --effective-at 2026-01-01T00:00:00Z
morgan remember "Read 10 minutes" --event-id reading-b --revises-event-id reading-a --source user_stated --author-id person:owner --effective-at 2026-06-01T00:00:00Z
morgan recall "reading duration" --effective-at 2026-03-01T00:00:00Z --json
morgan evidence reading-a reading-b --effective-at 2026-07-01T00:00:00Z --json
```

MCP `remember` accepts optional `event_id`, `effective_at` (timezone-aware ISO string), and
`revises_event_ids` (list of at most eight distinct source IDs). The CLI repeats
`--revises-event-id` to name multiple parents explicitly. `recall` and `evidence` accept
the same optional effective-time cutoff. An empty parent list is an independent assertion.
Invalid IDs, naive timestamps, unknown correction source, or missing reported author are
refused before opening the database or preparing embeddings.

Parents must already be stored in the same owner/context/source/author/scope and revision
family; effective time cannot precede them. Unknown legacy attribution requires a new
independent assertion. Reusing an event ID with identical source content is an idempotent
retry. Changing source content or parents under an existing ID is refused. Server-derived
recorded time and revision root are not client permissions or editable history.
An identical `remember` retry from the same reported client may reconnect or move working
directories: its original session/CWD capture metadata is preserved. A different origin
or reported client still conflicts. These labels are provenance, not authentication.

Concurrent siblings remain a fork. Resolving it requires a new correction explicitly naming
both eligible sibling branches. A later concurrent correction may create another fork;
recorded timestamp does not choose a winner. Future or quarantined corrections do not
suppress an eligible parent. A correction changes effective history after it is known;
the cutoff is not a recorded-time knowledge snapshot.

Recall/evidence retain existing payload fields and evidence version `morgan.evidence.v1`,
adding effective time, parent/root IDs, revision state, eligible leaf IDs/count, truncation
indicator, and support state. Inspect conflict references even when one branch ranks first.
Follow bounded support IDs through scoped evidence calls; raw historical records remain
visible with their eligibility labels. Do not treat inactive or conflicted source support
as a current grounded fact. Unsupported legacy facts have no invented provenance.
The `facts` CLI/MCP view also omits inactive/conflicted supported derivations and labels
remaining records with `support_state`. Unsupported legacy facts remain visible with
that label. Exact scoped evidence still returns the historical fact and its support state.

Recall applies scoped effective leaf eligibility inside both SQLite ranked searches,
before their candidate limits and before the vector relevance floor. Superseded, future,
and quarantined sources cannot occupy those slots. Eligible siblings still carry fork
metadata. Each recall keeps one query embedding and two ranked SQL queries; the queries
select source IDs and correction metadata in SQLite without hydrating the full history.

The eligibility subquery touches in-scope source metadata, so its work grows with history
size, not only `top_k`. It currently runs once for each ranked signal: two metadata passes
per recall. The cutoff is bound as exact integer microseconds; stored event times still
need normalization during each pass. No source text is fetched by those eligibility scans.

All writers sharing a database must implement schema 11 revision, erasure, and evidence
semantics. Do not run pre-revision or pre-erasure writers concurrently against it.
Compatibility with additive read fields does not imply compatibility with this lifecycle.
This is an operational requirement: Morgan does not credential-enforce a block on older
binaries. Do not downgrade in place. Preserve a full snapshot and raw lineage metadata
in any export, then replay into a separate synthetic database and verify the result before
cutover. Restore a snapshot with its matching build. Older binaries cannot be repaired
retroactively by these contracts.

This build rejects a database whose recorded `user_version` exceeds its supported schema
before normal writable initialization, migration, or source-only evidence admission. The
refusal preserves the source database and reports both versions. These are checks at open,
migration, and admission; they do not continuously police already-open connections.
Older running instances still require the operational rule against mixed-version writers.
