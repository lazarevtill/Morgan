# Read-only continuation inspection contract

This contract is reviewed before implementation. It assembles bounded exact sources using
`MemoryGate.evidence`; it does not retrieve by relevance, generate an answer, decode a checkpoint,
read history, write a summary or authorize an action. Existing checkpoints and their semantic
reports remain separate; experimental PR #59 organizer/resume behavior is not imported.

SDK: `MemoryGate.inspect_context(user_id, project, evidence_ids, selections=[], effective_at=None)`.
CLI: `morgan context inspect ID ... --selections FILE --effective-at ISO --project NAME --json`.
MCP: `inspect_context(ids, project, selections=[], effective_at=None)` using configured owner.
The database is opened through the existing read-only evidence composition, without migration,
indexes, embedding/model construction or model calls. All source access remains through the gate.

## Exact contract

Request one named owner/project, 1..16 distinct nonblank source identities of <=256 characters,
and at most 16 explicit selections. Duplicate source IDs or duplicate spans are rejected.
An aware effective cutoff is captured once if omitted and passed to the single evidence read.
No implicit root/branch/source closure or session history is added. Only requested source bodies
can appear; unresolved support and branch IDs are pointers, not silently fetched content.

A selection has exactly `section`, `event_id`, `start`, `end`, `quote` fields. Section is one of
`current_facts`, `completed_progress`, `unresolved_questions`, `relevant_constraints`. Start/end
are strict integer Unicode code-point offsets, 0 <= start < end. Quote is 1..240 nonblank
characters and must exactly equal the original source's content[start:end]. Its event ID must
occur in requested IDs. A span cannot be duplicated across sections. Caller/model semantic
selection is never verified; no generated text or extra paraphrase is accepted.

The result version is `morgan.continuation_context.v1`, with owner/project, exact captured cutoff,
requested/missing IDs, explicit coverage `requested_ids_only`, `action_authority: none`, and:

- `sources`: original text with ID/kind, source/author/origin/scope labels, instruction/quarantine
  metadata, effective/recorded times, revision state/parents/root/eligible-leaf pointers and
  support/validity metadata. Historical, future, conflicted and quarantined requested records
  remain visible as evidence, not permission. No embeddings, entities or history appear.
- `sections`: all four named sections, each with `status` and bounded `items`. Every accepted
  item is the unchanged exact selection plus `classification: unverified_selection`; progress
  also carries `verification: unverified_report`. Sections are `unverified` when items exist,
  `contested` if any requested selection is structurally conflicted, otherwise `unknown` when
  none can be accepted. Unknown means no eligible proposed item, not no fact/task/question.
- `withheld`: ineligible requested selections with a source ID, section and deterministic reason:
  missing, quarantined, inactive/future, conflicted/truncated lineage, inactive interval or
  unsupported/revised/conflicted fact support. Historical quotes are not copied into sections.
- `corrections`: explicit revision metadata copied from requested sources with named parents;
  no inferred relationship or natural-language interpretation.
- `unresolved_support_ids` and `unresolved_branch_ids`: referenced identities not present among
  returned source records. They cannot be used as supplied evidence; callers must inspect them
  explicitly in the same scope if needed.
- `bounds`: 16 source IDs, 16 selections, 240 code points/quote, 16384 total serialized UTF-8 bytes.

Only stored active episodics with nontruncated revision state are eligible selection sources.
Semantic fact sources require an effective half-open interval and `support_state=current`;
unsupported facts can be inspected but cannot populate sections. Exact source IDs/spans prove
identity and copied text, not entailment, truth, completeness, correct category, actual completion
or current permission. An active source may contain stale or contradictory natural-language claims.
`current_facts` is a name for proposed organization, not a promotion to authoritative fact.

Validate malformed/oversized request before opening the surface database. Validate spans against
any returned source even if later withheld; missing requested sources have a missing diagnostic.
Compute the entire envelope and refuse if its canonical compact JSON exceeds 16384 UTF-8 bytes.
Never crop a source, summarize it, drop an item or silently return a partial view to fit.
Model/agent proposals remain proposals; `action_authority: none` is unconditional.

## Explicit event lifecycle

Revision edges are explicit whole-event corrections. Parents must already exist as stored
episodics in the same owner/project, with matching source, author and scope, a shared family
root, and an event time no later than the correction. A correction names at most eight
distinct parents; their IDs are canonicalized in sorted order on admission. New writes
without an asserted event time receive the store clock time. Existing undated records can
still be read; they are not the result of an ordinary new write with an omitted time.
An event's words do not create additional edges or grant action permission.

At one captured cutoff, stored events with event time at or before the cutoff are activated;
undated roots are also eligible. Every activated stored correction suppresses only its named
parents. Suppression persists when that correction is itself superseded: ancestors do not
resurface. Future and quarantined corrections do not suppress stored parents.

Family leaves are activated events minus all explicitly suppressed parents. A record that
is not a leaf is `inactive`, including the historical parent of a fork. Only actual leaves
are `conflicted` when more than one family leaf exists; a unique leaf is `active`.
For example, `R -> A` and `R -> B` yield inactive `R` and conflicted `A`, `B`.
A later correction of both leaves resolves that structural conflict. Independent contradictory
statements without revision edges remain independent records; their meaning is not adjudicated.

All requested family members expose the same sorted eligible-leaf pointers and full count.
Pointers retain at most the first 32 IDs; `revision_truncated` marks a larger family. An active
unrequested child can suppress a requested parent, but its body is not implicitly returned.
Returned historical and quarantined bodies remain evidence, with their labels intact.

For episodic selections, diagnostic priority is `quarantined`, then
`conflicted_or_truncated_lineage`, then `inactive_or_future`; a missing requested record is
`missing`. Thus a fork's inactive parent has `inactive_or_future` unless pointer truncation
requires the stronger structural warning. These labels describe eligibility, not semantic truth.
Activation includes the exact microsecond boundary. `Z`, `+00:00`, and other offsets naming the
same instant compare equally; canonical ISO serialization need not retain input spelling.

## Validation and release boundary

Integration must exercise actual SDK, CLI JSON and MCP public tools over synthetic SQLite,
including read-only/no-migration/no-provider behavior, owner/project isolation, explicit historical
corrections, fork/truncation, fact intervals/support, exact Unicode spans, missing pointers,
revocation-shaped text, unknown selections, duplicate/malformed/overflow refusal and restart.
Fresh independently authored varied cases measure requested-source preservation and deterministic
lifecycle/selection eligibility separately from proposed classification or downstream answers.
Coverage is requested IDs, not recall@k or discovery of omitted relevant records. Human-curated
selection success is not model-selection quality. If downstream reader calls are evaluated, use
frozen criteria, no gold in requests, and report their useful-answer gate independently. Passing
this deterministic context contract cannot override failed V11r/V12 or admit autonomous native
continuation. No real-memory migration, new model/service or weaker product gate is introduced.

## Usage

Inspect explicit sources without selections:

```bash
morgan context inspect event-1 event-2 --project personal --json
```

Optional selections are a JSON array in a UTF-8 file:

```json
[{"section":"completed_progress","event_id":"event-2","start":0,"end":6,"quote":"Done ✓"}]
```

```bash
morgan context inspect event-2 --project personal --selections selections.json \
  --effective-at 2026-10-03T12:00:00Z --json
```

Offsets count Unicode code points in the original content. SDK callers pass `selections`
as the same array and an aware `datetime`; MCP callers pass an ISO timestamp. Empty sections
remain unknown. Historical sources remain inspectable even when proposed quotes are withheld.

The fixed source projection is `id`, `user_id`, `project`, `kind`, `content`, `source`,
`author_id`, `origin_kind`, `scope`, `instruction_like`, `status`, `created_at`,
`effective_at` (`valid_from` for facts, asserted `created_at` for events), `recorded_at`,
`revises_event_ids`,
`revision_root_id`, `revision_state`, `eligible_leaf_ids`, `eligible_leaf_count`,
`revision_truncated`, `support_event_ids`, `support_state`, `valid_from`, `valid_to`,
`superseded_by`, `last_confirmed`, `confidence`. Fact `content` is the existing deterministic
`fact_memory` evidence representation of
subject/predicate/object, rather than an original stored raw text field. Fact selection spans
refer to that representation; episodic spans refer to original stored content. Classification
and consolidation remain unverified.

Legacy stored timestamps without a timezone are compared as UTC using the existing revision
normalization. Their original serialized fields remain unchanged in the source projection.
