# Read-only context inspection: integration proofs and aborted evaluation

This records a main-based, read-only inspection interface, separately from useful-reader
quality. It adds no experimental PR59 organizer, checkpoint resume, model selection, new
storage, history, migration or action permission. V11r/V12 failures and the useful-reader
acceptance rule (candidate >=14/16, zero critical errors, no worse than equal-budget baseline)
remain unchanged. This record is not an acceptance pass.

## Product evidence

`MemoryGate.inspect_context`, the actual `morgan context inspect --json` subprocess, and the
actual MCP tool are exercised over synthetic SQLite. Requested scoped source text and
provenance remain visible; selected exact Unicode spans are unverified proposals. Completed
progress remains an unverified report, and `action_authority` is always `none`. The whole
bounded response is refused rather than cropped. See [contract](../CONTINUATION_CONTEXT.md).

Independent implementation review found and verified two fixes before the panel: legacy-naive
stored timestamps are normalized only for comparisons; fact `effective_at` represents
`valid_from`. Semantic fact content is the existing deterministic evidence representation,
rather than a raw original fact text. Scope, single snapshot/cutoff, missing pointers,
quarantine, historical revisions, structural forks, fact support/half-open intervals, overflow,
exact quotes and read-only behavior have integration proofs. After the Codacy structural
refactor, the full offline suite passed **1199 tests, 4 live tests skipped** (93.79 seconds).
Ruff, format, strict mypy and Bandit passed. These prove tested contracts, not answer quality.

## First frozen panel and invalid instrument

An independent author froze 16 cases (8 EN/8 RU; 9 personal/7 project; 64 events/62 selections).
Nine personal cases used null project values, violating the named-project contract. The first
preparation attempt therefore produced no successful inspections: nine fixture validation
errors and seven footprint errors from counting sqlite-vec tables without loading their
extension. All 16 failed receipts and original hashes were retained.

Before any successful API/model output, independent author and protocol reviews permitted one
versioned schema correction: 45 null project fields (9 case/36 event) became `personal` without
changing scope relationships, source content, IDs or gold. The footprint connection switched
to the existing supported read-only SQLite composition. Original files remained unchanged.

The actual corrected CLI run returned **16/16 contexts**, with **16/16 identical database
bytes, schema versions, schema definitions and table counts** before/after inspection. No
model calls occurred. This is successful public-API/read-only delivery, not a graded
continuation result or recall/discovery measurement. Caller-curated spans do not measure
model selection quality.

The independent grader froze a literal **0/16** result, then reported that its instrument was
invalid: equivalent UTC spellings (`Z`/`+00:00`) were compared as strings, and one documented
combined withholding reason was incorrectly mapped. Two gold revision-state expectations
also disagreed with existing quarantine behavior. Independent public-core review confirmed
that quarantined events are inactive revision sources and cannot suppress stored parents;
the interface preserves them and separately withholds their selections as quarantined.
No original judgment was edited or rescored. The invalid 0/16 is neither a valid integrity
verdict nor a reader-quality score; no passing score is substituted.

## One new bounded offline panel: stopped at preflight

A prospective review allowed exactly one new panel with clarified named scopes, timestamp
instant equivalence, reason vocabulary and existing quarantine semantics. A different author,
without old cases/outputs/grades/prompts, froze 16 cases (8 EN/8 RU; 10 personal/6 project;
64 events/64 selections). Independent gold preflight used existing `MemoryGate.evidence`
reference behavior before any new candidate inspection output.

Preflight was **invalid**: one case had two linked lifecycle/withheld-label discrepancies.
Scope, source IDs, exact spans and timestamp checks otherwise agreed, but no subset score is
claimed. The stopping rule was applied: no candidate run, gold repair, case replacement,
rescoring or additional panel. No downstream reader or LAN model calls were made.

The paired downstream runner was developed and mock-tested outside production, but remains
unadmitted. Its results are not runtime or reader-quality evidence. A valid instrument and
fresh independent acceptance evidence are still blockers; PR59 must remain unmerged. This
small read-only interface may be reviewed on its explicit tested contracts without promoting
reported facts, completion or permission, or weakening the failed reader gate.

## Reproducibility identifiers

Private synthetic inputs/gold and physical receipts remain local; they are not published or
sent to models. The adjacent summary preserves hashes and outcome denominators. Git history
preserves the tested implementation. The original Windows checkpoint and private GB5 runtime
were unavailable; this work does not claim their migration or recovery.
