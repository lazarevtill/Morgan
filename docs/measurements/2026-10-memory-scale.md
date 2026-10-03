# Bounded memory-core scale measurement, 2026-10-03

Merged main `1b9c1559f50eee8555708301bc4d10468a6fbec1` completed one fixed,
independently reviewed, zero-model synthetic run on the always-on M1 Mac with 16 GiB
physical RAM, Python 3.14.6 arm64. At the largest checkpoint, inspecting a small named
source set was practical: SDK sample p95 12.90 ms, fresh composition plus inspection
14.51 ms. This establishes a usable bounded memory component at the measured size,
not a complete personal agent or autonomous answer quality. PR59 remains open,
unmerged, with its semantic quality gate failed.

## Frozen workload and budgets

Before dispatch the independent reviewer approved the instrument. Seventy-six input
files, including protocol, generator, comparator, independent event oracle and core
sources, were frozen and checked again at finalization. All 20,000 generated events
passed pure fixture validation before product calls. Three injected finalization
failures proved that missing/hash/deadline failures cannot leave admitted success.
There was one actual scale run, with no replacement cases, warmup removal or rescoring.

The fixed checkpoints were 100, 1,000, 5,000 and 20,000 seeded memories. A bilingual
synthetic January–September history spans five explicit owner/project scopes, a
personal correction chain of about 2,000 events, stable sources, forks, a resolved
fork, quarantine and a future correction. Bodies total 2,839,996 UTF-8 bytes, roughly
142 bytes per seeded event. Every event has an explicit aware asserted instant and
provenance. Synthetic statements are opaque reports, not independently verified truth.

Each checkpoint has 25 SDK calls through an existing reader, five newly composed
readers, three real CLI subprocesses and three real in-memory MCP ClientSessions.
The same request names 14 source IDs with caller-selected exact spans; it deliberately
includes a wrong-owner ID and a missing ID. The independent event oracle checks
scope, explicit lifecycle, closed requested sources, provenance bound to persisted
`recorded_at`, exact spans, unknown/contested/unverified states and authority `none`.
The reader series includes its first read; nothing is discarded as a warmup.

Each stage then adds ten corrections to a separate update family and ten temporal
fact upserts. The ten fact keys use one stable support source. Seventy-two fresh
fact inspections check exact content/provenance, current support and half-open
validity/withholding before/at replacement or closure; separate `current_facts` and
`evidence` checks exercise those APIs. Revised-support behavior belongs to the earlier
structural panel, not this fixed fact fixture.

Prospective p95 targets were SDK 100 ms, fresh composition/read 150 ms, CLI/MCP
1,500 ms, stores/corrections 50 ms and fact updates 75 ms. Forget targets were SDK
10 s and CLI including snapshot 15 s. Hard budgets: parent RSS 512 MiB, largest child
RSS 256 MiB, reclaimable RAM at least 4 GiB, retained disk at most 768 MiB and total
phase at most 1,200 s. Ordinary operations reserve 30 s for finalization and 96 MiB
for diagnostics, plus an actual database/WAL size reserve before each snapshot.
All work was sequential except the three small controlled consistency schedules.
No user apps were stopped, services/models installed or private records used.

## Observed API latency and storage

| Seeded rows | Actual memories / facts at read | SDK p95 | Fresh reader p95 | CLI sample p95 | MCP sample p95 | Main DB / WAL bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 100 / 0 | 1.36 ms | 2.11 ms | 305.74 ms | 33.52 ms | 4,616,192 / 4,783,352 |
| 1,000 | 1,010 / 10 | 2.15 ms | 2.95 ms | 307.18 ms | 21.14 ms | 5,648,384 / 4,783,352 |
| 5,000 | 5,020 / 20 | 3.21 ms | 4.99 ms | 304.68 ms | 26.02 ms | 27,181,056 / 8,301,832 |
| 20,000 | 20,030 / 30 | 12.90 ms | 14.51 ms | 317.16 ms | 31.90 ms | 108,040,192 / 8,528,432 |

Prior-stage corrections explain extra memory rows; facts are counted separately.
The last ten corrections/facts follow the last read checkpoint, and three controlled
corrections follow them: 20,043 memories and 40 facts before forget. SQLite shared-memory
sidecars were 32,768 bytes at each checkpoint. Read-lane main DB/WAL hashes, complete
schema and table counts were unchanged before/after each series.

All 20,000 store durations are retained, including offline hash embedding: median
4.26 ms, p95 7.04 ms, p99 7.77 ms, maximum 88.49 ms; total timed stores 85.52 s.
Stage correction sample p95 ranges from 1.96 to 8.77 ms; fact upsert sample p95 from
0.63 to 0.73 ms. The frozen quantile convention uses sorted index `floor(n*p)` capped
at the last sample. Three CLI/MCP samples and ten updates per stage are descriptive
small samples, not stable population-tail estimates. CLI timing includes process
startup and the denial/capture receipt harness; MCP includes server construction,
initialization and actual in-memory tool dispatch, without stdio/HTTP transport.

Every measured canonical context was 13,381 bytes, below the 16,384-byte contract cap.
Physical CLI JSON was 18,122 bytes; MCP serialized result was 33,771 bytes, including
its textual and structured representations. SDK returns a Python object with no
inherent wire encoding. The cap applies to canonical complete context, not these
transport wrappers. Source request size remains fixed as the database grows.

Observed parent peak RSS was 388,743,168 bytes (370.7 MiB), including fixtures,
comparators and scoped vector-row audits. Largest child peak RSS was 76,120,064 bytes
(72.6 MiB). Minimum sampled free+inactive+speculative RAM was 5,778,554,880 bytes
(5.38 GiB). Headroom is sampled at chunk/lane boundaries, not continuously, and is
not a guarantee about another application's future workload. The phase through
summary was 110.04 s; directory birth to final export by filesystem timestamps was
110.28 s. The latter is not a monotonic stopwatch; the absolute 1,200 s deadline
covered finalization as well. No target or hard budget was exceeded.

## Consistency and functional forget

All 144 regular inspections matched the frozen independent event reference and one
another across SDK/CLI/MCP/reload. Including six before/after concurrency inspections
and the final collateral inspection, 151 structural comparisons passed. In each of
three controlled schedules, an existing source SELECT completed before a later fact
SELECT paused the reader. A separate prebuilt writer context committed an eligible
correction while the reader held its snapshot. The first result matched the complete
pre-update state; a freshly opened reader matched the complete post-update state.
Ordered SQL/commit receipts are retained. This proves those schedules, not every
possible race, shared transaction arrangement or transport.

SDK forget removed 3,997 workshop records in 1.76 s after a separate 0.31 s public
snapshot; repeated forget reported zero. A prepared store with a stale erasure
generation was refused with the specific `StoreInterruptedByForget` exception.
CLI forget removed another owner's 3,997 personal records in 2.24 s including its
built-in snapshot. All 7,994 deleted IDs were inspected in 500 batches after reopening
and returned missing. Scoped rows were absent from all audited tables carrying both
owner and project; retained scope row counts/content hashes were identical, including
vectors, facts and FTS. Final live cardinality was 12,049 memories and 40 facts.

The live database after forget was 99,139,584 bytes (94.55 MiB). Deleting roughly 40%
of memory rows did not shrink its file proportionally; this run's allocated vector
chunks remain substantial. Default-width 1,024-float vectors dominate storage relative
to short text, and this test gives no semantic value assessment for those vectors.
Two retained snapshots total 211,234,816 bytes. Final database, snapshots and original
receipts total 323,588,175 bytes (308.6 MiB). Snapshot retention therefore materially
affects total cost. These results establish functional deletion/invalidation, not
secure physical erasure from snapshots, backups, filesystem copies or SSD remnants.

## Practical boundary and evidence

For a caller that already retains relevant IDs, main is useful for durable, explicit
scoped continuation at these tested sizes. Read cost increased with the long correction
family; no capacity or latency claim is made beyond 20,000 seeded events. This is one
short-body workload with the offline hash backend at default width 1024. It does not
measure live embedding latency/cost, semantic discovery, capture selection, compression,
automatic consolidation, permission inference or long-running service operation.

Network and provider construction attempts were zero under denial guards, and 20,043
actual hash embedding calls were counted. Product model calls were zero; product tokens
are not applicable. Platform review/model and operator effort were not metered.
CI status is a regression check, not the semantic quality gate.

Use [the integration example](../ALWAYS_ON_MEMORY_EXAMPLE.md) to retain source IDs and
inspect current, contested and unknown metadata without a chat model. The
[machine-readable summary](2026-10-memory-scale-summary.json) contains exact timings,
resources, cardinalities and freeze hashes. The
[evidence archive](2026-10-memory-scale-evidence.zip) preserves
original instrument/protocol, fixtures, raw durations, original SDK/CLI/MCP outputs,
trace and deletion receipts and independent review. Large database/snapshot files remain
on the selected Mac and are bound by the original export manifest rather than committed
as binary memory copies. This report does not justify merging PR59.
