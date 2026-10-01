# Bounded preparation resource accounting

Synthetic offline measurements on 2026-10-01 used fake embeddings for setup, an isolated
in-memory SQLite database, 5 warmups and 30 timings for three alternating-order repeats.
Allocation peaks were measured separately using tracemalloc: these are Python allocations,
not total process RSS. No inference, network or actual personal data was used.

| Current facts / source inputs | Scoped reads without CAS | Basis capture | Check under write lock |
|---|---:|---:|---:|
| 10 / 10 | 1.04-1.08 ms | 1.06-1.10 ms | 1.08-1.12 ms |
| 100 / 20 | 3.77-3.99 ms | 4.23-4.69 ms | 4.40-5.14 ms |
| 256 / 50 | 9.14-9.74 ms | 11.21-11.66 ms | 10.82-11.66 ms |

The baseline reads scoped current facts and evidence but provides no compare-and-swap
guarantee, so it is a cost reference, not a semantically equivalent replacement. Capture
and check allocation peaks at the largest input were about 0.80 MB and 0.61 MB. Database
bytes and production source hashes matched before/after all measurements.

Reproduce with `python scripts/benchmark_consolidation_basis.py --output-dir <new-directory>`
from an installed checkout. It only creates synthetic in-memory DBs and refuses to reuse
an output directory; it accepts no user database path.

The final integrated run is preserved externally in
`consolidation-resource-final-integrated-v3/results.json` in the development task evidence.
Earlier source snapshots and harness-refactor runs are retained separately; the table reports
all three repeats after the final revision eligibility and preparation-transaction guards,
with matching production and harness before/after hashes.
These numbers apply to the measured input ceiling, not to
unbounded archives or model/end-to-end latency. The pending 256-fact scalability gate is
explicitly described in decision 0012. Capture/check add no dependency, migration or
service and retain only an in-process typed basis while generation runs.
