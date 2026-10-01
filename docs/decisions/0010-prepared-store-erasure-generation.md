# Prepared stores and committed erasure

An episodic store prepares its embedding before taking SQLite's write lock.
Another connection can commit a forget during that wait. The original store
must then stop before writing its prepared event or indexes.

Morgan records a single global integer in `erasure_state`. It contains no owner,
context, timestamp, event ID or deleted content. Each store captures this integer
at entry and compares it under the write lock before writing. A forget advances
the integer in the same transaction as deletion, including a forget of an empty
scope. A deletion transaction that fails rolls back both the deletion and the
integer. Maintenance after commit, such as vacuum, cannot undo this committed
invalidation even if maintenance fails.

A mismatch raises `StoreInterruptedByForget` with an explicit retry message.
Morgan does not retry automatically. A new call after forget can store new
intentional input. Because the integer is global, forgetting one owner/context
also cancels stores prepared for another owner/context. This conservative
cancellation avoids persisting scoped tombstones and keeps the mechanism small.

This protection covers preparation inside `store()`, including its embedding
wait. It does not invalidate chat or consolidation model work that finishes and
only then calls a new store. Broader operations would need to capture and pass
the same generation before their own model work. Snapshot restore and external
database replacement are separate operations from this forget boundary.

The schema addition is light: one new table and its singleton initialized to
zero, without rewriting existing memories, facts or indexes. Existing data
remains unchanged; the global metadata is deliberately outside the project
erasure registry. Source-only evidence reads must not create this metadata.
