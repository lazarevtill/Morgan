# Multiline journal compatibility and recovery

This synthetic audit compares pre-feature commit
`f6d6f645d5c7ecfc5b0f9bf5fdd5a9657e4d886b` with merged PR #122 commit
`1630e73819ddeb03061e1e8ac8866e7208a14e98`. No deployed workspace was read or changed.

## Result

No SQLite migration or journal rewrite is needed to read old data with the new version.
However, old code cannot read a retained multiline `RenewalUpdate`, even when its inbox
message is already applied. Clearing the session's pending update does not fix this:
the inbox still preserves the original reviewed update. Do not delete that receipt.

A pre-feature binary rollback is safe only before incompatible data has been written.
Afterward, preserve the workspace and return to a compatible runtime. Restoring an older
snapshot alone omits subsequent work and is not lossless recovery. Do not flatten line
breaks: exact source content participates in review digests and source verification.

## Reader matrix

Two languages, ten states, four reader paths, two actual source revisions: 160 probes.
Each process asserted its loaded module came from the selected exported revision.
All 80 new-version reads succeeded. Every probe preserved session, inbox and memory bytes.

| State | Old inbox/UI/worker loader | Old guided status | New readers |
|---|---|---|---|
| Single-line applied baseline | Read | Read | Read |
| Multiline message, not reviewed | Read | Read | Read |
| Multiline new task, reviewed | Read | Read | Read |
| Multiline new task, applied | Read | Read | Read |
| Multiline renewal, reviewed | Reject | Read | Read |
| Multiline renewal, applied | Reject | Read | Read |
| Crash before first revision write | Reject | Reject | Read |
| Crash after source commit | Reject | Reject | Read |
| Crash after task commit | Reject | Reject | Read |
| Crash after clearing session pending | Reject | Read | Read |

Reader paths were `owner_input.request(status)`, `OwnerUI.call(status)`, the worker's
`load_pilot` path and `guided_plan._print_status`. The worker and UI intentionally share
inbox validation. Guided status only reads the session, so its success alone cannot prove
that the whole workspace is compatible. Status/find/confirmation in the inbox share its
loader. The model adapter reads source-keyed draft artifacts, not `RenewalUpdate` objects;
the launch-plan renderer reads paths, not journals.

Six separate STOP-startup states per revision exercised `OwnerLoop.run`, with inference
forbidden by a test adapter. The new worker returned `stopped`; the old worker rejected
incompatible data before admission. Protected journals, STOP and expiry were unchanged.
These were isolated counters, not deployment launches. With a seeded count of eight,
the new stopped startup recorded nine; the old loader rejected before incrementing. Even
a STOP-startup probe consumes a launch, so run it on a copy, never as a live preflight.
Source-keyed model-cache status returned `stopped` in all twelve copies. No model process or network call was made.
The old emergency `owner_input stop` command was separately verified to bypass parsing
and set STOP without modifying the incompatible journals or memory.

## Crash recovery

Two synthetic EN/RU inputs were stopped with `os._exit(71)` at four boundaries:

1. Inbox marked applying and session pending saved, before the first memory write.
2. Source record committed, before the task revision.
3. Task revision committed, before clearing session pending.
4. Session pending cleared, before marking the inbox applied.

All eight copied workspaces resumed on the new version. Repeating exact confirmation
produced one current task family with the reviewed date, preserved exact source and
original expiry, and left the inbox applied. STOP blocked confirmation without writing;
a separate expired copy also refused confirmation and retained pending evidence unchanged.
These are completed-write boundaries, not power-failure or torn-filesystem simulations.
They do not test a crash inside an open SQLite transaction or the real model process.

## Upgrade and rollback rehearsal

A separate baseline was created by the actual old source version. With STOP present and
all writers quiescent, a complete workspace copy was made and hashed. Old/new/old/new
readers all opened it without changing protected bytes or clearing STOP. No schema or
journal transformation was invoked.

A second copy received a confirmed multiline renewal through the new version. Old readers
then refused it without changing anything; new readers reopened it with all work intact.
A third copy restored the full pre-change snapshot and was readable by old code, but had
only the old state. The newer branch was retained and verified unchanged. This proves why
snapshot restoration must not be described as a lossless rollback of subsequent writes.
No automatic merge of the two branches was attempted or claimed.

## Deployment conditions

- Quiesce all writers before a complete workspace snapshot; do not copy a live SQLite
  file without a consistent backup of its related journals and receipts.
- Verify the pinned runtime and all entry points, including UI, worker, status and review.
  Do not activate a new UI against an old worker or roll back only one reader.
- Preserve the original STOP marker, expiry, launch counter, source artifacts and attempt
  receipts. Never reset admission limits to make a failed upgrade appear successful.
- Test reads and exact recovery on a copied workspace with the candidate runtime before
  activation. Keep STOP until explicitly authorized to resume the real workspace.
- Before incompatible writes, the unchanged snapshot and old runtime remain usable.
  After incompatible writes, use a compatible runtime over the preserved current data.
  Retain older snapshots for investigation, not as replacements that silently lose work.

The audit used exported Git sources and existing Python dependencies, not a fresh wheel
installation or an atomic multi-process launch-agent switch. Native GUI behavior, external
notification delivery, device reboot and filesystem durability remain outside this test.
No pilot activation, expiry extension, credential change or private-data publication occurred.
