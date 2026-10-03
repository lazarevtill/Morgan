# Memory guarantees and reader boundaries

Morgan can serve scoped, attributed source records to an external agent independently of its native
model-backed answers. This document describes the existing core on `main`, not experimental working
context from PR #59. The [reader capability contract](READER_CAPABILITY.md) describes the additional
semantic requirements. Passing storage tests or a tokenizer calibration does not establish them.

- **Owner and project.** Exact `evidence` reads require a named owner/project and bound IDs to that
  scope. Caller identities are attribution, not authentication. The surrounding client must
  authenticate.

- **Exact versus ranked reads.** `evidence` returns requested IDs and lifecycle metadata; `recall`
  ranks a scope. A retrieved record can be historical. Ranking does not make it the current answer
  or complete evidence.

- **Single-record `get`.** The gate checks the owner. `get` has no project argument. Check the
  returned project, or use exact project-scoped `evidence`.

- **Corrections.** Immutable whole-event revisions validate parent scope, source, actor, chronology
  and family; expose active/inactive/conflicted state. Do not merge clauses from a superseded whole
  note unless explicitly retained. The reader must interpret what the words mean.

- **Effective time.** Aware cutoffs and exact half-open fact intervals determine effective state.
  Source family reads use a snapshot. Effective time is not recorded time or proof of an external
  observation. Receiving a historical source is not current permission.

- **Provenance.** Source IDs and origin/author/scope labels survive outward reads. Strict answers
  reject invalid or unavailable support IDs. Membership and exact text are not semantic entailment,
  correct adjacent attribution or truthful prose.

- **Turn atomicity.** Two matching events and history entries commit under one write lock. Supplied
  evidence basis is re-read and compared before commit. Basis completeness is the caller's
  responsibility. Main appends to sessions; it has no occupied-session freshness guard.

- **Forget.** Registered project stores are erased atomically; erasure generation invalidates
  prepared work. File cleanup reports blocked checkpoint conditions. CLI snapshots preserve an undo
  copy. Erasure cannot recall external exports, earlier model inputs or other backups.

- **Permissions and revocation.** An explicit correction/revocation is retained with its source and
  revision state. Morgan is not a natural-language permission engine. An active event is not an
  executable authorization token.

- **Strict generation.** Complete native input counts, output caps, identity/usage checks, stop
  reasons and unsupported tool returns are validated. This path cannot execute returned tools; it
  can still produce false claims of sending, booking or verification.

These guarantees depend on callers using `MemoryGate` and the corresponding complete-basis options.
They are not promises about clients that bypass the gate, omit a basis or ignore lifecycle labels.
No real-memory migration, deployment or experimental answer-quality admission follows from these
boundaries.

## Existing regression evidence

The following existing tests exercise real SQLite and deterministic providers, without LAN model
calls. On main base `4f6a0cf`, the eight selected files passed **158 tests** on Python 3.14.6 /
macOS arm64. The complete correctly isolated main suite also passed **1185 tests, 4 skipped**.
Run from a main checkout with its declared development dependencies:

```sh
python -m pytest -q \
  tests/unit/memory/test_recall_effective_clock.py \
  tests/unit/memory/store/test_effective_fact_clock.py \
  tests/unit/memory/test_event_revisions.py \
  tests/unit/memory/test_durable_evidence.py \
  tests/unit/memory/test_forget_prepared_store.py \
  tests/unit/memory/test_forget_reaches_every_project_keyed_table.py \
  tests/integration/test_forget_atomicity.py \
  tests/unit/app/test_strict_context.py
```

The suite includes exact temporal boundaries, scope refusal, immutable source replay, concurrent
revision forks and snapshot reads, forgetting every registered table without deleting another scope,
rollback after partial failure, prepared-write invalidation, invalid-citation refusal, correction or
fact expiration during generation, and atomic turn persistence. These are behavioral regression
proofs of bounded code contracts, not formal security proofs or tests of semantic reader quality.

PR #59's optional fresh-session guard and working-context show/list cutoff fix are branch-only.
The cutoff repair cannot be cherry-picked independently: its show/list services and schemas are
absent from main. Importing their dependencies would import experimental behavior. Keep that repair
with its feature; do not rename a release gate to admit it. No independently reproduced main bug
currently warrants a new memory API or an extra persisted mechanism for these boundaries.

Safe current use is an authenticated client's scoped source storage and evidence inspection, with
visible provenance/current-state labels and review of consequential conclusions. Autonomous native
continuation, interpretation of permissions, completeness of life/project outputs and
external-action
claims remain reader-dependent and must meet the separate capability contract.
