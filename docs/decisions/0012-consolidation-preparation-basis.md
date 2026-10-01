# Consolidation proposals carry an explicit preparation basis

An atomic apply lock alone does not establish that a proposal still describes the facts
and sources it saw before generation. A synthetic fixed proposal prepared from event A
was able to resume after forget and create an unsupported fact; after explicit correction
it could still create the old value; after another inferred fact it could supersede that
newer value. All three occurred on the integrated erasure-generation/revision base.

The consolidator now captures the global erasure generation before awaiting retrieval,
then obtains scoped candidate inputs and an ephemeral immutable basis in one SQLite
read snapshot. The basis contains source IDs, reported provenance fingerprints, revision
roots and complete current leaf sets, plus a digest and IDs of the complete current fact
inventory. Fact digests include value, source, author, scope, intervals, confidence,
last-confirmed and support IDs. Derived display state is excluded from the digest;
ineligible/conflicted facts are excluded consistently on both sides.

Preparation refuses an already active transaction before retrieval or model work, preserving
the caller's pending transaction. No read transaction is held during embeddings or generation.
Before any apply effect,
the existing write transaction checks the generation, source eligibility, provenance,
revision leaves and fact inventory again. A scheduled correction becoming effective
during generation invalidates the basis without another DB write. The actual fact
inventory is checked again before target selection, so a clock boundary between checks
cannot retarget DELETE at a newly effective row; DELETE is restricted to captured IDs.
Stale proposals are rejected explicitly. Only a fresh caller-requested preparation may
retry; the old batch is never automatically retried.

`FactOp.support_event_ids` is additive with an empty parsing default. ADD/UPDATE/DELETE
need one to 32 distinct IDs naming the particular trusted sources shown in their prompt.
The lexical surprise filter still selects at most 30 of the captured candidates for the
prompt; citations outside that shown subset refuse. Changes to another captured candidate
may conservatively cancel the proposal because they could change that selection.
Unknown, inferred, quarantined, unavailable, wrong-context, inactive and conflicted
sources are not automatic write authority. Non-NOOP `apply` without a basis refuses.
Unsupported operations reject the whole batch before effects; malformed lists fail the
bounded structured-output parser. Pure NOOP still changes nothing. New inferred facts
retain only the declared supports, with `source=agent_inferred` and `author_id=model:...`.
The model chooses those links: they are declared lineage, not proof of semantic entailment.
Manual gate fact writes retain their existing contract, including honest unsupported
legacy facts. Typed source-protection refusals continue to skip protected operations
without swallowing unexpected integrity failures; the latter roll back the batch.

Prompt source records include actual IDs/source/reported author in quoted JSON data.
Instructions explicitly treat record content as untrusted and require particular support
IDs. This is a prompt boundary and persisted admission check, not a model-obedience proof
or a credential-based agent isolation claim.

This first slice explicitly refuses preparation above 50 source inputs or 256 current
fact inputs, without truncation. Unresolved source forks cause abstention rather than
inventing a winner. Global forget conservatively invalidates prepared work in another
scope too; an empty successful forget still invalidates. A failed transactional forget
rolls back generation and permits a still-valid proposal. No personal tombstones,
consumption ledger, schema migration, service or framework is introduced.

The 256-fact ceiling is a documented scalability gate, not a long-term memory capacity
claim. A later measured slice may stream a digest over all scoped current metadata while
packing bounded relevant prompt inputs. Source IDs outside the retrieved inventory and
new independent unrelated event roots are not a semantic completeness guarantee; support
eligibility/freshness does not automatically resolve all natural-language contradictions.

Reproducible synthetic coverage includes two-connection pauses across erase, correction,
fork, scheduled activation, new fact keys and same-ID confidence/confirmation changes;
forget during retrieval; atomic mixed unsupported batches; parser list bounds; fresh
preparation; rolled-back forget; and source eligibility boundaries. A two-process test
with 200 proposed keys allows exactly one prepared basis to commit, rejects the other
as stale, then proves a fresh explicit preparation deduplicates with no write.
