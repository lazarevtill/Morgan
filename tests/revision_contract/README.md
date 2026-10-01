# Frozen production revision contract

The fixture and its SHA-256 are byte-identical copies of acceptance v1. The scorer is
copied from its existing independent declared-gold scorer. Do not amend this frozen
fixture to make production pass; create a new version for changed contracts.

`production_adapter.observe` rejects any nested `expected` key. The caller strips gold
and supplies operation-only data. Actual state comes from production `gate.evidence`,
`gate.recall`, the native prompt renderer and persisted SQLite rows. No leaf-selection
algorithm or expected-ID lookup exists in the adapter.

New events and candidate writes use the gate with injected recorded/applied clocks.
Legacy null-recorded events use the immutable episodic store; historical fact snapshots
use the public temporal store without re-admitting past proposals. Candidate fact writes
use the gate and its current revision-basis validation. Rejected writes assert byte-exact
database preservation. Replay measures embedding calls and unchanged recorded times.

The legacy fixture field `executable_instruction_ids` is a conservative exposure proxy:
instruction-like recalled records appearing in the native system prompt. Empty output
for quarantined evidence is not a model-obedience or tool-execution test. Fake embeddings
exercise plumbing only; this is authored contract acceptance, not retrieval evaluation.

Run `python -m tests.revision_contract.run_actual --output <new-file.json>` from the repo,
then `python -m tests.revision_contract.score_gold --actual <new-file.json>`. The runner
refuses overwrite; the scorer verifies the fixture hash before comparison.
