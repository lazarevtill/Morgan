# Optional resumable checkpoints

The SDK offers an opt-in `morgan.checkpoint.v1` codec for a personal goal, task or project.
It stores ordinary `TemporalFact` JSON under `checkpoint:<stable_id>` and
`resumable_state_v1`. There is no schema migration, scheduler, embedding request or new
service. Ownership/project and reported scope stay in the fact envelope; subject entity
and applicability describe the work and grant no access. Planned steps grant no permission.

```python
from morgan_brain.memory.checkpoints import Checkpoint, CheckpointContext, CheckpointItem

state = Checkpoint(
    kind="goal", title="Read German independently",
    objective="Read independently by December",
    next_steps=[CheckpointItem(text="Choose the next reading passage", evidence_ids=[source_id])],
)
fact_id = await gate.put_checkpoint(
    state, checkpoint_id="german-reading", context=CheckpointContext(user_id=owner),
    support_event_ids=[source_id],
)
result = await gate.get_checkpoint("german-reading", user_id=owner)
# Prepare an update against this exact head; a stale writer raises StaleCheckpoint.
updated_id = await gate.put_checkpoint(
    state, checkpoint_id="german-reading", context=CheckpointContext(user_id=owner),
    support_event_ids=[source_id], expected_fact_id=fact_id,
)
```

`CheckpointContext` bundles owner, project, reported author and scope labels; it is immutable
and rejects unknown fields. These labels do not authorize access. Default context is `personal`. A missing expected ID means create-only; replaying a
completed create or update with its old basis is explicitly refused, not silently written
again. Head comparison, source admission and persistence use one writer transaction and
one cutoff. Historical facts remain available through scoped evidence. A stale support
revision or fork produces `needs_rebuild` without a resumable state; future corrections
leave support current until activation. New checkpoint writes retain ordinary refusal of
conflicted, missing, quarantined or agent-authored support. Raw branch IDs may be inspected
progressively through evidence; checkpoints do not resolve ambiguities by rank.

The reserved checkpoint predicate is excluded from automatic consolidation prompts and
surprise gating. ADD/UPDATE/DELETE proposals targeting it reject the entire batch before
any write, including earlier ordinary operations. The complete fact inventory remains in
the preparation digest so concurrent checkpoint changes still invalidate stale proposals.

Agent-written checkpoints are always `agent_inferred`. Progress is explicitly
`unverified_agent_report`; its `reference_ids` are history pointers, not trusted support.
`current` means the source basis is eligible, not that the JSON's claims are entailed or
verified. Status, including `completed`, remains an inferred summary; it does not replace
the original user/tool statements or establish permission to act.
Item `evidence_ids` must be a subset of at most sixteen fact-level support IDs. Empty
support returns `unsupported`, never grounded. This typed surface does not adopt manually
written user/tool checkpoint facts as trusted typed state: their intrinsic basis yields
`needs_rebuild`, preserving source protection against inferred replacement.

`get_checkpoint` raises `AmbiguousCheckpoint` with the competing fact IDs when legacy
finite intervals overlap at the current cutoff. It returns no selected state; inspect those
IDs through scoped evidence. No historical record is rewritten.

Unknown versions return `unsupported_version`; malformed or oversized payloads return
`invalid_state`. The original fact remains available for exact evidence access. No version
is silently rewritten. Per-field/list bounds and a 32,768-byte JSON bound limit abuse;
these do not constitute a token budget for the complete fact envelope or model prompt.

Export `state.model_dump_json()` for a plain checkpoint file. Its original support IDs
remain meaningful only in the corresponding source archive; recreating with missing IDs
is refused. A full SQLite snapshot preserves JSON, fact identities, original sources and
historical lineage; synthetic copy/reopen tests verify this. This codec is not an archive
import format or an authorization contract. Forget removes checkpoint facts with their
context, and old source IDs cannot authorize their reconstruction after erasure.

## Validation and comparison limits

The frozen four RU/EN cases provide eight temporal views: six require rebuilding an old
checkpoint and two retain its current basis. Tests exercise those outcomes through public
gate calls without using a model. They additionally cover two-connection stale writers,
source corrections, future activation, source/owner/context separation, invalid lineage,
unknown schema, size bounds, user protection, synthetic snapshot reopening and forgetting.
The fixture's supplied bytes and 850 Cyrillic codepoints are preserved.

Plain files can encode the same state and can implement the same revision checks and
atomic compare-and-swap policy. These examples prove automatic refusal and an explicit
schema for existing Morgan users; they do not show superior resumption answers or lower
token use than a well-maintained file. The retained implementation is a thin SDK adapter,
with no CLI/MCP commands or new state engine. Further surfaces or per-item salvage require
measured use, not an assumed graph benefit.

Reproduce synthetic overhead separately:

```sh
python tests/checkpoint_contract/measure_checkpoint_v1.py --db /tmp/new-checkpoint.db --output /tmp/checkpoint-results.json
```

The run measures codec, read and CAS-update medians over 100 sequential warm repetitions
with one source and 101 preserved checkpoint versions. It excludes model latency, token
packing, native/Python RSS, simultaneous contention and large histories. No benchmark
result establishes broad memory quality or future-model compatibility.
