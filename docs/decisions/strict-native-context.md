# Opt-in strict native context

Keep the existing ChatClient and ordinary ask compatible. Strict ask requires an injected
backend capability that counts the complete templated request and generates with the same
model/template/options and an output limit. No counter or an unverifiable counter fails
before recall/generation. This slice provides the contract and deterministic fake-backend
unit tests. Installed-backend calibration is recorded separately; it does not establish
answer quality or entailment. No model download or service.

Budget = complete input tokens + output reserve + safety margin. Counter results bind the
model, template identity and canonical request fingerprint, including the response schema.
Accept only positive integer counts (excluding bool) and exact=true. Count final serialized
messages, including all history and JSON wrappers. At most four counter calls; remove whole
history turns and then whole evidence groups, never fragments. Refuse oversized base input.

Only fixed policy occupies SYSTEM. Memory and historical roles become JSON data in USER
messages. Preserve durable IDs, provenance, applicability/access labels, effective/recorded
times, source status, revision/conflict references and support/validity metadata. This
reduces privilege exposure; it is not proof of resistance to model prompt injection.

Progressive scoped evidence closure is bounded to 32 records/four reads and finite depth.
Group each derived record with its roots and every eligible branch needed for a fork.
Quarantined/inactive/unsupported evidence cannot ground positive memory answers. Missing,
truncated or unfit fork branches force explicit abstention before generation. Never select
one branch as the current value. Budget pressure can drop complete nonconflicting groups.

Structured answers have answer, evidence_ids and abstained. Validate unique supplied IDs,
scoped current source availability and complete supplied trusted roots for a cited inference.
Citation identity checks do not prove semantic entailment. Any unresolved retrieved family
conservatively abstains in the first slice. A valid abstention may persist atomically;
invalid citations do not persist a positive reply.

Carry bounded evidence snapshots into atomic store_turn. Under its existing write lock,
re-read the exact scoped records and compare current revision/support state and source
fields before committing any event/index/history row. A correction, conflict, fact update,
interval expiration or forget during generation rejects the stale turn; no auto retry.
This narrower answer basis does not reuse consolidation's whole-current-fact CAS blindly.

Acceptance: capability unavailable before model work; malformed/estimated/wrong model or
request count; complete request at limit and one over; RU/EN and wrappers; system-role
demotion; provenance roundtrip; complete fact/root group; top-one fork/missing branch;
bounded deep closure/truncation; fabricated/out-of-scope/unsupported citations; real
two-connection correction during generation cancels atomically; forget and normal ask
retain previous guarantees. Unit tests use a fake exact tokenizer. Frozen installed-backend
calibration evidence is
separate from these tests. Equal-budget held-out answer success and resource results remain
separate release gates.

Counter calls have a 10-second timeout; generation has a 60-second timeout. Reported
output exceeding the reserved cap fails without persistence. Missing/zero input usage or
disagreement with the count refuses persistence. Only
finish_reason=stop without tool calls is accepted. The adapter must validate raw
usage integers before coercion and enforce its advertised output cap.
Full serialized request bytes (including history, metadata and wrappers) are bounded
before counter transport; oversized context drops whole history/groups and recounts.

Use `morgan ask "Question" --strict-context` or MCP `ask_morgan(strict_context=true)`.
The default is false. Configure `MORGAN_STRICT_CONTEXT_BACKEND=llamacpp` explicitly;
`disabled` refuses before opening the ask context. Budgets are
`MORGAN_STRICT_CONTEXT_TOKENS=4096`, `MORGAN_STRICT_CONTEXT_OUTPUT_TOKENS=256`,
and `MORGAN_STRICT_CONTEXT_SAFETY_TOKENS=32`. The input budget must exceed both reserves.
The configured chat endpoint/model/key are reused without a constructor health probe.
Unsupported counters refuse; there is no estimate fallback. JSON CLI refusals include
reason and bounded evidence_ids; MCP error text preserves those references.
The ordinary legacy prompt remains unchanged and retains its previous memory/history
privilege behavior; these protections apply only to explicit strict asks.

`template_id` labels the adapter/calibration configuration; it is not a server template
attestation or byte hash. Per-response prompt usage equality verifies input length,
not identical template meaning or semantic entailment. Recalibrate after host, model,
template or option changes. The configured server is a trusted counting capability;
this contract does not authenticate a hot-swapped model behind a reused model ID.

Strict CLI/MCP success adds the `morgan.answer.v1` contract: schema_version, user_id,
project, model, answer, evidence_ids, abstained and budget. Existing response/provenance
fields remain available. The budget reports verified input_tokens, provider-reported
output tokens, total/output/safety reserves, template label and counter-call count.
Human CLI output prints citation IDs or explicit abstention. Clients can pass those IDs
to scoped evidence lookup for progressive verification. These checks establish identity
and availability, not entailment. `Chat.ask()` retains its string result; SDK callers
use `Chat.ask_evidence(TurnRequest(user_id=..., project=..., text=...))` for the detailed
contract, returned only after atomic commit. `TurnRequest` is an immutable per-call input.
Results are per-call values; no shared last-response state. Assistant raw evidence remains
plain answer text in this slice; the citation envelope is not persisted as source lineage.

The adapter classifies its count/generation deadlines at 10/60 seconds; the application
watchdogs remain bounded at 11/61 seconds, allowing those endpoint diagnostics to arrive.
Other counted backends still face these absolute watchdog bounds. The detailed SDK and
strict surfaces report commit-time `evidence_changed` or `store_interrupted_by_forget`
refusals with empty `evidence_ids`: no precise offending identity is available from the
atomic guard. Ordinary `Chat.ask` keeps its existing typed memory exceptions.

Strict native answers are experimental and disabled by default. The measured adapter
calibration established agreement between whole-request counts and server prompt usage;
it did not establish an answer-quality gain. Initial bilingual pilot answers frequently
abstained despite relevant supplied evidence. No prompt tuning based on held-out answers
is included here. Stable scoped evidence access remains useful to external agents
independently of this optional native answering path.
