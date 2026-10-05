# Explicit revisions and caller-declared consolidation request bytes

Consolidation previously discarded an explicit source correction when its words overlapped
existing facts. Lexical overlap cannot establish that a correction is redundant. After scoped,
trusted, active source capture, explicit revisions now precede ordinary novelty-ranked events.
Revision order follows captured input order; ordinary novelty thresholds and stable ties remain.
The combined cap stays at 30. Excess revisions are omitted and retained revisions can displace
novel ordinary events. Only recalled and captured eligible sources participate. Source admission,
shown-source citation checks and apply-time basis revalidation remain in place.

Retention establishes delivery to the proposer, not semantic truth, recognized revocation,
changed facts or useful reader answers. Inferred operations still cannot replace or delete a
protected user-stated fact. A fake NOOP demonstrates delivery with no durable effect.

Thirty records do not bound request size: source content has no byte ceiling. Public adversarial
selector/serializer controls produced multi-megabyte input at the same record cap. They measure
serialized bytes, not model tokens, inference cost or realistic answer quality.

## Optional SDK contract

`MemoryConsolidator(..., request_byte_limit=N)` and `generate_structured(...,
request_byte_limit=N)` accept a positive integer ceiling. The default is `None`, disabled.
No setting, surface flag, default limit or installed configuration changes. Existing callers
keep their behavior. The owner or SDK caller chooses an operational byte policy; the demo's
one-byte ceiling is deliberately unusable and is not a recommendation.

The guard measures compact UTF-8 JSON for complete ChatClient input arguments: model, message
roles/content/tool calls, response format and schema. The OpenAI-compatible adapter shares the
same payload builder. The guard includes JSON field names and escaping, as well as additional
schema messages in json-object/prompted modes. Every re-ask is checked after its validation-error
message is appended and before dispatch.

At exactly the limit, generation proceeds. Above it, `StructuredRequestTooLarge` reports
`measured_bytes`, `limit_bytes`, and one-based `attempt`. No generation occurs for that rejected
request, with no automatic retry, source/fact truncation or silent omission. Initial rejection
makes no database write. A later re-ask rejection does not undo or conceal earlier generation.
Invalid limits refuse explicitly. These exceptions propagate to the SDK caller; no new CLI/MCP
error rendering contract is introduced by this slice.

This is a canonical input JSON byte ceiling. It does not measure HTTP headers, internal SDK
serialization changes, provider-added fields, custom backend transformations, chat templates,
output, tokens or total cost. It must not be advertised as a provider wire or context-window
bound. Exact token/template readiness remains unresolved and no model request follows from
this change.

## Reproduce without a provider

```bash
PYTHONPATH=. python examples/consolidation_byte_limit_noop.py
pytest -q tests/unit/providers/test_request_byte_budget.py \
  tests/unit/memory/knowledge/test_revision_surprise.py
```

The example uses an in-memory public synthetic Gate, fake embeddings, explicit revision and
local fixed NOOP. It shows an initial caller-visible byte refusal with zero calls/state changes,
then unchanged default admission, revision delivery and NOOP with no durable effect. Tests cover
English/Russian UTF-8, all three JSON modes, schema overhead, exact boundaries, invalid caps,
re-ask growth, initial refusal and default compatibility. No provider or heldout data is needed.
