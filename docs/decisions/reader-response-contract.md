# Explicit reader status contract

This local candidate depends on the unpublished adapter request-controls changes and
`adapter-request-controls.md`; it must be reviewed with that scope, separately from PR69.

A caller using `MemoryGate.recall` and `inspect_context` must define what its answer
fields mean. Source inspection preserves attribution, revisions and scope; it does
not verify a progress claim or authorize an action. Schema-constrained JSON controls
format, while a strict local validator and the caller's evidence policy check meaning.
See [context inspection](../CONTINUATION_CONTEXT.md) and
[request controls](adapter-request-controls.md).

Use the same definitions in schema field descriptions and system instructions:

- `verified`: designated verification evidence confirms the scoped task. Another
  assertion of verification, a source label or a confidence score is insufficient.
- `unverified_report`: a scoped progress report exists without designated proof.
- `unknown`: neither a scoped progress report nor designated proof exists. This is
  different from a report lacking proof.
- `independently_verified`: true only for `verified`; otherwise false, never null.
- Permission is resolved from applicable owner/project/task evidence and active
  revisions. A revocation or explicit regrant replaces its referenced parent;
  conflicting active branches and absent permission evidence yield `unknown`.
- A proposed step is not a completed step. Completion needs designated evidence.
  Memory content and these classifications never grant execution authority.

A caller must validate proof designation and scope before presenting it to the
reader. Keep exact source IDs, provenance and artifact references; do not turn a
model-generated claim into proof. Caller-supplied attribution is not authentication.

For an already configured compatible endpoint, the opt-in SDK request is:

```python
from morgan_brain.providers.openai_compat import OpenAICompatAdapter
from morgan_brain.providers.request_budget import ChatRequestOptions
from morgan_brain.providers.structured import generate_structured

# Configure OpenAICompatAdapter with these explicit, compatible request options.
options = ChatRequestOptions(
    max_output_tokens=64, temperature=0.0, enable_thinking=False
)
client = OpenAICompatAdapter(
    configured_endpoint, configured_api_key, "llamacpp", timeout=60,
    setting="MORGAN_LLM_ENDPOINT", request_options=options,
)
# messages: caller-validated scoped evidence plus definitions;
# ReaderAnswer: caller-owned strict Pydantic schema with the definitions above.
answer = await generate_structured(
    client, messages, model=configured_model, schema=ReaderAnswer,
    json_mode="json_schema", max_reask=0, request_byte_limit=8192,
)
```

The snippet is an SDK integration outline, not a standalone executable or a new
CLI/MCP default. Apply a physical call limit, request deadline and process resource
limit in the caller. Never strip Markdown fences to retroactively pass a failed
strict JSON result. A field description does not enforce its evidence semantics;
reject unsupported verification, permission and authority claims locally.

Local development diagnostics on October 6, 2026 passed three fresh categories:
report without proof, no evidence, and a designated synthetic local artifact.
The previous matched comparison remained 0/4 and stopped on a no-memory permission
error. Those results neither establish general reader reliability nor demonstrate
memory benefit over raw evidence. The synthetic artifact is not real-world proof.
No earlier results, heldout data, factories or runtime defaults are changed.

## Offline executable example

Run `PYTHONPATH=. python examples/reader_policy_zero_model.py NEW_DIRECTORY`.
It uses the public MemoryGate with a separate hash-embedding database. Its typed
synthetic events demonstrate a caller-owned policy; it does not interpret arbitrary
natural-language memory or establish authenticated provenance. The example preserves
raw simulated reader output, rejects contradictions without repairing them, and
never executes a proposed action. Designated proof must match an explicit local
artifact digest, source identity, reported tool author and owner/project/task.
The artifact is a synthetic demonstration, not real-world verification.

The executable regressions are in
`tests/unit/providers/test_reader_policy_example.py`; the JSON proposal remains a
review checklist for broader caller integration, not a completed test claim.

The same example also exposes `read_with_policy`: a public adapter call recorded
before `generate_structured` parses it, followed by the deterministic caller policy.
Use a caller-owned new raw receipt path. The receipt stores `ChatResult.text`, usage
and finish reason; it is not the complete provider JSON or hidden reasoning.
Malformed schema, unsupported verification and incomplete output preserve the raw
receipt and fail closed. Invalid scope or proof designation is refused before HTTP.
This integration imports the unpublished `ChatRequestOptions` dependency; it does
not enable a provider or change any CLI/MCP default. Tests use the real SDK with
`httpx.MockTransport`, and demonstrate no real model or network call.

Receipt storage is exclusively reserved before dispatch: an invalid path consumes
no adapter call. If writing, flushing or closing fails after a response,
`ReaderReceiptError.raw_result` preserves the returned `ChatResult` for recovery.
That raw result has not passed schema or caller policy validation and never grants
authority. A transport failure may leave an empty reserved receipt; it is not a
valid response and is never overwritten or retried automatically.
