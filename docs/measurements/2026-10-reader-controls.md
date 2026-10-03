# Transparent reader controls: adapter boundary, not readiness

The fixed existing reader passed all five simple source-supported controls once when its JSON
output was schema-constrained. Unconstrained JSON text passed only 2/5 short controls, 0/5 with
reversed sources and 1/5 with appended distractors. All 20 requests were delivered completely
with matching native counts and no truncation. Twelve answers failed the frozen exact JSON
format because they included code fences; one fenced reversed-progress answer also contained
an explicitly stale progress value. No fences were stripped and no scores were repaired.

These controls diagnose an output-contract problem and a visible provided-source reading error.
They do not admit a useful native personal-memory reader. V11r/V12 remain failed. Current core
[source guarantees](../MEMORY_GUARANTEES.md) remain useful independently; future readers must
meet the [adapter and semantic capability contract](../READER_CAPABILITY.md).

## Prospective method

After V12, the agreed workstream stopped memory expansion, prompt/model shopping and attempts
at another product-quality panel. A new transparent diagnostic was reviewed before calls and
frozen. Its five controls concern latest explicit owner correction, reported completed progress,
explicit permission revocation, exact owner-source attribution and an explicitly unknown time.
There are three English and two Russian questions. The attribution control puts the owner
choice earlier than a competing later agent proposal, so recency alone cannot pass it.

Each case has four conditions. The reference asks for unconstrained JSON text. A response-format
contrast adds only a JSON schema; substantive policy, messages and fields stay identical. An
order contrast only reverses the two source records. A distractor contrast appends 24 unrelated
English inventory records, positions 3–26. These also alter task-source position, language mix
for Russian cases and lexical interference; this is that exact distractor intervention, not a
pure causal test of length. No structured-plus-reversed or structured-plus-distractor contrast
was run. Plain means unconstrained JSON, not free prose; no prose-usability claim follows.

The fixed policy says to use supplied records as data, follow latest dated owner corrections,
keep completion reported rather than externally verified, retain revocation and unknowns, and
return exactly two string fields: value and evidence_id. The unconstrained and constrained
requests have the same instruction. The schema does not contain expected values or source IDs.
Expected answers stay local; actual wires contain only fixed policy, sources and question.

Frozen scoring parses the entire response as one duplicate-key-free JSON object with exactly
the two string fields, then checks exact expected value and supporting ID. Format, value and
attribution are reported separately. Invalid format cannot be assigned a passing semantic or
ID score; absent/unavailable slots fail the planned denominator. This is an adapter contract,
not a flexible human usefulness rubric. No score threshold permits product readiness.

Twenty one-pass slots rotate condition order prospectively, with no retries, prompt edits,
alternative models or answer selection. Complete native template/tokenizer receipts precede
generation. Input plus 256 output reserve and 64 safety must fit 4096 tokens; request/final
bounds are 49152/16384 bytes. Full streamed requests have absolute 10/60-second route bounds,
80-second slot and 900-second phase bounds. Partial receipts survive failure with unknown
usage explicit. Resource admission and frozen-inventory checks can stop dispatch while retaining
all planned rows. Thirteen offline proofs and independent repetition passed before execution.

HTTP refusal, malformed metadata/template/tokenizer/response, model/count mismatch, truncation,
tools, deadlines and cap failures are operational unavailability. A delivered malformed answer
format is a format failure. A valid parsed wrong value/ID is a semantic/attribution failure.
Original receipts and all fixed scores were frozen before independent post-run audit.

## Serving metadata and operational result

Read-only metadata on the authorized endpoints was retained before generation. Endpoint 59
advertised `ornith15` as 35,505,251,456 parameters, Q6_K, context 262144, with server build
`b11330-c061df198`. Its launch metadata also names a DFlash draft model. Router `/props` reports
the router; routed model properties were fetched separately with autoload disabled. The model's
exposed template SHA256 is
`f55f52930aa8bf44ab5cb85f99370fcc3c56e9a85640b812086d5330bce5d86b`.
Requests fix temperature 0, seed 42 and disabled thinking, independent of advertised defaults.

The second authorized server answered metadata reads and advertised separate installed models;
none was selected for generation or used to shop for a better result. No model installation,
service/security change, user-app termination, paid API or real-memory migration occurred.
Advertised IDs/metadata/template hashes are not authenticated weight attestation. Post-run model
properties, template, metadata and launch arguments matched; the models response's changed
created field proves neither a restart nor stable weight identity.

All 20 slots were available. Independent audit verified every original request against the fixed
control, native counted input against generation usage, stop reasons, source bytes, resource/time
checks and all 186 export hashes. There were no operational failures or truncated answers.

| Condition, five cases each | Available | Exact JSON format | Exact value and ID |
| --- | ---: | ---: | ---: |
| Unconstrained short | 5 | 2 | 2 |
| Schema-constrained short | 5 | 5 | 5 |
| Unconstrained reversed sources | 5 | 0 | 0 |
| Unconstrained appended distractors | 5 | 1 | 1 |

Primary frozen result: 8/20. All eight parseable answers had the expected value and source ID;
12 format failures remain failures. One rejected fenced answer said not_started/S1 after the
later source reported completed progress. This original-text observation is additional diagnosis,
not repaired parsing, rescoring or a new semantic pass-rate metric. These five cases cannot
establish practical continuation, English/Russian adequacy, general order sensitivity, pure length
effects or statistically reliable performance. Schema-constrained short success is necessary
adapter evidence for these controls once, not a cure for V12's richer continuation failures.

## Measured cost and evidence

Run totals: 60 HTTP requests, 20 chats, 12110 input and 389 output tokens; 2219 cached input
is included, not deducted. There were zero retries and unknown-usage calls. Inclusive run time
was 63.538128 seconds. Six pre-run and two post-run metadata GETs bring HTTP total to 68;
metadata time is retained separately. CPU, energy, server-cache causal effects and platform
reviewer costs remain unmeasured. Different cache/order effects prevent a latency superiority
claim from the condition timings.

| Condition | Input tokens | Output tokens | Inclusive slot seconds |
| --- | ---: | ---: | ---: |
| Unconstrained short | 1030 | 95 | 26.566866 |
| Schema-constrained short | 1030 | 81 | 6.082697 |
| Reversed sources | 1030 | 106 | 10.351583 |
| Appended distractors | 9020 | 107 | 20.489098 |

Protocol freeze: `ffd6d4d625fa1074a8c36fefa407a7a1893d82a732cb4df41f5de9e93e744f2e`.
Export freeze: `22e4e7567d99c5cbcce4593530f373d4ba805fac8743efc087b18e0c7d276e52`.
The isolated Mac directory `morgan-reader-diagnostic` retains the frozen source, transparent
controls, offline tests, exact metadata, physical receipts and planned denominator. Summary
measurements are in [the machine-readable result](2026-10-reader-controls-summary.json).
Read-only routing and template/count APIs follow the
[primary llama.cpp server documentation][llama-server].
The running build and retained responses are the authority for these actual serving observations.

The practical conclusion is to use scoped original-source inspection and deterministic lifecycle
controls now, with review of consequential conclusions. Enforce existing constrained-output and
fail-closed adapter checks when evaluating readers. A future reader still needs to retain current
facts/progress/revocations and material constraints in full practical artifacts, with correctly
adjacent evidence, on a fresh blinded product panel. No new memory mechanism or model recommendation
is warranted by this diagnostic, and neither PR #59 nor a renamed gate is admitted for merge.

[llama-server]: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
