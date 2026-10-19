# Transport controls and literal resumption, October 2026

The existing LAN reader could not deliver the first tiny public transport control within its fixed
deadline. Further generations were stopped. Work then continued offline: nineteen new integration
cases and the focused regression suite passed. No production code, models, settings or real memories
changed.

## Transport result

The prospective diagnostic allowed four identical synthetic JSON controls in
nonstream/stream/stream/nonstream order. Limits were twenty physical HTTP attempts, a 160-second
overall clock with a hard owned-process watchdog, native input at most 128 tokens and output at most
sixteen per attempted generation. It used the already loaded ornith15 model and the previously fixed
native template. Every attempt, partial body and terminal receipt was retained; timeout meant
stopping later generations because remote cancellation and quiescence were unknown.

Actual usage was thirteen HTTP attempts and one generation attempt. The native input count was
nineteen tokens; it is not returned model usage. That generation timed out after 20.003082 seconds
with no headers or body. Its token usage remains unknown. Three planned controls were not run.
Streaming was therefore not tested and cannot be recommended from this diagnostic. Owned launch wall
was 20.470632 seconds; the child exited and was independently verified absent.

Metadata and native template/count requests returned quickly, and pre/post loaded-model and template
identity matched. Local reclaimable RAM was 6,414,745,600 bytes before and 6,458,474,496 after.
Queue metrics returned HTTP 400 and were not enabled. Server CPU/GPU, active inference work,
contention, cancellation and monetary cost remain unknown. The second authorized LAN endpoint
answered read-only metadata and had no loaded models; no model was loaded or queried for generation
there.

The old provider uses an incremental HTTP body reader while requesting a non-streaming completion,
then parses JSON after the body finishes. No body in the failed control means there is no evidence
of a local JSON-completion parser error. Queueing, model execution and server-side completion
handling are not separately identified. A twenty-second tiny control does not retrospectively prove
the cause of earlier 120-second failures. Prior PR67 first grades, unknown usage and all unavailable
cases remain unchanged. Operational failures are valid observed end-to-end outcomes; absent
responses cannot be classified as semantic errors. Reader quality remains blocked.

## Existing structured contract, offline integration

The existing inspect_context contract accepts source IDs and exact Unicode spans, rejects invented
quotes and extra freeform fields, and withholds unavailable, quarantined, future, superseded or
out-of-scope sources. Its selected quotation is unverified source text and grants no action
authority. Returning that quotation with attribution constrains this tool result; it proves neither
truth, relevance, entailment nor that a generic agent will use it correctly.

Nineteen new tests compose this API and the existing typed checkpoint SDK with an owned synthetic
SQLite task tool. They cover RU/EN quotation durability, malformed factual fields, source scope and
lifecycle, scheduled correction activation, unknown progress and instruction text, unsupported and
revised checkpoints, current permission, observed task version, skipped-step refusal and durable
replay protection. A checkpoint can have a current source basis while its reported completed status
remains unverified. Tests never adopt that report as the task tool's completion prefix: only a
separate current owned observation and current caller authority govern synthetic effects.

The owned task simulator is a test fixture reused from the earlier public stateful diagnostic, with
formatting and SQL whitespace changes only. It is not a production tool, new Morgan architecture, or
external execution guarantee. A previously prepared memory snapshot does not update itself after
correction; callers must inspect again. A point-in-time memory recheck does not make source
revocation and another tool's transaction atomic. Agent selection and supported action explanations
remain unmeasured.

The new integration file passed nineteen tests in 0.38 seconds. The focused suite passed 65 in 1.04
seconds. The outer timing wrapper observed 1.55 seconds wall, 1.07 user and 0.37 system CPU seconds,
but could not read additional kernel timing statistics in this sandbox and returned a wrapper error
after pytest passed. Peak RSS is unknown; no additional test run was used to fill it. All new tests
prohibit httpx sends, use local hash embeddings and temporary synthetic SQLite, and passed explicit
Ruff checks. This is deterministic contract evidence, not an equal-budget reader-quality comparison
or novel memory benefit.

## Preserved quality and merge conditions

PR67's schema contrast remains zero of eight strict useful responses; its rationale contrast remains
zero of ten, with seven critical unknown cases. The failed budget-v2 development gate and unused
sixteen-story heldout remain untouched. Across those five previously named LAN campaigns plus this
transport diagnostic, the scoped recorded lower bound is 1,004 HTTP attempts, 78 generation attempts
and 126,193 reported tokens; twelve generation usages are unknown. Earlier campaigns, source
preparation, native caller activity, platform review/grading, server compute, energy and money are
excluded or unknown.

PR65 and PR67 are ready, reviewed and have ten green exact-head checks. Publishing these results was
authorized and complete. The exact actions to merge PR65 or PR67 remain blocked by the existing
condition against merging before actual quality is justified; no additional publication approval is
missing. These offline tests do not waive that condition or promote reader readiness.
