# Optional counted model capability

The existing ChatClient API is unchanged. StrictChatBackend adds an explicit capability:
count_request(messages, request=StrictRequest) returns RequestCount; generate_counted
accepts that receipt and returns the existing ChatResult. The fingerprint binds the
complete messages, model, schema and options. Counting includes the generation prefix. Changing the request requires
a fresh count. Unknown capabilities refuse; no character-based estimate is advertised
as an exact count.

The opt-in llama adapter snapshots caller inputs before awaiting, checks the configured
model, applies the whole chat template and tokenizes it. Generation accepts only a recent
issued receipt for that exact request. It checks raw integer usage, model identity,
output reserve, a completed text response and equality between counted and reported
prompt tokens. Requests are bounded to262KB, responses to4MB, counting to10seconds and
generation to60seconds. There are no automatic retries or construction-time probes.
The factory is disabled by default. Its current enabled model is the locally calibrated
ornith15; another model/template/backend requires independent calibration and a reviewed
adapter. This is a replaceable typed capability, not a prediction about future models.

The template label is a calibration identifier, not a cryptographic server-template
attestation. Equality of prompt-token counts verifies length, not identical meaning,
semantic entailment or the identity of a hot-swapped model behind the same reported ID.
The configured server is trusted for counting. Credentials are not provenance labels.

## Measured synthetic calibration

Four frozen synthetic EN/RU cases on2026-10-01 used the native evidence-answer JSON schema,
temperature0, seed42, thinkingfalse, output128 and safety32. Cases included ordinary data,
quoted special delimiters/history and a longer Russian note. All four whole-template
counts matched actual generation usage:263,273,322,672. The stage used13 sequential HTTP
requests and23.525seconds. Source hashes were unchanged. Three answers abstained; this
is counting evidence and no answer-quality or prompt-injection-safety gain is claimed.
Evidence is preserved as strict-calibration-result-v1.json and calibrate_strict_v1.py in
the autonomous development task. Endpoint/model/template/option upgrades need renewed
calibration; mock provider tests alone do not establish real token exactness.

The bounded context packer and native cited-answer surface are a separate client slice.
This provider change introduces no memory migration, framework, service or automatic
write authorization. Adapter contract tests cover malformed counts/usage, request
mutation across suspension, expired receipts, wrong model, output/finish errors and
cleanup without contacting a model.

