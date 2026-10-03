# Minimum reader and adapter acceptance contract

A reader consumes authorized sources and makes a usable answer or continuation. Its semantic
capabilities are separate from Morgan's [deterministic memory guarantees](MEMORY_GUARANTEES.md).
Existing source delivery, JSON validation and CI do not establish that the reader is dependable.

## Required semantic capabilities

The reader must follow the latest explicit authorized correction, distinguish reported completion
from an unstarted task and external verification, retain current revocation, attribute each claim to
the actual supporting source, and preserve unknown facts. It must separate owner decisions from
agent proposals; historical permission from current permission; ability from agreement; an intended
or drafted action from execution; and a reported status from a checked artifact.

For useful life/project continuation it must deliver the requested draft, plan or handoff, preserve
all material active constraints, and ask only necessary unresolved questions. It must interpret
whole-note correction semantics without recovering facts that were not explicitly retained. These
requirements apply to English and Russian. Source IDs must support adjacent factual claims in the
correct time, actor and scope; a valid ID alone is insufficient. Creative proposals must be visibly
proposed and cannot assert unsupported suitability, current facts, permission or execution.

## Adapter admission before semantic evaluation

1. Record the existing endpoint's advertised model identity, serving build, context/output limits,
   template and relevant sampling/reasoning settings using read-only metadata. Distinguish an alias,
   a template label/hash and advertised weights from authenticated weight attestation. A router's
   own metadata can differ from the routed model's properties. Do not change models or settings.
2. Preserve canonical complete sources, query and fixed policy as actual wire receipts. Count the
   actual native templated input, including wrappers/schema/history; verify returned input usage.
   Bound full request bytes, input plus output/safety reserve, physical responses and final bytes.
3. Enforce physical time/call limits with no hidden retries or answer selection. Charge failures,
   all passes, repeated context and cached input. Unknown usage stays unknown. Separate inclusive
   latency from overlapping components; disclose unmeasured energy/platform costs.
4. Classify HTTP refusal, malformed template/tokenizer/response, count or model mismatch, tools,
   truncation and deadline/cap failures as operational unavailability. Valid delivered output with
   unsupported facts is semantic failure. Output-format failure is separately visible. None can
   disappear from the planned denominator or be repaired by an undocumented retry.
5. Run reviewed transparent source-supported controls before broader evaluation. Change only one
   declared factor per contrast, freeze exact scoring and preserve original output. Such controls
   diagnose an adapter/reader failure; even a perfect small score does not establish readiness.

## Product admission remains separate

A future reader needs a genuinely fresh independent source-grounded blinded practical continuation
panel with frozen criteria and unchanged denominator: at least 14/16 strict useful answers, zero
critical errors, and no worse than the equal-budget baseline. Useful fragments with failures do not
pass. An isolated pass still requires independent implementation review, production integration,
regressions, exact-head CI/Codex review and another fresh production validation before readiness.
Model suitability recommendations need independent primary evidence; an advertised model name or
one easy diagnostic is insufficient. No model installation or recommendation is implied here.

The Mac V11r and V12 failures remain unchanged. V12's mandatory same-model review was more
expensive,
produced 3/16 strict useful finals versus 5/16 baseline, and lost one of three correct drafts. Stop
expanding memory machinery or shopping for prompts/models/panels in response. A changed diagnostic
of simple controls is explicitly diagnostic, not another attempt to pass a product-quality gate or
a reuse of held-out scenarios. Current native reader quality remains unadmitted.

The [transparent control diagnostic](measurements/2026-10-reader-controls.md) separates
format/adapter issues from provided-source reading. Its short constrained condition passed
five simple controls once; this does not replace the unchanged product admission requirement.
