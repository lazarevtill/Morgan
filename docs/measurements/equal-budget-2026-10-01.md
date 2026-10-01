# Morgan equal-budget synthetic evaluation, 2026-10-01

The durable-memory work improves source identity and lifecycle safety. This evaluation does not
establish broad answer-quality superiority. Strict native answering should remain experimental and
disabled by default.

The frozen authored set contains twenty synthetic source events and twenty-four RU/EN questions
representing twelve paired tasks. Four candidate-selection arms share the same source inventory and
production whole-record packer: actual Qwen recall, simple lexical retrieval, full scoped history,
and a faithful source-file log with a processing checkpoint. The file baseline is not a manually
maintained personal profile. No real personal conversations were used.

Every arm received the same model, question, response schema, temperature zero, seed42, thinking
disabled, total budget2048, output reserve128 and safety32. All twenty nonempty-gold question views
had candidate-source coverage before packing in every arm. No more elaborate packer survived the
earlier equal-budget coverage screen with a demonstrated gain.

| Selection | Observations | Completed | Acceptable | Grounded positive answers | Unsupported answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen recall | 24 | 22 | 10 | 6 | 0 |
| Lexical | 24 | 22 | 7 | 3 | 0 |
| Full scoped history | 24 | 22 | 8 | 4 | 1 |
| Source-file checkpoint | 24 | 21 | 5 | 2 | 1 |

Across96 unique observations,87 completed, eight cross-scope requests were explicitly unsupported
and one counting request failed. The failed observation remains in the denominator; it was not
retried. Six untouched observations were completed under a separately frozen amendment. The counting
failure's cause was not recovered.

An independent reviewer scored shuffled packets without the arm mapping. Thirty observations were
acceptable: fifteen positive answers grounded in supplied evidence and fifteen appropriate UNKNOWN
answers. Forty-one unnecessary abstentions occurred despite sufficient delivered evidence. Two
conflict answers selected an unsupported resolution when the opposing source had been omitted by
packing. Citation identity and support-chain validation do not prove semantic entailment. A stricter
fixture policy requiring predecessor citations reduces positive provenance success from fifteen to
twelve; both results are preserved.

All87 completed generations matched counted prompt tokens to reported actual prompt usage. Median
final inputs were approximately1148-1156tokens: the simple halving packer often leaves available
space unused. The five generation reports total554 HTTP RPCs and389.67seconds, excluding separate
embedding capture, calibration and health requests. These stage totals are not isolated model
latency or host-memory measurements.

Reproduction evidence is local under token-pack-experiment/native-evaluation. Preserve
PROTOCOL_V1.md, PROTOCOL_AMENDMENT_V2.md, REMAINING_AMENDMENT_V2.md, all original failures and
source snapshots. Run python verify_equal_budget_evidence.py against the native-evaluation directory
for standard-library offline hash, schedule, denominator and token-accounting verification. This
verifies arithmetic and recorded evidence integrity; it does not independently rescore semantics.
real-qwen-capture-v2.json contains the approved actual embedding capture; generation_holdout_v1.py
freezes the generation schedule; generation_remaining_v2.py excludes every already attempted pair.
blind-full-packet-v1.json and blind-full-review-v2.json preserve the independent review;
root-only-full-mapping-v1.json links hidden arms after review. equal-budget-aggregate-v1.json lists
report SHA-256 values and denominators. The fixture SHA-256
is7fd4e99ab70ec5af7303abd08f5204fb5d3e18e5b0ae1aa7f28f8aa3357ff2d5. Current main-based changes must
be tested separately: the live-model run exercised a frozen reviewed pure
retrieval/packing/count/generation/parser reference, not final atomic Chat persistence.

No post-holdout prompt tuning was used to claim an improvement. Source lifecycle, migrations,
concurrency and checkpoint contracts have separate deterministic regression suites. Wider held-out
tasks, maintained-file baselines, scoped multi-context native answering and calibrated semantic
answer validation remain limitations rather than release promises.
