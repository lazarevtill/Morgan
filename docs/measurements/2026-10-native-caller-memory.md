# Existing owner assistant consuming Morgan evidence: fresh bounded evaluation

On the always-on M1 Mac, a fresh independently authored synthetic panel passed the
prospective native-memory-v1 quality gate. The existing owner assistant consumed
actual `MemoryGate` evidence through isolated platform callers, without an additional
LAN reader model. This is evidence for that bounded architecture, not production
readiness, on-device autonomy, lower total cost, or permission to merge or migrate
real memory. No production code or default backend changes accompany this report.

| Arm | Development useful | Closed useful | Closed delivered | Critical / unknown |
| --- | ---: | ---: | ---: | ---: |
| No memory | 2/8 | 4/16 | 16/16 | 0 / 0 |
| Raw retrieved sources | 8/8 | 15/16 | 15/16 | 0 / 0 |
| Morgan inspect | 8/8 | 16/16 | 16/16 | 0 / 0 |

The development gate required inspect at least 7/8, zero critical errors, and no worse
than either comparator; the closed gate required at least 14/16 with the same
conditions. Both passed. All delivered memory-arm answers were useful. Raw's sole
failure was parent collection after the original 180-second cutoff; its first output
remains preserved and the attempt stays failed. The one-point difference therefore
does **not** demonstrate an answer-quality advantage of inspect over raw. Evidence
helped versus no memory on this panel. The previous LAN-reader campaign's terminal
7/16 failure remains unchanged in [PR63](https://github.com/lazarevtill/Morgan/pull/63).
Different panels, architectures, and unknown platform compute prevent a causal or
compute-equivalent score comparison with that campaign.

## What was frozen and exercised

The repository base was `74315ba9ee6ef290c7f265a7853c32dbce29100e`. A fresh isolated
author received only the prospective protocol, validator, rubric and gold contract,
without previous fixtures, answers or candidate outputs. The new panel contains
24 disjoint histories: eight development and sixteen closed, twelve personal and
twelve project stories, balanced EN/RU, 288 events, twelve nonempty sessions per
story and at least thirty simulated days. Independent semantic review admitted all
24 before candidate outputs. The private gold and method mapping remain local.

The panel covers effective corrections and future boundaries, revocation, unsupported
claims, tentative alternatives, owner/project distractions, current permission denial,
plans versus reported completion, missing progress, and authorized continuation.
Each story uses twelve sequential store processes and one restarted query process.
The fixed local Unicode lexical index uses signed SHA-based 1024-dimensional L2
vectors; it is an experiment profile, not a semantic model or production default.
Top-eight selection and closure bounds (sixteen IDs, four rounds) were unchanged.
Raw and inspect receive identical full selected source content, IDs and metadata;
no-memory receives only the same current task. Arms rotate in a fixed order.

Each consumer starts without conversation history, sees only its assigned task,
context, frozen policy and response schema, and submits one original JSON payload.
There is no retry, repair, truncation, response selection or post-development tuning.
Exclusive submission preserves original UTF-8 bytes before parsing. Current permission
and the explicit simulator allowlist authorize action; remembered permission does not.
Continuation requires the matching actual effect in the isolated synthetic simulator.
No external action executes. A fresh blinded grader for each split sees opaque outputs,
all original payloads, relevant private gold and the fixed sufficient-evidence rubric,
without method identities, costs or previous results. Citation sets in the gold are
sufficient examples, not mandatory exhaustive ID lists; alternative sufficient evidence
is accepted uniformly. Unavailable attempts remain failures in the full denominator.

## Operational accounting

There were 24 development and 48 closed native caller **dispatches**, 72 retained
first originals and 71 available outputs. These are not exact physical model-call
counts: platform internal calls, model identity, tokens, reasoning, caching and monetary
cost are unavailable. Author, reviewer, grader, parent, hardware and electricity costs
are also unmetered. No new model override, paid API, model installation or LAN model
request was used. This does not establish zero cost, equal compute, or cheaper operation.

The source route used 312 owned sequential processes, 288 stored events and 312 local
index input items, with zero network attempts. Minimum measured reclaimable RAM was
6.167 GiB and maximum source peak RSS 63 MiB. Worker self-timed work totaled 16.599
seconds; that excludes Python startup, parent orchestration and platform time.
Retained run artifacts at accounting totaled 111,399,420 bytes (about 106.24 MiB).

| Closed arm | Total frame bytes | Maximum frame bytes | Median dispatch-to-collector seconds | Maximum seconds |
| --- | ---: | ---: | ---: | ---: |
| No memory | 42,298 | 2,774 | 57.49 | 107.89 |
| Raw retrieved sources | 163,572 | 11,373 | 60.46 | 284.15 |
| Morgan inspect | 173,014 | 12,099 | 59.37 | 100.39 |

These latencies include orchestration and collection, not just inference. Raw's maximum
is the failed late collection. Inspect used about 5.8% more closed frame bytes than raw;
bytes are not tokens and there is no equal-token cost claim. Full per-arm development
and closed accounting is in the accompanying aggregate JSON.

One original 7,200-second deadline began at `2026-10-04T17:42:33.466628Z` and covered
source preparation, callers, parent downtime and grading. It was never renewed for a
phase or restart. The grade was saved after about 5,901 seconds and integrity audit
after about 5,904 seconds. Collection used a 180-second admission cutoff, not a hard
platform termination guarantee. Only one consumer was active at a time; exact completed
platform snapshots were bound to frames before collection and finalization. The fixed
bounds were 4 GiB reclaimable RAM, parent RSS 512 MiB, worker RSS 256 MiB, shared
artifacts 512 MiB, context 16,384 UTF-8 bytes, whole frame 32,768 bytes, original output
16,384 bytes, and rationale 4,096 characters. No user applications were killed.

Independent audits verified all 646 development and 1,285 closed frozen files,
completion/frame bindings, once-only sequential dispatches, shared clock and complete
original-response masking. A pre-call tool-scope mistake is disclosed: a broad root
Ruff invocation parsed private `author.py` and emitted generic lint diagnostics, but
no semantic source lines, labels or expected answers. Independent integrity review
admitted continuation with unchanged freezes and zero candidate calls at that point.
Subsequent lint excluded the private directory. This is not a claim of zero private
source access by tools. Root never read private semantic gold JSON.

## Limits and next practical direction

These sparse twelve-event histories simulate time; they do not establish actual
multi-day reliability, consolidation, arbitrary project scale or real personal memory
behavior. Domain/category joint allocation is uneven, so there is no independent
personal-versus-project effect claim. All resumed steps use the same `advance` ID and
relative progress position. The unresolved alternatives are textual proposals, not
active revision-graph fork tests. Isolation is procedural in a shared filesystem, not
an operating-system sandbox. The consuming assistant is a platform caller, so this
is not evidence for autonomous on-device generation.

The practical direction supported here is a slim memory service for the owner's
existing assistant, with one `MemoryGate`, bounded source evidence, explicit current
permission and verified continuation effects. An additional unreliable LAN reader is
not necessary for this evaluated route. Inspect's answer benefit over paired raw
evidence is unproven; its extra metadata and bytes need separate justification.
Production integration and realistic robustness checks require a separate prospective
review. No real memory migration, production activation, automatic merger, or benchmark
replacement follows from this result. CI validates repository checks, not answer quality.

## Evidence identities

The aggregate [summary JSON](2026-10-native-caller-memory-summary.json) records the
protocol bounds, prior failures, costs, immutable hashes and limitations. Key identities:

- Public panel: `4d877cb643dc92af3a43c48d362873c5b8dc8e55a17db9f2fa15995f1035c6ae`.
- Instrument freeze: `ceceefdc792bf582e480faf1ee183e0a0445b855f9cdb2355979d9b4348e53e5`.
- Closed blinded input: `774e0e02bc5dc876cb5e95eab39c1e6efdfbd0714cb8128c0640bc6d33c51501`.
- Closed grade: `c38e603406f7e50ec30f7a80b03eca6f94c3cf7f5e0442cfc08e78b0c877a44e`.
- Closed audit: `7169d0d65585fe34f997e058af0fd93516b5c8272d258576f94732bc1cba4063`.

Gold, source payloads, closed expected answers and private method mapping are not
included in this publication. Earlier results remain preserved: V5 both 7/16; V9
source blocks 9/16 versus baseline 1/16 (gate failed); V10 public 3/4 with one exact
excerpt-copy failure and no heldout; V11 baseline 7/16 versus candidate 5/16 with
critical counts 3 versus 2; V12 baseline 5/16 versus candidate 3/16 with critical
counts 5 versus 6. The old invalid preflights and terminal long-lived failure are
not repaired or replaced by this fresh campaign.
