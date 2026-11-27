# Measurements and evidence

Use this index to find a result and its limits. Each report applies to its recorded revision,
environment and dataset. A later successful run does not erase an earlier failure.
Synthetic storage checks, live-model quality and deployed-user acceptance are separate evidence.

## Reliability and dependencies

| Record | What it establishes | Important limit |
| --- | --- | --- |
| [November reliability recheck](2026-11-reliability-recheck.md) | Reproduced defects, regression fixes, full CI and synthetic/live-provider checks for PR #129. | No independent consumer acceptance, physical delivery or production deployment. |
| [November dependency audit](2026-11-dependency-audit.md) | Compatible dependency/tool updates, tested graphs and scan applicability. | Residual OS/tooling findings remain; results are dated. |
| [Advisory inventory](2026-11-dependency-findings.json) | Sanitized advisory identifiers and package matches supporting that audit. | A scanner match is not proof of a reachable exploit. |

The reliability report links public CI runs. Some detailed logs and private historical-secret
triage were retained in the owner's workspace rather than committed. Public summaries do not
make those private artifacts available to a fresh clone.

## Memory, retrieval and readers

| Record | Scope and result |
| --- | --- |
| [September baseline](2026-09-phase0-baseline.md) | Original deterministic/live retrieval, fingerprint and archive-floor measurements. |
| [Historical retrieval summary](historical-retrieval-summary.md) | Earlier bundled-probe table and real-archive 0.6B ablation/floor results; raw 0.6B artifacts unavailable. |
| [September closing checks](2026-09-phase0-closing.md) | Migration, recovery, provider and configuration acceptance against that baseline. |
| [Temporal candidate filter](temporal-read-candidate-filter.md) | Bounded SQL candidate filtering with exact Python interval checks. |
| [Consolidation resource accounting](consolidation-basis-resource-v1.md) | Synthetic preparation timings and Python allocation measurements. |
| [Equal-budget evaluation](equal-budget-2026-10-01.md) | Frozen reader comparison; no broad answer-quality superiority established. Raw run material was retained locally. |
| [Context inspection](2026-10-context-inspection.md) | Deterministic source/lifecycle inspection; invalid instruments and corrections are retained. |
| [Memory scale](2026-10-memory-scale.md) | One bounded synthetic scale/restart/forget run; not autonomous-agent quality. |
| [Native caller](2026-10-native-caller-memory.md) | A bounded synthetic owner-assistant panel passed its frozen gate; not general readiness. |
| [Reader controls](2026-10-reader-controls.md) | Five simple constrained controls passed once; broader product admission remains separate. |

Machine-readable companion summaries:
[inspection](2026-10-context-inspection-summary.json),
[scale](2026-10-memory-scale-summary.json),
[native caller](2026-10-native-caller-memory-summary.json) and
[reader controls](2026-10-reader-controls-summary.json).

## Task and owner workflows

| Record | Scope and result |
| --- | --- |
| [Task recovery](2026-11-task-recovery.md) | Dated execution-journal review and crash-recovery requirements. Later task contracts are indexed under [decisions](../decisions/README.md). |
| [Native owner review](2026-11-06-native-owner-review.md) | Synthetic macOS review/lifecycle checks and remaining physical interaction limits. |
| [Integrated owner model](2026-11-06-integrated-owner-model.md) | One pinned-runtime draft/review scenario with measured resources. |
| [Finite owner loop](2026-11-06-finite-owner-loop.md) | Bounded drafting, restart and STOP probe on a pinned wheel. |
| [Multiline compatibility](2026-11-06-multiline-compatibility.md) | Reader compatibility, crash recovery and upgrade/rollback rehearsal. |
| [Ownership prompt comparison](owner-quality-2026-11-06/README.md) | Rejected candidate: fewer misses but more false positives and unchanged exact accuracy. |
| [Alternative models and validator](task-validation-2026-11-06/README.md) | Alternatives/second pass not adopted under the frozen deadline and non-regression rules. |

The two experiment directories include frozen protocols, cases, prompts, raw outputs, summaries
and reproduction scripts. Their README files explain which model files and pinned runtime are
required. Do not alter a frozen prompt or expected result to make a later implementation pass.

## Standalone workflow receipts

These JSON records accompany the linked decision documents. Preserve their observed results
and protocol fields; do not treat a receipt as proof of a different runtime or a later deployment.

| Contract | Records |
| --- | --- |
| [Current personal plan](../decisions/current-personal-plan-pilot.md) | [Pilot](2026-11-current-plan-pilot.json), [model check](2026-11-current-plan-model-check.json) |
| [Guided plan](../decisions/guided-personal-plan.md) | [Plan](2026-11-guided-plan.json), [direct path](2026-11-guided-direct.json) |
| [Guided status](../decisions/guided-status.md) | [Status](2026-11-guided-status.json) |
| [Unavailability](../decisions/confirmed-unavailability.md) | [Planner](2026-11-unavailability-pilot.json), [guided flow](2026-11-guided-unavailability.json) |
| [Owner input](../decisions/local-owner-input.md) | [Input receipt](2026-11-owner-input.json) |
| [Bounded model cycle](../decisions/bounded-model-cycle.md) | [Baseline](2026-11-model-cycle-baseline.json), [revised run](2026-11-model-cycle-revised.json) |
| [Reminder extraction](../decisions/missed-reminder-evaluation.md) | [Regression](2026-11-reminder-regression.json) |
| [Reviewed renewal](../decisions/reviewed-task-renewal.md) | [Rejected automatic matching](2026-11-task-update-matching.json) |
| [Pinned runtime](../decisions/pinned-runtime.md) | [Original runtime receipt](2026-11-pinned-runtime.json) |

## Reading and adding evidence

A useful record states the question, source revision, environment, exact command or protocol,
dataset identity, attempted denominator, failures and conclusion. Distinguish a recorded result
from illustrative output and a reproduction script from evidence that it was executed.

Report missing local artifacts as unavailable to the reader. Keep private data and credentials
out of public receipts. Hashes identify bytes; they do not authenticate a source or prove that
the measured system is safe. For current development checks, see
[contributing](../../CONTRIBUTING.md).
