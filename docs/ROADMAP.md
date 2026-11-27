# Status and roadmap

Morgan provides project-scoped source storage, ranked retrieval, exact evidence inspection and
on-demand consolidation. The CLI and seven MCP tools share one memory gate. Optional finite
task, note, model and owner workflows are implemented separately; they keep execution journals
and do not install services. See the [workflow map](../README.md#explicit-workflows).

## Verified baseline

The [November reliability recheck](measurements/2026-11-reliability-recheck.md) records the
tested revision, full-suite results, live Ollama checks and remaining gaps.
[Dependency findings](measurements/2026-11-dependency-audit.md) are tracked separately.

Storage correctness, model answer quality and deployment readiness are different claims.
Synthetic providers can verify transactions, scope and recovery; they cannot establish
semantic retrieval, truthful model answers or delivery to a person.

## Retrieval measurements

The November 26 run used 86 synthetic memories, 60 labelled probes, `qwen3-embedding:8b`
at 4,096 dimensions, k=8 and no relevance floor. Its
[full configuration and results](measurements/2026-11-reliability-recheck.md#end-to-end-evidence)
are preserved in the report.

| Probe kind | Count | recall@8 | MRR | Stale first | Abstain |
| --- | --- | --- | --- | --- | --- |
| Overall | 60 | 0.89 | 0.51 | 0.70 | 0.00 |
| Single hop | 20 | 0.95 | 0.59 | — | — |
| Temporal | 8 | 1.00 | 0.66 | — | — |
| Knowledge update | 10 | 0.90 | 0.51 | 0.70 | — |
| Multi-hop | 8 | 0.62 | 0.16 | — | — |
| Unanswerable | 14 | — | — | — | 0.00 |

Stale-first measures whether an older answer outranks the intended current answer in the
knowledge-update subset. Those probes do not supply explicit revision edges. Implemented
revision filtering removes explicitly superseded events; it does not infer corrections from
unrelated prose. Multi-hop recall measures retrieval of supporting records, not composition
of a correct answer.

Earlier [September archive measurements](measurements/2026-09-phase0-baseline.md) calibrated
a floor margin of 0.08 for this model and 0.11 for Qwen3-Embedding-0.6B. Those values depend on
the model and corpus. The November run above had the floor disabled and does not revalidate
either threshold. Keep dated comparisons in their original reports.

## Current limitations

| Area | Limit and practical consequence |
| --- | --- |
| Ranking | Fusion has no recency term; implicit corrections can leave stale answers ranked first. |
| Multi-hop | Recall returns records; it does not combine them into a supported answer. |
| Consolidation | Selects through an empty-query recall, up to 50 results, rather than recent events. Facts consume part of that window; a floor can decline the query. |
| Native answers | Citation and schema checks do not prove entailment. Strict mode is experimental and disabled by default. |
| Embedding changes | A database retains its fingerprinted model/width. No automatic space retirement or re-embedding exists. |
| Import canary | Periodic checks can miss a transient wrong vector; replay does not repair suspect stored vectors. |
| Project identity | Same folder names share a project. MCP cannot infer the caller's repository. |
| Repository metadata | Git config includes and URL rewriting are not followed; configured code roots are not scanned. |
| MCP startup | FastMCP can fail while decoding an unrelated non-UTF-8 cwd `.env`. |
| Entities | Deterministic cased-script extraction misses uncased scripts and some names at sentence starts. |
| Owner workflows | Model drafts require review; automatic task identity matching remains rejected by its frozen evaluation. |
| Deployment | Independent consumer acceptance, physical GUI/push delivery and production recovery require separate checks. |

## Priorities

1. Measure retrieval improvements against frozen probes and relevant real-corpus questions,
   retaining both historical and current-answer tests.
2. Establish useful reader quality under the [reader contract](READER_CAPABILITY.md) before
   expanding autonomous interpretation or action.
3. Validate independent consumer integration and bounded owner workflows on explicitly selected
   data before treating synthetic or model-only checks as operational acceptance.
4. Address compatible upstream security fixes as they become available. Broader MCP/Python
   migrations remain separate work, as documented in the dependency audit.

A self-tuning learning loop, general autonomous assistant and permanent scheduling service
are not implemented by the memory core. Retired kernel designs live in the
[archive](archive/README.md); they are not an implementation backlog or current API reference.
