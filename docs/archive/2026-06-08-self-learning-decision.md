# Decision Record — Self-Learning Mechanism

**Date:** 2026-06-08 · **Status:** Accepted · **Source:** Wave 0 research workflow `morgan-brain-research-wave0` (8 agents, web-researched, 2025–2026 sources)

## Context

Morgan must "learn from the owner" over time on **frozen, local, OpenAI-compatible models** (Ollama
is one example backend), single-owner, privacy-first, modest local GPU. The question: *how* should
learning change behavior — weights (LoRA), prompt (auto-optimized preprompt), memory (RAG), or hybrid?

## Decision

**Memory/RAG-first hybrid + an automatically-optimized "champion" preprompt. No LoRA by default.**
LoRA is a rare, eval-gated escalation, not the primary mechanism.

### Why (evidence)
- **RAG dominates LoRA for personalization.** LaMP/ICTIR 2025 (arXiv:2409.09510): RAG **+14.92%**,
  PEFT/LoRA **+1.07%**, hybrid **+15.98%** over baseline — LoRA adds only ~0.44pp on top of RAG.
- **Data scarcity.** LoRA needs ~500–10,000 clean instruction pairs; one user accrues these over
  months/years. Below ~100 it's just few-shot with extra steps.
- **Catastrophic forgetting** is real for vanilla LoRA even with frozen base weights.
- **Privacy.** RAG facts are editable/deletable rows (supports "forget that"); LoRA bakes personal
  data irreversibly into weights — destroys GDPR-style forgetting.
- **Behavior change without GPU.** A versioned preprompt/skill doc is the only no-train mechanism
  that genuinely changes global behavior; matches the "skills are trainable state" principle.

## The three substrates

1. **Memory (primary, always-on).** Bi-temporal fact rows (Qdrant + SQLite). Hot-path
   extract-as-filter; async nightly + idle "sleep" consolidation (Mem0 ADD/UPDATE/DELETE/NOOP,
   deterministic dedup, contradiction → `invalid_at` not delete, pinned episode timestamps, bounded
   forgetting via importance+recency+frequency + global downscaling). Reference: getzep/graphiti.
2. **Champion preprompt/skill docs (secondary, always-on).** `SOUL.md`/`USER.md` + per-skill `.md`,
   re-optimized offline by **`dspy.GEPA`** (reflection_lm = largest loadable local model; task model
   stays 7–14B). Adopt SkillOpt's *loop design* (bounded add/delete/replace on one champion, strict
   held-out gate) on the **real** `gepa-ai/gepa` — `skillopt` is **not** a package. Hard ~1,500-char
   budget; prefer delete/replace; push transient facts back to RAG.
3. **QLoRA (conditional escalation only).** See test below.

## LoRA escalation test (ALL FOUR must hold)
1. 500–1,000+ deduplicated, instruction-shaped, owner-curated pairs in a **stable** domain.
2. A **golden-eval-proven** persistent style/voice/format gap that prompt+RAG cannot close after a
   champion-preprompt cycle.
3. Preprompt token growth causing **measurable** latency/context pressure compaction can't relieve.
4. Acceptance of a batch versioned pipeline + loss of clean "forget-this" deletion.

If it fires: Unsloth QLoRA (4-bit NF4, rank 16–32, seq 2048; ~12 GB, RTX 3090/24 GB value pick) →
`save_pretrained_merged(16bit)` → `save_pretrained_gguf(q8_0|q4_k_m)` → `ollama create morgan:vN`.
**Always merge to a self-contained GGUF; never the standalone `ADAPTER` directive** (identity/quant/
template mismatch silently degrades output). Atomic version swap via model tag; no per-request
hot-swap (Ollama #9548) — a single user never needs it. Weekly-or-slower, from a frozen versioned
dataset, never online. If preference-tuning weights ever needed, use **KTO/ORPO, not DPO** (no clean
pairs for one user).

## Training-signal collection (instrument NOW, before any optimizer consumes it)
Typed records on the post-response async path: `(context, query, original_reply, user_edit, retry?,
thumb)`. Value order: **edits** (free ground-truth correction pairs, CIPHER-style → distill a NL
preference delta) > retries / "no, I meant…" > thumbs (**down** reliable; **up least reliable** —
correlates with sycophancy, arXiv:2507.23158). Feeds (a) memory consolidation and (b) GEPA example
mining (~20–50 examples + per-example NL feedback). **Eval/golden items are firewalled** from what
the assistant may consolidate, else the metric measures leakage.

## Validation gate — "beats-current-or-nothing"
3-layer offline pytest harness on local models → JSON scorecard (also CI regression protection):
- **L1 (every commit, no judge):** retrieval recall@k / F1 over vector+BM25+entity+RRF, LongMemEval
  taxonomy (+ optional LoCoMo).
- **L2 (the "did it learn ME" signal):** hand-authored 20–50 item per-user golden set tagged by
  probe type — explicit recall, implicit-trait inference, **preference UPDATE** (value after the
  owner changed their mind — the real test), long-gap decay, **over-personalization NEGATIVES**
  (stale pref must NOT be injected), and abstention. Binary preference-following accuracy; keep
  RealPref's 3 dims (Awareness/Alignment/Quality) **separate**.
- **L3:** held-out A/B (replay last week, memory-ON vs OFF, pairwise win-rate) + trait-incorporation
  rate **time series** (the evidence of learning).
- **Judge discipline:** different model family than the assistant, both answer orderings
  (order-invariance), rubric+CoT+length-normalization, **calibrate once** (~50 hand-labeled items,
  Cohen's κ ≥ ~0.6 to auto-trust; re-calibrate on judge swap). Run at short (~3K) and long (~30K)
  context.
- **Promotion flow:** optimize a *candidate*, never the live champion; gate on the full valset;
  ship behind a flag; A/B 24–48 h; auto-rollback on regression; keep N versioned champions/tags.

## Key risks (mitigations in the gate)
Sycophancy/over-personalization (downweight up-votes, OP-Bench negatives, keep correctness
independent of the user model) · optimizer regression (full-valset gate + flagged A/B + auto-
rollback) · uncalibrated local judge (cross-family + κ calibration) · eval leakage (firewall) ·
memory bloat / context decay (bounded forgetting + compact injected profile) · prompt bloat (char
budget) · premature LoRA (the 4-condition test) · local-LLM date hallucination (pin timestamps).

## Consequences for the roadmap
- **Pull forward** the memory learning-substrate + **eval harness** to the front of Phase 2 (they
  deliver +14.92% with no GPU). GEPA optimizer = second half of Phase 2 / start of Phase 3, gated on
  the eval harness existing.
- **Add a dedicated eval module** (`tests/eval/` / `morgan-eval`) — currently absent.
- **Reframe** "skills are the trainable state": memory/RAG primary, optimized skill-doc secondary,
  QLoRA conditional. (`ARCHITECTURE_V2.md` is deleted; superseded by this record, the design spec,
  and [the local-first reshape design](2026-08-02-local-first-reshape-design.md); its
  `pip install skillopt` row is void.)
- **Add** anti-sycophancy / over-personalization guardrails as first-class in Personalization &
  Proactivity, and the typed edit-capture signal layer.

## Implementation substrate — MLflow (local, off the hot path)

The self-learning lifecycle (optimize → version → evaluate → gate → promote → rollback → trace) is
implemented on **MLflow 3** running **fully local** (SQLite backend + local filesystem artifacts),
**not** hand-rolled. Adopt selectively; it lives only in the learning/eval plane, never the request
runtime.

- **GEPA via MLflow:** `mlflow.genai.optimize_prompts(predict_fn, train_data, prompt_uris,
  optimizer=GepaPromptOptimizer(reflection_model=<biggest local model>, max_metric_calls=…),
  scorers=[…])`. MLflow wraps GEPA — no separate DSPy orchestration needed. (Experimental as of
  MLflow 3.5 — isolate behind our own `Optimizer` seam.)
- **Champion preprompt = Prompt Registry:** `register_prompt` + `set_prompt_alias("champion", v)`;
  rollback = re-point the alias (atomic, instant). `load_prompt("prompts:/morgan-system@champion")`
  at inference.
- **Validation gate = `mlflow.genai.evaluate`** with custom `@scorer`s (L1 recall@k, L2 preference
  probes) + **Evaluation Datasets** (the golden set) + **`make_judge`** (3.4+) for the calibrated
  cross-family judge (measures judge accuracy → our Cohen's κ requirement). "Beats-current-or-nothing"
  = compare candidate vs `@champion` scores before re-pointing the alias.
- **LoRA (if escalated) = Model Registry** aliases for versioned `morgan:vN` + rollback (defer until
  the escalation test fires).
- **Tracing/observability:** `mlflow-tracing` **slim package** in the service layer only
  (~95% smaller; do not install full `mlflow` in the hot path).
- **Privacy hard rules:** set `MLFLOW_DISABLE_TELEMETRY=true` and `DO_NOT_TRACK=true`; keep the
  `reflection_model` and any judge models **local** (remote = a privacy-egress event); the MLflow
  store is owner data → lives under the same encryption + `delete_subject()` fan-out as the rest.

Alternatives ruled out: W&B/Weave & LangSmith (cloud-only — fail privacy), Langfuse (heavyweight
multi-service deploy; shallow eval), plain Git+pytest (viable but rebuilds the registry/eval/history
MLflow gives free). MLflow wins for *this* use case because it unifies GEPA + eval + champion
registry + tracking locally.

## Key citations
LaMP RAG-vs-PEFT arXiv:2409.09510 · GEPA arXiv:2507.19457 + dspy.ai/api/optimizers/GEPA +
github.com/gepa-ai/gepa · SkillOpt (design only) arXiv:2605.23904 · PrefEval arXiv:2502.09597 ·
MAPLE arXiv:2602.13258 · PersonaMem arXiv:2504.14225 · CIPHER/implicit-feedback arXiv:2507.23158,
arXiv:2404.15269 · Mem0 arXiv:2504.19413 · Graphiti #1489 · generative-agents arXiv:2304.03442 ·
Letta sleep-time · Unsloth→GGUF + Ollama #9548 · LongMemEval arXiv:2410.10813 · OP-Bench
arXiv:2601.13722 · judge-bias arXiv:2604.23178 + κ calibration arXiv:2510.09738.
