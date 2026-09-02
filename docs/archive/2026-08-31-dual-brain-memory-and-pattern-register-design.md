# Dual-brain memory + the pattern register — design

**Status:** design; implementation lands in four commits behind this document
**Date:** 2026-08-31
**Builds on:** [local-first reshape](2026-08-02-local-first-reshape-design.md)
**Sources:**
- Xie, Lang, An, Zhao, Yang, Li, Ma, Lin, Miao, Yan.
  *VoiceMem: Streaming Dual-Brain Memory for Real-Time Interaction.*
  arXiv:2608.26005v1 [eess.AS], 26 Aug 2026.
- [razzant/ouroboros](https://github.com/razzant/ouroboros) — `BIBLE.md`, principles 2
  (Meta-over-Patch), 3 (Immune Integrity), 12 (Epistemic Stability).

## 1. Why these two, and what is deliberately left behind

VoiceMem is a *voice* paper. Its voice half — streaming ASR, voiceprints, acoustic scene
memory, the 134 ms latency budget, the four-stage anticipatory retrieval that starts
matching before the speaker finishes — is out of scope here and stays out: Morgan's stated
principle is quality over speed, its perception is text, and GPU serving is deferred by
design. What transfers is the part that is not about audio at all, and the paper's own
ablation says it is the part that carries the result:

| Mechanism removed | Accuracy lost (LoCoMo / ES-MemEval / ChatMem / Memora) |
|---|---|
| Upper-layer index | **−9.9 / −5.3 / −6.7 / −4.4** |
| Right brain (persona) | −6.3 / −4.3 / −5.4 / −4.4 |
| Emergent clustering | −5.5 / −2.0 / −3.4 / −0.2 |
| Dual-horizon updating | −5.4 / −2.0 / −3.2 / −1.4 |
| Joint retrieval | −2.6 / −3.1 / −2.7 / −0.4 |

Two further results decide the shape below. The index is **backend-agnostic**: dropped onto
Mem0, LangMem and Zep unchanged it improved all three by 15.8–29.5 points, so it belongs
*above* Morgan's store, at the `MemoryGate`/`MemoryModule` seam, not inside SQLite. And the
gain comes from a **denser** candidate pool rather than a larger one: at top-5 VoiceMem beats
the strongest baseline by 8.1 points using 4.4× fewer tokens, and the curve is flat past
K=5. Morgan retrieves `top_k=8` today, so this is a precision change, not a budget change.

Ouroboros contributes nothing to retrieval. It contributes the governance a self-modifying
system needs, and Morgan already has the expensive half of it — an eval-gated optimizer that
never mutates the live champion. What it lacks is the two cheap halves: learning at the level
of *classes* rather than instances (Meta-over-Patch), and an explicit rule that the thing
being optimized may not weaken the gate that judges it (Immune Integrity). Ouroboros's
`BIBLE.md` states the second as a constitutional bound: *"Ouroboros may improve the immune
system; it may not weaken it."* For a system whose optimizer writes the prompt that its own
judge then reads, that is not philosophy — it is the reward-hacking guard.

Explicitly **not** taken: self-rewriting code, the swarm of specialist agents, the always-
loaded narrative identity file. The first two are a different product; the third contradicts
Morgan's hot-path context budget, and Morgan's champion preprompt already occupies that role.

## 2. What is broken today, and found while designing this

`MemoryModule.recall` fuses three rankings — vector, FTS5, entity overlap. **The entity
ranking is always empty in production.**

`MemoryModule.store` indexes `[e.name for e in memory.entities]`
(`modules/memory/store.py`), and no write path ever sets that field. `ConsolidationLearner.
process_session` constructs `Memory(...)` without `entities` (`learning/learner.py:72`), and
so does the CLI's `cmd_remember`. `Entity` objects are produced in exactly one place —
`TextPerception.analyze` — and that perception object is never carried into the memory that
gets stored. The only non-empty `memory_entities` rows in the entire test suite are ones the
tests insert by hand.

Two consequences. The reshape delivered "three signals that survive a restart"; in production
it is two. And the upper index below is an index *over entities*, so it cannot be built at
all until entities exist. Fixing the write path is therefore not adjacent work — it is step
one of §3, and it is a root-cause fix, not a graft prerequisite.

The extractor itself is also Latin-only: `_CAP_TOKEN` matches capitalised ASCII words
(`modules/perception/text/analyzer.py:39`). The reshape design established that a
substantial part of the intended corpus is non-English and rebuilt the keyword index on
FTS5 `unicode61` for exactly that reason. An entity extractor that cannot see non-Latin
scripts reproduces the bug one layer up.

## 3. Graft 1 — the semantic upper index (VoiceMem's left brain)

### 3.1 Shape

A two-level index *above* the store, exactly as the paper specifies: schemas for coarse
semantic routing, entities for locating concrete people, events and concepts.

```
G_L = (S, V, E)
  s = (description, N_macro, V_s)   schema: a coarse slot, holding entities
  v = (description, N_micro, I_v)   entity: belongs to exactly ONE schema; I_v indexes memories
  E = E_micro ∪ E_macro             entity↔entity and schema↔schema co-occurrence edges
```

Schema membership is stored on the entity row rather than as schema→entity edges — the
paper's own simplification, and the reason retrieval needs no recursive traversal.

`I_v` is not a new table: `memory_entities` already maps entity name → memory id, scoped by
`(user_id, project)`. Once §2 is fixed it is populated, and it becomes the leaf level of this
index for free.

### 3.2 Retrieval

Given the turn's query terms, inside one `(user_id, project)` scope:

1. **Match** query terms against entity names and schema names → `(V_t, S_t)`.
2. **Expand** to `Z_t = V_t ∪ V_{S_t} ∪ N_strong(V_t ∪ V_{S_t}) ∪ N_weak(V_t ∪ V_{S_t})`
   — one hop, strong edges (co-occurrence above a threshold) and weak edges, never two.
3. **Collect** `C_L = ⋃_{z ∈ Z_t} I_z` — the candidate memory ids.
4. **Search within `C_L`** rather than within the whole store.

Step 4 is where an implementation can quietly cheat, so it is pinned here. The narrowing is
pushed *into* each signal, not applied to its output: `FtsIndex.search`, `EntityIndex.search`
and `SqliteVectorIndex.search` take an optional `restrict_ids` and add it to their `WHERE`.
Filtering the top-16 after the fact is **not** the same mechanism — a relevant memory sitting
at rank 40 is never seen by a post-filter, and recovering it is the entire point of routing.
For a vector backend that lives outside this database (Qdrant), the restriction cannot join
the SQL, so that path over-fetches and filters in Python, and says so in its docstring.

**Routing never costs recall.** If matching finds nothing, or the candidate pool is empty,
`restrict_ids` is `None` and every signal searches unrestricted — the current behaviour,
exactly. The index can only narrow a search that had somewhere to narrow to. This is the
invariant the tests assert first.

### 3.3 Building it (cold path only)

A new `learning/semantic_index.py` job, driven off `RESPONSE_GENERATED` like consolidation:

1. Extract entities from the memory text — LLM (`reflection` role) with a deterministic
   script-aware fallback that, unlike `_CAP_TOKEN`, handles Cyrillic.
2. Write `memory_entities` rows — repairing §2.
3. Upsert entity nodes; assign each new entity to one schema from the preset slots
   (`work`, `health`, `daily_life`, `relationships`, `knowledge`, `goals` — the paper's six),
   plus whatever schemas have since emerged (§5).
4. Increment co-occurrence weights for every entity pair in the same memory, and for their
   schemas.

Nothing here runs on the request path. Step 7 of the orchestrator publishes; the worker does
the rest.

## 4. Graft 2 — the persona graph (VoiceMem's right brain)

The left brain records *what happened*; the right brain records *who the user is*. Morgan
already has the second idea as a flat `UserModel.traits` list scored by token overlap. The
paper's contribution is the distinction that list cannot express:

```
G_R = (V_I, V_C)
  v_I = (description, I)              intrinsic: an enduring disposition
  v_C = (description, I, ρ_{v,e})     cross-entity: an attitude, anchored to a left-brain entity
```

> *"This distinction is fundamental: `v_I` explains persistent user characteristics, whereas
> `v_C_e` preserves whom or what an emotion concerns. Collapsing the two would either mistake
> situational reactions for stable traits or remove the real-world causes that give affect its
> meaning."* — §3.2

"He is impatient" and "he is impatient **with the weekly Harbor sync**" are different claims,
and a flat trait list records the first when only the second is true. That is precisely the
over-personalization failure Morgan's own golden eval already probes
(`OVER_PERSONALIZATION_NEGATIVE`), which makes this measurable rather than decorative.

It also lands cleanly on an invariant Morgan already enforces. `MemorySource` exists so an
inference is never mistaken for a user's statement; a cross-entity node is the same discipline
applied to affect — the situational reading is kept situational until evidence promotes it.

**Two horizons, mapped onto the existing path split:**

| Paper | Morgan | Where |
|---|---|---|
| Short-horizon attribution (in-turn) | after the reply is sent | cold path of each turn, beside the signal recorder |
| Long-horizon attribution (post-session) | consolidate recurrent evidence into intrinsic nodes | nightly job |

The hot path only *reads*: `AdaptivePersonalizer` gains an optional persona-graph reader and
activates intrinsic nodes matching the turn plus cross-entity nodes anchored to the entities
the left brain already activated. That is the paper's joint retrieval, and it is why the two
grafts are one change: `Z_t` from §3.2 is the input to the right brain's expansion.

Promotion from cross-entity to intrinsic requires recurrence across sessions, never a single
turn. A stable trait asserted from one bad afternoon is the failure mode this whole structure
exists to prevent.

## 5. Graft 3 — cluster emergence

Preset slots go stale: the paper's own store ended with 49.8% of its items in two slots that
were never preset, each drawing from all six presets rather than refining within one. Rule-
based splitting fragments; so coherent sub-clusters are allowed to **emerge** from what is
actually retrieved together.

For a connected entity subset `H` and the queries `Q` seen in a window:

```
ρ(H) = (1/|Q|) · Σ_{q∈Q} |A_q ∩ H| / |A_q ∪ H|
```

where `A_q` is the entity set query `q` activated. High ρ means these entities are repeatedly
retrieved *together*. If the largest qualifying subgraph clears threshold α, an LLM judge
scores it for relevance, importance and completeness before it is promoted to a schema;
failing that, it is marked so it is not re-proposed every night.

This needs a co-retrieval log — `mem_query_activations`, written by the cold path, never the
hot one — and it runs nightly, not per turn.

## 6. Graft 4 — the pattern register and the gate-integrity invariant

### 6.1 Pattern register (Ouroboros, Principle 2)

Morgan mines high-value signals — edits beat retries beat thumbs — and hands them to the
reflection model as *instances*. Ouroboros's Meta-over-Patch says the useful unit is the
**class**: *"if this fix had existed six months ago, could today's failure still have reached
me through a different surface?"*, and its Pattern Register is the durable projection of
those classes, their counts, and the structural fix applied to each.

Morgan gets `learning/patterns.py` over a `learned_patterns` table: class id, title,
description, occurrence count, proposed structural fix, first/last seen, status. Signals are
grouped into classes by the reflection model; the register is then fed back *into* the
optimizer prompt, so a class seen eleven times is presented as a recurring class, not as
eleven unrelated edits. This is the mechanism by which the optimizer stops re-proposing the
same patch.

### 6.2 The gate may not be weakened (Ouroboros, Principle 3)

Morgan's optimizer writes a prompt. Morgan's judge reads the answer that prompt produced.
Nothing currently stops a candidate from scoring well by addressing the judge instead of the
user, and the promotion path has no memory of what the gate looked like when the champion was
scored. Two guards, in `eval/gate_integrity.py`:

- **The gate is fingerprinted.** A `GateSpec` — golden item count, a hash of the item ids,
  the judge model id, the scorer names, the thresholds — is captured when the champion is
  scored and again when the candidate is. A promotion whose two fingerprints disagree is
  refused: the candidate was measured against a different gate, so "beats current" is not a
  comparison. A gate with *fewer* items than the one that certified the champion is refused
  outright.
- **The candidate is screened for judge-directed text.** A candidate body containing
  instructions aimed at the evaluator rather than the user is refused before it is scored.

Neither guard can be satisfied by trying harder at the task, which is what makes them guards.

### 6.3 Decision receipts (Ouroboros, Principle 1)

> *"No optimization, compression, or caching strategy may destroy the ability to recover the
> exact prompt/context, tool schema, model route, and model output that shaped a decision."*

Every promotion decision writes a `decision_receipts` row: prompt name, champion version and
score, candidate body hash and score, the gate fingerprint, the per-scorer breakdown, the
judge model, the verdict and the reason. Promotions are currently a log line that scrolls
away; a receipt makes "why is the champion this?" answerable months later, and makes a
rollback an informed choice rather than a guess. Surfaced through `morgan receipts`.

## 7. Ordering, and how each step is proven

1. **Entity write path** (§2) — the entity ranking stops being empty. Proven by a test that
   stores a turn through the real path and asserts `memory_entities` is non-empty, and by a
   Cyrillic case the current extractor cannot pass.
2. **Upper index** (§3) — proven by the routing-never-costs-recall invariant, by a
   narrowing test showing a memory outside the pool is excluded, and against
   `tests/memory_quality/`, which is the harness CLAUDE.md requires for any memory change.
3. **Persona graph** (§4) — proven by a test that a single situational reaction does not
   become an intrinsic trait, and that a cross-entity node surfaces only when its anchor
   entity is active.
4. **Emergence, register, gate integrity, receipts** (§5, §6) — proven by unit tests per
   mechanism; the gate-integrity guards are proven by adversarial cases (a shrunken gate, a
   swapped judge, a candidate addressing the judge) that must all be refused.

Every step keeps `ruff check`, `ruff format --check`, `mypy --strict` and the full suite green,
and no step adds a synchronous write to the request path.
