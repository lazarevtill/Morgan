# Text E2E Wiring + Benchmark Report

**Date:** 2026-06-14
**Branch:** `bench/text-e2e` (HEAD `5dc37cb`)
**Scope:** The TEXT request path only — `/api/chat` and `/api/chat/stream`, perceive → personalize →
recall → skills → reason (+ bounded tool loop) → store/signal. Voice/vision out of scope (deferred).
**Method:** An additive, source-untouched benchmark harness (`morgan-brain/tests/e2e/`) drives the
**real orchestrator** via `composition._assemble` across multi-turn conversations, in two modes:
**deterministic** (fakes, zero external services) and **live** (real Ollama LLM + embeddings).

> Labels used throughout: **DETERMINISTIC** = fake LLM/embeddings, in-memory vector + in-proc bus;
> **LIVE** = real model inference against a configured OpenAI-compatible endpoint. Numbers are never
> mixed; every figure below is tagged with its mode.

---

## 1. Wiring verdict — is the full text request path actually wired end-to-end?

**Headline: PARTIAL.** The text path is wired and works end-to-end **only** in the single-process,
in-process-bus, blocking (`POST /api/chat`) configuration. The two configurations the docs present
as first-class — the **SSE streaming endpoint** and the **`MORGAN_EVENT_BUS=redis` two-process
production topology** — each have a real break that silently degrades the core promise. The entire
existing test suite exercises only the inproc + `handle_turn` happy path, so both breaks are green in CI.

| Subsystem | inproc + `/api/chat` | `/api/chat/stream` (SSE) | `MORGAN_EVENT_BUS=redis` (2-process) |
|---|---|---|---|
| Orchestrator cognitive loop (7 steps) | **YES** | PARTIAL — champion dropped | PARTIAL |
| Perception → Personalization → Reasoning injection | **YES** | **YES** (traits reach prompt) | YES (per turn) |
| Champion preprompt (`system_override`) threading | **YES** | **NO** — silently dropped | YES (but stale until restart) |
| Memory recall (episodic, hot path) | **YES** | **YES** | **NO** — process-local, cross-process recall dead |
| Memory facts (bi-temporal, consolidated) | **YES** | **YES** | YES (shared SQLite) — only after nightly consolidation |
| Session history (multi-turn threading) | YES (in-process only) | YES (in-process only) | **NO** — read but never written |
| Skills select + bounded tool loop | **YES** | YES (but effectively non-streaming) | YES |
| Cold-path store + signal (RESPONSE_GENERATED) | **YES** | **YES** | PARTIAL — base signal never recorded |
| Self-learning optimize loop closes at runtime | **NO** — restart required | **NO** | **NO** — restart required |

**Reading of the table.** The *cognitive computation* of a turn (perceive → personalize → recall →
reason) is correctly wired on every path; what fractures is **persistence and the learned-prompt
threading** — the parts that make the experience multi-turn and self-learning. In the documented
production topology (redis, two processes) session history is never written, cross-process episodic
recall is dead, and the per-turn base signal is never recorded — all because their only writers live
in an `InProcessBus`-gated subscriber that the redis path skips.

---

## 2. Benchmark results

Seven scenarios mirror the `tests/memory_quality` categories (single-hop, multi-hop, temporal,
preference, tools, personalization, champion). Each turn is timed; the report aggregates p50/p95 and
**recall accuracy** = fraction of recall-category scenarios passing. The bench runs entirely on the
**working in-process `/api/chat` path** — it threads history and the champion `system_override`
caller-side exactly as the blocking route does, deliberately avoiding the streaming/redis gaps.

### 2a. DETERMINISTIC mode (default, zero external services)

`python -m tests.e2e.run_bench` → **exit 0 — 7 passed, 0 failed, 0 skipped**

- Recall accuracy: **1.0**
- Latency (13 turns measured): **p50 = 0.205 ms, p95 = 0.312 ms** (fake LLM — measures wiring
  overhead only, **not** model speed)
- pytest entrypoint (`pytest -q tests/e2e`): **8 passed in 0.42s** (7 scenarios + aggregate smoke)

| scenario | category | status | detail |
|---|---|---|---|
| single_hop_recall | single-hop | PASS | 'Rust' present in turn-2 prompt |
| multi_hop_recall | multi-hop | PASS | both hops (acme & berlin) in final prompt |
| temporal_knowledge_update | temporal | PASS | current `lives_in={'Munich'}`; Berlin superseded |
| preference_learning_visible | preference | PASS | `length=terse` fragment in system prompt |
| tool_call_loop | tools | PASS | calculator invoked, result (42) threaded back, final `'6 x 7 = 42.'` |
| personalization_injection | personalization | PASS | `length=thorough` injected into system prompt |
| champion_preprompt_applied | personalization | PASS | champion preprompt present in system prompt |

### 2b. LIVE mode (real Ollama LLM + embeddings)

Live-service probe (run from this host on 2026-06-14):

- **LLM (Ollama OpenAI-compat) `http://localhost:11434/v1`** — **REACHABLE** (HTTP 200; chat via
  `llama3.2:latest` returned "OK"; `/v1/embeddings` with `nomic-embed-text:latest` returned a
  768-dim vector).
- **Qdrant `http://localhost:6333`** — **NOT reachable** (connection refused). Did not block live
  mode: bench used `MORGAN_VECTOR_BACKEND=memory`.
- **Redis `localhost:6379`** — **NOT reachable**. Not exercised by the bench (in-proc bus throughout).

> Config note: shipped defaults (`llm_model=qwen2.5:7b`, `embedding_model=qwen3-embedding:4b`) are
> not pulled on this host; the live run was pointed at models that exist (`llama3.2:latest`,
> `nomic-embed-text:latest`, dim 768).

`MORGAN_BENCH_LIVE=1 … python -m tests.e2e.run_bench` → **exit 0 — 7 passed, 0 failed, 0 skipped**

- Recall accuracy: **1.0**
- Latency (13 turns, real inference): **p50 = 1320 ms, p95 = 1919 ms**
- Genuine cross-turn recall confirmed in real model output:
  - single-hop: "Your favorite programming language is **Rust**…"
  - multi-hop: "Your employer, Acme Corp, is headquartered in **Berlin**!"
  - temporal: "…you've recently moved to **Munich**…" (latest fact wins)
- For preference/tool/personalization/champion scenarios, live mode asserts reachability + non-empty
  reply only (the real model decides tool use; internal fragments are not surfaced in output text).

### 2c. Supporting harnesses (DETERMINISTIC)

- Memory-quality (`pytest tests/memory_quality -v`): **exit 0 — 3 passed** (single-hop recall;
  knowledge-update latest-fact-wins; temporal history queryable).
- Eval gate (`pytest tests/unit/eval` + gated-promotion integration): **exit 0 — 100 passed**
  (88 in `tests/unit/eval`). Confirms the self-learning gate: better candidate promoted; worse/tie
  candidate rejected (registry unchanged); recall@k / F1@k / Cohen's kappa scorers green.
  Note: `tests/eval/` holds only the dataset `golden_set.json`; the runnable harness lives in
  `tests/unit/eval/`.

---

## 3. Confirmed gaps (with evidence and status)

All five subsystem audits independently confirmed the same two latent bugs. Severity per the audits.

### GAP-1 (MAJOR/BLOCKER per subsystem) — Streaming silently drops the learned champion preprompt

`orchestrator.stream_turn` has no `system_override` parameter and builds its `ReasoningRequest`
without it, so it defaults to `""`. The blocking path threads it via
`handle_turn_with_id(system_override=_champion_override)`; the SSE route calls `stream_turn(...)`
with no override. `build_messages` only prepends the champion when `system_override` is truthy, so
**every streamed turn runs on the base system prompt** — the eval-gated, promoted champion (the
entire visible output of the self-learning loop) is invisible on `/api/chat/stream`.

- Evidence: `core/orchestrator.py:184-229` (stream_turn signature + request build, no override);
  `apps/brain_api/app.py:77-82` (stream call, no override) vs `app.py:54-60` (blocking passes it);
  `interfaces/reasoning.py:27` (default `""`); `modules/reasoning/context/builder.py:24-26`.
- Test gap: `test_system_override.py` only asserts `handle_turn`/`handle_turn_with_id`;
  `test_stream.py` never inspects the system message.
- **Status: OPEN, not fixed.** The bench routes recall/champion assertions through the blocking
  path, so it does not mask this — but it also does not exercise SSE. Untested in CI.

### GAP-2 (BLOCKER) — Under `MORGAN_EVENT_BUS=redis`, session history is read but never written

brain-api builds `SessionHistoryStore()` and reads it on every turn (`history_store.recent`), but the
only writer (`_store_turn` inside `_register_turn_storage`) is registered **only when**
`isinstance(resolved_bus, InProcessBus)`. With redis, `get_event_bus()` returns `RedisStreamsBus`,
the guard is false, and the append subscriber is never wired in brain-api. The learning-worker — the
sole redis consumer of `RESPONSE_GENERATED` — has no `SessionHistoryStore` and never appends (its
handler only calls `process_session`). **Net: `history_store.recent()` always returns `[]` under
redis → every turn is treated as turn 1.** Multi-turn context silently collapses in the documented
production topology.

- Evidence: `composition.py:229-230` (subscriber gated on InProcessBus), `composition.py:311`
  (`SessionHistoryStore()` → `:memory:`); `bus/__init__.py:12-15` (redis → RedisStreamsBus);
  `apps/learning_worker/__main__.py:76-100` (worker handler never appends history);
  `composition.py:385-393` (worker `_assemble` called without `history_store`).
- **Status: OPEN, not fixed.** The bench uses the in-proc bus and so cannot and does not hit this.

### Related confirmed gaps (context for the two blockers)

- **MAJOR — cross-process episodic recall is dead under redis.** `MemoryModule` keeps the
  authoritative episodic record set in process-local dicts (`_by_id`/`_bm25`/`_entities`); `recall()`
  filters out any id not in *this* process's `_by_id` and never reconstructs a `Memory` from the
  Qdrant payload. Episodics the worker stores are invisible to brain-api recall regardless of
  `MORGAN_VECTOR_BACKEND=qdrant`. Only consolidated bi-temporal facts survive cross-process — and
  only after the nightly job. Evidence: `modules/memory/store.py:35-36,51-53,77`;
  `modules/memory/stores/vector.py:136-144`; `learning/learner.py:64-76`.
- **MAJOR — base per-turn signal never recorded under redis.** `recorder.record_turn` is only
  invoked from the same InProcessBus-gated subscriber; the worker handler never calls it. A later
  thumb-up then inserts a stub signal with empty `query`/`reply`, poisoning the optimizer's training
  set. Evidence: `composition.py:113-119,229`; `learning/recorder.py:108-123`; `learning/optimizer.py:127-135`.
- **MAJOR — promoted champion not served until brain-api restart.** `_champion_override` is read
  exactly once at startup and cached in the route closure; a worker-promoted champion never reaches
  live traffic without an operator restart. Evidence: `apps/brain_api/app.py:44,59`;
  `composition.py:266-286`; `learning/champion_trainer.py:141`.
- **MAJOR — optimize half of the loop never fires by default.** `enable_scheduling` defaults to
  `False`; consolidation + optimize cron jobs are built only inside the worker under that flag. Out
  of the box, the champion body stays empty forever. Evidence: `config.py:49`;
  `apps/learning_worker/__main__.py:260-272`.
- **MAJOR — edit/thumb/retry feedback never reaches the UserModel.** `UserProfileBuilder.build`
  derives the model exclusively from `current_facts()`; the injected `signals` dependency is assigned
  and never read; `preference_delta_from_edit`/`apply_edit_delta` have no production caller. The
  CIPHER learn-from-edits loop is dead code in production. Evidence: `learning/profile.py:102,108-208`;
  `apps/brain_api/routes.py:74` → `recorder.py:61-76`.
- **MAJOR — `SessionHistoryStore` is `:memory:` even on the working inproc path.** Constructed with
  no file path (unlike temporal/signals, which derive a sibling SQLite file), so all history is lost
  on every brain-api restart and can never be shared across processes. No `MORGAN_` setting points it
  at a durable file. Evidence: `composition.py:311`; `learning/history.py:50-60`; contrast
  `composition.py:300-304`.
- **MINOR — streaming is effectively non-streaming.** Three AUTO-granted builtin tools mean
  `ReasoningRequest.tools` is almost always non-empty; `ReasoningModule.stream()` then runs the full
  blocking loop and yields the whole answer as one chunk. The SSE seam exists but emits a single
  `data:` frame then `[DONE]`. Evidence: `modules/reasoning/reasoner.py:103-109`;
  `core/orchestrator.py:49-63,227`; `composition.py:147-165`.
- **MINOR — published-never-consumed events.** `PERCEPTION_COMPLETE` (3 publish sites) and
  `TOOL_INVOKED` (audit) have zero subscribers anywhere; `get_event_bus()` returns a fresh instance
  per call despite a "singleton" docstring; the champion registry path is relative `./data/prompts.db`
  (silent footgun if the two processes run from different cwds). Evidence: `core/orchestrator.py:79-81`;
  `modules/tools/executor.py:114-122`; `bus/__init__.py:9-18`; `learning_lifecycle/factory.py:48`.

### What is confirmed correctly wired (OK findings)

The inproc `/api/chat` path is genuinely end-to-end: `InProcessBus.publish` awaits subscribers
synchronously, so `_store_turn` persists episodic memory, records the base signal, and appends both
messages to history **before** the next request — making turn N see turns 1..N-1 within one process,
with the champion `system_override` correctly threaded. Signal-store / champion-registry sharing
across processes is correctly wired via shared SQLite file paths and a matching `morgan-system`
prompt name; the break is the live-reload and the base-signal/history writers, not store sharing.
Per-turn trait injection is correct on **both** blocking and streaming paths — only the champion
`system_override` is dropped on stream.

---

## 4. What it would take to get a fully LIVE benchmark

The live run above is "live LLM, single process, in-proc bus." A *fully* live benchmark — exercising
the documented production topology end-to-end — requires both infrastructure and source fixes:

**Infrastructure (operational, available today):**
1. Start **Qdrant** (`docker compose up -d qdrant`) and set `MORGAN_VECTOR_BACKEND=qdrant` for durable
   vectors.
2. Start **Redis** (`docker compose up -d redis`) and set `MORGAN_EVENT_BUS=redis`.
3. Run **two processes**: `brain_api` and `learning_worker` (the latter with
   `MORGAN_ENABLE_SCHEDULING=true`), sharing one cwd so `./data/prompts.db` resolves identically.
4. Pull the actual default models (or override to pulled ones) so config defaults match reality.

**Source fixes required before a 2-process live bench would actually pass (currently OPEN):**
1. **GAP-1:** add `system_override` to `stream_turn` and pass `_champion_override` from
   `chat_stream` — otherwise streamed turns fail any champion assertion.
2. **GAP-2:** register the history-append writer independent of bus type (e.g. always register
   `_store_turn` for the local-persistence concern, or give the worker a `SessionHistoryStore` and an
   append step) — otherwise every redis-mode turn is single-turn.
3. **Cross-process episodic recall:** have `recall()` reconstruct `Memory` objects from the Qdrant
   payload (or share the side metadata) — otherwise multi-turn episodic recall fails across processes.
4. **Base-signal write under redis** and **champion live-reload** (poll/refresh the registry, or
   subscribe to a promotion event) — otherwise the self-learning loop does not close at runtime.
5. Give `SessionHistoryStore` a durable file path from settings — otherwise restart wipes history.

**Then the bench would need a redis/streaming-aware mode** (currently it deliberately drives only the
working inproc blocking seam). That mode does not exist yet; adding it without the fixes above would
turn today's green bench red — which is the point.

---

## 5. Honest bottom line — proven vs unproven

**Proven (by this benchmark):**
- The **cognitive loop is correctly wired and computes the right thing** on the in-process blocking
  path: perception, personalization injection, multi-signal recall, temporal fact supersession,
  skill/tool selection, and the bounded tool loop all behave correctly — **DETERMINISTIC: 7/7 pass,
  recall 1.0**, and **LIVE: 7/7 pass, recall 1.0** with genuine real-model cross-turn recall
  (Rust / Berlin / Munich answered correctly by `llama3.2`).
- The **self-learning eval gate is real**: better candidates promote, worse/tie reject — **100
  tests pass** (DETERMINISTIC).
- Wiring overhead is negligible (**DETERMINISTIC p50 0.205 ms**); end-to-end latency is model-bound
  (**LIVE p50 1.32 s / p95 1.92 s** on `llama3.2:latest`), comfortably inside the "5–10 s
  thoughtful" budget.

**Unproven / known-broken (NOT covered by any passing test):**
- **Streaming with the learned champion** — `/api/chat/stream` drops it (GAP-1). The streamed
  assistant is *not* the learned one. **OPEN.**
- **The documented production topology (`redis`, two processes)** — session history is never written
  (GAP-2), cross-process episodic recall is dead, and the base per-turn signal is never recorded. A
  multi-turn conversation in production mode silently degrades to single-turn. **OPEN.**
- **Runtime closure of the self-learning loop** — a freshly promoted champion never reaches live
  traffic without a brain-api restart; the optimize half never runs without
  `MORGAN_ENABLE_SCHEDULING=true` + the worker; edit/thumb feedback never reaches the UserModel.
  **OPEN.**

**Net:** Morgan's headline — "self-learning, multi-turn, runs in production" — is **demonstrated only
in single-process inproc dev mode, on the blocking endpoint, with scheduling enabled, and only via
fact consolidation.** The bench is honest about this: it drives the one path that works, passes it in
both deterministic and live modes, and leaves the two production breaks visible and untouched (no
source was modified — `git diff --stat` against source is empty). The fixes are small and localized;
the report exists to keep them visible until they land.

---

### Appendix — how to reproduce

```bash
cd morgan-brain
# DETERMINISTIC (zero external services)
pytest -q tests/e2e
python -m tests.e2e.run_bench                  # → ./data/bench/text_e2e_report.{json,md}

# LIVE (configured LLM endpoint; SKIPs gracefully if unreachable)
MORGAN_BENCH_LIVE=1 MORGAN_LLM_ENDPOINT=http://localhost:11434/v1 \
  MORGAN_LLM_MODEL=llama3.2:latest MORGAN_LLM_FAST_MODEL=llama3.2:latest \
  MORGAN_EMBEDDING_MODEL=nomic-embed-text:latest MORGAN_EMBEDDING_DIM=768 \
  MORGAN_VECTOR_BACKEND=memory \
  python -m tests.e2e.run_bench --out ./data/bench/live
```

Harness source: `morgan-brain/tests/e2e/` (`harness.py`, `report.py`, `run_bench.py`,
`test_text_e2e.py`, `README.md`). Reports land in gitignored `data/bench/`.
