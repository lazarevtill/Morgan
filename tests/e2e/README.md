# Text E2E benchmark harness

An **additive, text-only** end-to-end benchmark that drives the **real orchestrator**
(assembled via `morgan_brain.composition._assemble`, the same seam the integration
suite uses) across multi-turn TEXT conversations and measures Morgan's core promise:
cross-turn recall, fact supersession, preference learning, tool execution, and
personalization injection — plus per-turn latency.

It proves **wiring**, not model quality. Nothing in `morgan_brain/` is modified.

## Modes

| Mode | Trigger | Backends | What it proves |
|------|---------|----------|----------------|
| **deterministic** (default) | none | `FakeChatClient`, `FakeEmbedder`, `InMemoryVectorIndex`, `InProcessBus`, in-memory SQLite | the 7-step loop is wired correctly, end to end, with zero external services |
| **live** (opt-in) | `MORGAN_BENCH_LIVE=1` | configured LLM/embedding endpoint (`MORGAN_LLM_ENDPOINT`), Qdrant if `MORGAN_VECTOR_BACKEND=qdrant` | the same loop works against real model + vectors |

Live mode **degrades gracefully**: if the LLM (or Qdrant when selected) is
unreachable, every scenario is marked `SKIP` and the run still exits `0`. It never
crashes on absent services.

## Scenarios (mirrors `tests/memory_quality` categories)

| Scenario | Category | Deterministic assertion |
|----------|----------|-------------------------|
| `single_hop_recall` | single-hop | a fact stated on turn 1 reaches a later turn's prompt |
| `multi_hop_recall` | multi-hop | two facts across turns both reach the final prompt to compose an answer |
| `temporal_knowledge_update` | temporal | a changed fact supersedes the old one; only the latest is current |
| `preference_learning_visible` | preference | a stated preference is consolidated and injected as a personalization fragment in a later turn |
| `tool_call_loop` | tools | a tool call executes and its result threads back into the next LLM call |
| `personalization_injection` | personalization | a known `UserModel` injects its fragment into the system prompt |
| `champion_preprompt_applied` | personalization | a promoted champion preprompt is prepended to the system prompt (blocking path) |

Each scenario records **per-turn latency**; the report aggregates p50/p95.

## Running

### As tests

```bash
pytest -q tests/e2e                              # deterministic (default)
MORGAN_BENCH_LIVE=1 pytest -q tests/e2e          # live (skips if services absent)
```

### As a script (emits JSON + markdown)

```bash
python -m tests.e2e.run_bench                     # → ./data/bench/text_e2e_report.{json,md}
python -m tests.e2e.run_bench --out ./reports     # custom output dir
MORGAN_BENCH_LIVE=1 python -m tests.e2e.run_bench # live mode
```

Exit code is `0` when every non-skipped scenario passes, `1` otherwise. Skipped
scenarios (live mode, services absent) do **not** fail the run.

## Why the deterministic path is the working path

The harness threads **session history** and the **champion `system_override`** on
the caller side, exactly the way the blocking `POST /api/chat` route does on the
in-process bus. It deliberately exercises the configuration that is wired today and
**avoids the known `redis`/streaming gaps** documented in the wiring analysis
(streaming drops the champion preprompt; under `MORGAN_EVENT_BUS=redis` session
history is read but never written). Those are out of scope for a green wiring gate;
this harness is the place to add a regression for them once they are fixed.

## Files

- `harness.py` — builders (deterministic/live), `ConversationHarness` driver,
  scenarios, scoring, the live reachability probe.
- `report.py` — JSON + markdown rendering.
- `run_bench.py` — script entrypoint.
- `test_text_e2e.py` — pytest entrypoint (one test per scenario + aggregate smoke).
