"""Text-only end-to-end benchmark harness for Morgan.

Drives the REAL orchestrator assembled via :mod:`morgan_brain.composition` over
multi-turn TEXT conversations and measures the platform's core promise:
cross-turn recall, fact supersession, preference learning, tool execution, and
personalization injection — plus per-turn latency.

Two modes:

* **deterministic** (default) — fakes/in-memory backends already used by the test
  suite (``FakeChatClient``, ``FakeEmbedder``, ``InMemoryVectorIndex``,
  ``InProcessBus``). Proves WIRING, not model quality. Zero external services.
* **live** (opt-in via ``MORGAN_BENCH_LIVE=1``) — the configured LLM/embedding
  endpoint + Qdrant if reachable. Degrades gracefully (skips) when absent.

Runnable two ways::

    pytest -q tests/e2e
    python -m tests.e2e.run_bench            # emits JSON + markdown report
"""

from __future__ import annotations
