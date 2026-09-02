"""Streaming self-evolving-memory benchmark — assertions (Evo-Memory style).

Validates the platform's "learns from accumulated interactions" promise over a stream longer
than the bounded session-history window: durable knowledge is recalled at every distance, and a
mid-stream update propagates.
"""

from __future__ import annotations

import pytest

from tests.e2e.streaming import run_streaming


@pytest.mark.asyncio
async def test_streaming_recall_is_distance_independent_via_facts() -> None:
    report = await run_streaming(stream_len=24)
    # The benchmark genuinely queries past the history window (otherwise it proves nothing).
    assert report.max_recall_distance > report.history_window
    # Knowledge established before the stream is recalled at EVERY distance — the durable fact
    # layer carries it even when the originating turns have scrolled out of the history window.
    assert report.recall_accuracy == 1.0


@pytest.mark.asyncio
async def test_streaming_mid_stream_update_propagates() -> None:
    report = await run_streaming(stream_len=24)
    # Post-update queries return the new value (and not the stale one), past the window.
    assert report.update_accuracy == 1.0
    # And the stale value is no longer a current fact — temporal supersession held under stream.
    assert report.stale_after_update is True
