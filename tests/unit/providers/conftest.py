"""Timing an embedding call's budget from where the budget starts.

Building the call's HTTP client loads a TLS context, 0.2-0.5 s on a loaded Windows machine,
and is not charged to the budget: its clock starts once the client is built. A test that
times a budget times it from there, so the build's jitter is not mistaken for a wait.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest

#: How far past its budget a call may end: the event loop's own lateness, not a wait.
TOLERANCE = 0.3


class ClientReady:
    """When the embedder's HTTP client last finished building (``at``), and how long each
    build is made to take on top of its own (``build_seconds``)."""

    def __init__(self) -> None:
        self.at = 0.0
        self.build_seconds = 0.0

    def since(self) -> float:
        """Seconds since the client was ready: the time the budget governs."""
        return time.monotonic() - self.at


@pytest.fixture
def client_ready(monkeypatch: pytest.MonkeyPatch) -> ClientReady:
    """Records when each ``httpx.AsyncClient`` is ready."""
    ready = ClientReady()
    build = httpx.AsyncClient.__init__

    def recorded(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        build(self, *args, **kwargs)
        time.sleep(ready.build_seconds)
        ready.at = time.monotonic()

    monkeypatch.setattr(httpx.AsyncClient, "__init__", recorded)
    return ready
