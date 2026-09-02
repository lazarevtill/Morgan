"""ChampionCache: a worker-promoted champion reaches live traffic within the TTL,
without a brain-api restart (it was previously read once at startup and cached forever)."""

from __future__ import annotations

import pytest

from morgan_brain.composition import ChampionCache


class _Version:
    def __init__(self, body: str) -> None:
        self.body = body


class _FakeRegistry:
    def __init__(self) -> None:
        self._body: str | None = None
        self._raise = False

    def promote(self, body: str) -> None:
        self._body = body

    def fail(self) -> None:
        self._raise = True

    async def champion(self, name: str) -> _Version | None:
        if self._raise:
            raise RuntimeError("registry down")
        return _Version(self._body) if self._body is not None else None


@pytest.mark.asyncio
async def test_champion_cache_serves_then_refreshes_after_ttl() -> None:
    t = [0.0]
    reg = _FakeRegistry()
    cache = ChampionCache(reg, "morgan-system", ttl_s=30.0, clock=lambda: t[0])  # type: ignore[arg-type]

    assert await cache.body() == ""  # no champion yet (first fetch at t=0)

    reg.promote("CHAMPION V1")  # the worker promotes a new champion
    assert await cache.body() == ""  # still cached within the TTL (t=0 < 30)

    t[0] = 31.0  # TTL elapsed
    assert await cache.body() == "CHAMPION V1"  # refreshed — live without a restart


@pytest.mark.asyncio
async def test_champion_cache_keeps_last_known_body_on_error() -> None:
    t = [0.0]
    reg = _FakeRegistry()
    reg.promote("GOOD")
    cache = ChampionCache(reg, "morgan-system", ttl_s=10.0, clock=lambda: t[0])  # type: ignore[arg-type]

    assert await cache.body() == "GOOD"

    reg.fail()  # registry hiccups
    t[0] = 20.0  # TTL elapsed → refresh attempted and fails
    assert await cache.body() == "GOOD"  # last-known body preserved, never blanked
