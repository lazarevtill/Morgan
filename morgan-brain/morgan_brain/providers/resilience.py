"""Retry and role-level fallback helpers.

``with_retry`` — generic async retry with injectable jitter/sleep (deterministic in tests).
``RoleFallback`` — iterates a role's bindings, advancing on exception and publishing
    an ``LLM_FALLBACK`` event via an injected ``EventBus``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Sequence, TypeVar

from morgan_brain.interfaces.events import Event, EventBus, EventType
from morgan_brain.providers.router import Binding, RoleRouter

T = TypeVar("T")

# Default per-attempt base sleep (seconds) — intentionally tiny so accidental real
# invocations don't block long; tests always inject their own jitter/sleep.
_DEFAULT_BASE: float = 0.2


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    jitter: Sequence[float] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> T:
    """Call *fn* up to *attempts* times, sleeping between failures.

    Args:
        fn:       Zero-argument async callable to retry.
        attempts: Maximum number of tries (must be >= 1).
        jitter:   Optional sequence of sleep durations (one per inter-attempt gap).
                  If shorter than ``attempts - 1``, the last value is reused.
                  If ``None``, uses a fixed ``_DEFAULT_BASE`` per gap.
        sleep:    Injectable async sleep (default: ``asyncio.sleep``).  Pass a
                  no-op (``lambda _: asyncio.sleep(0)``) or similar to keep tests
                  deterministic without actual wall-clock delays.

    Returns:
        The return value of *fn* on the first successful call.

    Raises:
        The last exception raised by *fn* if all attempts are exhausted.
    """
    _sleep = sleep if sleep is not None else asyncio.sleep
    last_exc: BaseException | None = None

    for attempt in range(attempts):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < attempts - 1:
                if jitter is not None:
                    idx = min(attempt, len(jitter) - 1)
                    delay = jitter[idx]
                else:
                    delay = _DEFAULT_BASE
                await _sleep(delay)

    raise last_exc  # type: ignore[misc]


class RoleFallback:
    """Iterates bindings for a role, advancing on exception and publishing fallback events.

    Args:
        router: ``RoleRouter`` whose ``bindings_for(role)`` provides the ordered list.
        bus:    ``EventBus`` that receives ``LLM_FALLBACK`` events on each failover.

    Example::

        fb = RoleFallback(router, bus)
        result = await fb.call("strong", lambda client, model: client.agenerate(..., model=model))
    """

    def __init__(self, router: RoleRouter, bus: EventBus) -> None:
        self._router = router
        self._bus = bus

    async def call(
        self,
        role: str,
        fn: Callable[[Any, str], Awaitable[T]],
        **caps: Any,
    ) -> T:
        """Try each binding for *role* in order, falling back on exception.

        Args:
            role: Role name to look up.
            fn:   Async callable ``(client, model) -> T``.
            **caps: Passed to ``router.chat_for`` as capability flags (currently unused
                    in iteration — iteration is over the raw binding list to allow the
                    caller's ``fn`` to decide feasibility).

        Returns:
            Result of the first successful ``fn(client, model)`` invocation.

        Raises:
            The last exception if every binding fails.
        """
        bindings: list[Binding] = self._router.bindings_for(role)
        if not bindings:
            raise LookupError(f"No bindings registered for role {role!r}")

        last_exc: BaseException | None = None
        for i, binding in enumerate(bindings):
            try:
                return await fn(binding.client, binding.model)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                # Emit fallback event so observers can log/alert/trace.
                if i < len(bindings) - 1:
                    await self._bus.publish(
                        Event(
                            type=EventType.LLM_FALLBACK,
                            user_id="system",
                            payload={
                                "role": role,
                                "failed_provider": binding.provider,
                                "failed_model": binding.model,
                                "next_provider": bindings[i + 1].provider,
                                "next_model": bindings[i + 1].model,
                                "error": str(exc),
                            },
                        )
                    )

        raise last_exc  # type: ignore[misc]
