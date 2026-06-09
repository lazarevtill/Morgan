"""FakeChannel — deterministic in-process channel adapter for tests.

Satisfies the ``Channel`` Protocol without any network dependency.
Lets tests push inbound messages and inspect outbound sends.

Usage::

    ch = FakeChannel(name="test")
    await ch.start()

    # Push an inbound message (tests call handle_inbound on the gateway directly;
    # FakeChannel mainly tracks outbound sends from gateway.handle_inbound).
    await ch.send(OutboundMessage(chat_id="chat1", text="hello"))

    assert ch.sent == [OutboundMessage(chat_id="chat1", text="hello")]
"""

from __future__ import annotations

from morgan_brain.channels.base import OutboundMessage


class FakeChannel:
    """In-process channel adapter for tests.

    Attributes:
        sent:    Ordered list of ``OutboundMessage`` objects delivered via ``send()``.
        started: Whether ``start()`` has been called.
        stopped: Whether ``stop()`` has been called.
    """

    def __init__(self, name: str = "fake") -> None:
        self._name = name
        self.sent: list[OutboundMessage] = []
        self.started: bool = False
        self.stopped: bool = False

    @property
    def name(self) -> str:
        return self._name

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def send(self, msg: OutboundMessage) -> None:
        self.sent.append(msg)
