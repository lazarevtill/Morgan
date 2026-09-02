"""The bind guard: an unauthenticated listener must never be reachable off-machine.

The guard is the only thing standing between the zero-config convenience (no key configured
=> no authentication) and an open memory store on the overlay network, so these tests assert
both directions -- it refuses what it must, and it does not refuse what it must not.
"""

from __future__ import annotations

import pytest

from morgan_brain.network import (
    UNSET_API_KEY_SENTINEL,
    api_key_is_configured,
    assert_safe_bind,
    is_loopback,
)


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "127.0.0.53", "localhost", "LOCALHOST", " localhost ", "::1", "[::1]"],
)
def test_loopback_hosts_are_recognised(host: str) -> None:
    assert is_loopback(host) is True


@pytest.mark.parametrize(
    "host",
    [
        "0.0.0.0",  # every interface -- the old hardcoded default
        "::",
        "",  # uvicorn reads an empty host as every interface
        "100.64.0.7",  # a NetBird overlay address, the real deployment case
        "192.168.1.10",
        "brain.internal",  # a name is never assumed to be loopback
    ],
)
def test_non_loopback_hosts_are_rejected(host: str) -> None:
    assert is_loopback(host) is False


def test_placeholder_key_counts_as_unconfigured() -> None:
    assert api_key_is_configured("") is False
    assert api_key_is_configured(UNSET_API_KEY_SENTINEL) is False
    assert api_key_is_configured("a-real-key") is True


def test_refuses_to_expose_an_unauthenticated_listener() -> None:
    with pytest.raises(SystemExit) as excinfo:
        assert_safe_bind(
            host="0.0.0.0", api_key=UNSET_API_KEY_SENTINEL, surface="morgan-mcp (http)"
        )
    message = str(excinfo.value)
    assert "morgan-mcp" in message
    assert "MORGAN_API_KEY" in message


def test_empty_key_is_refused_too() -> None:
    with pytest.raises(SystemExit):
        assert_safe_bind(host="100.64.0.7", api_key="", surface="morgan-mcp (http)")


def test_the_message_names_the_placeholder_not_a_credential() -> None:
    """A startup refusal lands in logs and terminals, so it names the env var and the
    placeholder, never a key value.

    The guard only raises when no real key is configured, so there is no configured credential
    at that moment -- this asserts what the message *does* say. (An earlier version of this
    test claimed to prove the message "never echoes the configured key", which is unfalsifiable
    here for exactly that reason.)
    """
    with pytest.raises(SystemExit) as excinfo:
        assert_safe_bind(host="0.0.0.0", api_key="", surface="morgan-mcp (http)")
    message = str(excinfo.value)
    assert UNSET_API_KEY_SENTINEL in message
    assert "MORGAN_API_KEY" in message


def test_loopback_without_a_key_is_allowed() -> None:
    """The zero-config path a fresh clone depends on."""
    assert_safe_bind(host="127.0.0.1", api_key=UNSET_API_KEY_SENTINEL, surface="morgan-mcp (http)")


def test_non_loopback_with_a_key_is_allowed() -> None:
    """The owner's real deployment: bound to the overlay address, authenticated."""
    assert_safe_bind(host="100.64.0.7", api_key="a-real-key", surface="morgan-mcp (http)")
