"""Where the inbound API key is defined, and the bind guard that makes it mandatory.

Morgan's two network listeners -- ``brain-api`` and the ``morgan-mcp`` streamable-HTTP
transport -- both skip authentication when ``MORGAN_API_KEY`` is empty or still the shipped
``"change-me"`` placeholder. That is a deliberate zero-config convenience: a fresh clone
should work on loopback without an env file, and locking the owner out of their own laptop
teaches nobody anything.

It stops being a convenience the moment the socket is reachable from another machine. On the
owner's real topology the laptops reach the homelab over a NetBird overlay, so a listener on
``0.0.0.0`` with authentication disabled is an open memory store on the overlay -- read, write
and *forget* on every project, to anyone who can route to it.

So the two conditions are coupled here, once, for both surfaces: bind beyond loopback and the
key becomes required. Not a warning -- warnings scroll past and the process keeps serving.
``assert_safe_bind`` refuses to start.
"""

from __future__ import annotations

import ipaddress

#: The placeholder shipped in ``.env.example``. Treated as "no key configured" everywhere --
#: ``apps/brain_api/auth.py``, ``ports/mcp_server.py`` and this module all import this name
#: rather than repeating the literal, so the policy has exactly one definition.
UNSET_API_KEY_SENTINEL = "change-me"

#: Hostnames that resolve to the loopback interface. Anything not proven loopback is treated
#: as remotely reachable -- an unresolvable name or an unfamiliar alias fails closed.
_LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain"})


def api_key_is_configured(api_key: str) -> bool:
    """True when *api_key* is a real credential rather than unset or the placeholder."""
    return bool(api_key) and api_key != UNSET_API_KEY_SENTINEL


def is_loopback(host: str) -> bool:
    """True only when binding *host* cannot accept a connection from another machine.

    Unrecognised hostnames return False. This is not name resolution: a DNS lookup here would
    make the safety of a bind depend on a resolver that can change under the process, and the
    failure direction of a wrong answer is an exposed listener.
    """
    candidate = host.strip().lower()
    if not candidate:
        # An empty bind host means "all interfaces" to uvicorn, the same as 0.0.0.0.
        return False
    if candidate in _LOOPBACK_NAMES:
        return True
    # Strip the brackets from an IPv6 literal written in URL form, e.g. "[::1]".
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return False
    # is_loopback covers 127.0.0.0/8 and ::1. The unspecified addresses (0.0.0.0, ::) are
    # explicitly not loopback -- they bind every interface.
    return address.is_loopback


def assert_safe_bind(*, host: str, api_key: str, surface: str) -> None:
    """Refuse to expose *surface* on a non-loopback *host* without an API key.

    Raises ``SystemExit`` with a message naming the fix. ``SystemExit`` rather than a custom
    exception because both callers are process entry points: this must read as a startup
    refusal, not a traceback from somewhere in the request path.
    """
    if is_loopback(host) or api_key_is_configured(api_key):
        return
    raise SystemExit(
        f"Refusing to start {surface} on {host}: MORGAN_API_KEY is not set "
        f"(or is still the '{UNSET_API_KEY_SENTINEL}' placeholder), so every request would be "
        f"accepted unauthenticated -- including forget(), which erases a project.\n"
        f"Set MORGAN_API_KEY to a real value, or bind 127.0.0.1 to serve this machine only."
    )
