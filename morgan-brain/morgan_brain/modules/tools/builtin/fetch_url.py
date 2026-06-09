"""FetchUrlTool — fetches the text content of a URL via an injectable HTTP client.

The HTTP client is injectable so unit tests can pass a fake that returns
canned text without hitting the network.  In production the default
``httpx.AsyncClient`` is used.

Permission default: ASK (side-effecting / network egress).

Security hardening
------------------
* **Scheme whitelist**: only ``http`` and ``https`` are allowed.
* **SSRF / private-IP block**: the target hostname is resolved and every
  resolved IP is checked; loopback, link-local, private (RFC 1918), unique-local,
  multicast, reserved, and the cloud-metadata IP ``169.254.169.254`` are all
  rejected before any request is sent.
* **No redirect following**: ``follow_redirects=False`` is always passed; a
  redirect response is treated as a failure.
* **Egress allowlist**: if an ``egress_allowlist`` is configured, the resolved
  hostname must appear in the list; anything else is rejected.
* **Streamed body cap**: the response body is read via ``stream()`` and
  accumulation stops once the running byte count exceeds ``max_chars * 4``;
  this prevents gzip-bombs and large-body DoS.
* **Timeout**: an ``httpx.Timeout(10)`` is used so the tool can never hang
  indefinitely.
* **Accept-Encoding: identity**: sent to disable transparent decompression by
  httpx, which could otherwise silently inflate a compressed body.

The resolver callable (``socket.getaddrinfo`` by default) is injectable so
tests can simulate private-IP resolution without real DNS/network.
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.parse
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from morgan_brain.interfaces.tools import ToolResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALLOWED_SCHEMES = frozenset({"http", "https"})

# cloud-metadata IP — explicit check because is_link_local covers it, but be
# explicit for clarity.
_CLOUD_METADATA_IP = ipaddress.ip_address("169.254.169.254")

# Default request timeout (seconds).
_DEFAULT_TIMEOUT = 10.0


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class AsyncHttpClient(Protocol):
    """Minimal interface satisfied by ``httpx.AsyncClient`` and test fakes."""

    async def get(self, url: str, **kwargs: Any) -> Any:
        """Perform a GET request and return a response-like object."""
        ...


# ---------------------------------------------------------------------------
# SSRF helpers
# ---------------------------------------------------------------------------

# Type alias for getaddrinfo-like callables.
_ResolverType = Callable[..., list[Any]]


def _is_blocked_ip(ip_str: str) -> bool:
    """Return True if *ip_str* represents a private/reserved/blocked address."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        # Malformed IP → block as a precaution.
        return True

    if addr == _CLOUD_METADATA_IP:
        return True

    return (
        addr.is_loopback
        or addr.is_link_local
        or addr.is_private
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def _resolve_and_check(
    hostname: str,
    resolver: _ResolverType,
    egress_allowlist: list[str] | None,
) -> str | None:
    """Resolve *hostname* and validate it against SSRF and allowlist rules.

    Returns
    -------
    None
        If the host passes all checks.
    str
        An error message describing why the host was blocked.
    """
    # Egress-allowlist check (before DNS resolution — fast fail).
    if egress_allowlist is not None and hostname not in egress_allowlist:
        return f"host {hostname!r} is not in the egress allowlist"

    # DNS resolution + IP validation.
    try:
        infos = resolver(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except OSError as exc:
        return f"DNS resolution failed for {hostname!r}: {exc}"

    for info in infos:
        ip_str: str = info[4][0]
        if _is_blocked_ip(ip_str):
            return f"blocked host: {hostname!r} resolves to {ip_str!r} (private/reserved)"

    return None


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class FetchUrlTool:
    """Fetch the text body of a URL.

    Parameters
    ----------
    http_client:
        An ``AsyncHttpClient``-compatible object.  Defaults to a lazily
        created ``httpx.AsyncClient`` instance.  Inject a fake in tests to
        avoid any network I/O.
    max_chars:
        Maximum number of characters to return (default 8 000).  The streamed
        body accumulation stops at ``max_chars * 4`` bytes.
    egress_allowlist:
        If provided, only hostnames in this list may be contacted.
    resolver:
        Callable with the same signature as ``socket.getaddrinfo``.  Defaults
        to ``socket.getaddrinfo``.  Inject a fake in tests.

    Notes
    -----
    * Default permission mode is ASK — callers should set AUTO only after
      an explicit egress-allowlist grant has been installed.
    * Response text is truncated to ``max_chars`` characters to prevent
      unbounded memory usage.
    """

    name = "fetch_url"
    description = "Fetch the text content of a URL (requires explicit permission)."
    default_permission = "ask"  # advisory; enforced by the PermissionGate

    def __init__(
        self,
        http_client: AsyncHttpClient | None = None,
        *,
        max_chars: int = 8_000,
        egress_allowlist: list[str] | None = None,
        resolver: _ResolverType | None = None,
    ) -> None:
        self._client = http_client
        self._max_chars = max_chars
        self._egress_allowlist = egress_allowlist
        self._resolver: _ResolverType = resolver if resolver is not None else socket.getaddrinfo

    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Fully-qualified URL to fetch.",
                }
            },
            "required": ["url"],
        }

    async def run(self, *, user_id: str, url: str, **_: Any) -> ToolResult:
        # ------------------------------------------------------------------
        # 1. Scheme validation
        # ------------------------------------------------------------------
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in _ALLOWED_SCHEMES:
            return ToolResult(
                ok=False,
                error=f"scheme not allowed: {parsed.scheme!r} (only http/https)",
            )

        hostname = parsed.hostname or ""
        if not hostname:
            return ToolResult(ok=False, error="URL has no hostname")

        # ------------------------------------------------------------------
        # 2. SSRF / private-IP check (DNS resolution)
        # ------------------------------------------------------------------
        block_reason = _resolve_and_check(hostname, self._resolver, self._egress_allowlist)
        if block_reason is not None:
            return ToolResult(ok=False, error=block_reason)

        # ------------------------------------------------------------------
        # 3. Build effective client
        # ------------------------------------------------------------------
        if self._client is not None:
            effective_client = self._client
        else:
            import httpx  # lazy import — keeps module importable without httpx

            effective_client = httpx.AsyncClient(  # type: ignore[assignment]
                timeout=httpx.Timeout(_DEFAULT_TIMEOUT),
                follow_redirects=False,
            )

        # ------------------------------------------------------------------
        # 4. Streamed fetch with body cap; no redirect following
        # ------------------------------------------------------------------
        request_headers = {"Accept-Encoding": "identity"}
        byte_cap = self._max_chars * 4

        try:
            # Prefer stream() if supported (covers production httpx and our
            # new _StreamFakeClient); fall back to .get() for legacy fakes.
            if hasattr(effective_client, "stream"):
                chunks: list[bytes] = []
                total_bytes = 0
                async with effective_client.stream(
                    "GET",
                    url,
                    headers=request_headers,
                    follow_redirects=False,
                ) as resp:
                    # Reject redirects immediately.
                    sc = getattr(resp, "status_code", 200)
                    if 300 <= sc < 400:
                        return ToolResult(
                            ok=False,
                            error=f"redirect not followed (status {sc})",
                        )
                    async for chunk in resp.aiter_bytes(4096):
                        chunks.append(chunk)
                        total_bytes += len(chunk)
                        if total_bytes > byte_cap:
                            break
                raw = b"".join(chunks)
                text = raw.decode("utf-8", errors="replace")
            else:
                # Legacy path for old-style fakes that only implement .get()
                response = await effective_client.get(
                    url,
                    headers=request_headers,
                    follow_redirects=False,
                )
                sc = getattr(response, "status_code", 200)
                if 300 <= sc < 400:
                    return ToolResult(
                        ok=False,
                        error=f"redirect not followed (status {sc})",
                    )
                text = response.text

            if len(text) > self._max_chars:
                text = text[: self._max_chars] + "\n[truncated]"
            return ToolResult(ok=True, output=text)

        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, error=str(exc))
