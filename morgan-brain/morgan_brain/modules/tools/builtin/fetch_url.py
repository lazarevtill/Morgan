"""FetchUrlTool — fetches the text content of a URL via an injectable HTTP client.

The HTTP client is injectable so unit tests can pass a fake that returns
canned text without hitting the network.  In production the default
``httpx.AsyncClient`` is used.

Permission default: ASK (side-effecting / network egress).
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from morgan_brain.interfaces.tools import ToolResult


@runtime_checkable
class AsyncHttpClient(Protocol):
    """Minimal interface satisfied by ``httpx.AsyncClient`` and test fakes."""

    async def get(self, url: str, **kwargs: Any) -> Any:
        """Perform a GET request and return a response-like object."""
        ...


class FetchUrlTool:
    """Fetch the text body of a URL.

    Parameters
    ----------
    http_client:
        An ``AsyncHttpClient``-compatible object.  Defaults to a lazily
        created ``httpx.AsyncClient`` instance.  Inject a fake in tests to
        avoid any network I/O.

    Notes
    -----
    * Default permission mode is ASK — callers should set AUTO only after
      an explicit egress-allowlist grant has been installed.
    * Response text is truncated to ``max_chars`` characters (default 8 000)
      to prevent unbounded memory usage.
    """

    name = "fetch_url"
    description = "Fetch the text content of a URL (requires explicit permission)."
    default_permission = "ask"  # advisory; enforced by the PermissionGate

    def __init__(
        self,
        http_client: AsyncHttpClient | None = None,
        *,
        max_chars: int = 8_000,
    ) -> None:
        self._client = http_client
        self._max_chars = max_chars

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
        effective_client: AsyncHttpClient
        if self._client is not None:
            effective_client = self._client
        else:
            # Lazy import — httpx is a core dependency; import here keeps the
            # module importable even if httpx is somehow absent in a test stub.
            import httpx

            effective_client = httpx.AsyncClient()  # type: ignore[assignment]

        try:
            response = await effective_client.get(url)
            text: str = response.text
            if len(text) > self._max_chars:
                text = text[: self._max_chars] + "\n[truncated]"
            return ToolResult(ok=True, output=text)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, error=str(exc))
