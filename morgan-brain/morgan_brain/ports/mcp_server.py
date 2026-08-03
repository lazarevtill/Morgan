"""The MCP server -- the second way to use Morgan.

The ``morgan`` CLI (Task 17) makes Morgan usable by a human at a terminal. This makes it
usable by every AI tool the owner already runs: Claude Code, Claude Desktop, or any other
client that speaks the Model Context Protocol. Both surfaces are thin adapters over the same
library facade -- every tool here calls the exact ``cli.__main__`` command handler the CLI
itself calls, which in turn goes through ``composition.build_memory_context`` /
``build_app_context`` and the one ``MemoryGate``. No memory logic is reimplemented here.

Five tools, deliberately capped
--------------------------------
``remember``, ``recall``, ``facts``, ``forget``, ``ask_morgan``. Every tool a server exposes
costs context window in every connected client on every request -- a five-tool server that
stays enabled beats a fifteen-tool one that gets switched off. Do not add a sixth.

Project scoping
----------------
An MCP client is usually working inside a repository, but this server is a long-lived daemon
on the owner's homelab -- its *own* working directory means nothing to a client running on a
laptop. Project therefore comes from the tool's explicit ``project`` argument; when omitted it
falls back to ``DEFAULT_PROJECT`` (the same system-wide default the CLI falls back to outside
a git repo). This is the one place the CLI's git-root ``detect_project`` must NOT be copied.

Two transports, because the deployment is remote-first
--------------------------------------------------------
* stdio -- a client on the same machine as the brain (a laptop running its own instance).
* streamable-HTTP with a bearer token -- the normal case: laptops reaching the homelab over
  NetBird. Reuses ``MORGAN_API_KEY``, the INBOUND key clients present to Morgan -- never
  ``MORGAN_LLM_API_KEY``, which is OUTBOUND to llama-server. They point in opposite directions.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from morgan_brain.cli.__main__ import cmd_ask, cmd_facts, cmd_forget, cmd_recall, cmd_remember
from morgan_brain.config import Settings, get_settings
from morgan_brain.models.memory import DEFAULT_PROJECT
from morgan_brain.security.network import api_key_is_configured, assert_safe_bind

TOOL_NAMES: tuple[str, ...] = ("remember", "recall", "facts", "forget", "ask_morgan")

_ToolFn = Callable[..., Awaitable[dict[str, Any]]]


@dataclass
class MorganMcpServer:
    """A ``FastMCP`` instance plus a direct dispatch table for the five tools.

    ``call_tool`` bypasses FastMCP's wire encoding (content blocks) and returns each tool's
    JSON-serializable dict straight through -- used both by real stdio/HTTP transports (via
    ``mcp``, whose own registered tools call the same underlying functions) and by any caller
    that wants a tool's result directly, no transport involved.
    """

    mcp: FastMCP
    settings: Settings
    _dispatch: dict[str, _ToolFn] = field(repr=False)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._dispatch:
            raise ValueError(f"Unknown tool: {name!r} (have: {', '.join(TOOL_NAMES)})")
        return await self._dispatch[name](**arguments)

    async def run_stdio_async(self) -> None:
        """Serve over stdio -- a client on the same machine as the brain."""
        await self.mcp.run_stdio_async()

    async def run_http_async(self, host: str, port: int) -> None:
        """Serve over streamable-HTTP with a bearer token -- laptops reaching the homelab
        over NetBird, the normal deployment case.

        Refuses to bind beyond loopback without an API key: the bearer middleware below is a
        no-op when none is configured, and these five tools include ``forget``.
        """
        import uvicorn

        assert_safe_bind(host=host, api_key=self.settings.api_key, surface="morgan-mcp (http)")
        self.mcp.settings.host = host
        self.mcp.settings.port = port
        app = self.mcp.streamable_http_app()
        app.add_middleware(_BearerAuthMiddleware, settings=self.settings)
        config = uvicorn.Config(
            app, host=host, port=port, log_level=self.mcp.settings.log_level.lower()
        )
        await uvicorn.Server(config).serve()


class _BearerAuthMiddleware(BaseHTTPMiddleware):
    """Enforce ``MORGAN_API_KEY`` as a bearer token on the streamable-HTTP transport, with the
    exact same policy ``apps/brain_api/auth.py`` applies to ``/api/*``: a request must present
    ``Authorization: Bearer <MORGAN_API_KEY>`` unless the key is empty or the ``"change-me"``
    sentinel, in which case the server is open."""

    def __init__(self, app: Any, settings: Settings) -> None:
        super().__init__(app)
        self._enforced = api_key_is_configured(settings.api_key)
        self._api_key = settings.api_key

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if self._enforced:
            authorization = request.headers.get("authorization", "")
            token = (
                authorization.removeprefix("Bearer ").strip()
                if authorization.startswith("Bearer ")
                else ""
            )
            if not token or token != self._api_key:
                return JSONResponse({"error": "Invalid or missing API key."}, status_code=401)
        return await call_next(request)


def build_server(settings: Settings | None = None) -> MorganMcpServer:
    """Build the MCP server over the same composition path the ``morgan`` CLI uses.

    ``settings`` defaults to a fresh ``Settings()`` read of the current environment -- not the
    cached ``get_settings()`` singleton -- so a caller that sets ``MORGAN_*`` env vars right
    before calling this function (tests; a future multi-instance host) is honored immediately.
    ``main()`` below explicitly passes ``get_settings()``, the one cached settings object every
    other production entry point in this repo uses.
    """
    settings = settings if settings is not None else Settings()
    mcp = FastMCP("morgan")

    async def remember(text: str, project: str | None = None) -> dict[str, Any]:
        """Store a memory in a project."""
        args = argparse.Namespace(text=text)
        return await cmd_remember(args, settings, project or DEFAULT_PROJECT)

    async def recall(
        query: str,
        project: str | None = None,
        all_projects: bool = False,
        top_k: int = 8,
    ) -> dict[str, Any]:
        """Multi-signal retrieval (vector + keyword + entity), project-scoped by default."""
        args = argparse.Namespace(query=query, all_projects=all_projects, top_k=top_k)
        return await cmd_recall(args, settings, project or DEFAULT_PROJECT)

    async def facts(
        project: str | None = None,
        subject: str | None = None,
        all_projects: bool = False,
    ) -> dict[str, Any]:
        """Currently-valid temporal facts for a project, optionally filtered by subject."""
        args = argparse.Namespace(subject=subject, all_projects=all_projects)
        return await cmd_facts(args, settings, project or DEFAULT_PROJECT)

    async def forget(project: str | None = None, all_projects: bool = False) -> dict[str, Any]:
        """Cascading erasure of everything stored under a project -- the same honest report
        (including which tables were skipped) the ``morgan forget`` CLI prints."""
        args = argparse.Namespace(all_projects=all_projects)
        return await cmd_forget(args, settings, project or DEFAULT_PROJECT)

    async def ask_morgan(text: str, project: str | None = None) -> dict[str, Any]:
        """A full turn through the orchestrator (requires a reachable LLM)."""
        args = argparse.Namespace(text=text)
        return await cmd_ask(args, settings, project or DEFAULT_PROJECT)

    dispatch: dict[str, _ToolFn] = {
        "remember": remember,
        "recall": recall,
        "facts": facts,
        "forget": forget,
        "ask_morgan": ask_morgan,
    }
    for name, fn in dispatch.items():
        mcp.tool(name=name)(fn)

    return MorganMcpServer(mcp=mcp, settings=settings, _dispatch=dispatch)


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="morgan-mcp",
        description="Morgan's MCP server -- five tools over the same memory the morgan CLI uses.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="stdio for a client on this machine (default); http for streamable-HTTP "
        "(bearer token from MORGAN_API_KEY), the normal case for laptops reaching the "
        "homelab over NetBird.",
    )
    parser.add_argument(
        "--host",
        default=settings.mcp_host,
        help="Bind host for --transport http (default MORGAN_MCP_HOST, loopback). Binding "
        "beyond loopback requires MORGAN_API_KEY.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.mcp_port,
        help="Bind port for --transport http (default MORGAN_MCP_PORT).",
    )
    args = parser.parse_args(argv)

    import asyncio

    server = build_server(settings)
    if args.transport == "stdio":
        asyncio.run(server.run_stdio_async())
    else:
        asyncio.run(server.run_http_async(args.host, args.port))
    return 0


if __name__ == "__main__":
    sys.exit(main())
