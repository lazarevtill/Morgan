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
falls back to ``PERSONAL_PROJECT`` (the same system-wide default the CLI falls back to outside
a git repo). This is the one place the CLI's git-root ``detect_project`` must NOT be copied.
``remember`` reports the fallback: a call with no ``project`` gets back
``"project_defaulted": true``, so a client can tell a silent default from a named one.

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
from uuid import uuid4

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from mcp.types import ToolAnnotations
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from morgan_brain.config import Settings, get_settings
from morgan_brain.logging_setup import configure_logging
from morgan_brain.models import PERSONAL_PROJECT
from morgan_brain.surfaces.cli.commands import (
    cmd_ask,
    cmd_facts,
    cmd_forget,
    cmd_recall,
    cmd_remember,
)
from morgan_brain.surfaces.network import (
    api_key_is_configured,
    assert_safe_bind,
    unauthenticated_peer_allowed,
)

TOOL_NAMES: tuple[str, ...] = ("remember", "recall", "facts", "forget", "ask_morgan")

#: The type FastMCP injects when a tool asks for it by annotation. Parameterized (rather than
#: bare ``Context``) because mypy --strict's ``disallow_any_generics`` rejects a generic used
#: without its arguments; ``ServerSession``/``Request`` match ``FastMCP.get_context``'s own
#: return type.
_ToolContext = Context[ServerSession, Any, Request]


def _client_name(ctx: _ToolContext | None) -> str:
    """The caller's ``clientInfo.name`` -- what a client identifies itself as at MCP
    handshake -- or ``"mcp"`` when none is available (a direct dispatch call in a test, or a
    client that skipped ``clientInfo``)."""
    if ctx is None or ctx.session.client_params is None:
        return "mcp"
    # str(): the SDK's ``Context.session`` property carries no return annotation, so mypy
    # infers the chain down to ``clientInfo.name`` as ``Any`` even though it is a ``str`` at
    # runtime.
    return str(ctx.session.client_params.clientInfo.name)


#: What each tool declares to a client, which decides from it whether a call may run without
#: asking. Read-only is the hint that lets it, so only the tools that read make that claim:
#: ``ask_morgan`` sounds like a query, but a turn stores both halves of the exchange. Every
#: tool states every hint, and registering a tool missing from this table fails, so none is
#: left to a client's defaults. Opening the database runs only its light migration steps
#: (``memory.migrations``), which re-derive stored data and change no memory a caller wrote.
#: While a heavy step waits for ``morgan migrate``, every write tool raises
#: ``DatabaseNeedsMigration``, and the SDK hands its message to the client as an error result.
TOOL_ANNOTATIONS: dict[str, ToolAnnotations] = {
    "remember": ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
    ),
    "recall": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
    "facts": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
    "forget": ToolAnnotations(
        readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False
    ),
    "ask_morgan": ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
    ),
}

#: The tools a client may run without asking, derived from the declarations above.
READ_ONLY_TOOLS: tuple[str, ...] = tuple(
    name for name in TOOL_NAMES if TOOL_ANNOTATIONS[name].readOnlyHint
)

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
    sentinel -- and in that open case, only from a loopback peer. These five tools include
    ``forget``, so an unauthenticated remote caller can erase a project."""

    def __init__(self, app: Any, settings: Settings) -> None:
        super().__init__(app)
        self._enforced = api_key_is_configured(settings.api_key)
        self._api_key = settings.api_key

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not self._enforced:
            if unauthenticated_peer_allowed(request.client.host if request.client else None):
                return await call_next(request)
            return JSONResponse({"error": "Invalid or missing API key."}, status_code=401)
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
    #: One id for this server process's whole lifetime -- every memory any client has this
    #: process store shares it, the way the CLI's empty ``session_id`` marks a one-shot process.
    session_id = uuid4().hex

    async def remember(
        text: str, project: str | None = None, ctx: _ToolContext | None = None
    ) -> dict[str, Any]:
        """Store a memory in a project. *project* is passed through as given -- ``None`` and
        all -- so ``cmd_remember`` is the one place that resolves it to ``PERSONAL_PROJECT``
        and reports whether it had to."""
        args = argparse.Namespace(text=text)
        return await cmd_remember(
            args,
            settings,
            project,
            client=_client_name(ctx),
            session_id=session_id,
        )

    async def recall(
        query: str,
        project: str | None = None,
        all_projects: bool = False,
        top_k: int = 8,
    ) -> dict[str, Any]:
        """Search memories by meaning and by keyword, project-scoped by default."""
        args = argparse.Namespace(query=query, all_projects=all_projects, top_k=top_k)
        return await cmd_recall(args, settings, project or PERSONAL_PROJECT)

    async def facts(
        project: str | None = None,
        subject: str | None = None,
        all_projects: bool = False,
    ) -> dict[str, Any]:
        """Currently-valid temporal facts for a project, optionally filtered by subject."""
        args = argparse.Namespace(subject=subject, all_projects=all_projects)
        return await cmd_facts(args, settings, project or PERSONAL_PROJECT)

    async def forget(project: str | None = None, all_projects: bool = False) -> dict[str, Any]:
        """Cascading erasure of everything stored under a project -- the same honest report
        (including which tables were skipped, and the path of the snapshot taken first, the
        undo) the ``morgan forget`` CLI prints."""
        args = argparse.Namespace(all_projects=all_projects)
        return await cmd_forget(args, settings, project or PERSONAL_PROJECT)

    async def ask_morgan(
        text: str, project: str | None = None, ctx: _ToolContext | None = None
    ) -> dict[str, Any]:
        """A full turn through the orchestrator (requires a reachable LLM)."""
        args = argparse.Namespace(text=text)
        return await cmd_ask(
            args,
            settings,
            project or PERSONAL_PROJECT,
            client=_client_name(ctx),
            session_id=session_id,
        )

    dispatch: dict[str, _ToolFn] = {
        "remember": remember,
        "recall": recall,
        "facts": facts,
        "forget": forget,
        "ask_morgan": ask_morgan,
    }
    for name, fn in dispatch.items():
        mcp.tool(name=name, annotations=TOOL_ANNOTATIONS[name])(fn)

    return MorganMcpServer(mcp=mcp, settings=settings, _dispatch=dispatch)


def main(argv: list[str] | None = None) -> int:
    # Over stdio, stdout *is* the JSON-RPC channel: a log line there is a framing error in
    # the client. Configured before anything that might log -- including reading settings.
    configure_logging()
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
