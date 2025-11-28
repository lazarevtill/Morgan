"""
Click-based CLI for Morgan.

This wraps the existing command handlers in ``morgan.cli.app`` so users
get a modern Click UX while keeping the battle-tested logic.
"""

from types import SimpleNamespace
from typing import Optional

import click
import requests

from morgan import create_assistant
from morgan.cli import app as legacy_app
from morgan.utils.logger import setup_logging


class RemoteMorganClient:
    """Lightweight HTTP client that talks to a remote Morgan server."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def chat(self, message: str, conversation_id: Optional[str], user_id: Optional[str]):
        payload = {
            "message": message,
            "conversation_id": conversation_id,
            "user_id": user_id,
        }
        resp = self.session.post(
            f"{self.base_url}/api/chat", json=payload, timeout=60
        )
        resp.raise_for_status()
        return resp.json()


def _get_remote(ctx: click.Context) -> Optional[RemoteMorganClient]:
    return ctx.obj.get("remote_client")


def _get_morgan(ctx: click.Context):
    """Lazily create and cache the Morgan assistant."""
    if "morgan" not in ctx.obj:
        ctx.obj["morgan"] = create_assistant(config_path=ctx.obj.get("config"))
    return ctx.obj["morgan"]


@click.group()
@click.option("--config", type=click.Path(exists=False), help="Path to config file")
@click.option(
    "--remote-url",
    envvar="MORGAN_REMOTE_URL",
    help="Use a remote Morgan server (e.g., http://localhost:8080)",
)
@click.option(
    "--remote-api-key",
    envvar="MORGAN_REMOTE_API_KEY",
    help="Optional API key for the remote server",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logs")
@click.option("--debug", is_flag=True, help="Enable debug logs")
@click.pass_context
def cli(
    ctx: click.Context,
    config: Optional[str],
    remote_url: Optional[str],
    remote_api_key: Optional[str],
    verbose: bool,
    debug: bool,
):
    """Morgan - human-first AI assistant."""
    ctx.ensure_object(dict)
    ctx.obj.update(
        {
            "config": config,
            "verbose": verbose,
            "debug": debug,
        }
    )
    if remote_url:
        ctx.obj["remote_client"] = RemoteMorganClient(remote_url, remote_api_key)
    log_level = "DEBUG" if debug else ("INFO" if verbose else "WARNING")
    setup_logging(level=log_level)


@cli.command()
@click.option("--topic", help="Initial topic for the conversation")
@click.pass_context
def chat(ctx: click.Context, topic: Optional[str]):
    """Start interactive chat with Morgan."""
    remote = _get_remote(ctx)
    if remote:
        conversation_id = None
        user_id = None
        click.echo("Connected to remote Morgan server.")
        click.echo("Type your questions. Type 'quit' to exit.")
        while True:
            try:
                question = click.prompt("You", type=str, default="").strip()
            except (click.Abort, KeyboardInterrupt):
                break
            if not question:
                continue
            if question.lower() in {"quit", "exit", "q"}:
                break
            try:
                result = remote.chat(
                    message=question,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
                conversation_id = result.get("conversation_id", conversation_id)
                click.echo(f"Morgan: {result.get('answer')}")
            except Exception as exc:
                click.echo(f"[error] Remote chat failed: {exc}")
                break
    else:
        args = SimpleNamespace(topic=topic)
        legacy_app.cmd_chat(args, _get_morgan(ctx))


@cli.command()
@click.argument("question")
@click.option("--sources", is_flag=True, help="Include source references")
@click.option("--stream", is_flag=True, help="Stream the response")
@click.pass_context
def ask(ctx: click.Context, question: str, sources: bool, stream: bool):
    """Ask Morgan a single question."""
    remote = _get_remote(ctx)
    if remote:
        try:
            result = remote.chat(message=question, conversation_id=None, user_id=None)
            answer = result.get("answer") or result
            click.echo(answer)
            if sources and result.get("sources"):
                click.echo("\nSources:")
                for src in result["sources"]:
                    click.echo(f"- {src}")
        except Exception as exc:
            raise click.ClickException(f"Remote ask failed: {exc}") from exc
    else:
        args = SimpleNamespace(question=question, sources=sources, stream=stream)
        legacy_app.cmd_ask(args, _get_morgan(ctx))


@cli.command()
@click.argument("source", required=False)
@click.option("--url", help="URL to learn from")
@click.option(
    "--type",
    "doc_type",
    type=click.Choice(["auto", "pdf", "web", "code", "markdown", "text"]),
    default="auto",
    show_default=True,
    help="Document type",
)
@click.option("--progress/--no-progress", default=True, show_default=True)
@click.pass_context
def learn(
    ctx: click.Context,
    source: Optional[str],
    url: Optional[str],
    doc_type: str,
    progress: bool,
):
    """Teach Morgan from documents or URLs."""
    args = SimpleNamespace(source=source, url=url, type=doc_type, progress=progress)
    legacy_app.cmd_learn(args, _get_morgan(ctx))


@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8080, show_default=True, type=int)
@click.option("--api-only", is_flag=True, help="Start only the API server")
@click.pass_context
def serve(ctx: click.Context, host: str, port: int, api_only: bool):
    """Start Morgan's API/web server (optional)."""
    remote = _get_remote(ctx)
    if remote:
        raise click.ClickException("Serve command is only for local mode.")
    args = SimpleNamespace(
        host=host, port=port, api_only=api_only, debug=ctx.obj.get("debug", False)
    )
    legacy_app.cmd_serve(args, _get_morgan(ctx))


@cli.command()
@click.option("--detailed", is_flag=True, help="Show detailed health info")
@click.pass_context
def health(ctx: click.Context, detailed: bool):
    """Check system health."""
    remote = _get_remote(ctx)
    if remote:
        try:
            resp = remote.session.get(f"{remote.base_url}/docs", timeout=10)
            resp.raise_for_status()
            click.echo("Remote server reachable.")
        except Exception as exc:
            raise click.ClickException(f"Remote health check failed: {exc}") from exc
    else:
        args = SimpleNamespace(detailed=detailed)
        legacy_app.cmd_health(args, _get_morgan(ctx))


@cli.command()
@click.argument(
    "action",
    type=click.Choice(
        ["analyze", "plan", "execute", "validate", "rollback", "list-backups", "cleanup"]
    ),
)
@click.option("--collection", help="Collection name (analyze)")
@click.option("--source-collection", "--source", help="Source collection")
@click.option("--target-collection", "--target", help="Target collection")
@click.option("--batch-size", type=int, default=1000, show_default=True)
@click.option("--confirm", is_flag=True, help="Confirm destructive actions")
@click.option("--keep-days", type=int, default=30, show_default=True)
@click.pass_context
def migrate(
    ctx: click.Context,
    action: str,
    collection: Optional[str],
    source_collection: Optional[str],
    target_collection: Optional[str],
    batch_size: int,
    confirm: bool,
    keep_days: int,
):
    """Run migration utilities (knowledge collections, backups, cleanup)."""
    if _get_remote(ctx):
        raise click.ClickException("Migrations are only supported in local mode.")

    args = SimpleNamespace(
        migrate_action=action,
        collection=collection,
        source_collection=source_collection,
        target_collection=target_collection,
        batch_size=batch_size,
        confirm=confirm,
        keep_days=keep_days,
    )
    legacy_app.cmd_migrate(args, _get_morgan(ctx))


@cli.command()
@click.option("--stats", is_flag=True, help="Show memory statistics")
@click.option("--search", help="Search conversation history")
@click.option("--cleanup", type=int, help="Clean conversations older than N days")
@click.pass_context
def memory(ctx: click.Context, stats: bool, search: Optional[str], cleanup: Optional[int]):
    """Manage conversation memory."""
    args = SimpleNamespace(stats=stats, search=search, cleanup=cleanup)
    legacy_app.cmd_memory(args, _get_morgan(ctx))


@cli.command()
@click.option("--stats", is_flag=True, help="Show knowledge stats")
@click.option("--search", help="Search the knowledge base")
@click.option("--clear", is_flag=True, help="Clear all knowledge (confirm inside)")
@click.pass_context
def knowledge(ctx: click.Context, stats: bool, search: Optional[str], clear: bool):
    """Manage knowledge base."""
    args = SimpleNamespace(stats=stats, search=search, clear=clear)
    legacy_app.cmd_knowledge(args, _get_morgan(ctx))


@cli.command()
@click.option("--stats", is_flag=True, help="Show cache statistics")
@click.option("--metrics", is_flag=True, help="Show detailed cache metrics")
@click.option("--efficiency", is_flag=True, help="Show cache efficiency report")
@click.option("--clear", is_flag=True, help="Clear cache metrics (confirm inside)")
@click.option("--cleanup", type=int, help="Cleanup cache entries older than N days")
@click.option("--confirm", is_flag=True, help="Skip interactive confirmation prompts")
@click.pass_context
def cache(
    ctx: click.Context,
    stats: bool,
    metrics: bool,
    efficiency: bool,
    clear: bool,
    cleanup: Optional[int],
    confirm: bool,
):
    """Manage Git hash cache."""
    args = SimpleNamespace(
        stats=stats,
        metrics=metrics,
        efficiency=efficiency,
        clear=clear,
        cleanup=cleanup,
        confirm=confirm,
    )
    legacy_app.cmd_cache(args, _get_morgan(ctx))


@cli.command()
@click.option("--force", is_flag=True, help="Overwrite existing configuration")
@click.pass_context
def init(ctx: click.Context, force: bool):
    """Initialize Morgan in current directory."""
    args = SimpleNamespace(force=force)
    legacy_app.cmd_init(args)


def run():
    """Entry point for ``python -m morgan``."""
    cli(prog_name="morgan")
