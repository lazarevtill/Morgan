"""
Click CLI for Morgan Client.

This module provides the command-line interface for interacting with Morgan.
It includes commands for chat, learning, memory management, and more.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from morgan_cli.client import (
    MorganClient,
    ClientConfig,
    ConnectionError as MorganConnectionError,
    RequestError,
    TimeoutError as MorganTimeoutError,
)
from morgan_cli.config import get_config
from morgan_cli import ui


console = Console()


# ============================================================================
# Command History Support
# ============================================================================


class CommandHistory:
    """Simple command history manager."""

    def __init__(self, max_size: int = 1000):
        """
        Initialize command history.

        Args:
            max_size: Maximum number of commands to store
        """
        self.history: list[str] = []
        self.max_size = max_size
        self.position = 0

    def add(self, command: str) -> None:
        """
        Add a command to history.

        Args:
            command: Command to add
        """
        # Only add non-empty commands (after stripping whitespace)
        if (
            command
            and command.strip()
            and (not self.history or self.history[-1] != command)
        ):
            self.history.append(command)
            if len(self.history) > self.max_size:
                self.history.pop(0)
        self.position = len(self.history)

    def get_previous(self) -> Optional[str]:
        """
        Get previous command in history.

        Returns:
            Previous command or None
        """
        if self.position > 0:
            self.position -= 1
            return self.history[self.position]
        return None

    def get_next(self) -> Optional[str]:
        """
        Get next command in history.

        Returns:
            Next command or None
        """
        if self.position < len(self.history) - 1:
            self.position += 1
            return self.history[self.position]
        elif self.position == len(self.history) - 1:
            self.position = len(self.history)
            return ""
        return None

    def get_all(self) -> list[str]:
        """
        Get all commands in history.

        Returns:
            List of commands
        """
        return self.history.copy()


# Global command history instance
_command_history = CommandHistory()


def get_command_history() -> CommandHistory:
    """
    Get the global command history instance.

    Returns:
        CommandHistory instance
    """
    return _command_history


# ============================================================================
# Helper Functions
# ============================================================================


def handle_error(error: Exception, server_url: str) -> None:
    """
    Handle and display errors appropriately.

    Args:
        error: The exception that occurred
        server_url: Server URL for context
    """
    if isinstance(error, MorganConnectionError):
        ui.render_connection_error(server_url, error)
    elif isinstance(error, MorganTimeoutError):
        ui.render_timeout_error(60)
    elif isinstance(error, RequestError):
        ui.render_server_error(error.status_code or 500, str(error), error.details)
    else:
        ui.render_error(error_message=str(error), error_type=type(error).__name__)


async def create_client(
    server_url: Optional[str], api_key: Optional[str], user_id: Optional[str]
) -> MorganClient:
    """
    Create and connect a Morgan client.

    Args:
        server_url: Server URL override
        api_key: API key override
        user_id: User ID override

    Returns:
        Connected MorganClient instance
    """
    config = get_config(server_url, api_key, user_id)
    client_config = ClientConfig(
        server_url=config.server_url,
        api_key=config.api_key,
        user_id=config.user_id,
        timeout_seconds=config.timeout_seconds,
        retry_attempts=config.retry_attempts,
        retry_delay_seconds=config.retry_delay_seconds,
    )

    client = MorganClient(client_config)
    await client.http.connect()
    return client


# ============================================================================
# CLI Group
# ============================================================================


@click.group()
@click.version_option(version="0.1.0", prog_name="morgan")
@click.option("--server-url", envvar="MORGAN_SERVER_URL", help="Morgan server URL")
@click.option("--api-key", envvar="MORGAN_API_KEY", help="API key for authentication")
@click.option("--user-id", envvar="MORGAN_USER_ID", help="User identifier")
@click.pass_context
def cli(
    ctx: click.Context,
    server_url: Optional[str],
    api_key: Optional[str],
    user_id: Optional[str],
):
    """
    Morgan CLI - Your personal AI assistant.

    A terminal client for interacting with Morgan, featuring empathic
    intelligence and knowledge capabilities.
    """
    # Store options in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["server_url"] = server_url
    ctx.obj["api_key"] = api_key
    ctx.obj["user_id"] = user_id


# ============================================================================
# Chat Command
# ============================================================================


@cli.command()
@click.pass_context
def chat(ctx: click.Context):
    """
    Start an interactive chat session with Morgan.

    This command opens an interactive terminal interface where you can
    have a conversation with Morgan. Type your messages and press Enter
    to send them. Use /help for available commands during the chat.
    """
    asyncio.run(
        _chat_interactive(
            ctx.obj.get("server_url"), ctx.obj.get("api_key"), ctx.obj.get("user_id")
        )
    )


async def _chat_interactive(
    server_url: Optional[str], api_key: Optional[str], user_id: Optional[str]
):
    """Interactive chat implementation."""
    ui.render_welcome()

    try:
        client = await create_client(server_url, api_key, user_id)
    except Exception as e:
        handle_error(e, server_url or "http://localhost:8080")
        sys.exit(1)

    conversation_id = None

    try:
        while True:
            try:
                # Get user input
                user_message = ui.get_input("You")

                if not user_message:
                    continue

                # Add to history
                _command_history.add(user_message)

                # Handle special commands
                if user_message.startswith("/"):
                    command = user_message[1:].lower().strip()

                    if command in ("exit", "quit"):
                        ui.render_status("info", "Goodbye! 👋")
                        break
                    elif command == "help":
                        ui.render_help()
                        continue
                    elif command == "clear":
                        ui.clear_screen()
                        continue
                    elif command == "status":
                        status = await client.http.get_status()
                        ui.render_message(
                            f"**Status:** {status.get('status', 'unknown')}",
                            sender="System",
                        )
                        continue
                    elif command == "profile":
                        profile = await client.http.get_profile()
                        ui.render_message(
                            f"**User:** {profile.get('user_id', 'unknown')}\n"
                            f"**Trust Level:** {profile.get('trust_level', 0):.2f}\n"
                            f"**Interactions:** {profile.get('interaction_count', 0)}",
                            sender="System",
                        )
                        continue
                    elif command == "memory":
                        stats = await client.http.get_memory_stats()
                        ui.render_message(
                            f"**Total Conversations:** {stats.get('total_conversations', 0)}\n"
                            f"**Total Messages:** {stats.get('total_messages', 0)}",
                            sender="System",
                        )
                        continue
                    else:
                        ui.render_error(
                            f"Unknown command: /{command}",
                            error_type="Command Error",
                            suggestions=["Type /help for available commands"],
                        )
                        continue

                # Send message to Morgan
                with ui.show_progress("Morgan is thinking..."):
                    response = await client.http.chat(
                        message=user_message, conversation_id=conversation_id
                    )

                # Update conversation ID
                conversation_id = response.get("conversation_id")

                # Display response
                ui.render_message(
                    response.get("answer", ""),
                    sender="Morgan",
                    metadata={
                        "emotional_tone": response.get("emotional_tone"),
                        "confidence": response.get("confidence"),
                        "sources": response.get("sources", []),
                    },
                )

            except KeyboardInterrupt:
                ui.render_status("info", "\nGoodbye! 👋")
                break
            except Exception as e:
                handle_error(e, client.config.server_url)
                continue

    finally:
        await client.close()


# ============================================================================
# Ask Command
# ============================================================================


@cli.command()
@click.argument("question", nargs=-1, required=True)
@click.pass_context
def ask(ctx: click.Context, question: tuple[str, ...]):
    """
    Ask Morgan a single question and get a response.

    This command sends a single question to Morgan and displays the answer.
    It's useful for quick queries without starting an interactive session.

    Example:
        morgan ask "What is the weather like today?"
    """
    question_text = " ".join(question)
    asyncio.run(
        _ask_question(
            question_text,
            ctx.obj.get("server_url"),
            ctx.obj.get("api_key"),
            ctx.obj.get("user_id"),
        )
    )


async def _ask_question(
    question: str,
    server_url: Optional[str],
    api_key: Optional[str],
    user_id: Optional[str],
):
    """Ask a single question implementation."""
    try:
        client = await create_client(server_url, api_key, user_id)

        with ui.show_progress("Morgan is thinking..."):
            response = await client.http.chat(message=question)

        ui.render_message(
            response.get("answer", ""),
            sender="Morgan",
            metadata={
                "emotional_tone": response.get("emotional_tone"),
                "confidence": response.get("confidence"),
            },
        )

        await client.close()

    except Exception as e:
        handle_error(e, server_url or "http://localhost:8080")
        sys.exit(1)


# ============================================================================
# Learn Command
# ============================================================================


@cli.command()
@click.option(
    "--file",
    "-f",
    "file_path",
    type=click.Path(exists=True),
    help="Path to file to learn from",
)
@click.option("--url", "-u", help="URL to fetch and learn from")
@click.option("--content", "-c", help="Direct content to learn")
@click.option(
    "--type",
    "-t",
    "doc_type",
    default="auto",
    help="Document type (auto, pdf, markdown, text, html)",
)
@click.pass_context
def learn(
    ctx: click.Context,
    file_path: Optional[str],
    url: Optional[str],
    content: Optional[str],
    doc_type: str,
):
    """
    Add documents to Morgan's knowledge base.

    This command allows you to teach Morgan new information by providing
    documents, URLs, or direct content. Morgan will process and store
    this information for future reference.

    Examples:
        morgan learn --file document.pdf
        morgan learn --url https://example.com/article
        morgan learn --content "Important information to remember"
    """
    if not any([file_path, url, content]):
        click.echo("Error: Must provide --file, --url, or --content")
        sys.exit(1)

    asyncio.run(
        _learn_content(
            file_path,
            url,
            content,
            doc_type,
            ctx.obj.get("server_url"),
            ctx.obj.get("api_key"),
            ctx.obj.get("user_id"),
        )
    )


async def _learn_content(
    file_path: Optional[str],
    url: Optional[str],
    content: Optional[str],
    doc_type: str,
    server_url: Optional[str],
    api_key: Optional[str],
    user_id: Optional[str],
):
    """Learn content implementation."""
    try:
        client = await create_client(server_url, api_key, user_id)

        # Read file content if file path provided
        source = None
        if file_path:
            source = str(Path(file_path).absolute())
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

        with ui.show_progress("Processing document..."):
            response = await client.http.learn(
                source=source, url=url, content=content, doc_type=doc_type
            )

        ui.render_status("success", "Document learned successfully!")
        ui.render_message(
            f"**Documents Processed:** {response.get('documents_processed', 0)}\n"
            f"**Chunks Created:** {response.get('chunks_created', 0)}\n"
            f"**Processing Time:** {response.get('processing_time_seconds', 0):.2f}s",
            sender="System",
        )

        await client.close()

    except Exception as e:
        handle_error(e, server_url or "http://localhost:8080")
        sys.exit(1)


# ============================================================================
# Memory Command
# ============================================================================


@cli.command()
@click.option("--stats", is_flag=True, help="Show memory statistics")
@click.option("--search", "-s", help="Search conversation history")
@click.option("--cleanup", is_flag=True, help="Clean up old conversations")
@click.pass_context
def memory(ctx: click.Context, stats: bool, search: Optional[str], cleanup: bool):
    """
    Manage Morgan's conversation memory.

    This command provides tools for viewing memory statistics, searching
    past conversations, and cleaning up old data.

    Examples:
        morgan memory --stats
        morgan memory --search "python programming"
        morgan memory --cleanup
    """
    if not any([stats, search, cleanup]):
        click.echo("Error: Must provide --stats, --search, or --cleanup")
        sys.exit(1)

    asyncio.run(
        _manage_memory(
            stats,
            search,
            cleanup,
            ctx.obj.get("server_url"),
            ctx.obj.get("api_key"),
            ctx.obj.get("user_id"),
        )
    )


async def _manage_memory(
    show_stats: bool,
    search_query: Optional[str],
    do_cleanup: bool,
    server_url: Optional[str],
    api_key: Optional[str],
    user_id: Optional[str],
):
    """Memory management implementation."""
    try:
        client = await create_client(server_url, api_key, user_id)

        if show_stats:
            stats = await client.http.get_memory_stats()
            ui.render_message(
                f"**Total Conversations:** {stats.get('total_conversations', 0)}\n"
                f"**Active Conversations:** {stats.get('active_conversations', 0)}\n"
                f"**Total Messages:** {stats.get('total_messages', 0)}",
                sender="Memory Stats",
            )

        if search_query:
            with ui.show_progress("Searching memory..."):
                results = await client.http.search_memory(search_query)

            if results:
                ui.render_status("success", f"Found {len(results)} results")
                for result in results[:5]:  # Show top 5
                    ui.render_message(
                        f"**Q:** {result.get('message', '')}\n\n"
                        f"**A:** {result.get('response', '')}",
                        sender="Memory",
                        metadata={"relevance": result.get("relevance_score", 0)},
                    )
            else:
                ui.render_status("info", "No results found")

        if do_cleanup:
            if ui.confirm("Are you sure you want to clean up old conversations?"):
                with ui.show_progress("Cleaning up..."):
                    result = await client.http.cleanup_memory()
                ui.render_status("success", "Cleanup complete!")
                ui.render_message(
                    f"**Conversations Removed:** {result.get('removed', 0)}",
                    sender="System",
                )

        await client.close()

    except Exception as e:
        handle_error(e, server_url or "http://localhost:8080")
        sys.exit(1)


# ============================================================================
# Knowledge Command
# ============================================================================


@cli.command()
@click.option("--stats", is_flag=True, help="Show knowledge base statistics")
@click.option("--search", "-s", help="Search knowledge base")
@click.pass_context
def knowledge(ctx: click.Context, stats: bool, search: Optional[str]):
    """
    Manage Morgan's knowledge base.

    This command provides tools for viewing knowledge base statistics
    and searching stored documents.

    Examples:
        morgan knowledge --stats
        morgan knowledge --search "machine learning"
    """
    if not any([stats, search]):
        click.echo("Error: Must provide --stats or --search")
        sys.exit(1)

    asyncio.run(
        _manage_knowledge(
            stats,
            search,
            ctx.obj.get("server_url"),
            ctx.obj.get("api_key"),
            ctx.obj.get("user_id"),
        )
    )


async def _manage_knowledge(
    show_stats: bool,
    search_query: Optional[str],
    server_url: Optional[str],
    api_key: Optional[str],
    user_id: Optional[str],
):
    """Knowledge management implementation."""
    try:
        client = await create_client(server_url, api_key, user_id)

        if show_stats:
            stats = await client.http.get_knowledge_stats()
            ui.render_message(
                f"**Total Documents:** {stats.get('total_documents', 0)}\n"
                f"**Total Chunks:** {stats.get('total_chunks', 0)}\n"
                f"**Collections:** {', '.join(stats.get('collections', []))}",
                sender="Knowledge Stats",
            )

        if search_query:
            with ui.show_progress("Searching knowledge base..."):
                results = await client.http.search_knowledge(search_query)

            if results:
                ui.render_status("success", f"Found {len(results)} results")
                for result in results[:5]:  # Show top 5
                    ui.render_message(
                        result.get("content", ""),
                        sender="Knowledge",
                        metadata={
                            "source": result.get("source"),
                            "score": result.get("score", 0),
                        },
                    )
            else:
                ui.render_status("info", "No results found")

        await client.close()

    except Exception as e:
        handle_error(e, server_url or "http://localhost:8080")
        sys.exit(1)


# ============================================================================
# Health Command
# ============================================================================


@cli.command()
@click.option("--detailed", "-d", is_flag=True, help="Show detailed status information")
@click.pass_context
def health(ctx: click.Context, detailed: bool):
    """
    Check Morgan server health status.

    This command checks if the Morgan server is running and healthy.
    Use --detailed for comprehensive component status.

    Examples:
        morgan health
        morgan health --detailed
    """
    asyncio.run(
        _check_health(
            detailed,
            ctx.obj.get("server_url"),
            ctx.obj.get("api_key"),
            ctx.obj.get("user_id"),
        )
    )


async def _check_health(
    detailed: bool,
    server_url: Optional[str],
    api_key: Optional[str],
    user_id: Optional[str],
):
    """Health check implementation."""
    try:
        client = await create_client(server_url, api_key, user_id)

        if detailed:
            with ui.show_progress("Checking server status..."):
                status = await client.http.get_status()

            ui.render_status("success", "Server is healthy!")
            ui.render_message(
                f"**Status:** {status.get('status', 'unknown')}\n"
                f"**Version:** {status.get('version', 'unknown')}\n"
                f"**Uptime:** {status.get('uptime_seconds', 0):.0f}s",
                sender="Server Status",
            )

            # Show component status
            components = status.get("components", {})
            if components:
                ui.print_separator()
                console.print("[bold]Component Status:[/bold]")
                for name, comp_status in components.items():
                    status_str = comp_status.get("status", "unknown")
                    if status_str == "up":
                        console.print(f"  ✓ {name}: [green]{status_str}[/green]")
                    else:
                        console.print(f"  ✗ {name}: [red]{status_str}[/red]")
        else:
            health_status = await client.http.health_check()
            status_value = health_status.get("status", "unknown")

            if status_value == "healthy":
                ui.render_status("success", "Server is healthy! ✓")
            else:
                ui.render_status("warning", f"Server status: {status_value}")

        await client.close()

    except Exception as e:
        handle_error(e, server_url or "http://localhost:8080")
        sys.exit(1)


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted by user[/dim]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
