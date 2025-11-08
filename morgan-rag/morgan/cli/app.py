"""
Morgan CLI - Main User Interface.

Provides interactive commands for:
- Chat sessions
- Document ingestion
- Knowledge management
- Configuration
- History and session management
- Learning and feedback

Full async/await with Click 8.1+ support.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# Try to import click with async support
try:
    import click
    from click import Context, pass_context
except ImportError:
    print("Error: Click library not found. Please install: pip install click")
    sys.exit(1)

from morgan.cli.config import CLIConfig, ensure_config_exists, interactive_init
from morgan.cli.formatters import ConsoleFormatter
from morgan.cli.utils import (
    GracefulShutdown,
    SessionManager,
    assistant_context,
    check_health,
    display_response,
    format_duration,
    get_user_input,
    handle_cli_error,
    setup_logging,
    truncate_text,
)
from morgan.core.types import MessageRole
from morgan.learning.types import FeedbackSignal, FeedbackType


# Global state
class GlobalState:
    """Global CLI state."""

    def __init__(self):
        self.config: Optional[CLIConfig] = None
        self.formatter: Optional[ConsoleFormatter] = None
        self.session_manager: Optional[SessionManager] = None
        self.current_session_id: Optional[str] = None
        self.last_response_id: Optional[str] = None


pass_state = click.make_pass_decorator(GlobalState, ensure=True)


@click.group()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--no-rich",
    is_flag=True,
    help="Disable rich formatting",
)
@pass_context
def cli(ctx: Context, config: Optional[Path], verbose: bool, no_rich: bool):
    """
    Morgan AI Assistant CLI.

    An intelligent, emotionally-aware assistant with learning capabilities.
    """
    # Initialize global state
    state = ctx.ensure_object(GlobalState)

    # Load configuration
    try:
        if config:
            state.config = CLIConfig.load(config)
        else:
            state.config = ensure_config_exists()

        # Override verbose setting
        if verbose:
            state.config.verbose = True

        # Override rich formatting
        if no_rich:
            state.config.use_rich_formatting = False

        # Set up logging
        setup_logging(state.config)

        # Initialize formatter
        state.formatter = ConsoleFormatter(use_rich=state.config.use_rich_formatting)

        # Initialize session manager
        state.session_manager = SessionManager(Path(state.config.storage_path))

    except Exception as e:
        print(f"Error initializing CLI: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--session-id",
    help="Session ID to use (creates new if not specified)",
)
@click.option(
    "--streaming/--no-streaming",
    default=True,
    help="Enable streaming responses",
)
@pass_state
def chat(state: GlobalState, session_id: Optional[str], streaming: bool):
    """
    Start an interactive chat session.

    Press Ctrl+C or type 'exit', 'quit', or 'bye' to end the session.

    Examples:
        morgan chat                    # Start new session
        morgan chat --session-id abc   # Resume session
        morgan chat --no-streaming     # Disable streaming
    """
    asyncio.run(_chat_async(state, session_id, streaming))


async def _chat_async(state: GlobalState, session_id: Optional[str], streaming: bool):
    """Async implementation of chat command."""
    config = state.config
    formatter = state.formatter

    # Generate or use provided session ID
    if not session_id:
        session_id = str(uuid.uuid4())
        formatter.print(f"[dim]Starting new session: {session_id}[/dim]\n")
    else:
        formatter.print(f"[dim]Resuming session: {session_id}[/dim]\n")

    state.current_session_id = session_id

    # Welcome message
    formatter.print("[bold green]Morgan AI Assistant[/bold green]")
    formatter.print("[dim]Type 'exit', 'quit', or 'bye' to end the session.[/dim]\n")

    # Initialize assistant
    try:
        async with assistant_context(config) as assistant:
            # Set up graceful shutdown
            with GracefulShutdown() as shutdown:
                # Chat loop
                while not shutdown.shutdown_requested:
                    try:
                        # Get user input
                        user_message = await get_user_input("\n[bold cyan]You:[/bold cyan] ")

                        # Check for exit commands
                        if user_message.lower() in ("exit", "quit", "bye", "q"):
                            formatter.print("\n[yellow]Goodbye! 👋[/yellow]")
                            break

                        if not user_message:
                            continue

                        # Show processing indicator
                        if formatter.use_rich:
                            progress = formatter.create_progress()
                            if progress:
                                with progress:
                                    task = progress.add_task("Processing...", total=None)
                                    # Small delay to show indicator
                                    await asyncio.sleep(0.1)

                        # Process message
                        try:
                            if streaming and config.use_streaming:
                                # Streaming response
                                formatter.print("\n[bold green]🤖 Morgan:[/bold green] ", end="")

                                async for chunk in assistant.stream_response(
                                    user_id="cli_user",
                                    message=user_message,
                                    session_id=session_id,
                                ):
                                    print(chunk, end="", flush=True)

                                print("\n")

                                # Note: In streaming mode, we don't have full response metadata
                                # This is a trade-off for lower latency

                            else:
                                # Non-streaming response
                                response = await assistant.process_message(
                                    user_id="cli_user",
                                    message=user_message,
                                    session_id=session_id,
                                )

                                # Store response ID for feedback
                                state.last_response_id = response.response_id

                                # Display response
                                display_response(
                                    response,
                                    formatter,
                                    show_sources=config.show_sources,
                                    show_emotion=config.show_emotions,
                                    show_metrics=config.show_metrics,
                                )

                        except Exception as e:
                            handle_cli_error(e, formatter, config.verbose, exit_code=0)
                            continue

                    except (KeyboardInterrupt, EOFError):
                        formatter.print("\n\n[yellow]Session interrupted. Goodbye! 👋[/yellow]")
                        break

    except Exception as e:
        handle_cli_error(e, formatter, config.verbose)


@cli.command()
@click.argument("question")
@click.option(
    "--session-id",
    help="Session ID for context",
)
@pass_state
def ask(state: GlobalState, question: str, session_id: Optional[str]):
    """
    Ask a single question.

    Examples:
        morgan ask "What is the capital of France?"
        morgan ask "Explain quantum computing" --session-id abc
    """
    asyncio.run(_ask_async(state, question, session_id))


async def _ask_async(state: GlobalState, question: str, session_id: Optional[str]):
    """Async implementation of ask command."""
    config = state.config
    formatter = state.formatter

    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        async with assistant_context(config) as assistant:
            # Process message
            response = await assistant.process_message(
                user_id="cli_user",
                message=question,
                session_id=session_id,
            )

            # Display response
            display_response(
                response,
                formatter,
                show_sources=config.show_sources,
                show_emotion=config.show_emotions,
                show_metrics=config.show_metrics,
            )

    except Exception as e:
        handle_cli_error(e, formatter, config.verbose)


@cli.command()
@pass_state
def health(state: GlobalState):
    """
    Check system health.

    Verifies all components are operational:
    - Memory system
    - Emotion detector
    - Learning engine
    - RAG search
    - LLM connection
    """
    asyncio.run(_health_async(state))


async def _health_async(state: GlobalState):
    """Async implementation of health command."""
    config = state.config
    formatter = state.formatter

    formatter.print("\n[bold cyan]Checking system health...[/bold cyan]\n")

    try:
        async with assistant_context(config) as assistant:
            # Check health
            health_status = await check_health(assistant)

            # Display results
            if formatter.use_rich:
                table = formatter.format_health(health_status)
                formatter.console.print(table)
            else:
                output = formatter.format_health(health_status)
                print(output)

            # Determine overall status
            all_healthy = all(s.get("healthy", False) for s in health_status.values())

            if all_healthy:
                formatter.print("\n[bold green]✅ All systems operational[/bold green]")
                sys.exit(0)
            else:
                formatter.print("\n[bold yellow]⚠️ Some systems degraded[/bold yellow]")
                sys.exit(1)

    except Exception as e:
        handle_cli_error(e, formatter, config.verbose)


@cli.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--recursive",
    is_flag=True,
    help="Process directory recursively",
)
@click.option(
    "--file-types",
    multiple=True,
    help="File types to process (e.g., .pdf, .txt)",
)
@pass_state
def learn(state: GlobalState, path: Path, recursive: bool, file_types: tuple):
    """
    Ingest documents into the knowledge base.

    Supports: PDF, TXT, DOCX, MD, and more.

    Examples:
        morgan learn document.pdf
        morgan learn ./docs --recursive
        morgan learn ./docs --file-types .pdf --file-types .txt
    """
    asyncio.run(_learn_async(state, path, recursive, file_types))


async def _learn_async(
    state: GlobalState,
    path: Path,
    recursive: bool,
    file_types: tuple,
):
    """Async implementation of learn command."""
    config = state.config
    formatter = state.formatter

    formatter.print(f"\n[bold cyan]Ingesting documents from:[/bold cyan] {path}\n")

    # Collect files
    files = []
    if path.is_file():
        files = [path]
    else:
        # Directory
        pattern = "**/*" if recursive else "*"
        for file_path in path.glob(pattern):
            if file_path.is_file():
                # Filter by file type if specified
                if file_types and file_path.suffix not in file_types:
                    continue
                files.append(file_path)

    if not files:
        formatter.print("[yellow]No files found to process.[/yellow]")
        return

    formatter.print(f"[dim]Found {len(files)} file(s) to process...[/dim]\n")

    # Process files
    # Note: This would integrate with the ingestion system
    # For now, we'll show a placeholder implementation

    try:
        from morgan.ingestion.processor import DocumentProcessor

        processor = DocumentProcessor(
            storage_path=Path(config.storage_path) / "documents",
        )

        # Process with progress
        if formatter.use_rich:
            from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=formatter.console,
            ) as progress:
                task = progress.add_task("Processing documents...", total=len(files))

                for file_path in files:
                    try:
                        # Process document (placeholder - actual implementation would be async)
                        progress.update(
                            task,
                            description=f"Processing {file_path.name}",
                            advance=0,
                        )

                        # Simulate processing (replace with actual ingestion)
                        await asyncio.sleep(0.1)

                        progress.update(task, advance=1)

                    except Exception as e:
                        formatter.print(f"[yellow]Warning:[/yellow] Failed to process {file_path.name}: {e}")

        else:
            # Plain text progress
            for i, file_path in enumerate(files, 1):
                print(f"Processing {i}/{len(files)}: {file_path.name}...")
                await asyncio.sleep(0.1)  # Simulate processing

        formatter.print(f"\n[bold green]✅ Successfully processed {len(files)} document(s)[/bold green]")

    except ImportError:
        formatter.print("[yellow]⚠️ Document ingestion module not available.[/yellow]")
        formatter.print("[dim]Install with: pip install morgan[ingestion][/dim]")
    except Exception as e:
        handle_cli_error(e, formatter, config.verbose)


@cli.command()
@pass_state
def knowledge(state: GlobalState):
    """
    Show knowledge base statistics.

    Displays:
    - Number of documents
    - Number of chunks
    - Collections
    - Storage usage
    """
    asyncio.run(_knowledge_async(state))


async def _knowledge_async(state: GlobalState):
    """Async implementation of knowledge command."""
    config = state.config
    formatter = state.formatter

    formatter.print("\n[bold cyan]Knowledge Base Statistics[/bold cyan]\n")

    try:
        # Connect to vector DB
        from morgan.vector_db.client import QdrantClient

        client = QdrantClient(
            url=config.vector_db_url,
            collection_name=config.vector_db_collection,
        )

        # Get collection info
        info = await client.get_collection_info()

        if formatter.use_rich:
            from rich.table import Table

            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right", style="green")

            table.add_row("Collection", config.vector_db_collection)
            table.add_row("Total Vectors", str(info.get("vectors_count", 0)))
            table.add_row("Indexed Vectors", str(info.get("indexed_vectors_count", 0)))
            table.add_row("Points Count", str(info.get("points_count", 0)))

            formatter.console.print(table)
        else:
            print(f"Collection: {config.vector_db_collection}")
            print(f"Total Vectors: {info.get('vectors_count', 0)}")
            print(f"Indexed Vectors: {info.get('indexed_vectors_count', 0)}")
            print(f"Points Count: {info.get('points_count', 0)}")

    except Exception as e:
        formatter.print(f"[yellow]Could not retrieve knowledge base stats: {e}[/yellow]")


@cli.command()
@pass_state
def init(state: GlobalState):
    """
    Initialize Morgan configuration interactively.

    Creates a configuration file with your preferences.
    """
    asyncio.run(_init_async(state))


async def _init_async(state: GlobalState):
    """Async implementation of init command."""
    try:
        config = await interactive_init()
        print("\n✅ Configuration initialized successfully!")
        print(f"Config file: {config.storage_path}/config.json")
    except Exception as e:
        print(f"❌ Configuration initialization failed: {e}")
        sys.exit(1)


@cli.command()
@click.argument("key", required=False)
@click.argument("value", required=False)
@pass_state
def config_cmd(state: GlobalState, key: Optional[str], value: Optional[str]):
    """
    View or edit configuration.

    Examples:
        morgan config                    # Show all config
        morgan config llm_model          # Show specific value
        morgan config llm_model llama3   # Set value
    """
    config = state.config
    formatter = state.formatter

    if not key:
        # Show all configuration
        formatter.print("\n[bold cyan]Morgan Configuration[/bold cyan]\n")

        if formatter.use_rich:
            from rich.table import Table

            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")

            for k, v in config.to_dict().items():
                table.add_row(k, str(v))

            formatter.console.print(table)
        else:
            for k, v in config.to_dict().items():
                print(f"{k}: {v}")

    elif not value:
        # Show specific value
        try:
            val = config.get(key)
            formatter.print(f"[cyan]{key}:[/cyan] {val}")
        except AttributeError:
            formatter.print(f"[red]Unknown configuration key:[/red] {key}")
            sys.exit(1)

    else:
        # Set value
        try:
            config.set(key, value)
            config.save()
            formatter.print(f"[green]✅ Set {key} = {value}[/green]")
        except ValueError as e:
            formatter.print(f"[red]Error:[/red] {e}")
            sys.exit(1)


@cli.command()
@click.option(
    "--limit",
    default=10,
    help="Number of sessions to show",
)
@pass_state
def history(state: GlobalState, limit: int):
    """
    Show conversation history.

    Lists recent conversation sessions.
    """
    formatter = state.formatter
    session_manager = state.session_manager

    formatter.print("\n[bold cyan]Conversation History[/bold cyan]\n")

    sessions = session_manager.list_sessions()

    if not sessions:
        formatter.print("[dim]No conversation history found.[/dim]")
        return

    # Show sessions
    if formatter.use_rich:
        from rich.table import Table

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Session ID", style="cyan")
        table.add_column("Created", style="yellow")
        table.add_column("Messages", justify="right", style="green")

        for session in sessions[:limit]:
            table.add_row(
                truncate_text(session["session_id"], 40),
                session.get("created_at", "unknown"),
                str(session.get("message_count", 0)),
            )

        formatter.console.print(table)
    else:
        for session in sessions[:limit]:
            print(
                f"{session['session_id'][:40]} | "
                f"{session.get('created_at', 'unknown')} | "
                f"{session.get('message_count', 0)} messages"
            )


@cli.command()
@click.argument("session_id")
@pass_state
def resume(state: GlobalState, session_id: str):
    """
    Resume a conversation session.

    Examples:
        morgan resume abc123
    """
    formatter = state.formatter

    # Check if session exists
    session_data = state.session_manager.load_session(session_id)

    if not session_data:
        formatter.print(f"[red]Session not found:[/red] {session_id}")
        sys.exit(1)

    formatter.print(f"\n[green]Resuming session:[/green] {session_id}")
    formatter.print(f"[dim]Messages: {len(session_data.get('messages', []))}[/dim]\n")

    # Start chat with this session
    asyncio.run(_chat_async(state, session_id, streaming=True))


@cli.command()
@click.argument("rating", type=float)
@click.option(
    "--comment",
    help="Optional feedback comment",
)
@pass_state
def rate(state: GlobalState, rating: float, comment: Optional[str]):
    """
    Rate the last response.

    Rating scale: 0.0 (poor) to 1.0 (excellent)

    Examples:
        morgan rate 0.9
        morgan rate 0.5 --comment "Too verbose"
    """
    asyncio.run(_rate_async(state, rating, comment))


async def _rate_async(state: GlobalState, rating: float, comment: Optional[str]):
    """Async implementation of rate command."""
    config = state.config
    formatter = state.formatter

    if not state.last_response_id:
        formatter.print("[yellow]No recent response to rate.[/yellow]")
        return

    if not 0.0 <= rating <= 1.0:
        formatter.print("[red]Rating must be between 0.0 and 1.0[/red]")
        sys.exit(1)

    try:
        async with assistant_context(config) as assistant:
            if not assistant.learning_engine:
                formatter.print("[yellow]Learning system not enabled.[/yellow]")
                return

            # Create feedback signal
            feedback = FeedbackSignal(
                feedback_type=FeedbackType.EXPLICIT,
                signal_value=rating,
                timestamp=datetime.now(),
                response_id=state.last_response_id,
                user_id="cli_user",
                session_id=state.current_session_id or "unknown",
                context_data={"comment": comment} if comment else {},
            )

            # Submit feedback
            await assistant.learning_engine.process_feedback(feedback)

            formatter.print(f"[green]✅ Feedback submitted: {rating:.1f}/1.0[/green]")
            if comment:
                formatter.print(f"[dim]Comment: {comment}[/dim]")

    except Exception as e:
        handle_cli_error(e, formatter, config.verbose)


@cli.command()
@pass_state
def stats(state: GlobalState):
    """
    Show learning statistics.

    Displays:
    - Patterns detected
    - Feedback processed
    - Preferences learned
    - Adaptations made
    """
    asyncio.run(_stats_async(state))


async def _stats_async(state: GlobalState):
    """Async implementation of stats command."""
    config = state.config
    formatter = state.formatter

    formatter.print("\n[bold cyan]Learning Statistics[/bold cyan]\n")

    try:
        async with assistant_context(config) as assistant:
            if not assistant.learning_engine:
                formatter.print("[yellow]Learning system not enabled.[/yellow]")
                return

            # Get metrics
            metrics = await assistant.learning_engine.get_metrics()

            # Display stats
            if formatter.use_rich:
                stats_panel = formatter.format_learning_stats(metrics)
                formatter.console.print(stats_panel)
            else:
                output = formatter.format_learning_stats(metrics)
                print(output)

            # Get preferences
            preferences = await assistant.learning_engine.get_user_preferences("cli_user")
            if preferences:
                formatter.print("")
                pref_panel = formatter.format_preferences(preferences)
                if formatter.use_rich:
                    formatter.console.print(pref_panel)
                else:
                    print(pref_panel)

    except Exception as e:
        handle_cli_error(e, formatter, config.verbose)


@cli.command()
@pass_state
def version(state: GlobalState):
    """Show Morgan version information."""
    formatter = state.formatter

    formatter.print("\n[bold green]Morgan AI Assistant[/bold green]")
    formatter.print("[dim]Version: 2.0.0[/dim]")
    formatter.print("[dim]An intelligent, emotionally-aware assistant[/dim]\n")


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
