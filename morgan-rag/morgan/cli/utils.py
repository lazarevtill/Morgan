"""
CLI Utilities.

Helper functions for the Morgan CLI including:
- Assistant initialization
- Stream handling
- User prompts
- Error handling
- Session management
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from morgan.cli.config import CLIConfig
from morgan.cli.formatters import ConsoleFormatter
from morgan.core.assistant import AssistantError, MorganAssistant
from morgan.core.types import AssistantResponse
from morgan.emotions.exceptions import EmotionDetectionError
from morgan.emotions.types import EmotionResult
from morgan.jina.reranking.service import RerankingService
from morgan.learning.exceptions import LearningError
from morgan.services.embedding_service import EmbeddingService
from morgan.vector_db.client import QdrantClient

logger = logging.getLogger(__name__)


class GracefulShutdown:
    """Handle graceful shutdown on SIGINT/SIGTERM."""

    def __init__(self):
        """Initialize shutdown handler."""
        self.shutdown_requested = False
        self._original_sigint = None
        self._original_sigterm = None

    def __enter__(self):
        """Set up signal handlers."""
        self._original_sigint = signal.signal(signal.SIGINT, self._handle_signal)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._handle_signal)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original signal handlers."""
        if self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm:
            signal.signal(signal.SIGTERM, self._original_sigterm)

    def _handle_signal(self, signum, frame):
        """Handle shutdown signal."""
        self.shutdown_requested = True
        print("\n\nShutdown requested. Cleaning up...")


async def create_assistant(config: CLIConfig) -> MorganAssistant:
    """
    Create and initialize Morgan Assistant instance.

    Args:
        config: CLI configuration

    Returns:
        Initialized MorganAssistant

    Raises:
        AssistantError: If initialization fails
    """
    try:
        # Initialize vector DB if RAG is enabled
        vector_db = None
        if config.enable_rag:
            vector_db = QdrantClient(
                url=config.vector_db_url,
                collection_name=config.vector_db_collection,
            )

        # Initialize embedding service
        embedding_service = None
        if config.enable_rag:
            embedding_service = EmbeddingService(
                model_name=config.embedding_model,
                batch_size=config.embedding_batch_size,
            )

        # Initialize reranking service
        reranking_service = None
        if config.enable_rag and config.reranking_enabled:
            reranking_service = RerankingService(
                model_name=config.reranking_model,
                top_n=config.reranking_top_n,
            )

        # Create assistant
        assistant = MorganAssistant(
            storage_path=Path(config.storage_path),
            llm_base_url=config.llm_base_url,
            llm_model=config.llm_model,
            vector_db=vector_db,
            embedding_service=embedding_service,
            reranking_service=reranking_service,
            enable_emotion_detection=config.enable_emotion_detection,
            enable_learning=config.enable_learning,
            enable_rag=config.enable_rag,
            max_concurrent_operations=config.max_concurrent_operations,
        )

        # Initialize
        await assistant.initialize()

        logger.info("Assistant created and initialized successfully")
        return assistant

    except Exception as e:
        logger.error(f"Failed to create assistant: {e}", exc_info=True)
        raise AssistantError(
            f"Failed to initialize assistant: {str(e)}",
            recoverable=False,
        )


@asynccontextmanager
async def assistant_context(config: CLIConfig) -> AsyncIterator[MorganAssistant]:
    """
    Async context manager for assistant lifecycle.

    Args:
        config: CLI configuration

    Yields:
        Initialized assistant

    Example:
        async with assistant_context(config) as assistant:
            response = await assistant.process_message(...)
    """
    assistant = await create_assistant(config)
    try:
        yield assistant
    finally:
        await assistant.cleanup()


async def handle_stream_response(
    response_stream: AsyncIterator[str],
    formatter: ConsoleFormatter,
    show_emotion: bool = True,
) -> str:
    """
    Display streaming response with rich formatting.

    Args:
        response_stream: Async iterator of response chunks
        formatter: Console formatter
        show_emotion: Display emotion indicators

    Returns:
        Complete response text
    """
    full_response = []

    # Print prefix
    formatter.print("\n[bold green]🤖 Morgan:[/bold green] ", end="")

    try:
        async for chunk in response_stream:
            full_response.append(chunk)
            # Print chunk without newline
            print(chunk, end="", flush=True)

        # Print newline after complete response
        print("\n")

        return "".join(full_response)

    except Exception as e:
        formatter.print(f"\n[red]Error during streaming: {str(e)}[/red]")
        raise


def display_emotion(emotion: EmotionResult, formatter: ConsoleFormatter) -> None:
    """
    Display emotion with visual indicators.

    Args:
        emotion: Emotion result
        formatter: Console formatter
    """
    if formatter.use_rich:
        panel = formatter._format_emotion_panel(emotion)
        formatter.console.print(panel)
    else:
        print(f"\nEmotion: {emotion.primary_emotion} (intensity: {emotion.intensity:.2f})")
        if emotion.emotions:
            print("All emotions:")
            for emo, score in list(emotion.emotions.items())[:3]:
                print(f"  - {emo}: {score:.2f}")


def display_response(
    response: AssistantResponse,
    formatter: ConsoleFormatter,
    show_sources: bool = True,
    show_emotion: bool = True,
    show_metrics: bool = False,
) -> None:
    """
    Display formatted assistant response.

    Args:
        response: Assistant response
        formatter: Console formatter
        show_sources: Display RAG sources
        show_emotion: Display emotion
        show_metrics: Display performance metrics
    """
    if formatter.use_rich:
        # Rich formatting returns multiple panels
        parts = formatter.format_response(
            response,
            show_sources=show_sources,
            show_metrics=show_metrics,
        )

        if isinstance(parts, list):
            for part in parts:
                formatter.console.print(part)
        else:
            formatter.console.print(parts)

    else:
        # Plain text
        output = formatter.format_response(
            response,
            show_sources=show_sources,
            show_metrics=show_metrics,
        )
        print(output)


async def confirm(message: str, default: bool = False) -> bool:
    """
    Async confirmation prompt.

    Args:
        message: Confirmation message
        default: Default value if user just presses enter

    Returns:
        True if confirmed, False otherwise
    """
    # Run input in thread pool to avoid blocking
    loop = asyncio.get_event_loop()

    default_text = "Y/n" if default else "y/N"
    prompt = f"{message} [{default_text}]: "

    try:
        response = await loop.run_in_executor(
            None,
            input,
            prompt,
        )
        response = response.strip().lower()

        if not response:
            return default

        return response in ("y", "yes", "true", "1")

    except (KeyboardInterrupt, EOFError):
        print()
        return False


def handle_cli_error(
    error: Exception,
    formatter: ConsoleFormatter,
    verbose: bool = False,
    exit_code: int = 1,
) -> None:
    """
    Handle and display CLI error with appropriate formatting.

    Args:
        error: Exception to handle
        formatter: Console formatter
        verbose: Show verbose error details
        exit_code: Exit code (0 = success, 1 = error)
    """
    # Format error message based on type
    if isinstance(error, AssistantError):
        if verbose and error.correlation_id:
            formatter.print(
                f"[red]❌ Error:[/red] {str(error)}\n"
                f"[dim]Correlation ID: {error.correlation_id}[/dim]"
            )
        else:
            formatter.print(f"[red]❌ Error:[/red] {str(error)}")

        if error.recoverable:
            formatter.print("[yellow]💡 This error is recoverable. Please try again.[/yellow]")

    elif isinstance(error, EmotionDetectionError):
        formatter.print(
            "[yellow]⚠️ Emotion detection failed.[/yellow] "
            "Continuing without emotion analysis."
        )
        if verbose:
            formatter.print(f"[dim]Details: {str(error)}[/dim]")

    elif isinstance(error, LearningError):
        formatter.print(
            "[yellow]⚠️ Learning system error.[/yellow] "
            "Continuing without learning updates."
        )
        if verbose:
            formatter.print(f"[dim]Details: {str(error)}[/dim]")

    elif isinstance(error, KeyboardInterrupt):
        formatter.print("\n[yellow]Interrupted by user.[/yellow]")
        exit_code = 130  # Standard exit code for SIGINT

    else:
        # Generic error
        formatter.print(f"[red]❌ Unexpected error:[/red] {str(error)}")

        if verbose:
            formatted_error = formatter.format_error(error, verbose=True)
            if formatter.use_rich:
                formatter.console.print(formatted_error)
            else:
                print(formatted_error)

    # Exit if requested
    if exit_code > 0:
        sys.exit(exit_code)


async def get_user_input(prompt: str = "You: ") -> str:
    """
    Get user input asynchronously.

    Args:
        prompt: Input prompt

    Returns:
        User input string

    Raises:
        KeyboardInterrupt: If user interrupts
        EOFError: If input stream ends
    """
    loop = asyncio.get_event_loop()

    try:
        user_input = await loop.run_in_executor(None, input, prompt)
        return user_input.strip()
    except (KeyboardInterrupt, EOFError):
        raise


def setup_logging(config: CLIConfig) -> None:
    """
    Set up logging configuration.

    Args:
        config: CLI configuration
    """
    # Determine log level
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)

    # Basic configuration
    logging_config = {
        "level": log_level,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }

    # Add file handler if specified
    if config.log_file:
        logging_config["filename"] = config.log_file
        logging_config["filemode"] = "a"

    logging.basicConfig(**logging_config)

    # Adjust third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    if config.verbose:
        # In verbose mode, show more details
        logging.getLogger("morgan").setLevel(logging.DEBUG)


async def check_health(assistant: MorganAssistant) -> dict:
    """
    Check health of all assistant components.

    Args:
        assistant: Morgan assistant instance

    Returns:
        Health status dictionary
    """
    import time

    health_status = {}

    # Check memory system
    try:
        start = time.time()
        # Simple health check - just verify it's initialized
        is_healthy = hasattr(assistant.memory_system, "_initialized")
        latency = (time.time() - start) * 1000

        health_status["memory"] = {
            "healthy": is_healthy,
            "status": "operational" if is_healthy else "error",
            "latency_ms": latency,
        }
    except Exception as e:
        health_status["memory"] = {
            "healthy": False,
            "status": "error",
            "message": str(e),
        }

    # Check emotion detector
    if assistant.emotion_detector:
        try:
            start = time.time()
            health = await assistant.emotion_detector.health_check()
            latency = (time.time() - start) * 1000

            health_status["emotion"] = {
                "healthy": health.healthy,
                "status": health.status.value,
                "latency_ms": latency,
                "message": health.message or "",
            }
        except Exception as e:
            health_status["emotion"] = {
                "healthy": False,
                "status": "error",
                "message": str(e),
            }

    # Check learning engine
    if assistant.learning_engine:
        try:
            start = time.time()
            health = await assistant.learning_engine.health_check()
            latency = (time.time() - start) * 1000

            health_status["learning"] = {
                "healthy": health.healthy,
                "status": health.status.value,
                "latency_ms": latency,
                "message": health.message or "",
            }
        except Exception as e:
            health_status["learning"] = {
                "healthy": False,
                "status": "error",
                "message": str(e),
            }

    # Check RAG search
    if assistant.search_engine:
        try:
            start = time.time()
            # Simple check - verify components are initialized
            is_healthy = assistant.search_engine.vector_db is not None
            latency = (time.time() - start) * 1000

            health_status["rag"] = {
                "healthy": is_healthy,
                "status": "operational" if is_healthy else "error",
                "latency_ms": latency,
            }
        except Exception as e:
            health_status["rag"] = {
                "healthy": False,
                "status": "error",
                "message": str(e),
            }

    # Check LLM
    try:
        start = time.time()
        # Check if response generator is ready
        is_healthy = assistant.response_generator is not None
        latency = (time.time() - start) * 1000

        health_status["llm"] = {
            "healthy": is_healthy,
            "status": "operational" if is_healthy else "error",
            "latency_ms": latency,
        }
    except Exception as e:
        health_status["llm"] = {
            "healthy": False,
            "status": "error",
            "message": str(e),
        }

    return health_status


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


class SessionManager:
    """Manage CLI sessions."""

    def __init__(self, storage_path: Path):
        """
        Initialize session manager.

        Args:
            storage_path: Path for session storage
        """
        self.storage_path = storage_path / "sessions"
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def list_sessions(self) -> list[dict]:
        """List all available sessions."""
        sessions = []

        for session_file in self.storage_path.glob("*.json"):
            try:
                import json
                with open(session_file, "r") as f:
                    session_data = json.load(f)
                    sessions.append({
                        "session_id": session_data.get("session_id"),
                        "created_at": session_data.get("created_at"),
                        "message_count": len(session_data.get("messages", [])),
                    })
            except Exception:
                continue

        return sorted(sessions, key=lambda s: s.get("created_at", ""), reverse=True)

    def load_session(self, session_id: str) -> Optional[dict]:
        """Load session data."""
        session_file = self.storage_path / f"{session_id}.json"

        if not session_file.exists():
            return None

        try:
            import json
            with open(session_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def save_session(self, session_id: str, session_data: dict) -> None:
        """Save session data."""
        session_file = self.storage_path / f"{session_id}.json"

        try:
            import json
            with open(session_file, "w") as f:
                json.dump(session_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")

    def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        session_file = self.storage_path / f"{session_id}.json"

        if not session_file.exists():
            return False

        try:
            session_file.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
