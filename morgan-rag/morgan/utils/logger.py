"""
Simple, human-friendly logging for Morgan RAG.

KISS Principle: Easy to use, clear output, helpful for debugging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO", log_file: Optional[str] = None, rich_console: bool = True
) -> None:
    """
    Set up logging for Morgan RAG.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to log to
        rich_console: Use Rich for beautiful console output
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatters
    if rich_console:
        # Rich formatter for console (beautiful output)
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
        )
        console_handler.setLevel(numeric_level)
    else:
        # Simple formatter for console
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(numeric_level)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)

    # File handler (if specified)
    handlers = [console_handler]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)  # Always debug level for files
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("This is a log message")
    """
    return logging.getLogger(name)


# Human-friendly logging helpers
def log_user_action(logger: logging.Logger, action: str, details: str = ""):
    """Log a user action in a human-friendly way."""
    message = f"👤 User action: {action}"
    if details:
        message += f" - {details}"
    logger.info(message)


def log_morgan_response(logger: logging.Logger, response_type: str, details: str = ""):
    """Log Morgan's response in a human-friendly way."""
    message = f"🤖 Morgan {response_type}"
    if details:
        message += f" - {details}"
    logger.info(message)


def log_system_event(logger: logging.Logger, event: str, details: str = ""):
    """Log a system event in a human-friendly way."""
    message = f"⚙️  System: {event}"
    if details:
        message += f" - {details}"
    logger.info(message)


def log_error_with_context(logger: logging.Logger, error: Exception, context: str = ""):
    """Log an error with helpful context."""
    message = "❌ Error"
    if context:
        message += f" in {context}"
    message += f": {str(error)}"
    logger.error(message, exc_info=True)


if __name__ == "__main__":
    # Demo logging setup
    setup_logging(level="DEBUG", rich_console=True)

    logger = get_logger(__name__)

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Demo human-friendly helpers
    log_user_action(logger, "asked question", "How do I deploy Docker?")
    log_morgan_response(logger, "answered", "Provided Docker deployment guide")
    log_system_event(logger, "knowledge updated", "Added 5 new documents")

    try:
        raise ValueError("This is a test error")
    except Exception as e:
        log_error_with_context(logger, e, "demo function")
