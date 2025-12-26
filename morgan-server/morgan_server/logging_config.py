"""
Logging Configuration for Morgan Server

This module provides comprehensive logging configuration with:
- Structured logging (JSON format)
- Log level filtering
- Context fields (request_id, user_id, conversation_id)
- Log rotation
- Multiple output handlers

**Validates: Requirements 10.2, 10.4, 10.5**
"""

import logging
import logging.handlers
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone


# ============================================================================
# JSON Formatter
# ============================================================================


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.

    Outputs log records as JSON objects with standard fields:
    - timestamp: ISO 8601 timestamp
    - level: Log level name
    - logger: Logger name
    - message: Log message
    - Additional context fields from extra parameter

    **Validates: Requirements 10.5**
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON string representation of log record
        """
        log_data = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            log_data["stack_trace"] = self.formatException(record.exc_info)

        # Add extra fields from record
        # Skip standard fields and internal fields
        skip_fields = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "thread",
            "threadName",
            "exc_info",
            "exc_text",
            "stack_info",
        }

        for key, value in record.__dict__.items():
            if key not in skip_fields and not key.startswith("_"):
                log_data[key] = value

        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """
    Human-readable text formatter for development.

    Includes context fields when available.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as human-readable text.

        Args:
            record: Log record to format

        Returns:
            Formatted text string
        """
        # Base format
        base_msg = super().format(record)

        # Add context fields if present
        context_parts = []

        if hasattr(record, "request_id"):
            context_parts.append(f"request_id={record.request_id}")

        if hasattr(record, "user_id") and record.user_id:
            context_parts.append(f"user_id={record.user_id}")

        if hasattr(record, "conversation_id") and record.conversation_id:
            context_parts.append(f"conversation_id={record.conversation_id}")

        if context_parts:
            base_msg += f" [{', '.join(context_parts)}]"

        return base_msg


# ============================================================================
# Logging Configuration
# ============================================================================


def configure_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    enable_rotation: bool = True,
    max_bytes: int = 100 * 1024 * 1024,  # 100 MB
    backup_count: int = 30,  # Keep 30 days
    enable_console: bool = True,
) -> None:
    """
    Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format ("json" or "text")
        log_file: Path to log file (optional)
        enable_rotation: Whether to enable log rotation
        max_bytes: Maximum log file size before rotation (default: 100 MB)
        backup_count: Number of backup files to keep (default: 30)
        enable_console: Whether to log to console

    **Validates: Requirements 10.4, 10.5**
    """
    # Validate log level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level.upper() not in valid_levels:
        raise ValueError(
            f"Invalid log level: {log_level}. Must be one of {valid_levels}"
        )

    # Get root logger for morgan
    logger = logging.getLogger("morgan")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Choose formatter
    if log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Add console handler if enabled
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if log file specified
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if enable_rotation:
            # Use rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
        else:
            # Use regular file handler
            file_handler = logging.FileHandler(filename=log_file, encoding="utf-8")

        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    # Log configuration
    logger.info(
        "Logging configured",
        extra={
            "log_level": log_level,
            "log_format": log_format,
            "log_file": log_file,
            "enable_rotation": enable_rotation,
            "enable_console": enable_console,
        },
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    # Ensure logger is under morgan namespace
    if not name.startswith("morgan"):
        name = f"morgan.{name}"

    return logging.getLogger(name)


# ============================================================================
# Context Manager for Adding Context Fields
# ============================================================================


class LogContext:
    """
    Context manager for adding context fields to log records.

    Usage:
        with LogContext(request_id="123", user_id="user_456"):
            logger.info("Processing request")
    """

    def __init__(self, **context):
        """
        Initialize log context.

        Args:
            **context: Context fields to add to log records
        """
        self.context = context
        self.old_factory = None

    def __enter__(self):
        """Enter context and set up log record factory."""
        old_factory = logging.getLogRecordFactory()
        self.old_factory = old_factory

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore old log record factory."""
        if self.old_factory:
            logging.setLogRecordFactory(self.old_factory)


# ============================================================================
# Helper Functions
# ============================================================================


def add_context_to_record(record: logging.LogRecord, **context) -> None:
    """
    Add context fields to a log record.

    Args:
        record: Log record to modify
        **context: Context fields to add
    """
    for key, value in context.items():
        setattr(record, key, value)


def filter_by_level(record: logging.LogRecord, min_level: str) -> bool:
    """
    Filter log records by minimum level.

    Args:
        record: Log record to filter
        min_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        True if record should be logged, False otherwise

    **Validates: Requirements 10.4**
    """
    min_level_num = getattr(logging, min_level.upper())
    return record.levelno >= min_level_num


class LevelFilter(logging.Filter):
    """
    Filter that only allows records at or above a certain level.

    **Validates: Requirements 10.4**
    """

    def __init__(self, min_level: str):
        """
        Initialize level filter.

        Args:
            min_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        super().__init__()
        self.min_level = getattr(logging, min_level.upper())

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log record by level.

        Args:
            record: Log record to filter

        Returns:
            True if record should be logged, False otherwise
        """
        return record.levelno >= self.min_level


# ============================================================================
# Utility Functions
# ============================================================================


def log_with_context(
    logger: logging.Logger, level: str, message: str, **context
) -> None:
    """
    Log a message with context fields.

    Args:
        logger: Logger instance
        level: Log level (debug, info, warning, error, critical)
        message: Log message
        **context: Context fields to include
    """
    log_func = getattr(logger, level.lower())
    log_func(message, extra=context)


def setup_request_logging(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create context dictionary for request logging.

    Args:
        request_id: Request ID
        user_id: User ID
        conversation_id: Conversation ID

    Returns:
        Dictionary of context fields

    **Validates: Requirements 10.1**
    """
    context = {}

    if request_id:
        context["request_id"] = request_id

    if user_id:
        context["user_id"] = user_id

    if conversation_id:
        context["conversation_id"] = conversation_id

    return context
