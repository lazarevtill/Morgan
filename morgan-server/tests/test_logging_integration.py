"""
Integration Tests for Logging System

Tests the logging system in realistic scenarios.
"""

import pytest
import logging
import json
import tempfile
from pathlib import Path

from morgan_server.logging_config import (
    configure_logging,
    get_logger,
    LogContext,
    setup_request_logging,
)


def test_logging_to_file_with_rotation():
    """Test that logging to file works with rotation enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"

        # Configure logging with file output
        configure_logging(
            log_level="INFO",
            log_format="json",
            log_file=str(log_file),
            enable_rotation=True,
            max_bytes=1024,  # Small size for testing
            backup_count=3,
            enable_console=False,
        )

        # Get logger and log messages
        logger = get_logger("test.integration")
        logger.info("Test message 1")
        logger.info("Test message 2")

        # Close all handlers to release file locks
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        # Also close handlers on the root morgan logger
        morgan_logger = logging.getLogger("morgan")
        for handler in morgan_logger.handlers[:]:
            handler.close()
            morgan_logger.removeHandler(handler)

        # Verify log file exists
        assert log_file.exists()

        # Read and verify log content
        with open(log_file, "r") as f:
            lines = f.readlines()

        assert len(lines) >= 2

        # Verify each line is valid JSON
        for line in lines:
            log_entry = json.loads(line.strip())
            assert "timestamp" in log_entry
            assert "level" in log_entry
            assert "logger" in log_entry
            assert "message" in log_entry


def test_logging_with_context_fields():
    """Test logging with request context fields."""
    # Configure logging
    configure_logging(log_level="DEBUG", log_format="json", enable_console=False)

    # Get logger
    logger = get_logger("test.context")

    # Create request context
    context = setup_request_logging(
        request_id="req_123", user_id="user_456", conversation_id="conv_789"
    )

    # Log with context
    logger.info("Processing request", extra=context)

    # Test passes if no exceptions are raised


def test_log_context_manager():
    """Test LogContext manager for adding context to all logs."""
    # Configure logging
    configure_logging(log_level="DEBUG", log_format="json", enable_console=False)

    # Get logger
    logger = get_logger("test.context_manager")

    # Use context manager
    with LogContext(request_id="req_123", user_id="user_456"):
        logger.info("Message 1")
        logger.info("Message 2")

    # Log without context
    logger.info("Message 3")

    # Test passes if no exceptions are raised


def test_different_log_levels():
    """Test logging at different levels."""
    # Configure logging at INFO level
    configure_logging(log_level="INFO", log_format="json", enable_console=False)

    # Get logger
    logger = get_logger("test.levels")

    # Log at different levels
    logger.debug("Debug message")  # Should not appear
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    # Test passes if no exceptions are raised


def test_text_format_logging():
    """Test logging with text format instead of JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test_text.log"

        # Configure logging with text format
        configure_logging(
            log_level="INFO",
            log_format="text",
            log_file=str(log_file),
            enable_rotation=False,
            enable_console=False,
        )

        # Get logger and log messages
        logger = get_logger("test.text")
        logger.info("Test message", extra={"request_id": "req_123"})

        # Close all handlers to release file locks
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        # Also close handlers on the root morgan logger
        morgan_logger = logging.getLogger("morgan")
        for handler in morgan_logger.handlers[:]:
            handler.close()
            morgan_logger.removeHandler(handler)

        # Verify log file exists
        assert log_file.exists()

        # Read and verify log content
        with open(log_file, "r") as f:
            content = f.read()

        assert "Test message" in content
        assert "request_id=req_123" in content


def test_exception_logging():
    """Test that exceptions are properly logged."""
    # Configure logging
    configure_logging(log_level="ERROR", log_format="json", enable_console=False)

    # Get logger
    logger = get_logger("test.exception")

    # Log exception
    try:
        raise ValueError("Test exception")
    except ValueError:
        logger.error("An error occurred", exc_info=True)

    # Test passes if no exceptions are raised


def test_multiple_loggers():
    """Test that multiple loggers work independently."""
    # Configure logging
    configure_logging(log_level="INFO", log_format="json", enable_console=False)

    # Get multiple loggers
    logger1 = get_logger("test.module1")
    logger2 = get_logger("test.module2")

    # Log from different loggers
    logger1.info("Message from module 1")
    logger2.info("Message from module 2")

    # Verify they have different names
    assert logger1.name != logger2.name
    assert "module1" in logger1.name
    assert "module2" in logger2.name


def test_logging_configuration_validation():
    """Test that invalid configuration is rejected."""
    with pytest.raises(ValueError, match="Invalid log level"):
        configure_logging(log_level="INVALID")


def test_console_and_file_logging():
    """Test logging to both console and file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test_both.log"

        # Configure logging with both console and file
        configure_logging(
            log_level="INFO",
            log_format="json",
            log_file=str(log_file),
            enable_rotation=False,
            enable_console=True,
        )

        # Get logger and log message
        logger = get_logger("test.both")
        logger.info("Test message to both outputs")

        # Close all handlers to release file locks
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        # Also close handlers on the root morgan logger
        morgan_logger = logging.getLogger("morgan")
        for handler in morgan_logger.handlers[:]:
            handler.close()
            morgan_logger.removeHandler(handler)

        # Verify log file exists and has content
        assert log_file.exists()
        with open(log_file, "r") as f:
            content = f.read()
        assert "Test message to both outputs" in content
