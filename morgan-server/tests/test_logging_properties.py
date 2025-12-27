"""
Property-Based Tests for Logging System

Tests universal properties of the logging system using Hypothesis.

**Feature: client-server-separation**
**Validates: Requirements 10.4, 10.5**
"""

import pytest
import logging
import json
import io
from hypothesis import given, strategies as st, settings
from datetime import datetime

from morgan_server.logging_config import (
    configure_logging,
    JSONFormatter,
    TextFormatter,
    LevelFilter,
    filter_by_level,
    get_logger,
    LogContext,
    setup_request_logging,
)


# ============================================================================
# Test Strategies
# ============================================================================

# Valid log levels
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

log_level_strategy = st.sampled_from(LOG_LEVELS)

# Log messages
log_message_strategy = st.text(min_size=1, max_size=200)

# Context fields
# Reserved LogRecord attributes that should not be used as context keys
RESERVED_LOG_FIELDS = {
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
    "asctime",
    "taskName",
    "level",
    "logger",
    "timestamp",
}

context_key_strategy = st.text(
    min_size=1,
    max_size=50,
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        min_codepoint=ord("a"),
        max_codepoint=ord("z"),
    ),
).filter(lambda x: x and not x.startswith("_") and x not in RESERVED_LOG_FIELDS)

context_value_strategy = st.one_of(
    st.text(max_size=100),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.none(),
)

context_dict_strategy = st.dictionaries(
    keys=context_key_strategy, values=context_value_strategy, min_size=0, max_size=10
)


# ============================================================================
# Helper Functions
# ============================================================================


def get_log_level_number(level: str) -> int:
    """Get numeric value for log level."""
    return getattr(logging, level.upper())


def create_test_logger(name: str, level: str, stream: io.StringIO) -> logging.Logger:
    """Create a test logger with a string stream handler."""
    logger = logging.getLogger(name)
    logger.setLevel(get_log_level_number(level))
    logger.handlers.clear()

    handler = logging.StreamHandler(stream)
    handler.setLevel(get_log_level_number(level))
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.propagate = False

    return logger


# ============================================================================
# Property 25: Log Level Filtering
# ============================================================================


@given(
    configured_level=log_level_strategy,
    message_level=log_level_strategy,
    message=log_message_strategy,
)
@settings(max_examples=100, deadline=None)
def test_log_level_filtering_property(
    configured_level: str, message_level: str, message: str
):
    """
    **Feature: client-server-separation, Property 25: Log level filtering**
    **Validates: Requirements 10.4**

    For any configured log level (DEBUG, INFO, WARNING, ERROR, CRITICAL),
    only log messages at that level or higher should be written to the log output.

    Property: If configured_level_num <= message_level_num, then message appears in output.
              If configured_level_num > message_level_num, then message does NOT appear in output.
    """
    # Create a string stream to capture log output
    stream = io.StringIO()

    # Create test logger with configured level
    logger_name = f"test_logger_{configured_level}_{message_level}"
    logger = create_test_logger(logger_name, configured_level, stream)

    # Get numeric levels
    configured_level_num = get_log_level_number(configured_level)
    message_level_num = get_log_level_number(message_level)

    # Log message at message_level
    log_func = getattr(logger, message_level.lower())
    log_func(message)

    # Get output
    output = stream.getvalue()

    # Verify filtering behavior
    if message_level_num >= configured_level_num:
        # Message should appear in output
        assert len(output) > 0, (
            f"Expected message to be logged when message_level ({message_level}) >= "
            f"configured_level ({configured_level}), but output was empty"
        )

        # Parse JSON and verify message is present
        try:
            log_entry = json.loads(output.strip())
            assert (
                message in log_entry["message"]
            ), f"Expected message '{message}' in log output, but got: {log_entry['message']}"
            assert (
                log_entry["level"] == message_level
            ), f"Expected level '{message_level}' in log output, but got: {log_entry['level']}"
        except json.JSONDecodeError:
            pytest.fail(f"Log output is not valid JSON: {output}")
    else:
        # Message should NOT appear in output
        assert len(output) == 0, (
            f"Expected no output when message_level ({message_level}) < "
            f"configured_level ({configured_level}), but got: {output}"
        )

    # Cleanup
    logger.handlers.clear()


@given(min_level=log_level_strategy, test_level=log_level_strategy)
@settings(max_examples=100, deadline=None)
def test_level_filter_class_property(min_level: str, test_level: str):
    """
    **Feature: client-server-separation, Property 25: Log level filtering**
    **Validates: Requirements 10.4**

    For any LevelFilter with a minimum level, it should only allow records
    at or above that level.
    """
    # Create filter
    level_filter = LevelFilter(min_level)

    # Create a log record at test_level
    record = logging.LogRecord(
        name="test",
        level=get_log_level_number(test_level),
        pathname="test.py",
        lineno=1,
        msg="test message",
        args=(),
        exc_info=None,
    )

    # Get numeric levels
    min_level_num = get_log_level_number(min_level)
    test_level_num = get_log_level_number(test_level)

    # Test filter
    result = level_filter.filter(record)

    # Verify filtering behavior
    if test_level_num >= min_level_num:
        assert result is True, (
            f"Expected filter to allow record at level {test_level} "
            f"when min_level is {min_level}"
        )
    else:
        assert result is False, (
            f"Expected filter to block record at level {test_level} "
            f"when min_level is {min_level}"
        )


@given(min_level=log_level_strategy, test_level=log_level_strategy)
@settings(max_examples=100, deadline=None)
def test_filter_by_level_function_property(min_level: str, test_level: str):
    """
    **Feature: client-server-separation, Property 25: Log level filtering**
    **Validates: Requirements 10.4**

    For any filter_by_level call, it should return True only when
    the record level is at or above the minimum level.
    """
    # Create a log record at test_level
    record = logging.LogRecord(
        name="test",
        level=get_log_level_number(test_level),
        pathname="test.py",
        lineno=1,
        msg="test message",
        args=(),
        exc_info=None,
    )

    # Get numeric levels
    min_level_num = get_log_level_number(min_level)
    test_level_num = get_log_level_number(test_level)

    # Test filter function
    result = filter_by_level(record, min_level)

    # Verify filtering behavior
    expected = test_level_num >= min_level_num
    assert result == expected, (
        f"Expected filter_by_level to return {expected} for "
        f"test_level={test_level}, min_level={min_level}"
    )


# ============================================================================
# Property 26: Structured Logging Format
# ============================================================================


@given(
    level=log_level_strategy,
    message=log_message_strategy,
    context=context_dict_strategy,
)
@settings(max_examples=100, deadline=None)
def test_structured_logging_format_property(level: str, message: str, context: dict):
    """
    **Feature: client-server-separation, Property 26: Structured logging format**
    **Validates: Requirements 10.5**

    For any log entry when JSON logging is enabled, the log output should be
    valid JSON containing standard fields (timestamp, level, message, logger name)
    and any additional context fields.
    """
    # Create a string stream to capture log output
    stream = io.StringIO()

    # Create test logger
    logger_name = f"test_logger_{level}"
    logger = create_test_logger(logger_name, "DEBUG", stream)

    # Log message with context
    log_func = getattr(logger, level.lower())
    log_func(message, extra=context)

    # Get output
    output = stream.getvalue().strip()

    # Verify output is not empty
    assert len(output) > 0, "Expected log output but got empty string"

    # Verify output is valid JSON
    try:
        log_entry = json.loads(output)
    except json.JSONDecodeError as e:
        pytest.fail(f"Log output is not valid JSON: {output}\nError: {e}")

    # Verify standard fields are present
    assert "timestamp" in log_entry, "Log entry missing 'timestamp' field"
    assert "level" in log_entry, "Log entry missing 'level' field"
    assert "logger" in log_entry, "Log entry missing 'logger' field"
    assert "message" in log_entry, "Log entry missing 'message' field"

    # Verify standard field values
    assert (
        log_entry["level"] == level
    ), f"Expected level '{level}' but got '{log_entry['level']}'"
    assert (
        message in log_entry["message"]
    ), f"Expected message '{message}' in log output, but got: {log_entry['message']}"
    assert (
        logger_name in log_entry["logger"]
    ), f"Expected logger name '{logger_name}' in log output, but got: {log_entry['logger']}"

    # Verify timestamp is valid ISO 8601 format
    try:
        datetime.fromisoformat(log_entry["timestamp"].replace("Z", "+00:00"))
    except ValueError as e:
        pytest.fail(
            f"Timestamp is not valid ISO 8601 format: {log_entry['timestamp']}\nError: {e}"
        )

    # Verify context fields are present
    for key, value in context.items():
        assert (
            key in log_entry
        ), f"Expected context field '{key}' in log output, but it's missing"
        # Convert both to strings for comparison to handle type differences
        assert str(log_entry[key]) == str(value), (
            f"Expected context field '{key}' to have value '{value}', "
            f"but got '{log_entry[key]}'"
        )

    # Cleanup
    logger.handlers.clear()


@given(
    message=log_message_strategy,
    request_id=st.text(min_size=1, max_size=50),
    user_id=st.text(min_size=1, max_size=50),
    conversation_id=st.text(min_size=1, max_size=50),
)
@settings(max_examples=100, deadline=None)
def test_context_fields_in_json_output_property(
    message: str, request_id: str, user_id: str, conversation_id: str
):
    """
    **Feature: client-server-separation, Property 26: Structured logging format**
    **Validates: Requirements 10.5**

    For any log entry with context fields (request_id, user_id, conversation_id),
    the JSON output should include these fields.
    """
    # Create a string stream to capture log output
    stream = io.StringIO()

    # Create test logger
    logger = create_test_logger("test_context_logger", "DEBUG", stream)

    # Log message with context fields
    logger.info(
        message,
        extra={
            "request_id": request_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
        },
    )

    # Get output
    output = stream.getvalue().strip()

    # Parse JSON
    try:
        log_entry = json.loads(output)
    except json.JSONDecodeError as e:
        pytest.fail(f"Log output is not valid JSON: {output}\nError: {e}")

    # Verify context fields are present and correct
    assert (
        log_entry["request_id"] == request_id
    ), f"Expected request_id '{request_id}' but got '{log_entry.get('request_id')}'"
    assert (
        log_entry["user_id"] == user_id
    ), f"Expected user_id '{user_id}' but got '{log_entry.get('user_id')}'"
    assert (
        log_entry["conversation_id"] == conversation_id
    ), f"Expected conversation_id '{conversation_id}' but got '{log_entry.get('conversation_id')}'"

    # Cleanup
    logger.handlers.clear()


@given(message=log_message_strategy, context=context_dict_strategy)
@settings(max_examples=100, deadline=None)
def test_json_formatter_produces_valid_json_property(message: str, context: dict):
    """
    **Feature: client-server-separation, Property 26: Structured logging format**
    **Validates: Requirements 10.5**

    For any message and context, the JSONFormatter should always produce
    valid JSON output that can be parsed.
    """
    # Create formatter
    formatter = JSONFormatter()

    # Create log record
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )

    # Add context fields to record
    for key, value in context.items():
        setattr(record, key, value)

    # Format record
    output = formatter.format(record)

    # Verify output is valid JSON
    try:
        log_entry = json.loads(output)
    except json.JSONDecodeError as e:
        pytest.fail(f"Formatter output is not valid JSON: {output}\nError: {e}")

    # Verify standard fields
    assert "timestamp" in log_entry
    assert "level" in log_entry
    assert "logger" in log_entry
    assert "message" in log_entry

    # Verify context fields
    for key, value in context.items():
        assert key in log_entry, f"Context field '{key}' missing from output"


# ============================================================================
# Additional Property Tests
# ============================================================================


@given(context=context_dict_strategy)
@settings(max_examples=100, deadline=None)
def test_log_context_manager_property(context: dict):
    """
    Test that LogContext manager properly adds context fields to all log records.
    """
    # Create a string stream to capture log output
    stream = io.StringIO()

    # Create test logger
    logger = create_test_logger("test_context_manager", "DEBUG", stream)

    # Log with context manager
    with LogContext(**context):
        logger.info("test message")

    # Get output
    output = stream.getvalue().strip()

    # Parse JSON
    try:
        log_entry = json.loads(output)
    except json.JSONDecodeError as e:
        pytest.fail(f"Log output is not valid JSON: {output}\nError: {e}")

    # Verify all context fields are present
    for key, value in context.items():
        assert key in log_entry, f"Context field '{key}' missing from output"
        assert str(log_entry[key]) == str(
            value
        ), f"Context field '{key}' has wrong value: expected '{value}', got '{log_entry[key]}'"

    # Cleanup
    logger.handlers.clear()


@given(
    request_id=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
    user_id=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
    conversation_id=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
)
@settings(max_examples=100, deadline=None)
def test_setup_request_logging_property(
    request_id: str, user_id: str, conversation_id: str
):
    """
    Test that setup_request_logging creates correct context dictionary.
    """
    # Create context
    context = setup_request_logging(
        request_id=request_id, user_id=user_id, conversation_id=conversation_id
    )

    # Verify context contains only non-None values
    if request_id is not None:
        assert "request_id" in context
        assert context["request_id"] == request_id
    else:
        assert "request_id" not in context

    if user_id is not None:
        assert "user_id" in context
        assert context["user_id"] == user_id
    else:
        assert "user_id" not in context

    if conversation_id is not None:
        assert "conversation_id" in context
        assert context["conversation_id"] == conversation_id
    else:
        assert "conversation_id" not in context


# ============================================================================
# Unit Tests for Edge Cases
# ============================================================================


def test_configure_logging_invalid_level():
    """Test that configure_logging raises error for invalid log level."""
    with pytest.raises(ValueError, match="Invalid log level"):
        configure_logging(log_level="INVALID")


def test_json_formatter_with_exception():
    """Test that JSONFormatter properly formats exceptions."""
    formatter = JSONFormatter()

    try:
        raise ValueError("Test exception")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        output = formatter.format(record)
        log_entry = json.loads(output)

        assert "exception" in log_entry
        assert "stack_trace" in log_entry
        assert "ValueError" in log_entry["exception"]
        assert "Test exception" in log_entry["exception"]


def test_get_logger_adds_morgan_prefix():
    """Test that get_logger adds 'morgan' prefix if not present."""
    logger = get_logger("test.module")
    assert logger.name == "morgan.test.module"

    logger2 = get_logger("morgan.test.module")
    assert logger2.name == "morgan.test.module"
