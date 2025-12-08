"""
Example: Using the Morgan Server Logging System

This example demonstrates how to use the logging system with various features:
- Structured JSON logging
- Log level filtering
- Context fields
- Log rotation
- Exception logging
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from morgan_server.logging_config import (
    configure_logging,
    get_logger,
    LogContext,
    setup_request_logging,
)


def example_basic_logging():
    """Example 1: Basic logging with JSON format."""
    print("\n=== Example 1: Basic Logging ===\n")
    
    # Configure logging
    configure_logging(
        log_level="INFO",
        log_format="json",
        enable_console=True
    )
    
    # Get logger
    logger = get_logger("example.basic")
    
    # Log at different levels
    logger.debug("This won't appear (below INFO level)")
    logger.info("Application started")
    logger.warning("This is a warning")
    logger.error("This is an error")


def example_context_fields():
    """Example 2: Logging with context fields."""
    print("\n=== Example 2: Context Fields ===\n")
    
    # Configure logging
    configure_logging(
        log_level="INFO",
        log_format="json",
        enable_console=True
    )
    
    # Get logger
    logger = get_logger("example.context")
    
    # Create request context
    context = setup_request_logging(
        request_id="req_12345",
        user_id="user_67890",
        conversation_id="conv_abcde"
    )
    
    # Log with context
    logger.info("Processing user request", extra=context)
    logger.info("Request completed successfully", extra=context)


def example_log_context_manager():
    """Example 3: Using LogContext manager."""
    print("\n=== Example 3: LogContext Manager ===\n")
    
    # Configure logging
    configure_logging(
        log_level="INFO",
        log_format="json",
        enable_console=True
    )
    
    # Get logger
    logger = get_logger("example.context_manager")
    
    # All logs within this context will include these fields
    with LogContext(request_id="req_99999", user_id="user_11111"):
        logger.info("Starting operation")
        logger.info("Operation in progress")
        logger.info("Operation complete")
    
    # This log won't have the context fields
    logger.info("Outside context")


def example_exception_logging():
    """Example 4: Logging exceptions with stack traces."""
    print("\n=== Example 4: Exception Logging ===\n")
    
    # Configure logging
    configure_logging(
        log_level="ERROR",
        log_format="json",
        enable_console=True
    )
    
    # Get logger
    logger = get_logger("example.exception")
    
    # Log an exception
    try:
        result = 1 / 0
    except ZeroDivisionError:
        logger.error(
            "Division by zero error",
            exc_info=True,
            extra={"operation": "divide", "numerator": 1, "denominator": 0}
        )


def example_text_format():
    """Example 5: Human-readable text format for development."""
    print("\n=== Example 5: Text Format ===\n")
    
    # Configure logging with text format
    configure_logging(
        log_level="INFO",
        log_format="text",
        enable_console=True
    )
    
    # Get logger
    logger = get_logger("example.text")
    
    # Log with context
    logger.info(
        "Processing request",
        extra={
            "request_id": "req_12345",
            "user_id": "user_67890"
        }
    )


def example_file_logging():
    """Example 6: Logging to file with rotation."""
    print("\n=== Example 6: File Logging with Rotation ===\n")
    
    # Create temporary directory for logs
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "morgan.log"
        
        # Configure logging with file output
        configure_logging(
            log_level="INFO",
            log_format="json",
            log_file=str(log_file),
            enable_rotation=True,
            max_bytes=1024 * 1024,  # 1 MB
            backup_count=5,
            enable_console=True
        )
        
        # Get logger
        logger = get_logger("example.file")
        
        # Log messages
        for i in range(10):
            logger.info(f"Log message {i}", extra={"iteration": i})
        
        print(f"\nLog file created at: {log_file}")
        print(f"Log file size: {log_file.stat().st_size} bytes")
        
        # Close handlers to release file locks (Windows requirement)
        import logging
        morgan_logger = logging.getLogger("morgan")
        for handler in morgan_logger.handlers[:]:
            handler.close()
            morgan_logger.removeHandler(handler)


def example_multiple_loggers():
    """Example 7: Using multiple loggers for different modules."""
    print("\n=== Example 7: Multiple Loggers ===\n")
    
    # Configure logging
    configure_logging(
        log_level="INFO",
        log_format="json",
        enable_console=True
    )
    
    # Get loggers for different modules
    api_logger = get_logger("example.api")
    db_logger = get_logger("example.database")
    cache_logger = get_logger("example.cache")
    
    # Log from different modules
    api_logger.info("API request received")
    db_logger.info("Database query executed")
    cache_logger.info("Cache hit")


def example_log_levels():
    """Example 8: Demonstrating log level filtering."""
    print("\n=== Example 8: Log Level Filtering ===\n")
    
    # Configure at WARNING level
    configure_logging(
        log_level="WARNING",
        log_format="text",
        enable_console=True
    )
    
    # Get logger
    logger = get_logger("example.levels")
    
    print("Configured at WARNING level - only WARNING, ERROR, CRITICAL will appear:\n")
    
    logger.debug("Debug message (won't appear)")
    logger.info("Info message (won't appear)")
    logger.warning("Warning message (will appear)")
    logger.error("Error message (will appear)")
    logger.critical("Critical message (will appear)")


def main():
    """Run all examples."""
    print("=" * 70)
    print("Morgan Server Logging System Examples")
    print("=" * 70)
    
    try:
        example_basic_logging()
        example_context_fields()
        example_log_context_manager()
        example_exception_logging()
        example_text_format()
        example_file_logging()
        example_multiple_loggers()
        example_log_levels()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
