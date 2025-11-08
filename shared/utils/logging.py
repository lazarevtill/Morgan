"""
Modern logging utilities for Morgan AI Assistant
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class MorganLogger:
    """Enhanced logger for Morgan services"""

    def __init__(
        self,
        name: str,
        level: str = "INFO",
        log_file: Optional[str] = None,
        json_format: bool = False,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Clear existing handlers
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))

        if json_format:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5
            )
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """Get the underlying logger instance"""
        return self.logger

    def info(self, message: str, **kwargs):
        """Log info message with optional metadata"""
        if kwargs:
            message = f"{message} {kwargs}"
        self.logger.info(message)

    def error(self, message: str, **kwargs):
        """Log error message with optional metadata"""
        if kwargs:
            message = f"{message} {kwargs}"
        self.logger.error(message)

    def warning(self, message: str, **kwargs):
        """Log warning message with optional metadata"""
        if kwargs:
            message = f"{message} {kwargs}"
        self.logger.warning(message)

    def debug(self, message: str, **kwargs):
        """Log debug message with optional metadata"""
        if kwargs:
            message = f"{message} {kwargs}"
        self.logger.debug(message)


def setup_logging(
    service_name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
) -> logging.Logger:
    """Setup logging for a service"""
    morgan_logger = MorganLogger(service_name, level, log_file, json_format)
    return morgan_logger.get_logger()


class Timer:
    """Context manager for timing operations"""

    def __init__(
        self, logger: Optional[logging.Logger] = None, operation: str = "operation"
    ):
        self.logger = logger
        self.operation = operation
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        if self.logger:
            self.logger.info(f"{self.operation} completed in {duration:.3f}s")

        return False

    @property
    def duration(self) -> float:
        """Get duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0.0


def log_function_call(logger: Optional[logging.Logger] = None):
    """Decorator to log function calls"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            if logger:
                logger.debug(
                    f"Calling {func.__name__} with args={args}, kwargs={kwargs}"
                )
            result = func(*args, **kwargs)
            if logger:
                logger.debug(f"{func.__name__} returned {result}")
            return result

        return wrapper

    return decorator
