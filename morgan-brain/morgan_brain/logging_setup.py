"""One logging configuration for every entrypoint: everything goes to stderr.

Two of Morgan's four surfaces own their stdout as a *protocol*. The ``morgan`` CLI's
``--json`` output is parsed by scripts, and ``morgan-mcp --transport stdio`` speaks JSON-RPC
over it -- one log line on stdout is a corrupted JSON document in the first case and a
framing error in the MCP client in the second. structlog's default logger factory prints to
stdout, and the standard library's last-resort handler prints unformatted; this routes both
to stderr, formatted the same way, and is called once from each ``main()``.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: int = logging.INFO) -> None:
    """Route structlog and stdlib logging to stderr. Idempotent."""
    logging.basicConfig(
        stream=sys.stderr,
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        force=True,
    )
    # One line per model call is not diagnostics, it is volume: the SDK's HTTP client logs
    # every request at INFO. Morgan's own warnings on the same path already say what failed.
    for noisy in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )
