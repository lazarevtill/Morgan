"""One output configuration for every entrypoint: stdout is UTF-8, logs go to stderr.

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


def _use_utf8_streams() -> None:
    """Make this process's stdout and stderr UTF-8, whatever the platform chose.

    Both protocols carried on stdout are UTF-8 by specification: JSON (RFC 8259) and the
    MCP stdio framing. Python picks the encoding from the locale instead, which on a
    Windows console is a single-byte codepage -- so printing a Cyrillic memory raised
    UnicodeEncodeError and the command exited non-zero with a traceback. Encoding is a
    property of the protocol, not of the machine that happens to run it.
    """
    for stream in (sys.stdout, sys.stderr):
        # Absent under pytest's capture and any other replaced stream; nothing to fix there.
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def configure_logging(level: int = logging.INFO) -> None:
    """Fix the output encodings, then route structlog and stdlib logging to stderr.

    One call per entrypoint configures everything this process writes. Idempotent.
    """
    _use_utf8_streams()
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
