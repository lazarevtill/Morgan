"""
Morgan CLI Package.

Provides command-line interfaces for:
- Interactive chat (morgan)
- Admin operations (morgan-admin)
- Configuration management
- Document ingestion
- System monitoring
"""

from morgan.cli.app import cli, main
from morgan.cli.config import CLIConfig, ensure_config_exists, interactive_init
from morgan.cli.formatters import ConsoleFormatter
from morgan.cli.utils import (
    assistant_context,
    check_health,
    confirm,
    create_assistant,
    display_response,
    handle_cli_error,
    setup_logging,
)

__all__ = [
    # Main CLI
    "cli",
    "main",
    # Configuration
    "CLIConfig",
    "ensure_config_exists",
    "interactive_init",
    # Formatting
    "ConsoleFormatter",
    # Utilities
    "assistant_context",
    "check_health",
    "confirm",
    "create_assistant",
    "display_response",
    "handle_cli_error",
    "setup_logging",
]
