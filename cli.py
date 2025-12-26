#!/usr/bin/env python3
"""
⚠️ DEPRECATED - This CLI is deprecated and will be removed in a future release.

Please use the new morgan-cli package instead:

    pip install -e ./morgan-cli
    morgan chat

For migration instructions, see MIGRATION.md in the root directory.
"""

import sys
from rich.console import Console

console = Console()

console.print("[bold red]⚠️  DEPRECATED CLI[/bold red]\n")
console.print(
    "This CLI (cli.py) is deprecated and will be removed in a future release.\n"
)
console.print("[bold cyan]Please use the new morgan-cli package instead:[/bold cyan]\n")
console.print("  1. Install: [green]pip install -e ./morgan-cli[/green]")
console.print("  2. Use: [green]morgan chat[/green]\n")
console.print(
    "For migration instructions, see [yellow]MIGRATION.md[/yellow] in the root directory.\n"
)

sys.exit(1)
