#!/usr/bin/env python3
"""
Simple Morgan Chat Interface

Usage:
    export PYTHONPATH=./local_libs:$PYTHONPATH
    python3 scripts/run_morgan.py
"""

import sys
from pathlib import Path

# Add local libs
local_libs = Path(__file__).parent.parent / "local_libs"
if local_libs.exists():
    sys.path.insert(0, str(local_libs))

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich import print as rprint

console = Console()

def main():
    console.print(Panel.fit(
        "[bold cyan]Morgan v2-0.0.1[/bold cyan]\n"
        "Complete Emotional AI Assistant",
        border_style="blue"
    ))

    from morgan.core.assistant import MorganAssistant

    console.print("\n[yellow]Initializing Morgan...[/yellow]")
    assistant = MorganAssistant()
    console.print("[green]✅ Morgan is ready![/green]\n")

    console.print("[dim]Type 'exit' or 'quit' to end the conversation[/dim]\n")

    while True:
        try:
            query = Prompt.ask("\n[bold blue]You[/bold blue]")

            if query.lower() in ['exit', 'quit', 'bye']:
                console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break

            console.print("\n[yellow]Morgan is thinking...[/yellow]")

            response = assistant.process_query(
                query=query,
                user_id="default_user"
            )

            answer = response.get("answer", "I'm sorry, I couldn't process that.")

            console.print(Panel(
                answer,
                title="[bold green]Morgan[/bold green]",
                border_style="green"
            ))

            # Show emotions if detected
            if response.get("emotions"):
                emotions_str = ", ".join([
                    f"{e}: {i:.0%}"
                    for e, i in list(response["emotions"].items())[:3]
                ])
                console.print(f"[dim]Emotions detected: {emotions_str}[/dim]")

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted. Goodbye! 👋[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
