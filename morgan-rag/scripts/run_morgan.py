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

            response = assistant.ask(
                question=query,
                user_id="default_user",
                include_sources=True
            )

            # Response object has .answer attribute
            answer = response.answer if hasattr(response, 'answer') else str(response)

            console.print(Panel(
                answer,
                title="[bold green]Morgan[/bold green]",
                border_style="green"
            ))

            # Show sources if available (from Response.sources)
            if hasattr(response, 'sources') and response.sources:
                console.print(f"\n[dim]📚 Sources: {len(response.sources)} documents[/dim]")
                for i, source in enumerate(response.sources[:3], 1):
                    console.print(f"[dim]  {i}. {source[:60]}...[/dim]")

            # Show emotional tone if available
            if hasattr(response, 'emotional_tone') and response.emotional_tone:
                console.print(f"[dim]💙 Emotional tone: {response.emotional_tone}[/dim]")

            # Show empathy level if available
            if hasattr(response, 'empathy_level') and response.empathy_level > 0:
                console.print(f"[dim]🤝 Empathy: {response.empathy_level:.0%}[/dim]")

            # Show suggestions if available
            if hasattr(response, 'suggestions') and response.suggestions:
                console.print(f"\n[dim]💡 Suggestions:[/dim]")
                for suggestion in response.suggestions[:3]:
                    console.print(f"[dim]  • {suggestion}[/dim]")

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted. Goodbye! 👋[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
