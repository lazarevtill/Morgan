#!/usr/bin/env python3
"""
Morgan CLI - Interactive Command Line Interface

Simple, human-first CLI for interacting with Morgan assistant.
Supports emotional awareness, memory, and personalized responses.
"""

import argparse
import asyncio
import sys
from datetime import datetime
from typing import Optional

import aiohttp
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()


class MorganCLI:
    """Interactive CLI for Morgan AI Assistant"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.user_id = "cli_user"
        self.conversation_id: Optional[str] = None
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def check_health(self) -> bool:
        """Check if Morgan service is healthy"""
        try:
            async with self.session.get(f"{self.base_url}/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("status") == "healthy"
        except Exception as e:
            console.print(f"[red]Failed to connect to Morgan: {e}[/red]")
            return False
        return False

    async def send_message(self, text: str, show_sources: bool = False) -> dict:
        """Send message to Morgan and get response"""
        try:
            payload = {"text": text, "user_id": self.user_id, "metadata": {}}

            if self.conversation_id:
                payload["metadata"]["conversation_id"] = self.conversation_id

            if show_sources:
                payload["metadata"]["show_sources"] = show_sources

            async with self.session.post(
                f"{self.base_url}/api/text",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    # Store conversation ID
                    if "metadata" in data and "conversation_id" in data["metadata"]:
                        self.conversation_id = data["metadata"]["conversation_id"]

                    return data
                else:
                    error_text = await resp.text()
                    console.print(f"[red]Error {resp.status}: {error_text}[/red]")
                    return {}
        except Exception as e:
            console.print(f"[red]Failed to send message: {e}[/red]")
            return {}

    async def chat_interactive(self, topic: Optional[str] = None):
        """Start interactive chat session"""
        console.print(
            Panel.fit(
                "[bold cyan]Morgan AI Assistant[/bold cyan]\n"
                "Type your message and press Enter. Type 'exit' or 'quit' to end.\n"
                "Type 'help' for available commands.",
                title="🤖 Welcome to Morgan",
            )
        )

        if topic:
            console.print(f"[dim]Starting conversation about: {topic}[/dim]\n")
            # Send initial topic message
            await self.send_message(f"I'd like to talk about {topic}")

        while True:
            try:
                user_input = Prompt.ask("\n[bold green]You[/bold green]")

                if not user_input.strip():
                    continue

                if user_input.lower() in ["exit", "quit", "bye"]:
                    console.print(
                        "[cyan]Thanks for chatting! Goodbye! 👋[/cyan]"
                    )
                    break

                if user_input.lower() == "help":
                    self.show_help()
                    continue

                if user_input.lower() == "reset":
                    self.conversation_id = None
                    console.print("[yellow]Conversation reset.[/yellow]")
                    continue

                # Show typing indicator
                console.print("\n[dim]Morgan is thinking...[/dim]")

                # Send message
                response = await self.send_message(user_input)

                if response and "text" in response:
                    # Display response with markdown rendering
                    console.print(
                        f"\n[bold magenta]Morgan[/bold magenta]: ", end=""
                    )
                    md = Markdown(response["text"])
                    console.print(md)

                    # Show metadata if available
                    if "metadata" in response and response["metadata"]:
                        console.print(
                            f"[dim]Conversation ID: {response['metadata'].get('conversation_id', 'N/A')}[/dim]"
                        )

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")

    async def ask_question(self, question: str, show_sources: bool = False):
        """Ask a single question and display answer"""
        console.print(f"[bold green]Question:[/bold green] {question}\n")
        console.print("[dim]Morgan is thinking...[/dim]\n")

        response = await self.send_message(question, show_sources=show_sources)

        if response and "text" in response:
            console.print(Panel(Markdown(response["text"]), title="🤖 Morgan's Answer"))

            if show_sources and "metadata" in response:
                console.print("\n[dim]Sources and metadata available in API response[/dim]")

    def show_help(self):
        """Display help information"""
        help_text = """
        **Available Commands:**

        - Type any message to chat with Morgan
        - `help` - Show this help message
        - `reset` - Reset the conversation
        - `exit` or `quit` - End the chat session

        **Tips:**
        - Morgan remembers context within the conversation
        - Be specific for better answers
        - Feel free to ask follow-up questions
        """
        console.print(Panel(Markdown(help_text), title="Help"))

    async def get_memory_stats(self):
        """Get conversation memory statistics"""
        try:
            async with self.session.get(f"{self.base_url}/status") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if "conversations" in data:
                        console.print(
                            Panel.fit(
                                f"**Total Conversations:** {data['conversations'].get('total', 0)}\n"
                                f"**Active Conversations:** {data['conversations'].get('active', 0)}\n"
                                f"**Total Messages:** {data['conversations'].get('total_messages', 0)}",
                                title="📊 Memory Statistics",
                            )
                        )
                    return data
        except Exception as e:
            console.print(f"[red]Failed to get stats: {e}[/red]")
        return {}


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog="morgan",
        description="🤖 Morgan - Your AI Assistant with Emotional Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py chat                           Start interactive chat
  python cli.py ask "What is Docker?"          Ask a single question
  python cli.py ask "Explain AI" --sources     Ask with source references
  python cli.py memory --stats                 View memory statistics
  python cli.py health                         Check system health

For best experience, ensure Morgan service is running on http://localhost:8000
        """,
    )

    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Morgan service URL (default: http://localhost:8000)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Chat command
    chat_parser = subparsers.add_parser(
        "chat", help="Start interactive chat with Morgan"
    )
    chat_parser.add_argument("--topic", type=str, help="Initial conversation topic")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask Morgan a single question")
    ask_parser.add_argument("question", type=str, help="Question to ask")
    ask_parser.add_argument(
        "--sources", action="store_true", help="Show source references"
    )

    # Health command
    health_parser = subparsers.add_parser("health", help="Check system health")

    # Memory command
    memory_parser = subparsers.add_parser("memory", help="View memory statistics")
    memory_parser.add_argument(
        "--stats", action="store_true", default=True, help="Show statistics"
    )

    args = parser.parse_args()

    # Default to chat if no command specified
    if not args.command:
        args.command = "chat"

    async with MorganCLI(base_url=args.url) as cli:
        # Check health first
        if not await cli.check_health():
            console.print(
                "[red]Morgan service is not available. Please ensure it's running.[/red]"
            )
            console.print(
                f"[yellow]Expected URL: {args.url}[/yellow]"
            )
            return

        if args.command == "chat":
            await cli.chat_interactive(topic=getattr(args, "topic", None))
        elif args.command == "ask":
            await cli.ask_question(args.question, show_sources=getattr(args, "sources", False))
        elif args.command == "health":
            console.print("[green]✓ Morgan service is healthy and ready![/green]")
        elif args.command == "memory":
            await cli.get_memory_stats()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
        sys.exit(0)
