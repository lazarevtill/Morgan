"""
Morgan RAG CLI Application

Human-first command line interface with intuitive commands.

KISS Principle: Simple commands that do exactly what humans expect.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from morgan import create_assistant
from morgan.config import get_settings
from morgan.utils.logger import setup_logging

console = Console()


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with human-friendly commands."""
    parser = argparse.ArgumentParser(
        prog="morgan",
        description="🤖 Morgan - Your Human-First AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  morgan chat                           Start interactive chat with Morgan
  morgan ask "How do I deploy Docker?"  Ask Morgan a single question
  morgan learn ./docs                   Teach Morgan from documents
  morgan learn --url https://docs.python.org  Learn from a website
  morgan serve                          Start web interface
  morgan health                         Check system health
  morgan memory --stats                 View conversation statistics

For more help on a specific command, use: morgan <command> --help
        """
    )

    # Global options
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="COMMAND"
    )

    # Chat command
    chat_parser = subparsers.add_parser(
        "chat",
        help="Start interactive chat with Morgan",
        description="Start an interactive conversation with Morgan. Perfect for exploring topics and getting help."
    )
    chat_parser.add_argument(
        "--topic",
        type=str,
        help="Initial topic for the conversation"
    )

    # Ask command
    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask Morgan a single question",
        description="Ask Morgan a question and get an immediate answer. Great for quick queries."
    )
    ask_parser.add_argument(
        "question",
        type=str,
        help="The question to ask Morgan"
    )
    ask_parser.add_argument(
        "--sources",
        action="store_true",
        help="Include source references in the answer"
    )
    ask_parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response in real-time"
    )

    # Learn command
    learn_parser = subparsers.add_parser(
        "learn",
        help="Teach Morgan from documents or websites",
        description="Add new knowledge to Morgan's knowledge base from various sources."
    )
    learn_group = learn_parser.add_mutually_exclusive_group(required=True)
    learn_group.add_argument(
        "source",
        nargs="?",
        type=str,
        help="Path to documents or directory to learn from"
    )
    learn_group.add_argument(
        "--url",
        type=str,
        help="URL to learn from (website, documentation, etc.)"
    )
    learn_parser.add_argument(
        "--type",
        type=str,
        default="auto",
        choices=["auto", "pdf", "web", "code", "markdown", "text"],
        help="Type of documents (auto-detect by default)"
    )
    learn_parser.add_argument(
        "--progress",
        action="store_true",
        default=True,
        help="Show progress during learning"
    )

    # Serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start Morgan's web interface",
        description="Start the web server for Morgan's chat interface and API."
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to (default: 8080)"
    )
    serve_parser.add_argument(
        "--api-only",
        action="store_true",
        help="Start only the API server (no web interface)"
    )

    # Health command
    health_parser = subparsers.add_parser(
        "health",
        help="Check Morgan's system health",
        description="Check if all of Morgan's components are working properly."
    )
    health_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed health information"
    )

    # Memory command
    memory_parser = subparsers.add_parser(
        "memory",
        help="Manage Morgan's conversation memory",
        description="View and manage Morgan's conversation history and learning."
    )
    memory_group = memory_parser.add_mutually_exclusive_group()
    memory_group.add_argument(
        "--stats",
        action="store_true",
        help="Show memory and learning statistics"
    )
    memory_group.add_argument(
        "--search",
        type=str,
        help="Search through conversation history"
    )
    memory_group.add_argument(
        "--cleanup",
        type=int,
        metavar="DAYS",
        help="Clean up conversations older than N days"
    )

    # Knowledge command
    knowledge_parser = subparsers.add_parser(
        "knowledge",
        help="Manage Morgan's knowledge base",
        description="View and manage Morgan's knowledge base."
    )
    knowledge_group = knowledge_parser.add_mutually_exclusive_group()
    knowledge_group.add_argument(
        "--stats",
        action="store_true",
        help="Show knowledge base statistics"
    )
    knowledge_group.add_argument(
        "--search",
        type=str,
        help="Search the knowledge base"
    )
    knowledge_group.add_argument(
        "--clear",
        action="store_true",
        help="Clear all knowledge (requires confirmation)"
    )

    # Init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize Morgan in current directory",
        description="Set up Morgan configuration and data directories."
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing configuration"
    )

    return parser


def cmd_chat(args, morgan):
    """Handle the chat command."""
    console.print(Panel.fit(
        f"🤖 [bold blue]Morgan[/bold blue] - Your AI Assistant\n"
        f"Type your questions naturally. Type 'quit' to exit.",
        title="Interactive Chat",
        border_style="blue"
    ))
    
    conversation_id = morgan.start_conversation(topic=args.topic)
    
    try:
        while True:
            # Get user input
            try:
                question = console.input("\n[bold green]You:[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if question.lower() in ['quit', 'exit', 'bye', 'q']:
                break
            
            if not question:
                continue
            
            # Get Morgan's response
            console.print("\n[bold blue]Morgan:[/bold blue] ", end="")
            
            try:
                if args.stream if hasattr(args, 'stream') else True:
                    # Stream response for natural conversation feel
                    for chunk in morgan.ask_stream(question, conversation_id):
                        console.print(chunk, end="", highlight=False)
                    console.print()  # New line after response
                else:
                    # Get complete response
                    response = morgan.ask(question, conversation_id)
                    console.print(response.answer)
                    
                    if response.sources and len(response.sources) > 0:
                        console.print(f"\n[dim]Sources: {', '.join(response.sources[:3])}[/dim]")
                        
            except Exception as e:
                console.print(f"[red]Sorry, I encountered an error: {e}[/red]")
    
    except KeyboardInterrupt:
        pass
    
    console.print(f"\n[blue]Morgan:[/blue] Goodbye! Feel free to chat with me anytime. 👋")


def cmd_ask(args, morgan):
    """Handle the ask command."""
    try:
        if args.stream:
            # Stream response
            console.print(f"[bold blue]Morgan:[/bold blue] ", end="")
            for chunk in morgan.ask_stream(args.question):
                console.print(chunk, end="", highlight=False)
            console.print()
        else:
            # Get complete response
            response = morgan.ask(args.question, include_sources=args.sources)
            
            console.print(f"[bold blue]Morgan:[/bold blue] {response.answer}")
            
            if args.sources and response.sources:
                console.print(f"\n[dim]Sources:[/dim]")
                for i, source in enumerate(response.sources[:5], 1):
                    console.print(f"  [dim]{i}. {source}[/dim]")
            
            if response.suggestions:
                console.print(f"\n[dim]Related topics:[/dim]")
                for suggestion in response.suggestions[:3]:
                    console.print(f"  [dim]• {suggestion}[/dim]")
                    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def cmd_learn(args, morgan):
    """Handle the learn command."""
    source = args.source or args.url
    
    console.print(f"[blue]Teaching Morgan from:[/blue] {source}")
    
    try:
        if args.url:
            # Learn from URL
            result = morgan.learn_from_documents(
                source_path=args.url,
                document_type="web",
                show_progress=args.progress
            )
        else:
            # Learn from local source
            result = morgan.learn_from_documents(
                source_path=source,
                document_type=args.type,
                show_progress=args.progress
            )
        
        if result["success"]:
            console.print(Panel.fit(
                f"✅ [green]Learning Complete![/green]\n\n"
                f"📚 Documents processed: {result['documents_processed']}\n"
                f"🧩 Knowledge chunks: {result['chunks_created']}\n"
                f"⏱️  Processing time: {result['learning_time']:.1f}s\n"
                f"🎯 Knowledge areas: {', '.join(result['knowledge_areas'][:5])}",
                title="Learning Results",
                border_style="green"
            ))
        else:
            console.print(f"[red]Learning failed: {result.get('message', 'Unknown error')}[/red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]Error during learning: {e}[/red]")
        sys.exit(1)


def cmd_serve(args, morgan):
    """Handle the serve command."""
    console.print(f"[blue]Starting Morgan web server...[/blue]")
    
    try:
        from morgan.web.app import create_app
        
        app = create_app(morgan)
        
        console.print(Panel.fit(
            f"🌐 [green]Morgan Web Interface[/green]\n\n"
            f"🔗 Web Interface: http://{args.host}:{args.port}\n"
            f"🔗 API Endpoint: http://{args.host}:{args.port}/api\n"
            f"📖 API Docs: http://{args.host}:{args.port}/docs\n\n"
            f"Press Ctrl+C to stop the server",
            title="Server Started",
            border_style="green"
        ))
        
        # Start the server
        import uvicorn
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info" if not args.debug else "debug"
        )
        
    except ImportError:
        console.print("[red]Web interface dependencies not installed. Install with: pip install 'morgan[web]'[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Failed to start server: {e}[/red]")
        sys.exit(1)


def cmd_health(args, morgan):
    """Handle the health command."""
    console.print("[blue]Checking Morgan's health...[/blue]")
    
    try:
        from morgan.utils.health import HealthChecker
        
        health_checker = HealthChecker()
        health_status = health_checker.check_all_systems(detailed=args.detailed)
        
        if health_status["overall_status"] == "healthy":
            status_color = "green"
            status_icon = "✅"
        elif health_status["overall_status"] == "warning":
            status_color = "yellow"
            status_icon = "⚠️"
        else:
            status_color = "red"
            status_icon = "❌"
        
        console.print(Panel.fit(
            f"{status_icon} [bold {status_color}]Overall Status: {health_status['overall_status'].upper()}[/bold {status_color}]\n\n"
            f"🧠 Knowledge Base: {health_status['components']['knowledge']['status']}\n"
            f"💾 Memory System: {health_status['components']['memory']['status']}\n"
            f"🔍 Search Engine: {health_status['components']['search']['status']}\n"
            f"🤖 LLM Service: {health_status['components']['llm']['status']}\n"
            f"📊 Vector Database: {health_status['components']['vector_db']['status']}",
            title="System Health",
            border_style=status_color
        ))
        
        if args.detailed:
            for component, details in health_status['components'].items():
                if details.get('details'):
                    console.print(f"\n[bold]{component.title()}:[/bold]")
                    for key, value in details['details'].items():
                        console.print(f"  {key}: {value}")
        
        if health_status["overall_status"] != "healthy":
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]Health check failed: {e}[/red]")
        sys.exit(1)


def cmd_memory(args, morgan):
    """Handle the memory command."""
    try:
        if args.stats:
            insights = morgan.memory.get_learning_insights()
            
            console.print(Panel.fit(
                f"🧠 [bold blue]Memory Statistics[/bold blue]\n\n"
                f"💬 Total conversations: {insights['total_conversations']}\n"
                f"🔄 Total turns: {insights['total_turns']}\n"
                f"⭐ Average rating: {insights['average_rating']:.1f}/5\n"
                f"📊 Feedback rate: {insights['feedback_percentage']:.1f}%\n"
                f"🏷️  Common topics: {', '.join(insights['common_topics'][:5])}",
                title="Conversation Memory",
                border_style="blue"
            ))
            
        elif args.search:
            results = morgan.memory.search_conversations(args.search, max_results=10)
            
            console.print(f"[blue]Found {len(results)} relevant conversations:[/blue]\n")
            
            for i, result in enumerate(results, 1):
                console.print(f"[bold]{i}. Score: {result['score']:.2f}[/bold]")
                console.print(f"   Q: {result['question']}")
                console.print(f"   A: {result['answer'][:100]}...")
                if result['feedback_rating']:
                    console.print(f"   Rating: {result['feedback_rating']}/5")
                console.print()
                
        elif args.cleanup:
            cleaned = morgan.memory.cleanup_old_conversations(days_to_keep=args.cleanup)
            console.print(f"[green]Cleaned up {cleaned} old conversations (older than {args.cleanup} days)[/green]")
            
    except Exception as e:
        console.print(f"[red]Memory operation failed: {e}[/red]")
        sys.exit(1)


def cmd_knowledge(args, morgan):
    """Handle the knowledge command."""
    try:
        if args.stats:
            stats = morgan.get_knowledge_stats()
            
            console.print(Panel.fit(
                f"📚 [bold blue]Knowledge Base Statistics[/bold blue]\n\n"
                f"📄 Total documents: {stats['total_documents']}\n"
                f"🧩 Knowledge chunks: {stats['knowledge_chunks']}\n"
                f"🏷️  Knowledge areas: {', '.join(stats['knowledge_areas'][:5])}\n"
                f"💾 Storage size: {stats['storage_size']:.1f} MB\n"
                f"🕒 Last updated: {stats['last_updated']}",
                title="Knowledge Base",
                border_style="blue"
            ))
            
        elif args.search:
            results = morgan.knowledge.search_knowledge(args.search, max_results=10)
            
            console.print(f"[blue]Found {len(results)} relevant knowledge chunks:[/blue]\n")
            
            for i, result in enumerate(results, 1):
                console.print(f"[bold]{i}. Score: {result['score']:.2f}[/bold]")
                console.print(f"   Source: {result['source']}")
                console.print(f"   Content: {result['content'][:150]}...")
                console.print()
                
        elif args.clear:
            confirm = console.input("[red]Are you sure you want to clear ALL knowledge? Type 'yes' to confirm: [/red]")
            if confirm.lower() == 'yes':
                success = morgan.knowledge.clear_knowledge(confirm=True)
                if success:
                    console.print("[green]Knowledge base cleared successfully[/green]")
                else:
                    console.print("[red]Failed to clear knowledge base[/red]")
                    sys.exit(1)
            else:
                console.print("[yellow]Operation cancelled[/yellow]")
                
    except Exception as e:
        console.print(f"[red]Knowledge operation failed: {e}[/red]")
        sys.exit(1)


def cmd_init(args):
    """Handle the init command."""
    console.print("[blue]Initializing Morgan in current directory...[/blue]")
    
    try:
        from morgan.utils.init import initialize_morgan
        
        result = initialize_morgan(force=args.force)
        
        if result["success"]:
            console.print(Panel.fit(
                f"✅ [green]Morgan Initialized Successfully![/green]\n\n"
                f"📁 Data directory: {result['data_dir']}\n"
                f"⚙️  Config file: {result['config_file']}\n"
                f"📝 Log directory: {result['log_dir']}\n\n"
                f"Next steps:\n"
                f"1. Edit .env to configure your LLM endpoint\n"
                f"2. Run 'morgan health' to check system status\n"
                f"3. Run 'morgan learn ./docs' to add knowledge\n"
                f"4. Run 'morgan chat' to start chatting!",
                title="Initialization Complete",
                border_style="green"
            ))
        else:
            console.print(f"[red]Initialization failed: {result.get('message', 'Unknown error')}[/red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]Initialization failed: {e}[/red]")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else ("INFO" if args.verbose else "WARNING")
    setup_logging(level=log_level)
    
    # Handle init command separately (doesn't need Morgan instance)
    if args.command == "init":
        cmd_init(args)
        return
    
    # Show help if no command provided
    if not args.command:
        parser.print_help()
        return
    
    # Create Morgan instance
    try:
        morgan = create_assistant(config_path=args.config)
    except Exception as e:
        console.print(f"[red]Failed to initialize Morgan: {e}[/red]")
        console.print(f"[yellow]Try running 'morgan init' first, or check your configuration.[/yellow]")
        sys.exit(1)
    
    # Route to appropriate command handler
    try:
        if args.command == "chat":
            cmd_chat(args, morgan)
        elif args.command == "ask":
            cmd_ask(args, morgan)
        elif args.command == "learn":
            cmd_learn(args, morgan)
        elif args.command == "serve":
            cmd_serve(args, morgan)
        elif args.command == "health":
            cmd_health(args, morgan)
        elif args.command == "memory":
            cmd_memory(args, morgan)
        elif args.command == "knowledge":
            cmd_knowledge(args, morgan)
        else:
            console.print(f"[red]Unknown command: {args.command}[/red]")
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Command failed: {e}[/red]")
        if args.debug:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()