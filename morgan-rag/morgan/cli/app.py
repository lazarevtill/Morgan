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

    # Cache command (implements R1.3, R9.1)
    cache_parser = subparsers.add_parser(
        "cache",
        help="Manage and monitor Git hash cache",
        description="View cache performance metrics and manage cache storage."
    )
    cache_group = cache_parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--stats",
        action="store_true",
        help="Show cache performance statistics"
    )
    cache_group.add_argument(
        "--metrics",
        action="store_true",
        help="Show detailed cache metrics"
    )
    cache_group.add_argument(
        "--efficiency",
        action="store_true",
        help="Show cache efficiency report"
    )
    cache_group.add_argument(
        "--clear",
        action="store_true",
        help="Clear cache metrics (requires confirmation)"
    )
    cache_group.add_argument(
        "--cleanup",
        type=int,
        metavar="DAYS",
        help="Clean up cache entries older than N days"
    )

    # Migration command (implements R10.4, R10.5)
    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Migrate knowledge bases to hierarchical format",
        description="Migrate existing knowledge bases from legacy single-vector to hierarchical multi-scale embeddings."
    )
    migrate_subparsers = migrate_parser.add_subparsers(
        dest="migrate_action",
        help="Migration actions"
    )
    
    # Migrate analyze
    analyze_parser = migrate_subparsers.add_parser(
        "analyze",
        help="Analyze collections for migration readiness"
    )
    analyze_parser.add_argument(
        "collection",
        nargs="?",
        help="Collection name to analyze (analyzes all if not specified)"
    )
    
    # Migrate plan
    plan_parser = migrate_subparsers.add_parser(
        "plan",
        help="Create migration plan for a collection"
    )
    plan_parser.add_argument(
        "source_collection",
        help="Source collection name"
    )
    plan_parser.add_argument(
        "--target",
        help="Target collection name (auto-generated if not specified)"
    )
    plan_parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for migration (default: 50)"
    )
    
    # Migrate execute
    execute_parser = migrate_subparsers.add_parser(
        "execute",
        help="Execute migration for a collection"
    )
    execute_parser.add_argument(
        "source_collection",
        help="Source collection name"
    )
    execute_parser.add_argument(
        "--target",
        help="Target collection name (auto-generated if not specified)"
    )
    execute_parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for migration (default: 50)"
    )
    execute_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run without making changes"
    )
    execute_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm migration execution"
    )
    
    # Migrate validate
    validate_parser = migrate_subparsers.add_parser(
        "validate",
        help="Validate completed migration"
    )
    validate_parser.add_argument(
        "source_collection",
        help="Source collection name"
    )
    validate_parser.add_argument(
        "target_collection",
        help="Target collection name"
    )
    validate_parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of points to sample for validation (default: 100)"
    )
    
    # Migrate rollback
    rollback_parser = migrate_subparsers.add_parser(
        "rollback",
        help="Rollback migration using backup"
    )
    rollback_parser.add_argument(
        "backup_path",
        help="Path to backup file"
    )
    rollback_parser.add_argument(
        "--target",
        help="Target collection name (uses backup collection name if not specified)"
    )
    rollback_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm rollback execution"
    )
    
    # Migrate list-backups
    list_backups_parser = migrate_subparsers.add_parser(
        "list-backups",
        help="List available migration backups"
    )
    
    # Migrate cleanup
    cleanup_parser = migrate_subparsers.add_parser(
        "cleanup",
        help="Clean up old migration backups"
    )
    cleanup_parser.add_argument(
        "--keep-days",
        type=int,
        default=30,
        help="Number of days to keep backups (default: 30)"
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


def cmd_cache(args, morgan):
    """Handle the cache command (implements R1.3, R9.1)."""
    try:
        from pathlib import Path
        from morgan.caching.git_hash_tracker import GitHashTracker
        from morgan.caching.intelligent_cache import IntelligentCacheManager
        
        # Initialize cache components
        cache_dir = Path.home() / ".morgan" / "cache"
        git_tracker = GitHashTracker(cache_dir)
        cache_manager = IntelligentCacheManager(cache_dir)
        
        if args.stats:
            # Show basic cache statistics
            metrics = git_tracker.get_cache_metrics()
            cache_perf = metrics["cache_performance"]
            collection_stats = metrics["collection_stats"]
            
            console.print(Panel.fit(
                f"📊 [bold blue]Git Hash Cache Statistics[/bold blue]\n\n"
                f"🎯 Hit Rate: {cache_perf['hit_rate']:.1%}\n"
                f"📈 Total Requests: {cache_perf['total_requests']}\n"
                f"✅ Cache Hits: {cache_perf['cache_hits']}\n"
                f"❌ Cache Misses: {cache_perf['cache_misses']}\n"
                f"🔢 Hash Calculations: {cache_perf['hash_calculations']}\n"
                f"🗑️  Invalidations: {cache_perf['invalidations']}\n\n"
                f"📚 Collections: {collection_stats['total_collections']}\n"
                f"📄 Documents: {collection_stats['total_documents']}\n"
                f"💾 Storage: {collection_stats['total_size_bytes'] / (1024*1024):.1f} MB",
                title="Cache Performance",
                border_style="blue"
            ))
            
        elif args.metrics:
            # Show detailed metrics
            metrics = git_tracker.get_cache_metrics()
            cache_stats = cache_manager.get_cache_statistics()
            
            console.print(Panel.fit(
                f"📊 [bold blue]Detailed Cache Metrics[/bold blue]\n\n"
                f"[bold]Performance Metrics:[/bold]\n"
                f"  Hit Rate: {metrics['cache_performance']['hit_rate']:.1%}\n"
                f"  Miss Rate: {1 - metrics['cache_performance']['hit_rate']:.1%}\n"
                f"  Total Requests: {metrics['cache_performance']['total_requests']}\n"
                f"  Hash Calculations: {metrics['cache_performance']['hash_calculations']}\n\n"
                f"[bold]Collection Statistics:[/bold]\n"
                f"  Total Collections: {metrics['collection_stats']['total_collections']}\n"
                f"  Total Documents: {metrics['collection_stats']['total_documents']}\n"
                f"  Average Docs/Collection: {metrics['collection_stats']['average_documents_per_collection']:.1f}\n"
                f"  Total Storage: {metrics['collection_stats']['total_size_bytes'] / (1024*1024):.1f} MB\n\n"
                f"[bold]System Metrics:[/bold]\n"
                f"  Cache Manager Hit Rate: {cache_stats['metrics']['hit_rate']:.1%}\n"
                f"  Cache Manager Requests: {cache_stats['metrics']['total_requests']}\n"
                f"  Collections in Cache: {cache_stats['collections']['total_collections']}",
                title="Detailed Cache Metrics",
                border_style="blue"
            ))
            
        elif args.efficiency:
            # Show efficiency report
            efficiency_report = git_tracker.get_cache_efficiency_report()
            summary = efficiency_report["summary"]
            
            # Determine color based on efficiency level
            if summary["efficiency_level"] == "Excellent":
                color = "green"
                icon = "🟢"
            elif summary["efficiency_level"] == "Good":
                color = "yellow"
                icon = "🟡"
            elif summary["efficiency_level"] == "Fair":
                color = "orange"
                icon = "🟠"
            else:
                color = "red"
                icon = "🔴"
            
            console.print(Panel.fit(
                f"{icon} [bold {color}]Cache Efficiency: {summary['efficiency_level']}[/bold {color}]\n\n"
                f"📊 Hit Rate: {summary['hit_rate_percentage']}\n"
                f"📈 Total Requests: {summary['total_requests']}\n"
                f"✅ Cache Hits: {summary['cache_hits']}\n"
                f"❌ Cache Misses: {summary['cache_misses']}\n\n"
                f"[bold]Recommendations:[/bold]\n" +
                "\n".join(f"  • {rec}" for rec in efficiency_report["recommendations"]) if efficiency_report["recommendations"] else "  • No recommendations - cache is performing well!",
                title="Cache Efficiency Report",
                border_style=color
            ))
            
        elif args.clear:
            # Clear cache metrics
            confirm = console.input("[red]Are you sure you want to clear cache metrics? Type 'yes' to confirm: [/red]")
            if confirm.lower() == 'yes':
                success = git_tracker.reset_cache_metrics()
                if success:
                    console.print("[green]Cache metrics cleared successfully[/green]")
                else:
                    console.print("[red]Failed to clear cache metrics[/red]")
                    sys.exit(1)
            else:
                console.print("[yellow]Operation cancelled[/yellow]")
                
        elif args.cleanup:
            # Clean up old cache entries
            console.print(f"[blue]Cleaning up cache entries older than {args.cleanup} days...[/blue]")
            cleaned_count = cache_manager.cleanup_expired_cache(max_age_days=args.cleanup)
            console.print(f"[green]Cleaned up {cleaned_count} expired cache entries[/green]")
            
        else:
            # Default: show basic stats
            metrics = git_tracker.get_cache_metrics()
            cache_perf = metrics["cache_performance"]
            
            console.print(Panel.fit(
                f"📊 [bold blue]Cache Overview[/bold blue]\n\n"
                f"🎯 Hit Rate: {cache_perf['hit_rate']:.1%}\n"
                f"📈 Total Requests: {cache_perf['total_requests']}\n"
                f"📚 Collections: {metrics['collection_stats']['total_collections']}\n\n"
                f"Use --stats, --metrics, or --efficiency for more details",
                title="Git Hash Cache",
                border_style="blue"
            ))
            
    except Exception as e:
        console.print(f"[red]Cache operation failed: {e}[/red]")
        sys.exit(1)


def cmd_migrate(args, morgan):
    """Handle the migrate command (implements R10.4, R10.5)."""
    from morgan.migration import KnowledgeBaseMigrator, MigrationValidator, RollbackManager
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    
    try:
        if args.migrate_action == "analyze":
            # Analyze collections for migration readiness
            migrator = KnowledgeBaseMigrator()
            
            if args.collection:
                # Analyze specific collection
                console.print(f"[blue]Analyzing collection '{args.collection}'...[/blue]")
                analysis = migrator.analyze_collection(args.collection)
                
                if not analysis.get("exists", False):
                    console.print(f"[red]Error: {analysis.get('error', 'Collection not found')}[/red]")
                    return
                
                # Display analysis results
                table = Table(title=f"Migration Analysis: {args.collection}")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="white")
                
                table.add_row("Total Points", str(analysis["total_points"]))
                table.add_row("Has Legacy Format", str(analysis["has_legacy_format"]))
                table.add_row("Has Hierarchical Format", str(analysis["has_hierarchical_format"]))
                table.add_row("Migration Needed", analysis["migration_needed"])
                table.add_row("Estimated Time (min)", f"{analysis['estimated_migration_time_minutes']:.1f}")
                table.add_row("Disk Usage (MB)", f"{analysis['disk_usage_mb']:.1f}")
                
                console.print(table)
                
                # Show recommendations
                if analysis["migration_needed"] == "legacy_to_hierarchical":
                    console.print("\n[yellow]Recommendation: This collection can be migrated to hierarchical format[/yellow]")
                elif analysis["migration_needed"] == "already_hierarchical":
                    console.print("\n[green]This collection is already in hierarchical format[/green]")
                elif analysis["migration_needed"] == "mixed_format":
                    console.print("\n[orange]Warning: This collection has mixed formats and may need cleanup[/orange]")
                
            else:
                # Analyze all collections
                console.print("[blue]Analyzing all collections...[/blue]")
                collections = migrator.list_collections()
                
                if not collections:
                    console.print("[yellow]No collections found[/yellow]")
                    return
                
                # Display results table
                table = Table(title="Collection Migration Analysis")
                table.add_column("Collection", style="cyan")
                table.add_column("Points", justify="right")
                table.add_column("Format", style="white")
                table.add_column("Migration Needed", style="yellow")
                table.add_column("Est. Time (min)", justify="right")
                
                for analysis in collections:
                    if analysis.get("exists", False):
                        format_str = ""
                        if analysis["has_legacy_format"]:
                            format_str += "Legacy"
                        if analysis["has_hierarchical_format"]:
                            format_str += " + Hierarchical" if format_str else "Hierarchical"
                        
                        table.add_row(
                            analysis["collection_name"],
                            str(analysis["total_points"]),
                            format_str,
                            analysis["migration_needed"],
                            f"{analysis['estimated_migration_time_minutes']:.1f}"
                        )
                
                console.print(table)
        
        elif args.migrate_action == "plan":
            # Create migration plan
            migrator = KnowledgeBaseMigrator()
            
            console.print(f"[blue]Creating migration plan for '{args.source_collection}'...[/blue]")
            
            try:
                plan = migrator.create_migration_plan(
                    source_collection=args.source_collection,
                    target_collection=args.target,
                    batch_size=args.batch_size
                )
                
                # Display plan
                table = Table(title="Migration Plan")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="white")
                
                table.add_row("Source Collection", plan.source_collection)
                table.add_row("Target Collection", plan.target_collection)
                table.add_row("Total Points", str(plan.total_points))
                table.add_row("Batch Size", str(plan.batch_size))
                table.add_row("Estimated Time (min)", f"{plan.estimated_time_minutes:.1f}")
                table.add_row("Backup Path", plan.backup_path)
                
                console.print(table)
                console.print("\n[green]Migration plan created successfully[/green]")
                console.print(f"[yellow]Use 'morgan migrate execute {args.source_collection}' to execute[/yellow]")
                
            except Exception as e:
                console.print(f"[red]Failed to create migration plan: {e}[/red]")
        
        elif args.migrate_action == "execute":
            # Execute migration
            if not args.confirm and not args.dry_run:
                console.print("[red]Migration requires --confirm flag or --dry-run for safety[/red]")
                console.print("[yellow]Use --dry-run to test without making changes[/yellow]")
                return
            
            migrator = KnowledgeBaseMigrator()
            
            console.print(f"[blue]{'Dry run' if args.dry_run else 'Executing'} migration for '{args.source_collection}'...[/blue]")
            
            try:
                # Create plan
                plan = migrator.create_migration_plan(
                    source_collection=args.source_collection,
                    target_collection=args.target,
                    batch_size=args.batch_size,
                    dry_run=args.dry_run
                )
                
                # Show plan summary
                console.print(f"Source: {plan.source_collection} -> Target: {plan.target_collection}")
                console.print(f"Points to migrate: {plan.total_points}")
                
                if not args.dry_run:
                    console.print(f"[yellow]Backup will be created at: {plan.backup_path}[/yellow]")
                
                # Execute with progress
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Migrating...", total=None)
                    result = migrator.execute_migration(plan)
                
                # Display results
                if result.success:
                    console.print(f"[green]Migration {'simulation' if args.dry_run else 'completed'} successfully![/green]")
                    console.print(f"Points migrated: {result.points_migrated}")
                    if result.points_failed > 0:
                        console.print(f"[yellow]Points failed: {result.points_failed}[/yellow]")
                    console.print(f"Execution time: {result.execution_time_seconds:.1f}s")
                    
                    if result.backup_created:
                        console.print(f"[green]Backup created successfully[/green]")
                    
                    if not args.dry_run:
                        console.print(f"[yellow]Use 'morgan migrate validate {plan.source_collection} {plan.target_collection}' to verify[/yellow]")
                else:
                    console.print(f"[red]Migration failed: {result.error_message}[/red]")
                    if result.backup_created:
                        console.print("[yellow]Backup was created and can be used for rollback[/yellow]")
                
            except Exception as e:
                console.print(f"[red]Migration execution failed: {e}[/red]")
        
        elif args.migrate_action == "validate":
            # Validate migration
            validator = MigrationValidator()
            
            console.print(f"[blue]Validating migration: {args.source_collection} -> {args.target_collection}...[/blue]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Validating...", total=None)
                result = validator.validate_migration(
                    args.source_collection,
                    args.target_collection,
                    sample_size=args.sample_size
                )
            
            # Display validation results
            if result.is_valid:
                console.print("[green]Migration validation passed![/green]")
            else:
                console.print("[red]Migration validation failed![/red]")
            
            table = Table(title="Validation Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Points Checked", str(result.total_points_checked))
            table.add_row("Valid Points", str(result.points_valid))
            table.add_row("Invalid Points", str(result.points_invalid))
            table.add_row("Validation Rate", f"{(result.points_valid / result.total_points_checked * 100):.1f}%" if result.total_points_checked > 0 else "N/A")
            
            console.print(table)
            
            if result.validation_errors:
                console.print("\n[red]Validation Errors:[/red]")
                for error in result.validation_errors[:10]:  # Show first 10 errors
                    console.print(f"  • {error}")
                if len(result.validation_errors) > 10:
                    console.print(f"  ... and {len(result.validation_errors) - 10} more errors")
        
        elif args.migrate_action == "rollback":
            # Execute rollback
            if not args.confirm:
                console.print("[red]Rollback requires --confirm flag for safety[/red]")
                return
            
            rollback_manager = RollbackManager()
            
            console.print(f"[blue]Executing rollback from {args.backup_path}...[/blue]")
            
            # Validate backup first
            validation = rollback_manager.validate_backup_for_rollback(args.backup_path)
            if not validation["is_valid"]:
                console.print(f"[red]Backup validation failed: {validation.get('error', 'Unknown error')}[/red]")
                return
            
            # Show warnings
            if validation.get("warnings"):
                for warning in validation["warnings"]:
                    console.print(f"[yellow]Warning: {warning}[/yellow]")
            
            # Execute rollback
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Rolling back...", total=None)
                result = rollback_manager.execute_rollback(
                    args.backup_path,
                    target_collection=args.target,
                    confirm_overwrite=True
                )
            
            # Display results
            if result.success:
                console.print("[green]Rollback completed successfully![/green]")
                console.print(f"Points restored: {result.points_restored}")
                console.print(f"Collection: {result.collection_restored}")
                console.print(f"Execution time: {result.execution_time_seconds:.1f}s")
            else:
                console.print(f"[red]Rollback failed: {result.error_message}[/red]")
        
        elif args.migrate_action == "list-backups":
            # List available backups
            rollback_manager = RollbackManager()
            
            console.print("[blue]Listing available migration backups...[/blue]")
            backups = rollback_manager.list_available_backups()
            
            if not backups:
                console.print("[yellow]No migration backups found[/yellow]")
                return
            
            table = Table(title="Available Migration Backups")
            table.add_column("File Name", style="cyan")
            table.add_column("Collection", style="white")
            table.add_column("Points", justify="right")
            table.add_column("Size (MB)", justify="right")
            table.add_column("Created", style="dim")
            
            for backup in backups:
                table.add_row(
                    backup["file_name"],
                    backup["collection_name"],
                    str(backup["total_points"]),
                    f"{backup['file_size_mb']:.1f}",
                    backup["backup_timestamp"][:19]  # Remove microseconds
                )
            
            console.print(table)
        
        elif args.migrate_action == "cleanup":
            # Clean up old backups
            rollback_manager = RollbackManager()
            
            console.print(f"[blue]Cleaning up backups older than {args.keep_days} days...[/blue]")
            result = rollback_manager.cleanup_old_backups(keep_days=args.keep_days)
            
            if "error" in result:
                console.print(f"[red]Cleanup failed: {result['error']}[/red]")
            else:
                console.print(f"[green]{result['message']}[/green]")
        
        else:
            console.print("[red]Unknown migration action[/red]")
            console.print("[yellow]Use 'morgan migrate --help' for available actions[/yellow]")
    
    except Exception as e:
        console.print(f"[red]Migration command failed: {e}[/red]")
        logger.error(f"Migration command error: {e}")


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
        elif args.command == "cache":
            cmd_cache(args, morgan)
        elif args.command == "migrate":
            cmd_migrate(args, morgan)
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