"""
Morgan Server Entry Point

Run the Morgan server directly using:
    python -m morgan_server

Or with custom configuration:
    python -m morgan_server --host 0.0.0.0 --port 8080
"""

import argparse
import sys
from pathlib import Path

import uvicorn

from morgan_server.app import create_app
from morgan_server.config import ConfigurationError


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Morgan Server - Personal AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults
  python -m morgan_server

  # Run on specific host and port
  python -m morgan_server --host 127.0.0.1 --port 9000

  # Run with custom config file
  python -m morgan_server --config config.yaml

  # Run with debug logging
  python -m morgan_server --log-level DEBUG

Environment Variables:
  MORGAN_HOST              Server host (default: 0.0.0.0)
  MORGAN_PORT              Server port (default: 8080)
  MORGAN_LLM_ENDPOINT      LLM endpoint URL (required)
  MORGAN_VECTOR_DB_URL     Vector database URL (required)
  MORGAN_LOG_LEVEL         Log level (default: INFO)
  MORGAN_LOG_FORMAT        Log format: json or text (default: json)

For more information, visit: http://localhost:8080/docs
        """
    )
    
    parser.add_argument(
        "--host",
        type=str,
        help="Server host address (default: from config or 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        help="Server port (default: from config or 8080)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (YAML, JSON, or .env)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-format",
        type=str,
        choices=["json", "text"],
        help="Log output format"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker processes (default: 4)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the server."""
    args = parse_args()
    
    try:
        # Build configuration overrides from command-line args
        config_overrides = {}
        
        if args.host:
            config_overrides["host"] = args.host
        
        if args.port:
            config_overrides["port"] = args.port
        
        if args.log_level:
            config_overrides["log_level"] = args.log_level
        
        if args.log_format:
            config_overrides["log_format"] = args.log_format
        
        if args.workers:
            config_overrides["workers"] = args.workers
        
        # Create application
        config_file = Path(args.config) if args.config else None
        app = create_app(
            config_file=config_file,
            **config_overrides
        )
        
        # Get final configuration
        config = app.state.config
        
        print(f"Starting Morgan Server v{app.version}")
        print(f"  Host: {config.host}")
        print(f"  Port: {config.port}")
        print(f"  Log Level: {config.log_level}")
        print(f"  LLM Provider: {config.llm_provider}")
        print(f"  LLM Endpoint: {config.llm_endpoint}")
        print(f"  Vector DB: {config.vector_db_url}")
        print()
        print(f"Documentation: http://{config.host}:{config.port}/docs")
        print(f"Health Check: http://{config.host}:{config.port}/health")
        print()
        
        # Run server
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            log_level=config.log_level.lower(),
            reload=args.reload,
        )
    
    except ConfigurationError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        print("\nPlease check your configuration and try again.", file=sys.stderr)
        print("For help, run: python -m morgan_server --help", file=sys.stderr)
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nShutting down Morgan Server...")
        sys.exit(0)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
