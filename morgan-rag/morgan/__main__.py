"""
Morgan RAG - Command Line Interface

Human-first CLI that makes it easy to interact with Morgan.

Usage:
    python -m morgan chat                    # Start interactive chat
    python -m morgan ask "How do I...?"     # Ask a single question
    python -m morgan learn ./docs           # Teach Morgan from documents
    python -m morgan serve                  # Start web interface
    python -m morgan health                 # Check system health
"""

from morgan.cli.click_cli import run

if __name__ == "__main__":
    run()
