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

import sys
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import morgan
sys.path.insert(0, str(Path(__file__).parent.parent))

from morgan.cli.app import main

if __name__ == "__main__":
    main()