"""
Main entry point for Morgan CLI.

This module provides the entry point when running the package as a module
(python -m morgan_cli) or via the installed console script (morgan).
"""

from morgan_cli.cli import main

if __name__ == "__main__":
    main()
