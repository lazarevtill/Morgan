"""
Minimal web app shim for Morgan.

This module wraps the existing FastAPI-based interface under
``morgan.interfaces.web_interface`` so the CLI ``serve`` command can
import ``create_app`` without failing.
"""

from .app import create_app  # noqa: F401
