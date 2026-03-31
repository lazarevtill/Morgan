"""
Built-in tools for the Morgan tool system.

Provides a curated set of tools that ship with Morgan out of the box.
"""

from __future__ import annotations

from typing import List

from morgan.tools.base import BaseTool
from morgan.tools.builtin.bash_tool import BashTool
from morgan.tools.builtin.calculator import CalculatorTool
from morgan.tools.builtin.file_read import FileReadTool
from morgan.tools.builtin.memory_search import MemorySearchTool
from morgan.tools.builtin.web_search import WebSearchTool

# All built-in tool classes, instantiated and ready to register
ALL_BUILTIN_TOOLS: List[BaseTool] = [
    CalculatorTool(),
    FileReadTool(),
    BashTool(),
    WebSearchTool(),
    MemorySearchTool(),
]

__all__ = [
    "ALL_BUILTIN_TOOLS",
    "BashTool",
    "CalculatorTool",
    "FileReadTool",
    "MemorySearchTool",
    "WebSearchTool",
]
