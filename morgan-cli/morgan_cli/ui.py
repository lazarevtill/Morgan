"""
Rich UI Components for Morgan CLI

This module provides UI components for the terminal interface using Rich library.
It handles markdown rendering, typing indicators, progress feedback, scrolling,
pagination, and error message display.
"""

import re
from typing import Optional, List, Dict, Any
from datetime import datetime

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.text import Text
from rich.table import Table
from rich.layout import Layout
from rich.style import Style
from rich import box


# ============================================================================
# Console Instance
# ============================================================================

console = Console()


# ============================================================================
# Markdown Rendering
# ============================================================================


def render_markdown(content: str, title: Optional[str] = None) -> None:
    """
    Render markdown content with Rich formatting.

    Args:
        content: Markdown content to render
        title: Optional title for the panel
    """
    if not content:
        return

    try:
        md = Markdown(content)
        if title:
            panel = Panel(md, title=title, border_style="blue", box=box.ROUNDED)
            console.print(panel)
        else:
            console.print(md)
    except Exception as e:
        # Fallback to plain text if markdown parsing fails
        console.print(f"[yellow]Warning: Failed to parse markdown: {e}[/yellow]")
        console.print(content)


def render_message(
    message: str,
    sender: str = "Morgan",
    timestamp: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Render a chat message with formatting.

    Args:
        message: Message content (supports markdown)
        sender: Sender name
        timestamp: Message timestamp
        metadata: Additional metadata to display
    """
    # Format timestamp
    time_str = ""
    if timestamp:
        time_str = f" [{timestamp.strftime('%H:%M:%S')}]"

    # Create title
    title = f"[bold cyan]{sender}[/bold cyan]{time_str}"

    # Render message
    md = Markdown(message)
    panel = Panel(
        md,
        title=title,
        border_style="cyan" if sender == "Morgan" else "green",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print(panel)

    # Display metadata if present
    if metadata and any(metadata.values()):
        render_metadata(metadata)


def render_metadata(metadata: Dict[str, Any]) -> None:
    """
    Render message metadata in a compact format.

    Args:
        metadata: Metadata dictionary
    """
    if not metadata:
        return

    # Filter out None values and empty lists
    filtered = {
        k: v for k, v in metadata.items() if v is not None and v != [] and v != {}
    }

    if not filtered:
        return

    # Create a compact table
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Key", style="dim")
    table.add_column("Value")

    for key, value in filtered.items():
        # Format key
        display_key = key.replace("_", " ").title()

        # Format value
        if isinstance(value, list):
            display_value = ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            display_value = ", ".join(f"{k}: {v}" for k, v in value.items())
        else:
            display_value = str(value)

        table.add_row(display_key, display_value)

    console.print(table)
    console.print()


# ============================================================================
# Typing Indicators and Progress
# ============================================================================


class TypingIndicator:
    """
    Typing indicator for showing that Morgan is thinking/responding.
    """

    def __init__(self, message: str = "Morgan is thinking"):
        """
        Initialize typing indicator.

        Args:
            message: Message to display
        """
        self.message = message
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        )
        self.task_id = None

    def __enter__(self):
        """Start the typing indicator."""
        self.progress.start()
        self.task_id = self.progress.add_task(self.message, total=None)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the typing indicator."""
        if self.task_id is not None:
            self.progress.stop()

    def update(self, message: str) -> None:
        """
        Update the indicator message.

        Args:
            message: New message to display
        """
        if self.task_id is not None:
            self.progress.update(self.task_id, description=message)


def show_progress(message: str) -> TypingIndicator:
    """
    Create and return a typing indicator.

    Args:
        message: Progress message

    Returns:
        TypingIndicator instance
    """
    return TypingIndicator(message)


# ============================================================================
# Scrolling and Pagination
# ============================================================================


def render_paginated(
    content: str, page_size: int = 20, title: Optional[str] = None
) -> None:
    """
    Render content with pagination for long responses.

    Args:
        content: Content to render
        page_size: Number of lines per page
        title: Optional title
    """
    lines = content.split("\n")

    if len(lines) <= page_size:
        # No pagination needed
        render_markdown(content, title)
        return

    # Paginate
    total_pages = (len(lines) + page_size - 1) // page_size
    current_page = 0

    while current_page < total_pages:
        start_idx = current_page * page_size
        end_idx = min(start_idx + page_size, len(lines))
        page_content = "\n".join(lines[start_idx:end_idx])

        # Clear screen and render page
        console.clear()
        page_title = (
            f"{title} (Page {current_page + 1}/{total_pages})"
            if title
            else f"Page {current_page + 1}/{total_pages}"
        )
        render_markdown(page_content, page_title)

        # Show navigation prompt
        if current_page < total_pages - 1:
            console.print("\n[dim]Press Enter for next page, 'q' to quit...[/dim]")
            user_input = input().strip().lower()
            if user_input == "q":
                break

        current_page += 1


def render_scrollable(content: str, title: Optional[str] = None) -> None:
    """
    Render content in a scrollable view.

    For now, this is the same as regular rendering since Rich handles
    terminal scrolling automatically. This function exists for future
    enhancement with custom scrolling controls.

    Args:
        content: Content to render
        title: Optional title
    """
    render_markdown(content, title)


# ============================================================================
# Error Display
# ============================================================================


def render_error(
    error_message: str,
    error_type: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    suggestions: Optional[List[str]] = None,
) -> None:
    """
    Render an error message with user-friendly formatting.

    Args:
        error_message: Main error message
        error_type: Type of error (e.g., "Connection Error", "Validation Error")
        details: Additional error details
        suggestions: List of suggested actions
    """
    # Create error content
    content_parts = []

    # Main error message
    error_text = Text()
    error_text.append("❌ ", style="bold red")
    error_text.append(error_message, style="red")
    content_parts.append(error_text)

    # Add details if present
    if details:
        content_parts.append(Text())  # Empty line
        details_text = Text("Details:", style="bold yellow")
        content_parts.append(details_text)

        for key, value in details.items():
            detail_line = Text()
            detail_line.append(f"  • {key}: ", style="yellow")
            detail_line.append(str(value), style="white")
            content_parts.append(detail_line)

    # Add suggestions if present
    if suggestions:
        content_parts.append(Text())  # Empty line
        suggestions_text = Text("Suggestions:", style="bold green")
        content_parts.append(suggestions_text)

        for suggestion in suggestions:
            suggestion_line = Text()
            suggestion_line.append("  → ", style="green")
            suggestion_line.append(suggestion, style="white")
            content_parts.append(suggestion_line)

    # Create panel
    title = f"[bold red]{error_type or 'Error'}[/bold red]"
    panel = Panel(
        Group(*content_parts),
        title=title,
        border_style="red",
        box=box.HEAVY,
        padding=(1, 2),
    )

    console.print(panel)


def render_connection_error(server_url: str, error: Exception) -> None:
    """
    Render a connection error with helpful suggestions.

    Args:
        server_url: Server URL that failed to connect
        error: The exception that occurred
    """
    suggestions = [
        f"Check if the server is running at {server_url}",
        "Verify the server URL is correct",
        "Check your network connection",
        "Try running: curl {}/health".format(server_url),
    ]

    render_error(
        error_message=f"Failed to connect to Morgan server",
        error_type="Connection Error",
        details={"Server URL": server_url, "Error": str(error)},
        suggestions=suggestions,
    )


def render_validation_error(field_errors: Dict[str, str]) -> None:
    """
    Render validation errors with field-level details.

    Args:
        field_errors: Dictionary mapping field names to error messages
    """
    suggestions = [
        "Check the format of your input",
        "Refer to the documentation for valid values",
    ]

    render_error(
        error_message="Invalid input provided",
        error_type="Validation Error",
        details=field_errors,
        suggestions=suggestions,
    )


def render_timeout_error(timeout_seconds: int) -> None:
    """
    Render a timeout error with suggestions.

    Args:
        timeout_seconds: Timeout duration that was exceeded
    """
    suggestions = [
        "The server might be processing a complex request",
        "Try again in a moment",
        "Consider increasing the timeout setting",
        "Check server logs for issues",
    ]

    render_error(
        error_message=f"Request timed out after {timeout_seconds} seconds",
        error_type="Timeout Error",
        details={"Timeout": f"{timeout_seconds}s"},
        suggestions=suggestions,
    )


def render_server_error(
    status_code: int, message: str, details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Render a server error with status code.

    Args:
        status_code: HTTP status code
        message: Error message from server
        details: Additional error details
    """
    suggestions = []

    if status_code >= 500:
        suggestions = [
            "This is a server-side error",
            "Try again in a moment",
            "Contact the administrator if the problem persists",
        ]
    elif status_code == 404:
        suggestions = [
            "Check that the resource exists",
            "Verify the endpoint URL is correct",
        ]
    elif status_code == 400:
        suggestions = ["Check your input format", "Refer to the API documentation"]

    error_details = {"Status Code": status_code}
    if details:
        error_details.update(details)

    render_error(
        error_message=message,
        error_type=f"Server Error ({status_code})",
        details=error_details,
        suggestions=suggestions,
    )


# ============================================================================
# Status Display
# ============================================================================


def render_status(status: str, message: Optional[str] = None) -> None:
    """
    Render a status message.

    Args:
        status: Status type ("success", "info", "warning", "error")
        message: Status message
    """
    status_styles = {
        "success": ("✓", "green"),
        "info": ("ℹ", "blue"),
        "warning": ("⚠", "yellow"),
        "error": ("✗", "red"),
    }

    icon, color = status_styles.get(status, ("•", "white"))

    text = Text()
    text.append(f"{icon} ", style=f"bold {color}")
    if message:
        text.append(message, style=color)

    console.print(text)


def render_welcome() -> None:
    """Render welcome message."""
    welcome_text = """
# Welcome to Morgan! 🤖

Your personal AI assistant with empathic intelligence and knowledge capabilities.

**Commands:**
- Type your message to chat with Morgan
- Type `/help` for available commands
- Type `/exit` or `/quit` to exit

Let's get started!
    """
    render_markdown(welcome_text.strip())
    console.print()


def render_help() -> None:
    """Render help information."""
    help_text = """
# Morgan CLI Help

## Chat Commands
- **Regular message**: Just type your message and press Enter
- **/help**: Show this help message
- **/exit** or **/quit**: Exit the application
- **/clear**: Clear the screen
- **/status**: Show server status
- **/profile**: Show your profile
- **/memory**: Show memory statistics

## Tips
- Morgan supports markdown formatting in responses
- Use arrow keys to navigate command history
- Long responses will be displayed with proper formatting
    """
    render_markdown(help_text.strip())
    console.print()


# ============================================================================
# Utility Functions
# ============================================================================


def clear_screen() -> None:
    """Clear the console screen."""
    console.clear()


def print_separator() -> None:
    """Print a visual separator."""
    console.print("─" * console.width, style="dim")


def confirm(message: str, default: bool = True) -> bool:
    """
    Ask for user confirmation.

    Args:
        message: Confirmation message
        default: Default value if user just presses Enter

    Returns:
        True if confirmed, False otherwise
    """
    suffix = " [Y/n]: " if default else " [y/N]: "
    response = console.input(f"[bold]{message}[/bold]{suffix}").strip().lower()

    if not response:
        return default

    return response in ("y", "yes")


def get_input(prompt: str = "You") -> str:
    """
    Get user input with a styled prompt.

    Args:
        prompt: Prompt text

    Returns:
        User input
    """
    return console.input(f"[bold green]{prompt}[/bold green] > ").strip()
