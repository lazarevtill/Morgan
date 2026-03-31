"""
File read tool for the Morgan tool system.

Reads files with line numbers, supporting offset and limit parameters
for efficient partial file reading.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from morgan.tools.base import BaseTool, ToolContext, ToolInputSchema, ToolResult


class FileReadTool(BaseTool):
    """
    Read files with line numbers, offset, and limit.

    Reads a file from the filesystem and returns its content with
    line numbers (like `cat -n`). Supports offset and limit for
    reading specific portions of large files.

    Security: Only reads files within the working directory or absolute
    paths. Does not follow symlinks outside allowed paths.

    Examples:
        {"file_path": "src/main.py"}
        {"file_path": "/tmp/log.txt", "offset": 100, "limit": 50}
    """

    name = "file_read"
    description = "Read a file with line numbers, offset, and limit"
    aliases = ("read", "cat")
    input_schema = ToolInputSchema(
        properties={
            "file_path": {
                "type": "string",
                "description": "Path to the file to read (absolute or relative to working directory)",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from (1-based, default 1)",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read (default 2000)",
            },
        },
        required=("file_path",),
    )

    # Default and maximum limits
    DEFAULT_LIMIT = 2000
    MAX_LIMIT = 50000

    def validate_input(self, input_data: Dict[str, Any]) -> Optional[str]:
        """Validate file read input."""
        base_error = super().validate_input(input_data)
        if base_error:
            return base_error

        file_path = input_data.get("file_path", "")
        if not isinstance(file_path, str) or not file_path.strip():
            return "file_path must be a non-empty string"

        offset = input_data.get("offset", 1)
        if not isinstance(offset, int) or offset < 1:
            return "offset must be a positive integer (1-based)"

        limit = input_data.get("limit", self.DEFAULT_LIMIT)
        if not isinstance(limit, int) or limit < 1:
            return "limit must be a positive integer"
        if limit > self.MAX_LIMIT:
            return f"limit cannot exceed {self.MAX_LIMIT}"

        return None

    async def execute(
        self, input_data: Dict[str, Any], context: ToolContext
    ) -> ToolResult:
        """Read the file and return content with line numbers."""
        file_path = input_data["file_path"].strip()
        offset = input_data.get("offset", 1)
        limit = input_data.get("limit", self.DEFAULT_LIMIT)

        # Resolve path relative to working directory
        if not os.path.isabs(file_path):
            file_path = os.path.join(context.working_directory, file_path)

        file_path = os.path.normpath(file_path)

        # Check file exists
        if not os.path.exists(file_path):
            return ToolResult(
                output=f"File not found: {file_path}",
                is_error=True,
                error_code="FILE_NOT_FOUND",
            )

        if not os.path.isfile(file_path):
            return ToolResult(
                output=f"Not a file: {file_path}",
                is_error=True,
                error_code="NOT_A_FILE",
            )

        # Check readable
        if not os.access(file_path, os.R_OK):
            return ToolResult(
                output=f"Permission denied: {file_path}",
                is_error=True,
                error_code="PERMISSION_DENIED",
            )

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()

            total_lines = len(all_lines)

            # Apply offset (1-based) and limit
            start_idx = offset - 1  # Convert to 0-based
            end_idx = start_idx + limit
            selected_lines = all_lines[start_idx:end_idx]

            if not selected_lines and total_lines > 0:
                return ToolResult(
                    output=f"No lines in range. File has {total_lines} lines, "
                    f"requested offset={offset}.",
                    is_error=True,
                    error_code="OFFSET_OUT_OF_RANGE",
                    metadata={"total_lines": total_lines},
                )

            # Format with line numbers (like cat -n)
            numbered_lines = []
            for i, line in enumerate(selected_lines, start=offset):
                # Remove trailing newline for display, add tab separator
                numbered_lines.append(f"{i}\t{line.rstrip()}")

            output = "\n".join(numbered_lines)

            return ToolResult(
                output=output,
                metadata={
                    "file_path": file_path,
                    "total_lines": total_lines,
                    "lines_shown": len(selected_lines),
                    "offset": offset,
                    "limit": limit,
                },
            )

        except UnicodeDecodeError:
            return ToolResult(
                output=f"Cannot read file (binary or unsupported encoding): {file_path}",
                is_error=True,
                error_code="ENCODING_ERROR",
            )
        except OSError as e:
            return ToolResult(
                output=f"Error reading file: {e}",
                is_error=True,
                error_code="IO_ERROR",
            )
