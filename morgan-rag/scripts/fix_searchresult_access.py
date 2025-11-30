#!/usr/bin/env python3
"""
Fix SearchResult attribute access across the codebase.

SearchResult (from vector_db.client) is a dataclass with attributes:
- result.payload (not result.get("payload"))
- result.score (not result.get("score"))
- result.id (not result.get("id"))
"""

import re
from pathlib import Path

def fix_file(file_path: Path):
    """Fix SearchResult access patterns in a file."""
    content = file_path.read_text()
    original = content

    # Fix result.get("payload", {}) -> result.payload
    content = re.sub(
        r'result\.get\("payload",\s*\{\}\)',
        'result.payload',
        content
    )

    # Fix result.get("score", ...) -> result.score
    content = re.sub(
        r'result\.get\("score",\s*[^)]+\)',
        'result.score',
        content
    )

    # Fix result.get("vector") -> result.vector (if vector field exists)
    #content = re.sub(
    #    r'result\.get\("vector"\)',
    #    'result.vector',
    #    content
    #)

    if content != original:
        print(f"Fixed: {file_path}")
        file_path.write_text(content)
        return True
    return False

def main():
    """Fix all files in morgan/core/."""
    base_dir = Path(__file__).parent.parent / "morgan" / "core"

    fixed_count = 0
    for py_file in base_dir.glob("*.py"):
        if fix_file(py_file):
            fixed_count += 1

    print(f"\nFixed {fixed_count} files")

if __name__ == "__main__":
    main()
