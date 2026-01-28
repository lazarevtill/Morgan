#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple wrapper to run refactoring with proper encoding
"""
import sys
import io

# Set stdout to UTF-8
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Now import and run
from refactor_to_v2 import main

if __name__ == "__main__":
    main()
