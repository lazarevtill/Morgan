# Copyright 2025 Morgan AI Assistant Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Memory Consolidation Module.

Provides daily memory logging, hybrid search (BM25 + vector),
and LLM-driven consolidation of daily logs into MEMORY.md.

Ported from OpenClaw's memory-core and Claude Code's memdir patterns.

Usage:
    from morgan.memory_consolidation import (
        DailyLogManager,
        HybridMemorySearch,
        MemoryConsolidator,
    )

    # Daily logging
    log_mgr = DailyLogManager(Path("memory"))
    log_mgr.append("Discovered new pattern in user queries")
    recent = log_mgr.list_recent(days=7)

    # Hybrid search over memory logs
    search = HybridMemorySearch(Path("memory"))
    results = search.hybrid_search("pattern recognition", limit=5)

    # Consolidate daily logs into MEMORY.md
    consolidator = MemoryConsolidator(Path("workspace"))
    consolidator.consolidate(days_to_review=7)
"""

from morgan.memory_consolidation.daily_log import DailyLogManager
from morgan.memory_consolidation.hybrid_search import HybridMemorySearch
from morgan.memory_consolidation.consolidator import MemoryConsolidator

__all__ = [
    "DailyLogManager",
    "HybridMemorySearch",
    "MemoryConsolidator",
]
