# Codebase Cleanup v2

## Overview

Comprehensive cleanup and reorganization of the Morgan codebase based on findings from 11 parallel exploration agents that identified 150+ issues.

## Status

| Document | Status |
|----------|--------|
| Requirements | Complete |
| Design | Complete |
| Tasks | Complete |
| Implementation | Ready to Start |

## Key Findings

### Critical Issues (2)
1. **Collection Name Mismatch** - Memory stored in `morgan_memories`, searched in `morgan_turns`
2. **Test Import Bugs** - Missing imports cause NameError in tests

### High Priority Issues (8)
- Exception hierarchy chaos (3 duplicate hierarchies)
- 6 unused exception classes
- Duplicate `setup_model_cache()` in 4 files
- Conflicting `HostRole` enum definitions
- Configuration classes with incompatible field names
- Inconsistent singleton patterns (15+ implementations)
- Client `cleanup_memory()` bug
- Silent error suppression in server

### Medium Priority Issues (10)
- 708 broad `except Exception` catches
- 5 duplicate deduplication implementations
- 4 duplicate text extraction functions
- 5 duplicate valence mappings
- Missing `health_check()` and `shutdown()` methods
- Global state anti-pattern in server
- Common client code duplication

## Estimated Impact

- **Lines of duplicate code removed**: ~2,500
- **Exception catches improved**: 708 → <200
- **Configuration sources unified**: 8 → 3
- **Singleton implementations**: 15+ → 1 factory

## Documents

- [requirements.md](requirements.md) - Detailed requirements with acceptance criteria
- [design.md](design.md) - Architecture and component designs
- [tasks.md](tasks.md) - Step-by-step implementation tasks

## Quick Start

1. Start with **Phase 1** (Critical Fixes) - no dependencies
2. Create **Phase 2** (Shared Module) - parallel with Phase 1
3. Continue phases in order based on dependencies

## Timeline

- **Phase 1-2**: Day 1 (4-6 hours)
- **Phase 3-4**: Day 2 (5-7 hours)
- **Phase 5-7**: Day 3 (6-9 hours)
- **Phase 8-10**: Day 4 (4-7 hours)

Total: 3-5 days

---

*Generated: 2025-12-26*
