"""morgan-brain — a personal assistant that knows and learns from you.

Package structure:
    config      — single MORGAN_-prefixed settings source
    interfaces  — Protocols every module implements (the contracts)
    models      — shared domain models, all user_id-keyed
    bus         — event bus (in-proc + Redis Streams, one interface)
    security    — MemoryGate + unified permissions
    modules     — perception, memory, learning, personalization, reasoning,
                  skills, tools, mcp, proactivity
    core        — thin cognitive-loop orchestrator
    apps        — brain_api, learning_worker, perception_gpu entrypoints
"""

__version__ = "0.1.0"
