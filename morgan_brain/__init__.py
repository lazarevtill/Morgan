"""Morgan — a project-scoped memory for your AI tools, consolidated into facts by a local model.

config        — the single MORGAN_-prefixed settings source
models        — the domain models; everything that persists is user- and project-keyed
memory        — one SQLite database behind one gate: stores, retrieval, consolidation
providers     — the chat and embedding adapters (the only place a model SDK is imported)
chat          — one chat turn: recall, answer, remember
composition   — opens the database and wires the above
cli           — the `morgan` terminal client
mcp_server    — the `morgan-mcp` server: the same operations for any MCP client
"""

__version__ = "0.2.0"
