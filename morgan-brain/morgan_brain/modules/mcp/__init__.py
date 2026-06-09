"""MCP Hub — external integrations as MCP servers (calendar, email, search).

Responsibility: connect to configured MCP servers and expose their tools as unified adapters.
Hardened host: tool descriptions are sanitized, fingerprint-pinned (rug-pull defense),
allowlisted, and default-deny until granted. New integrations = config, not code.
Service: brain-api. Built. Enabled via MORGAN_ENABLE_MCP + MORGAN_MCP_SERVERS; optional
dependency: morgan-brain[mcp].

Files: hub.py, client.py, security.py.
"""
