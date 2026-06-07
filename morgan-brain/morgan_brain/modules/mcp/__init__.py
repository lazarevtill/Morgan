"""MCP Hub — external integrations as MCP servers (calendar, email, search).

Responsibility: connect to configured MCP servers, discover their tools, expose unified adapters,
manage OAuth tokens (encrypted in Redis). New integrations = config, not code.
Service: brain-api. Phase: 3. Optional dependency: morgan-brain[mcp].

Planned files: hub.py, client.py, registry.py, adapters/{calendar,email,search}.py, auth/oauth.py.
"""
