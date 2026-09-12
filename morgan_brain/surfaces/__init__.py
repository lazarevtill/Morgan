"""Where requests come in: the ``morgan`` CLI and the ``morgan-mcp`` server.

Both are thin adapters over the same gate and the same command handlers -- a surface parses,
authorises and renders, and owns no memory logic of its own. ``network`` is the bind guard
that refuses to expose a listener beyond loopback without a real key.
"""
