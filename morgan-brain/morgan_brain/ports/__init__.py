"""Ports -- ways other software plugs into Morgan.

``mcp_server`` is the first port: other tools calling *in* to Morgan over the Model
Context Protocol (the opposite direction from the deleted ``modules/mcp/`` host/client
stub, which was Morgan calling *out* to other people's servers).
"""

from __future__ import annotations
