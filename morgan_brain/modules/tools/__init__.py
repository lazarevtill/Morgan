"""Tools module — implements ``interfaces.ToolExecutor``.

Responsibility: register and run built-in tools behind the single PermissionGate (default-deny for
side-effecting tools). fetch_url is SSRF/DoS-hardened.
Service: brain-api — built and wired into the reasoning tool loop; exposed at GET/POST /api/tools.

Files: executor.py (ToolExecutorImpl), builtin/{calculator,clock_tool,memory_search,fetch_url}.py.
"""
